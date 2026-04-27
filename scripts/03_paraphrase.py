"""Generate N paraphrases per question with GPT-4o-mini.

Resumable: re-running picks up where it left off.
Retries on rate-limit (429) and transient errors with exponential backoff.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm

from src.config import (N_PARAPHRASES, PARAPHRASE_MODEL, PARAPHRASES_FILE,
                        SUBSET_FILE)

load_dotenv()

PROMPT = """You are a paraphrasing assistant for a research evaluation.

Given a question about a video scene, produce {n} paraphrases that:
- Preserve the EXACT semantic meaning (the correct answer must be unchanged).
- Use clearly different surface forms (word choice, syntax, sentence structure).
- Are natural English questions.
- Do NOT add or remove any constraints, objects, or qualifiers.

Original question: {question}

Output ONLY a JSON object of the form:
{{"paraphrases": ["paraphrase 1", "paraphrase 2", ...]}}
with exactly {n} entries."""

MAX_RETRIES = 6


def paraphrase(client: OpenAI, question: str, n: int) -> list[str]:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=PARAPHRASE_MODEL,
                messages=[{"role": "user",
                           "content": PROMPT.format(n=n, question=question)}],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            content = resp.choices[0].message.content
            obj = json.loads(content)
            ps = obj.get("paraphrases", [])
            if len(ps) < n:
                ps = ps + [question] * (n - len(ps))
            return ps[:n]
        except RateLimitError as e:
            # OpenAI suggests waiting; back off with jitter.
            wait = min(60, (2 ** attempt) + random.random())
            print(f"  rate-limited; sleeping {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            last_err = e
        except (APIError, APIConnectionError) as e:
            wait = min(30, (2 ** attempt) + random.random())
            print(f"  transient API error; sleeping {wait:.1f}s: {e}")
            time.sleep(wait)
            last_err = e
    raise RuntimeError(f"paraphrase failed after {MAX_RETRIES} attempts: {last_err}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; create a .env from .env.example")

    items = json.loads(SUBSET_FILE.read_text())
    client = OpenAI()

    # Resume: load anything we already have
    done_by_id = {}
    if PARAPHRASES_FILE.exists():
        try:
            for r in json.loads(PARAPHRASES_FILE.read_text()):
                if "prompts" in r and len(r["prompts"]) >= N_PARAPHRASES + 1:
                    done_by_id[r["id"]] = r
        except Exception:
            done_by_id = {}
    if done_by_id:
        print(f"Resuming: {len(done_by_id)} items already done")

    out = list(done_by_id.values())
    pending = [it for it in items if it["id"] not in done_by_id]

    for item in tqdm(pending, desc="paraphrasing"):
        try:
            ps = paraphrase(client, item["question"], N_PARAPHRASES)
        except Exception as e:
            print(f"  WARN paraphrase failed for {item['id']} after retries: {e}")
            ps = [item["question"]] * N_PARAPHRASES
        item_out = dict(item)
        item_out["prompts"] = [item["question"]] + ps
        out.append(item_out)
        # Checkpoint after every item so we never lose progress
        PARAPHRASES_FILE.write_text(json.dumps(out, indent=2, default=str))

    print(f"Wrote {len(out)} items with {N_PARAPHRASES + 1} prompts each -> {PARAPHRASES_FILE}")


if __name__ == "__main__":
    main()
