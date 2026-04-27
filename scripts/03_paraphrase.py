"""Generate N paraphrases per question with GPT-4o-mini.

Resumable: re-running picks up where it left off.
Retries on rate-limit (429) and transient errors with exponential backoff.
Concurrent: uses a thread pool to actually exercise the org's RPM budget.
"""
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm

from src.config import (N_PARAPHRASES, PARAPHRASE_MODEL, PARAPHRASES_FILE,
                        SUBSET_FILE)

load_dotenv()

# Tier 1 gives 500 RPM for gpt-4o-mini. With ~1s latency per call,
# 30 workers ≈ 1800 RPM peak — too aggressive. 10 workers ≈ 600 RPM peak,
# which stays safely under the cap because most workers will be mid-flight.
WORKERS = 10
MAX_RETRIES = 6

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
            wait = min(60, (2 ** attempt) + random.random())
            time.sleep(wait)
            last_err = e
        except (APIError, APIConnectionError) as e:
            wait = min(30, (2 ** attempt) + random.random())
            time.sleep(wait)
            last_err = e
    raise RuntimeError(f"paraphrase failed after {MAX_RETRIES} attempts: {last_err}")


def process_one(client: OpenAI, item: dict) -> dict:
    try:
        ps = paraphrase(client, item["question"], N_PARAPHRASES)
        ok = True
    except Exception:
        ps = [item["question"]] * N_PARAPHRASES
        ok = False
    item_out = dict(item)
    item_out["prompts"] = [item["question"]] + ps
    item_out["_paraphrase_ok"] = ok
    return item_out


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; create a .env from .env.example")

    items = json.loads(SUBSET_FILE.read_text())
    client = OpenAI()

    # Resume: load anything we already have, but only count REAL paraphrases as done.
    done_by_id = {}
    if PARAPHRASES_FILE.exists():
        try:
            for r in json.loads(PARAPHRASES_FILE.read_text()):
                if "prompts" not in r or len(r["prompts"]) < N_PARAPHRASES + 1:
                    continue
                # Drop fallbacks (5 copies of the original) so we re-generate them.
                if all(p == r["question"] for p in r["prompts"][1:]):
                    continue
                done_by_id[r["id"]] = r
        except Exception:
            done_by_id = {}
    if done_by_id:
        print(f"Resuming: {len(done_by_id)} real paraphrases already done")

    out = list(done_by_id.values())
    pending = [it for it in items if it["id"] not in done_by_id]
    print(f"Pending: {len(pending)} items, using {WORKERS} workers")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process_one, client, it): it["id"] for it in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="paraphrasing"):
            result = fut.result()
            with lock:
                out.append(result)
                # Periodic checkpoint (every 10 items) to avoid a write per call
                if len(out) % 10 == 0:
                    PARAPHRASES_FILE.write_text(json.dumps(out, indent=2, default=str))

    # Final write
    PARAPHRASES_FILE.write_text(json.dumps(out, indent=2, default=str))
    failed = sum(1 for r in out if not r.get("_paraphrase_ok", True))
    print(f"Wrote {len(out)} items -> {PARAPHRASES_FILE}")
    if failed:
        print(f"  WARN: {failed} items still fell back after retries; consider rerunning")


if __name__ == "__main__":
    main()
