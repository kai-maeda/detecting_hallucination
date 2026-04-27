"""Generate N paraphrases per question with GPT-4o-mini."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
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


def paraphrase(client: OpenAI, question: str, n: int) -> list[str]:
    resp = client.chat.completions.create(
        model=PARAPHRASE_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(n=n, question=question)}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = resp.choices[0].message.content
    obj = json.loads(content)
    paraphrases = obj.get("paraphrases", [])
    if len(paraphrases) < n:
        paraphrases = paraphrases + [question] * (n - len(paraphrases))
    return paraphrases[:n]


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; create a .env from .env.example")

    items = json.loads(SUBSET_FILE.read_text())
    client = OpenAI()

    out = []
    for item in tqdm(items, desc="paraphrasing"):
        try:
            ps = paraphrase(client, item["question"], N_PARAPHRASES)
        except Exception as e:
            print(f"  WARN paraphrase failed for {item['id']}: {e}")
            ps = [item["question"]] * N_PARAPHRASES
        # First "paraphrase" is the original, then N variants
        item_out = dict(item)
        item_out["prompts"] = [item["question"]] + ps
        out.append(item_out)

    PARAPHRASES_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {len(out)} items with {N_PARAPHRASES + 1} prompts each -> {PARAPHRASES_FILE}")


if __name__ == "__main__":
    main()
