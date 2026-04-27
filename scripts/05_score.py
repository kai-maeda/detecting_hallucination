"""Compute consistency, correctness, and AUROC of consistency-as-detector per category."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import CATEGORIES, INFERENCE_FILE, SCORES_FILE
from src.normalize import consistency_score, is_correct


def main():
    items = json.loads(INFERENCE_FILE.read_text())

    rows = []
    for item in items:
        cat = item["category"]
        answers = item["answers"]
        gt = item["ground_truth"]
        opts = item.get("options")

        cons, modal = consistency_score(answers, cat)
        modal_correct = is_correct(modal, gt, cat, opts)
        per_prompt_correct = [is_correct(a, gt, cat, opts) for a in answers]
        accuracy = float(np.mean(per_prompt_correct))

        rows.append({
            "id": item["id"],
            "category": cat,
            "ground_truth": gt,
            "answers": answers,
            "modal_answer": modal,
            "consistency": cons,
            "modal_correct": modal_correct,
            "per_prompt_correct": per_prompt_correct,
            "accuracy": accuracy,
        })

    # Per-category aggregates
    per_cat = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        consistencies = np.array([r["consistency"] for r in cat_rows])
        correct = np.array([r["modal_correct"] for r in cat_rows], dtype=int)
        accuracy = float(np.mean(correct))

        # AUROC: can consistency separate correct (label=1) from hallucinated (label=0)?
        if 0 < correct.sum() < len(correct):
            auroc = float(roc_auc_score(correct, consistencies))
        else:
            auroc = None

        gap = float(consistencies[correct == 1].mean() - consistencies[correct == 0].mean()) \
            if 0 < correct.sum() < len(correct) else None

        per_cat[cat] = {
            "n": len(cat_rows),
            "modal_accuracy": accuracy,
            "mean_consistency_correct": float(consistencies[correct == 1].mean()) if correct.sum() else None,
            "mean_consistency_hallucinated": float(consistencies[correct == 0].mean()) if (len(correct) - correct.sum()) else None,
            "consistency_gap": gap,
            "auroc_consistency_vs_correct": auroc,
        }

    out = {"per_item": rows, "per_category": per_cat}
    SCORES_FILE.write_text(json.dumps(out, indent=2, default=str))

    print("\nPer-category results:")
    print(f"{'Category':<22}{'N':>4}{'Acc':>8}{'C|cor':>8}{'C|hal':>8}{'Gap':>8}{'AUROC':>8}")
    for cat, m in per_cat.items():
        def f(x): return f"{x:.3f}" if x is not None else "  -  "
        print(f"{cat:<22}{m['n']:>4}{f(m['modal_accuracy']):>8}"
              f"{f(m['mean_consistency_correct']):>8}{f(m['mean_consistency_hallucinated']):>8}"
              f"{f(m['consistency_gap']):>8}{f(m['auroc_consistency_vs_correct']):>8}")
    print(f"\nWrote {SCORES_FILE}")


if __name__ == "__main__":
    main()
