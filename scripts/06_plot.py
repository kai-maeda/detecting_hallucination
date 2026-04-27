"""Generate the figures and LaTeX table for the report."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.config import CATEGORIES, FIGURES_DIR, SCORES_FILE

CAT_LABEL = {
    "object_counting": "Object Count",
    "object_rel_distance": "Relative Distance",
    "object_rel_direction_easy": "Relative Direction (Easy)",
    "object_rel_direction_medium": "Relative Direction",
    "object_rel_direction_hard": "Relative Direction (Hard)",
}


def main():
    data = json.loads(SCORES_FILE.read_text())
    per_item = data["per_item"]
    per_cat = data["per_category"]

    # === Figure 1: violin/box of consistency, split by correctness, per category ===
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = []
    boxes_correct, boxes_hallu = [], []
    labels = []
    for i, cat in enumerate(CATEGORIES):
        cat_rows = [r for r in per_item if r["category"] == cat]
        c_cor = [r["consistency"] for r in cat_rows if r["modal_correct"]]
        c_hal = [r["consistency"] for r in cat_rows if not r["modal_correct"]]
        boxes_correct.append(c_cor if c_cor else [0])
        boxes_hallu.append(c_hal if c_hal else [0])
        labels.append(CAT_LABEL[cat])

    x = np.arange(len(CATEGORIES))
    w = 0.35
    bp1 = ax.boxplot(boxes_correct, positions=x - w/2, widths=w, patch_artist=True,
                     boxprops=dict(facecolor="#4C9AFF"), medianprops=dict(color="black"))
    bp2 = ax.boxplot(boxes_hallu, positions=x + w/2, widths=w, patch_artist=True,
                     boxprops=dict(facecolor="#FF6B6B"), medianprops=dict(color="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Self-consistency score")
    ax.set_title("Consistency of grounded vs. hallucinated answers (Qwen2.5-VL-7B on VSI-Bench)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Correct (grounded)", "Hallucinated"],
              loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "consistency_box.pdf"
    plt.savefig(fig_path)
    plt.savefig(fig_path.with_suffix(".png"), dpi=160)
    plt.close()
    print(f"Wrote {fig_path}")

    # === LaTeX table ===
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Category & N & Acc. & C(correct) & C(halluc.) & AUROC \\",
        r"\midrule",
    ]
    for cat in CATEGORIES:
        m = per_cat.get(cat, {})
        def f(x, p=2): return f"{x:.{p}f}" if x is not None else "--"
        lines.append(
            f"{CAT_LABEL[cat]} & {m.get('n','--')} & {f(m.get('modal_accuracy'))} & "
            f"{f(m.get('mean_consistency_correct'))} & {f(m.get('mean_consistency_hallucinated'))} & "
            f"{f(m.get('auroc_consistency_vs_correct'))} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-category accuracy and self-consistency. C(correct) and C(halluc.) are mean consistency scores for correctly answered and hallucinated items respectively. AUROC measures whether consistency separates correct from hallucinated answers.}",
        r"\label{tab:results}",
        r"\end{table}",
    ]
    table_path = FIGURES_DIR / "results_table.tex"
    table_path.write_text("\n".join(lines))
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
