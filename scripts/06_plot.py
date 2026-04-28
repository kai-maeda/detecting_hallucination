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

    # === Figure 2: grouped bar chart of consistency-level distribution ===
    # Consistency takes only 5 discrete values: {0.2, 0.4, 0.6, 0.8, 1.0}.
    # For each category, show % of items at each level, split by correct/hallucinated.
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(11, 3.5), sharey=True)
    if len(CATEGORIES) == 1:
        axes = [axes]
    bar_w = 0.38

    for ax, cat in zip(axes, CATEGORIES):
        cat_rows = [r for r in per_item if r["category"] == cat]
        cor = [r["consistency"] for r in cat_rows if r["modal_correct"]]
        hal = [r["consistency"] for r in cat_rows if not r["modal_correct"]]

        def pct_at(vals, level, tol=0.05):
            if not vals:
                return 0.0
            return 100.0 * sum(1 for v in vals if abs(v - level) < tol) / len(vals)

        cor_pct = [pct_at(cor, l) for l in levels]
        hal_pct = [pct_at(hal, l) for l in levels]

        x = np.arange(len(levels))
        ax.bar(x - bar_w/2, cor_pct, bar_w, color="#4C9AFF",
               label=f"Correct (n={len(cor)})", edgecolor="black", linewidth=0.5)
        ax.bar(x + bar_w/2, hal_pct, bar_w, color="#FF6B6B",
               label=f"Hallucinated (n={len(hal)})", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l:.1f}" for l in levels])
        ax.set_xlabel("Self-consistency score")
        ax.set_title(CAT_LABEL[cat], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("% of items at this consistency level")
    fig.suptitle("Per-category consistency distribution: Correct vs. Hallucinated answers",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = FIGURES_DIR / "consistency_bars.pdf"
    plt.savefig(fig2_path)
    plt.savefig(fig2_path.with_suffix(".png"), dpi=160)
    plt.close()
    print(f"Wrote {fig2_path}")

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
