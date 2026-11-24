"""Round-2 learning-curve figure from results/round2_stats.json.

fig_learning_curve.pdf: (a) accuracy and (b) gated misrouting cost (10,4,1) against the
number of labelled training claims, with the rule-aware comparators (RR1 N2, DA N1, R2 N3).
Palette: validated categorical slots (same as make_figures.py); lines also differ by dash and marker.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript/figures"
R = json.loads((ROOT / "experiments/results/round2_stats.json").read_text())["learning_curve"]
INK, INK2, GRID, REF = "#1a1a19", "#5c5b55", "#e6e5e0", "#8a8983"
SERIES = [  # key, label, colour, dash, marker
    ("v1", "Proposed (Decision 1 + combiner)", "#2a78d6", "-", "o"),
    ("b5fair", "B5-fair + combiner", "#6250d6", (0, (3, 1, 1, 1, 1, 1)), "v"),
    ("B4", "B4 early fusion", "#1baf7a", (0, (4, 1.5)), "s"),
    ("rule_lr_text_img", "Rule-aware LR: text + image stats", "#eda100", (0, (1, 1.2)), "^"),
    ("rule_lr_text", "Rule-aware LR: text only", "#e87ba4", (0, (5, 1.5, 1, 1.5)), "D"),
]
MM = 1 / 25.4
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})

fig, axes = plt.subplots(1, 2, figsize=(131 * MM, 92 * MM))
for ax, metric, ylab, lim in [(axes[0], "acc", "Accuracy", (0.55, 0.95)),
                              (axes[1], "cost", "Mean gated cost, weights (10,4,1)", (0.4, 1.05))]:
    for key, lab, col, dash, mk in SERIES:
        v = R[key]
        n = np.array([int(x) for x in v])
        m = np.array([v[x][metric]["mean"] for x in v])
        sd = np.array([v[x][metric]["sd"] for x in v])
        ax.plot(n, m, color=col, ls=dash, marker=mk, ms=3.5, lw=1.4, label=lab)
        if metric == "acc":
            ax.fill_between(n, m - sd, m + sd, color=col, alpha=0.10, lw=0)
    if metric == "acc":
        ref = R["text_rule (0 labels)"]["acc"]
        ax.axhline(ref, color=REF, lw=1, ls=(0, (2, 2)))
        ax.text(700, ref - 0.006, "text rule, 0 labels", color=INK2, fontsize=6.5, ha="right", va="top")
    if metric == "cost":
        esc = R["escalate_all"]["cost"]
        ax.axhline(esc, color=REF, lw=1, ls="-")
        ax.text(700, esc + 0.01, "escalate every claim", color=INK2, fontsize=6.5, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 400, 700], ["25", "50", "100", "200", "400", "700"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Labelled training claims")
    ax.set_ylabel(ylab)
    ax.set_ylim(*lim)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_title("(a) Accuracy", fontsize=8, loc="left", color=INK)
axes[1].set_title("(b) Gated cost, lower is better", fontsize=8, loc="left", color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.0)
fig.savefig(OUT / "fig_learning_curve.pdf")
fig.savefig(OUT / "fig_learning_curve.png", dpi=200)
print("figure written")
