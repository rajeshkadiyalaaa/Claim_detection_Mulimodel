"""Publication figures for the Results section (test split, n = 150).

fig_confusion.pdf     : confusion matrices for v1, v2 (post-hoc), B4
(The risk-coverage figure was dropped from the paper; AURC values are in results/revision_stats.json.)
Palette: validated categorical slots 1-5 (light mode); lines also differ by dash
pattern and carry direct labels, so identity never relies on colour alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from final_test import LABELS, combiner, load, ordered, ALL
from v2_test import V2_FEATS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript/figures"
INK, INK2, GRID = "#1a1a19", "#5c5b55", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUES = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MM = 1 / 25.4

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def risk_curve(conf, correct):
    order = np.argsort(-np.asarray(conf), kind="stable")
    err = 1 - np.asarray(correct, float)[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def main():
    rows = [json.loads(l) for l in (ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
    vids = [r["sample_id"] for r in rows if r["split"] == "val"]
    tids = [r["sample_id"] for r in rows if r["split"] == "test"]
    yt = np.array([LABELS.index(r["decision_label"]) for r in rows if r["split"] == "test"])

    _, P1, _ = combiner(load("decision1_train"), ordered(load("decision1_val"), vids), ordered(load("decision1_test"), tids), ALL)
    _, P2, _ = combiner(load("v2_train"), ordered(load("v2_val"), vids), ordered(load("v2_test"), tids), V2_FEATS)
    bl = json.loads((ROOT / "experiments/results/baselines_val_test.json").read_text())["models"]
    probs = {
        "Proposed v1": P1,
        "Proposed v2 (post-hoc)": P2,
        "B4 early fusion": np.array(bl["B4 early fusion (XGBoost)"]["test_proba"]),
        "B1 text-only": np.array(bl["B1 text-only (TF-IDF + LR)"]["test_proba"]),
        "B3 image-only": np.array(bl["B3 image-only (CLIP + LR)"]["test_proba"]),
    }

    # Figure 2: confusion matrices
    names = ["Proposed v1", "Proposed v2 (post-hoc)", "B4 early fusion"]
    short = ["Approve", "Audit", "Escalate"]
    fig, axes = plt.subplots(1, 3, figsize=(131 * MM, 48 * MM), sharey=True)
    cmap = matplotlib.colors.ListedColormap(BLUES)
    for ax, name in zip(axes, names):
        pred = probs[name].argmax(1)
        cm = np.zeros((3, 3), int)
        for a, b in zip(yt, pred):
            cm[a, b] += 1
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), short, fontsize=7)
        ax.set_yticks(range(3), short, fontsize=7)
        ax.set_title(f"{name}\naccuracy {np.mean(pred == yt):.3f}", fontsize=7.5, color=INK)
        ax.set_xlabel("Predicted", fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True", fontsize=7)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig_confusion.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "fig_confusion.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    print("figure written: fig_confusion")


if __name__ == "__main__":
    main()
