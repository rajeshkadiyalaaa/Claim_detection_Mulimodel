"""Small-sample check for B4 (round-1 revision; script added in round 2, RR1 N7).

Re-runs the learning curve of revision_stats.py for early fusion with shallower,
more regularised XGBoost models and with logistic regression on the same fused
features (CLIP + 64-component SVD of TF-IDF + standardised numbers). Same
stratified subsamples (seeds 1000-1009, ten draws per size) and the same 300
validation + test evaluation claims. No API calls.

Writes experiments/results/learning_curve_b4_regularised.json.

Reconstruction note: the original round-1 numbers were produced without a saved script.
This script reproduces them; the learning rate of the depth-2 model (0.1, against 0.05
for the others) was recovered by matching the cached file.
"""
import json

import numpy as np
from xgboost import XGBClassifier

import revision_stats as rs

CONFIGS = {
    # (kind, depth or C, trees, learning rate)
    "B4 depth6 n200 (paper)": ("xgb", 6, 200, 0.05),
    "B4 depth2 n50": ("xgb", 2, 50, 0.1),
    "B4 depth3 n100": ("xgb", 3, 100, 0.05),
    "LR on fused (C=0.1)": ("lr", 0.1, None, None),
    "LR on fused (C=1)": ("lr", 1, None, None),
}


def main(sizes=(25, 50, 100, 200), draws=10):
    y, split = rs.y, rs.split
    tr_all = np.where(split == "train")[0]
    ev = np.where(split != "train")[0]
    out = {}
    for n in sizes:
        out[str(n)] = {}
        for name, (kind, a, b, eta) in CONFIGS.items():
            accs = []
            for dsd in range(draws):
                r = np.random.default_rng(1000 + dsd)
                sub = np.concatenate([r.choice(tr_all[y[tr_all] == c], max(2, round(n * np.mean(y[tr_all] == c))), replace=False)
                                      for c in range(3)])
                Xtr, Xte = rs.build_baseline_X(sub, ev, "B4")
                m = rs.lr(a) if kind == "lr" else XGBClassifier(
                    n_estimators=b, max_depth=a, learning_rate=eta, subsample=0.9, colsample_bytree=0.9,
                    random_state=rs.SEED, eval_metric="mlogloss", n_jobs=4)
                accs.append((m.fit(Xtr, y[sub]).predict(Xte) == y[ev]).mean())
            out[str(n)][name] = round(float(np.mean(accs)), 3)
        print(n, out[str(n)], flush=True)
    (rs.RES / "learning_curve_b4_regularised.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
