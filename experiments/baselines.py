"""Supervised baselines B1-B4, trained on TRAIN, hyperparameters chosen on VAL by log loss.

B1 text only      : TF-IDF (1-2 grams) of claim_description + multinomial logistic regression
B2 numbers only   : XGBoost on the 11 numerical fields
B3 image only     : frozen CLIP ViT-B/32 embedding + multinomial logistic regression
B4 early fusion   : XGBoost on [CLIP embedding | truncated-SVD(64) of TF-IDF | standardised numbers]

Writes validation probabilities to experiments/results/baselines_val.json and
can also score the test split with --test (used only for the final run).
"""
import argparse
import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from decision1 import DATA, ROOT
from pilot_analysis import LABELS, NUM

SEED = 42


def xgb(depth, n):
    return XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, random_state=SEED, eval_metric="mlogloss", n_jobs=4)


def select(make_models, Xtr, ytr, Xva, yva):
    best = None
    for desc, m in make_models():
        m.fit(Xtr, ytr)
        ll = log_loss(yva, m.predict_proba(Xva), labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, desc, m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also score the test split (final run only)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    clip = np.load(ROOT / "experiments/cache/clip_vitb32.npz")
    emb = dict(zip(clip["ids"], clip["emb"]))
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    y = {s: np.array([LABELS.index(r["decision_label"]) for r in split[s]]) for s in split}

    tf = TfidfVectorizer(ngram_range=(1, 2)).fit([r["claim_description"] for r in split["train"]])
    T = {s: tf.transform([r["claim_description"] for r in split[s]]) for s in split}
    sc = StandardScaler().fit([[r[k] for k in NUM] for r in split["train"]])
    N_raw = {s: np.array([[r[k] for k in NUM] for r in split[s]], float) for s in split}
    N = {s: sc.transform(N_raw[s]) for s in split}
    I = {s: np.array([emb[r["sample_id"]] for r in split[s]]) for s in split}
    svd = TruncatedSVD(64, random_state=SEED).fit(T["train"])
    F = {s: np.hstack([I[s], svd.transform(T[s]), N[s]]) for s in split}

    lr_grid = lambda: [(f"C={C}", LogisticRegression(C=C, max_iter=10000, random_state=SEED)) for C in (0.01, 0.1, 1, 10, 100)]
    xgb_grid = lambda: [(f"depth={d},n={n}", xgb(d, n)) for d in (3, 4, 6) for n in (200, 400)]

    specs = {
        "B1 text-only (TF-IDF + LR)": (T, lr_grid),
        "B2 numbers-only (XGBoost)": (N_raw, xgb_grid),
        "B3 image-only (CLIP + LR)": (I, lr_grid),
        "B4 early fusion (XGBoost)": (F, xgb_grid),
    }
    out = {"labels": LABELS, "val_ids": [r["sample_id"] for r in split["val"]], "models": {}}
    for name, (X, grid) in specs.items():
        ll, desc, m = select(grid, X["train"], y["train"], X["val"], y["val"])
        entry = {"selected": desc, "val_logloss": ll, "val_proba": m.predict_proba(X["val"]).tolist()}
        if a.test:
            entry["test_proba"] = m.predict_proba(X["test"]).tolist()
            out["test_ids"] = [r["sample_id"] for r in split["test"]]
        out["models"][name] = entry
        print(f"{name:30} selected {desc:16} val log loss {ll:.3f}")
    fname = "baselines_val_test.json" if a.test else "baselines_val.json"
    (ROOT / "experiments/results" / fname).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
