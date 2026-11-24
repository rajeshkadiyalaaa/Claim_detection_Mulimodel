"""Pilot analysis on the validation split.

- Decision 1 (zero-shot) metrics from experiments/results/decision1_val.jsonl
- Baselines B1 (text TF-IDF + logistic regression) and B2 (numbers, XGBoost),
  trained on the training split and evaluated on the validation split.
- Gate behaviour with thresholds searched on the same validation split.
  NOTE: pilot gate numbers are in-sample (thresholds chosen and scored on val);
  the paper's protocol selects thresholds on val and reports on test.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Auto-Approve", "Reject_Fraud_Audit", "Escalate_Human"]
SHORT = {"Auto-Approve": "A", "Reject_Fraud_Audit": "R", "Escalate_Human": "E"}
NUM = ["total_claim_amount", "vehicle_claim", "injury_claim", "property_claim", "policy_deductable",
       "policy_annual_premium", "age", "months_as_customer", "auto_year", "bodily_injuries", "witnesses"]
COST = {"FA": 10, "FR": 4, "H": 1}
PRICE_IN = 0.10 / 1e6  # USD per input token, Decision 1 (Google launch blog, 11 Nov 2026)
SEED = 42


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


def brier(P, y):
    Y = np.eye(3)[[LABELS.index(v) for v in y]]
    return float(np.mean(np.sum((np.asarray(P) - Y) ** 2, axis=1)))


def cost(y, g):
    if g == y:
        return 0
    return {"Auto-Approve": COST["FA"], "Reject_Fraud_Audit": COST["FR"], "Escalate_Human": COST["H"]}[g]


def gate(pred, conf, supp, tc, ts):
    if pred in ("Auto-Approve", "Reject_Fraud_Audit") and conf >= tc and supp >= ts:
        return pred
    return "Escalate_Human"


def gate_search(y, pred, conf, supp, use_support):
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    best = None
    for tc in grid:
        for ts in (grid if use_support else [0.0]):
            g = [gate(p, c, s, tc, ts) for p, c, s in zip(pred, conf, supp)]
            mean_cost = np.mean([cost(a, b) for a, b in zip(y, g)])
            coverage = np.mean([x != "Escalate_Human" for x in g])
            key = (mean_cost, -coverage)
            if best is None or key < best[0]:
                best = (key, tc, ts, g)
    (_, tc, ts, g) = best
    return tc, ts, g


def report(name, y, pred, P=None, conf=None):
    r = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro"),
        **{f"recall_{SHORT[l]}": recall_score(y, pred, labels=[l], average="macro", zero_division=0) for l in LABELS},
    }
    if P is not None:
        correct = [a == b for a, b in zip(y, pred)]
        r["ece"] = ece(conf, correct)
        r["brier"] = brier(P, y)
    return r


def main():
    rows = [json.loads(l) for l in (ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    yv = [r["decision_label"] for r in va]

    # Decision 1
    d1 = {o["sample_id"]: o for o in map(json.loads, (ROOT / "experiments/results/decision1_val.jsonl").read_text().splitlines())}
    d = [d1[r["sample_id"]] for r in va]
    P = [[o["p_A"], o["p_R"], o["p_E"]] for o in d]
    pred = [o["pred"] for o in d]
    conf = [o["conf"] for o in d]
    results = [report("Decision 1 (q1 only, no gate)", yv, pred, P, conf)]

    # B1 text-only
    tf = TfidfVectorizer(ngram_range=(1, 2)).fit([r["claim_description"] for r in tr])
    b1 = LogisticRegression(max_iter=5000, random_state=SEED).fit(tf.transform([r["claim_description"] for r in tr]), [r["decision_label"] for r in tr])
    P1 = b1.predict_proba(tf.transform([r["claim_description"] for r in va]))
    P1 = P1[:, [list(b1.classes_).index(l) for l in LABELS]]
    pred1 = [LABELS[i] for i in P1.argmax(1)]
    results.append(report("B1 text-only (TF-IDF + LR)", yv, pred1, P1, P1.max(1)))

    # B2 numbers-only
    Xtr = np.array([[r[k] for k in NUM] for r in tr], float)
    Xva = np.array([[r[k] for k in NUM] for r in va], float)
    ytr = np.array([LABELS.index(r["decision_label"]) for r in tr])
    b2 = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                       colsample_bytree=0.9, random_state=SEED, eval_metric="mlogloss").fit(Xtr, ytr)
    P2 = b2.predict_proba(Xva)
    pred2 = [LABELS[i] for i in P2.argmax(1)]
    results.append(report("B2 numbers-only (XGBoost)", yv, pred2, P2, P2.max(1)))

    # Gates (in-sample on val)
    sA = [min(o["s2"], o["s3"], o["s4"], o["s5"]) for o in d]
    sR = [min(o["s3"], max(1 - o["s2"], 1 - o["s5"])) for o in d]
    supp = [a if p == "Auto-Approve" else (b if p == "Reject_Fraud_Audit" else 1.0) for p, a, b in zip(pred, sA, sR)]
    gates = []
    for nm, pr, cf, sp, use in [
        ("Decision 1 + confidence gate", pred, conf, supp, False),
        ("Decision 1 + confidence & support gate", pred, conf, supp, True),
        ("B1 + confidence gate", pred1, list(P1.max(1)), [1] * len(va), False),
        ("B2 + confidence gate", pred2, list(P2.max(1)), [1] * len(va), False),
    ]:
        tc, ts, g = gate_search(yv, pr, cf, sp, use)
        auto = [i for i, x in enumerate(g) if x != "Escalate_Human"]
        gates.append({
            "policy": nm, "tau_c": tc, "tau_s": ts,
            "mean_cost": float(np.mean([cost(a, b) for a, b in zip(yv, g)])),
            "accuracy": accuracy_score(yv, g),
            "macro_f1": f1_score(yv, g, labels=LABELS, average="macro"),
            "auto_coverage": len(auto) / len(va),
            "auto_error_rate": (float(np.mean([g[i] != yv[i] for i in auto])) if auto else 0.0),
            "wrong_approvals": sum(1 for a, b in zip(yv, g) if b == "Auto-Approve" and a != b),
        })

    # Efficiency
    lat = np.array([o["latency_s"] for o in d])
    tok = np.array([o["input_tokens"] for o in d])
    eff = {
        "latency_median_s": float(np.median(lat)), "latency_p90_s": float(np.percentile(lat, 90)),
        "latency_mean_s": float(lat.mean()),
        "input_tokens_mean": float(tok.mean()),
        "image_tokens_mean": float(np.mean([o["image_tokens"] for o in d])),
        "output_tokens_total": int(sum(o["output_tokens"] for o in d)),
        "cost_per_decision_usd": float(tok.mean() * PRICE_IN),
        "cost_per_1000_claims_usd": float(tok.mean() * PRICE_IN * 1000),
    }

    cm = confusion_matrix(yv, pred, labels=LABELS).tolist()
    check_means = {l: {k: float(np.mean([o[k] for o in d if o["label"] == l])) for k in ("s2", "s3", "s4", "s5")} for l in LABELS}

    out = {"n_val": len(va), "metrics": results, "gates_in_sample": gates, "efficiency": eff,
           "decision1_confusion_rows_true_cols_pred": {"labels": LABELS, "matrix": cm},
           "check_score_means_by_true_label": check_means}
    (ROOT / "experiments/results/pilot_val_summary.json").write_text(json.dumps(out, indent=2))

    print(f"n_val = {len(va)}\n")
    print(f"{'Model':38} {'Acc':>6} {'MacF1':>6} {'Rec A':>6} {'Rec R':>6} {'Rec E':>6} {'ECE':>6} {'Brier':>6}")
    for r in results:
        print(f"{r['model']:38} {r['accuracy']:6.3f} {r['macro_f1']:6.3f} {r['recall_A']:6.3f} {r['recall_R']:6.3f} {r['recall_E']:6.3f} {r.get('ece', float('nan')):6.3f} {r.get('brier', float('nan')):6.3f}")
    print("\nDecision 1 confusion (rows true A,R,E; cols pred A,R,E):", cm)
    print("\nMean check scores by true label:")
    for l, v in check_means.items():
        print(f"  {l:20} " + "  ".join(f"{k}={x:.2f}" for k, x in v.items()))
    print("\nGates (thresholds chosen on val, scored on val: optimistic):")
    for g in gates:
        print(f"  {g['policy']:40} tau_c={g['tau_c']:.2f} tau_s={g['tau_s']:.2f} cost={g['mean_cost']:.3f} "
              f"acc={g['accuracy']:.3f} macroF1={g['macro_f1']:.3f} coverage={g['auto_coverage']:.2f} "
              f"auto_err={g['auto_error_rate']:.3f} wrong_approvals={g['wrong_approvals']}")
    print("\nEfficiency:", json.dumps(eff, indent=1))


if __name__ == "__main__":
    main()
