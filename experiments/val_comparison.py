"""Validation-stage comparison of the proposed method, baselines and ablations.

All supervised parts are trained on TRAIN; hyperparameters and gate thresholds
are chosen on VAL, so the gated numbers below are in-sample for VAL and are used
only for model development. Final numbers are produced once on TEST.

Prices (USD per 1M tokens, Google AI pricing page, read 24 Nov 2026):
  gemini-decision-1: input 0.10, output n/a (0 output tokens)
  gemini-3.8-flash : input 0.75, output 3.75 (output price includes thinking tokens)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from pilot_analysis import LABELS, cost, gate_search, report

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
ALL = ["p_A", "p_R", "p_E", "s2", "s3", "s4", "s5"]
PRICE = {"d1_in": 0.10e-6, "flash_in": 0.75e-6, "flash_out": 3.75e-6}
SEED = 42


def load(name):
    f = RES / f"{name}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else None


def support(rows, pred):
    out = []
    for o, p in zip(rows, pred):
        if p == "Auto-Approve":
            out.append(min(o["s2"], o["s3"], o["s4"], o["s5"]))
        elif p == "Reject_Fraud_Audit":
            out.append(min(o["s3"], max(1 - o["s2"], 1 - o["s5"])))
        else:
            out.append(1.0)
    return out


def fit_combiner(tr, va, feats):
    Xtr = np.array([[o[f] for f in feats] for o in tr]); ytr = [LABELS.index(o["label"]) for o in tr]
    Xva = np.array([[o[f] for f in feats] for o in va]); yva = [LABELS.index(o["label"]) for o in va]
    best = None
    for C in (0.01, 0.1, 1, 10, 100):
        m = LogisticRegression(C=C, max_iter=10000, random_state=SEED).fit(Xtr, ytr)
        ll = log_loss(yva, m.predict_proba(Xva), labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, C, m)
    return best[2].predict_proba(Xva), best[1]


def evaluate(name, y, P=None, pred=None, conf=None, supp=None, extra=None):
    if P is not None:
        P = np.asarray(P)
        pred = [LABELS[i] for i in P.argmax(1)]
        conf = list(P.max(1))
    r = report(name, y, pred, P, conf) if P is not None else report(name, y, pred)
    if P is None and conf is not None:
        from pilot_analysis import ece
        r["ece"] = ece(conf, [a == b for a, b in zip(y, pred)])
    r["cost_ungated"] = float(np.mean([cost(a, b) for a, b in zip(y, pred)]))
    sp = supp if supp is not None else [1.0] * len(y)
    tc, ts, g = gate_search(y, pred, conf, sp, supp is not None)
    auto = [i for i, x in enumerate(g) if x != "Escalate_Human"]
    r.update({
        "tau_c": tc, "tau_s": ts, "cost_gated": float(np.mean([cost(a, b) for a, b in zip(y, g)])),
        "coverage": len(auto) / len(y),
        "auto_error": float(np.mean([g[i] != y[i] for i in auto])) if auto else 0.0,
        "wrong_approvals": sum(1 for a, b in zip(y, g) if b == "Auto-Approve" and a != b),
    })
    if extra:
        r.update(extra)
    return r


def main():
    tr, va = load("decision1_train"), load("decision1_val")
    ids = [o["sample_id"] for o in va]
    y = [o["label"] for o in va]
    rows = []

    # Proposed method
    P, C = fit_combiner(tr, va, ALL)
    pred = [LABELS[i] for i in np.asarray(P).argmax(1)]
    lat = [o["latency_s"] for o in va]
    tok = [o["input_tokens"] for o in va]
    eff = {"latency_median_s": float(np.median(lat)), "tokens_in_mean": float(np.mean(tok)), "tokens_out_mean": 0.0,
           "usd_per_1000": float(np.mean(tok) * PRICE["d1_in"] * 1000)}
    rows.append(evaluate("Proposed: Decision 1 + combiner + gate", y, P=P, supp=support(va, pred), extra=eff))
    rows.append(evaluate("Proposed, confidence-only gate", y, P=P))

    # Baselines B1-B4
    bl = json.loads((RES / "baselines_val.json").read_text())
    assert bl["val_ids"] == ids
    for name, e in bl["models"].items():
        rows.append(evaluate(name, y, P=e["val_proba"], extra={"selected": e["selected"]}))

    # B5 generative
    b5 = load("b5_gemini-3.8-flash_val")
    if b5:
        d = {o["sample_id"]: o for o in b5}
        if all(i in d for i in ids):
            o5 = [d[i] for i in ids]
            tin = np.mean([o["input_tokens"] for o in o5])
            tout = np.mean([(o["output_tokens"] or 0) + (o["thinking_tokens"] or 0) for o in o5])
            rows.append(evaluate("B5 generative (Gemini 3.8 Flash, JSON)", y, pred=[o["pred"] for o in o5],
                                 conf=[o["conf"] for o in o5], extra={
                                     "latency_median_s": float(np.median([o["latency_s"] for o in o5])),
                                     "tokens_in_mean": float(tin), "tokens_out_mean": float(tout),
                                     "usd_per_1000": float((tin * PRICE["flash_in"] + tout * PRICE["flash_out"]) * 1000)}))

    # Question ablations (same API responses, combiner refit on feature subsets)
    for name, feats in [("Ablation: q1 only, no combiner", None),
                        ("Ablation: combiner on q1 only", ["p_A", "p_R", "p_E"]),
                        ("Ablation: combiner on checks only", ["s2", "s3", "s4", "s5"])] + \
                       [(f"Ablation: drop {f}", [x for x in ALL if x != f]) for f in ("s2", "s3", "s4", "s5")]:
        if feats is None:
            rows.append(evaluate(name, y, P=[[o["p_A"], o["p_R"], o["p_E"]] for o in va]))
        else:
            Pq, _ = fit_combiner(tr, va, feats)
            rows.append(evaluate(name, y, P=Pq))

    # Modality ablations (separate API runs with one modality removed)
    for drop in ("image", "text", "numbers"):
        a_tr, a_va = load(f"decision1_train_no{drop}"), load(f"decision1_val_no{drop}")
        if a_tr and a_va and len(a_tr) == 700 and len(a_va) == 150:
            d = {o["sample_id"]: o for o in a_va}
            a_va = [d[i] for i in ids]
            Pm, _ = fit_combiner(a_tr, a_va, ALL)
            predm = [LABELS[i] for i in np.asarray(Pm).argmax(1)]
            rows.append(evaluate(f"Ablation: no {drop} in request", y, P=Pm, supp=support(a_va, predm)))

    (RES / "val_comparison.json").write_text(json.dumps({"combiner_C": C, "rows": rows}, indent=2))
    hdr = f"{'Model':42} {'Acc':>5} {'MF1':>5} {'ECE':>5} {'Brier':>5} {'Cost':>5} {'GCost':>5} {'Cov':>4} {'AErr':>5} {'WA':>3} {'Lat(s)':>6} {'$/1k':>6}"
    print(hdr)
    for r in rows:
        print(f"{r['model'][:42]:42} {r['accuracy']:5.3f} {r['macro_f1']:5.3f} {r.get('ece', float('nan')):5.3f} "
              f"{r.get('brier', float('nan')):5.3f} {r['cost_ungated']:5.2f} {r['cost_gated']:5.3f} {r['coverage']:4.2f} "
              f"{r['auto_error']:5.3f} {r['wrong_approvals']:3d} {r.get('latency_median_s', float('nan')):6.2f} "
              f"{r.get('usd_per_1000', float('nan')):6.3f}")


if __name__ == "__main__":
    main()
