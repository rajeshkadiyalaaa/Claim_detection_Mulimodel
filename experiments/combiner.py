"""Learned combiner over Gemini Decision 1 scores (validation-stage evaluation).

Features per claim: z = (p_A, p_R, p_E, s2, s3, s4, s5), the probabilities returned
by Decision 1 for the routing question and the four yes/no checks.
A multinomial logistic regression (L2) is trained on the TRAIN split.
Its regularisation strength C is chosen on VAL by log loss, the model is
refit on TRAIN only, and the gate thresholds (tau_c, tau_s) are chosen on VAL
by minimising expected misrouting cost. The TEST split is not used here.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from pilot_analysis import LABELS, cost, ece, brier, gate_search, report

ROOT = Path(__file__).resolve().parent.parent
FEATS = ["p_A", "p_R", "p_E", "s2", "s3", "s4", "s5"]
SEED = 42


def load(split):
    return [json.loads(l) for l in (ROOT / f"experiments/results/decision1_{split}.jsonl").read_text().splitlines()]


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


def main():
    tr, va = load("train"), load("val")
    Xtr = np.array([[o[f] for f in FEATS] for o in tr])
    ytr = np.array([LABELS.index(o["label"]) for o in tr])
    Xva = np.array([[o[f] for f in FEATS] for o in va])
    yva_idx = np.array([LABELS.index(o["label"]) for o in va])
    yva = [o["label"] for o in va]

    # choose C on validation log loss
    grid = [0.01, 0.1, 1, 10, 100]
    scores = {}
    for C in grid:
        m = LogisticRegression(C=C, max_iter=10000, random_state=SEED).fit(Xtr, ytr)
        scores[C] = log_loss(yva_idx, m.predict_proba(Xva), labels=[0, 1, 2])
    C_best = min(scores, key=scores.get)
    model = LogisticRegression(C=C_best, max_iter=10000, random_state=SEED).fit(Xtr, ytr)

    P = model.predict_proba(Xva)
    pred = [LABELS[i] for i in P.argmax(1)]
    conf = list(P.max(1))
    res = report("Decision 1 + learned combiner (no gate)", yva, pred, P, conf)

    supp = support(va, pred)
    gates = {}
    for name, use in [("confidence gate", False), ("confidence + support gate", True)]:
        tc, ts, g = gate_search(yva, pred, conf, supp, use)
        auto = [i for i, x in enumerate(g) if x != "Escalate_Human"]
        gates[name] = {
            "tau_c": tc, "tau_s": ts,
            "mean_cost": float(np.mean([cost(a, b) for a, b in zip(yva, g)])),
            "accuracy": accuracy_score(yva, g),
            "macro_f1": f1_score(yva, g, labels=LABELS, average="macro"),
            "auto_coverage": len(auto) / len(va),
            "auto_error_rate": float(np.mean([g[i] != yva[i] for i in auto])) if auto else 0.0,
            "wrong_approvals": sum(1 for a, b in zip(yva, g) if b == "Auto-Approve" and a != b),
        }
    ungated_cost = float(np.mean([cost(a, b) for a, b in zip(yva, pred)]))

    out = {
        "features": FEATS, "C_grid_val_logloss": scores, "C_selected": C_best,
        "coefficients": {LABELS[k]: dict(zip(FEATS, map(float, model.coef_[k]))) for k in range(3)},
        "intercepts": dict(zip(LABELS, map(float, model.intercept_))),
        "val_metrics_ungated": res, "val_ungated_mean_cost": ungated_cost,
        "val_confusion_rows_true_cols_pred": confusion_matrix(yva, pred, labels=LABELS).tolist(),
        "val_gates_thresholds_tuned_on_val": gates,
        "escalate_all_cost": float(np.mean([cost(a, "Escalate_Human") for a in yva])),
    }
    (ROOT / "experiments/results/combiner_val_summary.json").write_text(json.dumps(out, indent=2))

    print(f"C by val log loss: {scores} -> C={C_best}")
    print(f"Combiner on val (trained on train): acc={res['accuracy']:.3f} macroF1={res['macro_f1']:.3f} "
          f"recall A/R/E={res['recall_A']:.3f}/{res['recall_R']:.3f}/{res['recall_E']:.3f} "
          f"ECE={res['ece']:.3f} Brier={res['brier']:.3f} ungated cost={ungated_cost:.3f}")
    print("Confusion (rows true A,R,E):", out["val_confusion_rows_true_cols_pred"])
    for k, v in gates.items():
        print(f"  {k:28} tau_c={v['tau_c']:.2f} tau_s={v['tau_s']:.2f} cost={v['mean_cost']:.3f} "
              f"acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} coverage={v['auto_coverage']:.2f} "
              f"auto_err={v['auto_error_rate']:.3f} wrong_approvals={v['wrong_approvals']}")
    print(f"Escalate-all cost: {out['escalate_all_cost']:.3f}")
    print("Coefficients:", json.dumps(out["coefficients"], indent=1))


if __name__ == "__main__":
    main()
