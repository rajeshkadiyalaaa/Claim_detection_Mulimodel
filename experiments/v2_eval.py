"""Validation comparison of the Level-1 revised request (v2) against v1 and baselines.

v2 combiner: trained on TRAIN, C chosen on VAL; gate thresholds chosen on VAL
(gated numbers are therefore in-sample on VAL, exactly as for v1 in val_comparison.py).
"""
import json
from pathlib import Path

import numpy as np

from val_comparison import evaluate, fit_combiner, LABELS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
CHECKS = ["q2_damage_match", "q3_photo_clear", "q4_police_recorded", "q5_theft", "q6_vehicle_match", "q7_amounts_match"]
Q1 = ["p_A", "p_R", "p_E"]


def load(n):
    return [json.loads(l) for l in (RES / f"{n}.jsonl").read_text().splitlines()]


def main():
    tr, va = load("v2_train"), load("v2_val")
    rows = [json.loads(l) for l in (ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
    ids = [r["sample_id"] for r in rows if r["split"] == "val"]
    d = {o["sample_id"]: o for o in va}
    va = [d[i] for i in ids]
    y = [o["label"] for o in va]

    out = [evaluate("v2 zero-shot q1 (no combiner)", y, P=[[o[k] for k in Q1] for o in va])]
    for name, feats in [("v2 combiner, all answers", Q1 + CHECKS),
                        ("v2 combiner, q1 only", Q1),
                        ("v2 combiner, checks only", CHECKS),
                        ("v2 combiner, without vehicle check", Q1 + [c for c in CHECKS if c != "q6_vehicle_match"]),
                        ("v2 combiner, without amounts check", Q1 + [c for c in CHECKS if c != "q7_amounts_match"])]:
        P, C = fit_combiner(tr, va, feats)
        out.append(evaluate(name, y, P=P, extra={"C": C}))

    v1 = json.loads((RES / "val_comparison.json").read_text())["rows"]
    keep = {"Proposed: Decision 1 + combiner + gate", "B4 early fusion (XGBoost)",
            "Ablation: q1 only, no combiner", "B5 generative (Gemini 3.8 Flash, JSON)"}
    ref = [dict(r, model="v1 " + r["model"] if not r["model"].startswith("B") else r["model"]) for r in v1 if r["model"] in keep]

    means = {l: {c: float(np.mean([o[c] for o in va if o["label"] == l])) for c in CHECKS} for l in LABELS}
    tok = float(np.mean([o["input_tokens"] for o in va]))
    (RES / "v2_val_comparison.json").write_text(json.dumps({"v2": out, "reference": ref, "check_means": means,
                                                            "v2_tokens_in_mean": tok}, indent=2))
    print(f"{'Model (val, n=150)':46} {'Acc':>5} {'MF1':>5} {'RecA':>5} {'RecR':>5} {'RecE':>5} {'ECE':>5} {'GCost':>5} {'Cov':>4} {'WA':>3}")
    for r in ref + out:
        print(f"{r['model'][:46]:46} {r['accuracy']:5.3f} {r['macro_f1']:5.3f} {r['recall_A']:5.3f} {r['recall_R']:5.3f} "
              f"{r['recall_E']:5.3f} {r.get('ece', float('nan')):5.3f} {r['cost_gated']:5.3f} {r['coverage']:4.2f} {r['wrong_approvals']:3d}")
    print("\nMean yes-probability of v2 checks by true label:")
    for l, v in means.items():
        print(f"  {l:20} " + "  ".join(f"{k.split('_', 1)[1]}={x:.2f}" for k, x in v.items()))
    print(f"\nv2 mean input tokens per claim: {tok:.0f}")


if __name__ == "__main__":
    main()
