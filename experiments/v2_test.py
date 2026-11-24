"""Post-hoc test evaluation of the frozen v2 configuration (FROZEN_CONFIG_v2.md),
side by side with the pre-registered v1 main method and the B4 baseline."""
import json
from pathlib import Path

import numpy as np

from final_test import (System, LABELS, combiner, cost_arr, bootstrap, rng, DEFAULT_W, PRICE,
                        load, ordered, support, ALL)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
V2_FEATS = ["p_A", "p_R", "p_E", "q2_damage_match", "q3_photo_clear", "q4_police_recorded", "q5_theft", "q7_amounts_match"]


def main():
    rows = [json.loads(l) for l in (ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
    vids = [r["sample_id"] for r in rows if r["split"] == "val"]
    tids = [r["sample_id"] for r in rows if r["split"] == "test"]
    lab = {r["sample_id"]: r["decision_label"] for r in rows}
    yv, yt = [lab[i] for i in vids], [lab[i] for i in tids]

    tr2, va2, te2 = load("v2_train"), ordered(load("v2_val"), vids), ordered(load("v2_test"), tids)
    Pv, Pt, C = combiner(tr2, va2, te2, V2_FEATS)
    lat = [o["latency_s"] for o in te2]
    tok = [o["input_tokens"] for o in te2]
    v2 = System("v2 (post-hoc): policy instruction + revised questions + combiner + gate", yv, yt, Pv, Pt,
                extra={"C": C, "latency_median_s": float(np.median(lat)), "tokens_in_mean": float(np.mean(tok)),
                       "usd_per_1000": float(np.mean(tok) * PRICE["d1_in"] * 1000)})
    v2z = System("v2 zero-shot q1 (no combiner)", yv, yt,
                 [[o["p_A"], o["p_R"], o["p_E"]] for o in va2], [[o["p_A"], o["p_R"], o["p_E"]] for o in te2])

    tr1, va1, te1 = load("decision1_train"), ordered(load("decision1_val"), vids), ordered(load("decision1_test"), tids)
    Pv1, Pt1, _ = combiner(tr1, va1, te1, ALL)
    pv, pt = np.array(LABELS)[Pv1.argmax(1)], np.array(LABELS)[Pt1.argmax(1)]
    v1 = System("v1 (pre-registered main)", yv, yt, Pv1, Pt1, suppv=support(va1, pv), suppt=support(te1, pt), use_support=True)

    bl = json.loads((RES / "baselines_val_test.json").read_text())
    b4 = System("B4 early fusion (XGBoost)", yv, yt, bl["models"]["B4 early fusion (XGBoost)"]["val_proba"],
                bl["models"]["B4 early fusion (XGBoost)"]["test_proba"])

    systems = [v1, v2, v2z, b4]
    res = {"metrics": [s.metrics() for s in systems]}
    ci, idx, gated = bootstrap(systems)
    res["bootstrap_ci"] = ci
    diffs = {}
    for a, b in [(v2, v1), (v2, b4)]:
        da = (a.yt == a.predt)[idx].mean(1) - (b.yt == b.predt)[idx].mean(1)
        dc = cost_arr(a.yt, gated[a.name], DEFAULT_W)[idx].mean(1) - cost_arr(b.yt, gated[b.name], DEFAULT_W)[idx].mean(1)
        diffs[f"{a.name[:2]} minus {b.name[:2]}"] = {
            "acc_diff": float(da.mean()), "acc_ci": np.percentile(da, [2.5, 97.5]).tolist(),
            "cost_diff": float(dc.mean()), "cost_ci": np.percentile(dc, [2.5, 97.5]).tolist()}
    res["paired_differences"] = diffs
    (RES / "v2_test.json").write_text(json.dumps(res, indent=2))

    print(f"v2 combiner C = {C}")
    print(f"{'Model (test, n=150)':52} {'Acc':>5} {'MF1':>5} {'RecA':>5} {'RecR':>5} {'RecE':>5} {'ECE':>5} {'GCost':>5} {'Cov':>4} {'AErr':>5} {'WA':>3}")
    for r in res["metrics"]:
        print(f"{r['model'][:52]:52} {r['accuracy']:5.3f} {r['macro_f1']:5.3f} {r['recall_A']:5.3f} {r['recall_R']:5.3f} {r['recall_E']:5.3f} "
              f"{r['ece']:5.3f} {r['cost_gated']:5.3f} {r['coverage']:4.2f} {r['auto_error']:5.3f} {r['wrong_approvals']:3d}")
    print("\n95% CIs:")
    for k, v in ci.items():
        print(f"  {k[:52]:52} acc {v['acc_ci'][0]:.3f}-{v['acc_ci'][1]:.3f}  gated cost {v['cost_gated_ci'][0]:.3f}-{v['cost_gated_ci'][1]:.3f}")
    print("\nPaired differences:")
    for k, v in diffs.items():
        print(f"  {k}: acc {v['acc_diff']:+.3f} [{v['acc_ci'][0]:+.3f}, {v['acc_ci'][1]:+.3f}]  "
              f"cost {v['cost_diff']:+.3f} [{v['cost_ci'][0]:+.3f}, {v['cost_ci'][1]:+.3f}]")
    m = res["metrics"][1]
    print(f"\nv2 efficiency: median latency {m['latency_median_s']:.2f} s, {m['tokens_in_mean']:.0f} input tokens, ${m['usd_per_1000']:.3f} per 1,000 claims")


if __name__ == "__main__":
    main()
