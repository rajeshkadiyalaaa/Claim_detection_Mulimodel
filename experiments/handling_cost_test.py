"""Handling-cost variants H1 and H2 on the test split with the frozen two-threshold gate (round 2; R2 N2, DA-14).

Uses the same combiner, support score and joint (tau_c, tau_s) search as final_test.py, with thresholds
re-selected on validation for each cost. Adds the result to results/round2_stats.json under
fixed_split.handling_cost_frozen_gate. No API calls.
"""
import json

import numpy as np

import final_test as ft
import round2_stats as r2

L = ft.LABELS


def cost_h(y, g, p):
    yi = np.array([L.index(v) for v in y])
    gi = np.array([L.index(v) for v in g])
    return r2.cost(yi, gi, ("h", p))


def select(y, pred, conf, supp, use_support, p):
    best = None
    for tc in ft.GRID:
        for ts in (ft.GRID if use_support else [0.0]):
            g = ft.apply_gate(pred, conf, supp, tc, ts)
            key = (cost_h(y, g, p).mean(), -np.mean(g != "Escalate_Human"))
            if best is None or key < best[0]:
                best = (key, tc, ts)
    return best[1], best[2]


def main():
    rows = [json.loads(l) for l in (ft.ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
    vids = [r["sample_id"] for r in rows if r["split"] == "val"]
    tids = [r["sample_id"] for r in rows if r["split"] == "test"]
    lab = {r["sample_id"]: r["decision_label"] for r in rows}
    yv, yt = [lab[i] for i in vids], [lab[i] for i in tids]
    tr = ft.load("decision1_train")
    va, te = ft.ordered(ft.load("decision1_val"), vids), ft.ordered(ft.load("decision1_test"), tids)
    Pv, Pt, _ = ft.combiner(tr, va, te, ft.ALL)
    pv, pt = np.array(L)[Pv.argmax(1)], np.array(L)[Pt.argmax(1)]
    sv, st = ft.support(va, pv), ft.support(te, pt)
    out = {}
    for name, (_, p) in r2.SPECS.items():
        if _ != "h":
            continue
        tc, ts = select(yv, pv, Pv.max(1), sv, True, p)
        g = ft.apply_gate(pt, Pt.max(1), st, tc, ts)
        out[name] = {"tau_c": float(tc), "tau_s": float(ts), "cost": float(cost_h(yt, g, p).mean()),
                     "straight_through": float(np.mean(g == "Auto-Approve")),
                     "escalate_all": float(cost_h(yt, np.array(["Escalate_Human"] * len(yt)), p).mean())}
    res = json.loads(r2.OUT.read_text())
    res["fixed_split"]["handling_cost_frozen_gate"] = out
    r2.OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
