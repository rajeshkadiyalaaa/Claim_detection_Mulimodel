"""One-time test-split evaluation of the frozen configuration (see FROZEN_CONFIG.md).

Everything trainable is fitted on TRAIN; C and gate thresholds are chosen on VAL;
the test split is used only here, once, for reporting.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
LABELS = ["Auto-Approve", "Reject_Fraud_Audit", "Escalate_Human"]
ALL = ["p_A", "p_R", "p_E", "s2", "s3", "s4", "s5"]
PRICE = {"d1_in": 0.10e-6, "flash_in": 0.75e-6, "flash_out": 3.75e-6}
DEFAULT_W = (10, 4, 1)
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
SEED = 42
rng = np.random.default_rng(SEED)


def load(name):
    return [json.loads(l) for l in (RES / f"{name}.jsonl").read_text().splitlines()]


def ordered(rows, ids):
    d = {o["sample_id"]: o for o in rows}
    return [d[i] for i in ids]


def cost_arr(y, g, w):
    fa, fr, h = w
    y, g = np.asarray(y), np.asarray(g)
    c = np.zeros(len(y))
    wrong = y != g
    c[wrong & (g == "Auto-Approve")] = fa
    c[wrong & (g == "Reject_Fraud_Audit")] = fr
    c[wrong & (g == "Escalate_Human")] = h
    return c


def apply_gate(pred, conf, supp, tc, ts):
    pred, conf, supp = np.asarray(pred), np.asarray(conf), np.asarray(supp)
    auto = np.isin(pred, ["Auto-Approve", "Reject_Fraud_Audit"]) & (conf >= tc) & (supp >= ts)
    return np.where(auto, pred, "Escalate_Human")


def select_thresholds(y, pred, conf, supp, use_support, w):
    best = None
    for tc in GRID:
        for ts in (GRID if use_support else [0.0]):
            g = apply_gate(pred, conf, supp, tc, ts)
            key = (cost_arr(y, g, w).mean(), -np.mean(g != "Escalate_Human"))
            if best is None or key < best[0]:
                best = (key, tc, ts)
    return best[1], best[2]


def support(rows, pred):
    out = []
    for o, p in zip(rows, pred):
        if p == "Auto-Approve":
            out.append(min(o["s2"], o["s3"], o["s4"], o["s5"]))
        elif p == "Reject_Fraud_Audit":
            out.append(min(o["s3"], max(1 - o["s2"], 1 - o["s5"])))
        else:
            out.append(1.0)
    return np.array(out)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def brier(P, y):
    Y = np.eye(3)[[LABELS.index(v) for v in y]]
    return float(np.mean(np.sum((np.asarray(P) - Y) ** 2, axis=1)))


def aurc(conf, correct):
    order = np.argsort(-np.asarray(conf))
    err = 1 - np.asarray(correct, float)[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risks.mean()), risks.tolist()


def combiner(tr, va, te, feats):
    X = lambda rows: np.array([[o[f] for f in feats] for o in rows])
    yi = lambda rows: [LABELS.index(o["label"]) for o in rows]
    best = None
    for C in (0.01, 0.1, 1, 10, 100):
        m = LogisticRegression(C=C, max_iter=10000, random_state=SEED).fit(X(tr), yi(tr))
        ll = log_loss(yi(va), m.predict_proba(X(va)), labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, C, m)
    m = best[2]
    return m.predict_proba(X(va)), m.predict_proba(X(te)), best[1]


class System:
    """Holds val and test scores for one method; thresholds come from val only."""

    def __init__(self, name, yv, yt, Pv=None, Pt=None, predv=None, confv=None, predt=None, conft=None,
                 suppv=None, suppt=None, use_support=False, extra=None):
        self.name, self.yv, self.yt, self.use_support, self.extra = name, np.array(yv), np.array(yt), use_support, extra or {}
        self.Pt = None if Pt is None else np.asarray(Pt)
        if Pv is not None:
            Pv, Pt = np.asarray(Pv), np.asarray(Pt)
            predv, confv = np.array(LABELS)[Pv.argmax(1)], Pv.max(1)
            predt, conft = np.array(LABELS)[Pt.argmax(1)], Pt.max(1)
        self.predv, self.confv, self.predt, self.conft = map(np.asarray, (predv, confv, predt, conft))
        self.suppv = np.ones(len(yv)) if suppv is None else np.asarray(suppv)
        self.suppt = np.ones(len(yt)) if suppt is None else np.asarray(suppt)

    def gated_test(self, w=DEFAULT_W):
        tc, ts = select_thresholds(self.yv, self.predv, self.confv, self.suppv, self.use_support, w)
        return tc, ts, apply_gate(self.predt, self.conft, self.suppt, tc, ts)

    def metrics(self):
        y, p = self.yt, self.predt
        correct = y == p
        tc, ts, g = self.gated_test()
        auto = g != "Escalate_Human"
        r = {
            "model": self.name, "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=LABELS, average="macro"),
            **{f"recall_{l[0]}": recall_score(y, p, labels=[l], average="macro", zero_division=0) for l in LABELS},
            "ece": ece(self.conft, correct), "brier": brier(self.Pt, y) if self.Pt is not None else None,
            "aurc": aurc(self.conft, correct)[0],
            "cost_ungated": float(cost_arr(y, p, DEFAULT_W).mean()),
            "tau_c": float(tc), "tau_s": float(ts), "cost_gated": float(cost_arr(y, g, DEFAULT_W).mean()),
            "coverage": float(auto.mean()), "auto_error": float((g[auto] != y[auto]).mean()) if auto.any() else 0.0,
            "wrong_approvals": int(((g == "Auto-Approve") & (y != g)).sum()),
            "confusion": confusion_matrix(y, p, labels=LABELS).tolist(),
            "gated_confusion": confusion_matrix(y, g, labels=LABELS).tolist(),
        }
        r.update(self.extra)
        return r


def bootstrap(systems, B=10000):
    n = len(systems[0].yt)
    idx = rng.integers(0, n, size=(B, n))
    out = {}
    gated = {s.name: s.gated_test()[2] for s in systems}
    for s in systems:
        acc = (s.yt == s.predt)[idx].mean(1)
        f1 = np.array([f1_score(s.yt[i], s.predt[i], labels=LABELS, average="macro") for i in idx[:2000]])
        gc = cost_arr(s.yt, gated[s.name], DEFAULT_W)[idx].mean(1)
        out[s.name] = {"acc_ci": np.percentile(acc, [2.5, 97.5]).tolist(),
                       "macro_f1_ci_2000": np.percentile(f1, [2.5, 97.5]).tolist(),
                       "cost_gated_ci": np.percentile(gc, [2.5, 97.5]).tolist()}
    return out, idx, gated


def main():
    rows = [json.loads(l) for l in (ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
    vids = [r["sample_id"] for r in rows if r["split"] == "val"]
    tids = [r["sample_id"] for r in rows if r["split"] == "test"]
    lab = {r["sample_id"]: r["decision_label"] for r in rows}
    yv, yt = [lab[i] for i in vids], [lab[i] for i in tids]

    def d1(suffix):
        s = f"_{suffix}" if suffix else ""
        return (load(f"decision1_train{s}"), ordered(load(f"decision1_val{s}"), vids), ordered(load(f"decision1_test{s}"), tids))

    systems, Cs = [], {}
    tr, va, te = d1("")
    lat = [o["latency_s"] for o in te]
    tok = [o["input_tokens"] for o in te]
    eff_d1 = {"latency_median_s": float(np.median(lat)), "latency_p90_s": float(np.percentile(lat, 90)),
              "tokens_in_mean": float(np.mean(tok)), "tokens_out_mean": 0.0,
              "usd_per_1000": float(np.mean(tok) * PRICE["d1_in"] * 1000)}

    def proposed(name, trr, vaa, tee, feats=ALL, use_support=True, extra=None):
        Pv, Pt, C = combiner(trr, vaa, tee, feats)
        Cs[name] = C
        pv, pt = np.array(LABELS)[Pv.argmax(1)], np.array(LABELS)[Pt.argmax(1)]
        return System(name, yv, yt, Pv, Pt, suppv=support(vaa, pv), suppt=support(tee, pt),
                      use_support=use_support, extra=extra)

    main_sys = proposed("Proposed (tri-modal): Decision 1 + combiner + gate", tr, va, te, extra=eff_d1)
    systems.append(main_sys)
    trn, van, ten = d1("nonumbers")
    tokn = [o["input_tokens"] for o in ten]
    nonum = proposed("Proposed variant: image + text (no numbers)", trn, van, ten,
                     extra={"tokens_in_mean": float(np.mean(tokn)), "usd_per_1000": float(np.mean(tokn) * PRICE["d1_in"] * 1000)})
    systems.append(nonum)

    bl = json.loads((RES / "baselines_val_test.json").read_text())
    assert bl["val_ids"] == vids and bl["test_ids"] == tids
    bsys = {}
    for name, e in bl["models"].items():
        bsys[name] = System(name, yv, yt, e["val_proba"], e["test_proba"])
        systems.append(bsys[name])

    b5v, b5t = ordered(load("b5_gemini-3.8-flash_val"), vids), ordered(load("b5_gemini-3.8-flash_test"), tids)
    tin = np.mean([o["input_tokens"] for o in b5t])
    tout = np.mean([(o["output_tokens"] or 0) + (o["thinking_tokens"] or 0) for o in b5t])
    systems.append(System("B5 generative (Gemini 3.8 Flash, zero-shot)", yv, yt,
                          predv=[o["pred"] for o in b5v], confv=[o["conf"] for o in b5v],
                          predt=[o["pred"] for o in b5t], conft=[o["conf"] for o in b5t],
                          extra={"latency_median_s": float(np.median([o["latency_s"] for o in b5t])),
                                 "latency_p90_s": float(np.percentile([o["latency_s"] for o in b5t], 90)),
                                 "tokens_in_mean": float(tin), "tokens_out_mean": float(tout),
                                 "usd_per_1000": float((tin * PRICE["flash_in"] + tout * PRICE["flash_out"]) * 1000)}))

    ablations = [System("q1 only, no combiner", yv, yt,
                        [[o["p_A"], o["p_R"], o["p_E"]] for o in va], [[o["p_A"], o["p_R"], o["p_E"]] for o in te])]
    for name, feats in [("combiner on q1 only", ["p_A", "p_R", "p_E"]), ("combiner on checks only", ["s2", "s3", "s4", "s5"])] + \
                       [(f"drop {f}", [x for x in ALL if x != f]) for f in ("s2", "s3", "s4", "s5")]:
        ablations.append(proposed(name, tr, va, te, feats, use_support=False))
    for dname in ("noimage", "notext"):
        ablations.append(proposed(f"no {dname[2:]} in request", *d1(dname)))
    ablations.append(proposed("confidence-only gate (tau_s = 0)", tr, va, te, use_support=False))

    res = {"combiner_C": Cs, "main": [s.metrics() for s in systems], "ablations": [s.metrics() for s in ablations]}

    ci, idx, gated = bootstrap(systems)
    res["bootstrap_ci"] = ci
    b4 = bsys["B4 early fusion (XGBoost)"]
    diffs = {}
    for s in (main_sys, nonum):
        da = ((s.yt == s.predt)[idx].mean(1) - (b4.yt == b4.predt)[idx].mean(1))
        dc = (cost_arr(s.yt, gated[s.name], DEFAULT_W)[idx].mean(1) - cost_arr(b4.yt, gated[b4.name], DEFAULT_W)[idx].mean(1))
        diffs[f"{s.name} minus B4"] = {"acc_diff": float(da.mean()), "acc_diff_ci": np.percentile(da, [2.5, 97.5]).tolist(),
                                       "cost_diff": float(dc.mean()), "cost_diff_ci": np.percentile(dc, [2.5, 97.5]).tolist()}
    res["paired_differences_vs_B4"] = diffs

    sens = {}
    for w in [(5, 2, 1), (10, 4, 1), (20, 4, 1), (20, 8, 1), (50, 10, 1)]:
        row = {"escalate_all": float(cost_arr(yt, ["Escalate_Human"] * len(yt), w).mean())}
        for s in [main_sys, nonum, bsys["B4 early fusion (XGBoost)"], bsys["B1 text-only (TF-IDF + LR)"]]:
            tc, ts, g = s.gated_test(w)
            row[s.name] = {"cost": float(cost_arr(yt, g, w).mean()), "coverage": float(np.mean(g != "Escalate_Human")),
                           "wrong_approvals": int(((g == "Auto-Approve") & (np.array(yt) != g)).sum()), "tau_c": float(tc)}
        sens[str(w)] = row
    res["cost_sensitivity"] = sens

    res["risk_coverage_curves"] = {s.name: aurc(s.conft, s.yt == s.predt)[1] for s in systems[:6]}
    (RES / "final_test.json").write_text(json.dumps(res, indent=2))

    print(f"Combiner C: {Cs}")
    hdr = f"{'Model':46} {'Acc':>5} {'MF1':>5} {'ECE':>5} {'Brier':>5} {'AURC':>5} {'GCost':>5} {'Cov':>4} {'AErr':>5} {'WA':>3} {'Lat':>5} {'$/1k':>6}"
    for title, lst in (("MAIN (test, n=150)", res["main"]), ("ABLATIONS (test)", res["ablations"])):
        print("\n" + title + "\n" + hdr)
        for r in lst:
            b = r["brier"] if r["brier"] is not None else float("nan")
            print(f"{r['model'][:46]:46} {r['accuracy']:5.3f} {r['macro_f1']:5.3f} {r['ece']:5.3f} {b:5.3f} {r['aurc']:5.3f} "
                  f"{r['cost_gated']:5.3f} {r['coverage']:4.2f} {r['auto_error']:5.3f} {r['wrong_approvals']:3d} "
                  f"{r.get('latency_median_s', float('nan')):5.2f} {r.get('usd_per_1000', float('nan')):6.3f}")
    print("\nBootstrap 95% CIs:")
    for k, v in ci.items():
        print(f"  {k[:46]:46} acc {v['acc_ci'][0]:.3f}-{v['acc_ci'][1]:.3f}  cost {v['cost_gated_ci'][0]:.3f}-{v['cost_gated_ci'][1]:.3f}")
    print("\nPaired differences vs B4:", json.dumps(diffs, indent=1))
    print("\nCost sensitivity (test gated cost; thresholds re-selected on val):")
    for w, row in sens.items():
        print(f"  {w}: escalate-all {row['escalate_all']:.3f} | " +
              " | ".join(f"{k.split(':')[0].split('(')[0].strip()[:18]} {v['cost']:.3f} (cov {v['coverage']:.2f}, WA {v['wrong_approvals']})"
                         for k, v in row.items() if k != "escalate_all"))


if __name__ == "__main__":
    main()
