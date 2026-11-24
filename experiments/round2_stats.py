"""Round-2 reanalyses requested by the verification re-review (no API calls).

A. Repeated 5x5 CV kept at fold level, with the corrected resampled t-test
   (Nadeau & Bengio 2003) for every paired difference                 (RR1 N1, DA N5)
   - every misrouting weighting of the cost-sensitivity table          (DA-2)
   - two handling-cost variants with thresholds re-selected            (R2 N2, DA-14)
   - per-route rates: straight-through, audit, escalation              (R3 W1/N6)
   - the frozen combiner setting C = 100 alongside C = 10              (RR1 N4)
B. Learning curves with rule-aware comparators, A/R-only accuracy and
   gated cost, plus two-way bootstrap CIs at n = 25                    (RR1 N2, DA N1, R2 N3)
C. Fixed-split extras: route rates, wrongful audits by true class,
   handling cost, combiner weights, matched subset on test only,
   error composition                                                  (R3 N3-N6, RR1 N5/W10)
D. Efficiency decomposition of the generative baseline                 (DA-3, RR1 W5)
E. Stability re-test details                                           (R3, DA N3)
F. Label-rule precedence counts                                        (R2 N1)

Writes experiments/results/round2_stats.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import revision_stats as rs

ROOT, RES, y, split, rows, meta = rs.ROOT, rs.RES, rs.y, rs.split, rs.rows, rs.meta
N = len(rows)
OUT = RES / "round2_stats.json"

# ---------------------------------------------------------------- cost specifications
SPECS = {
    "misroute (5,2,1)": ("m", (5, 2, 1)),
    "misroute (10,4,1)": ("m", (10, 4, 1)),
    "misroute (20,4,1)": ("m", (20, 4, 1)),
    "misroute (20,8,1)": ("m", (20, 8, 1)),
    "misroute (50,10,1)": ("m", (50, 10, 1)),
    # handling cost: every claim a human touches costs a review; wrong approvals cost 10
    "handling H1 (audit 1, escalation 1, wrong approval 10)": ("h", (1, 1, 10, 0)),
    # audits cost more than desk reviews, and auditing an honest claim adds harm
    "handling H2 (audit 2, escalation 1, wrong approval 10, wrongful audit +4)": ("h", (2, 1, 10, 4)),
}
DEFAULT = "misroute (10,4,1)"


def cost(yv, g, spec):
    kind, p = spec
    if kind == "m":
        return rs.cost_arr(yv, g, p)
    hR, hE, cFA, cWA = p
    c = np.zeros(len(yv))
    c[g == 1] = hR
    c[g == 2] = hE
    c[(g == 0) & (yv != 0)] = cFA
    c[(g == 1) & (yv == 0)] += cWA
    return c


def pick(P, yv, spec):
    best = None
    for tc in rs.GRID:
        g = rs.gate(P, tc)
        key = (cost(yv, g, spec).mean(), -np.mean(g != 2))
        if best is None or key < best[0]:
            best = (key, tc)
    return best[1]


def corrected_t(d, k=5, r=5, test_frac=0.2):
    """Nadeau-Bengio corrected resampled t-test over r x k fold-level differences."""
    d = np.asarray(d, float)
    J = len(d)
    m, s2 = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + test_frac / (1 - test_frac)) * s2)
    t = m / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), J - 1)
    h = stats.t.ppf(0.975, J - 1) * se
    return {"mean": float(m), "ci": [float(m - h), float(m + h)], "p": float(p)}


# ---------------------------------------------------------------- model fitting (adds the frozen C = 100)
def fit_predict(kind, tr_idx, te_idx):
    if kind == "v1_C100":
        return rs.lr(100).fit(rs.D1[tr_idx], y[tr_idx]).predict_proba(rs.D1[te_idx])
    return rs.fit_predict(kind, tr_idx, te_idx)


def oof(kind, idx, seed, inner=5):
    out = np.zeros((len(idx), 3))
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(idx, y[idx]):
        out[b] = fit_predict(kind, idx[a], idx[b])
    return out


# ---------------------------------------------------------------- A. repeated CV at fold level
def cv_fold_level(kinds, repeats=5, folds=5):
    allidx = np.arange(N)
    rec = {k: [] for k in kinds + ["escalate_all"]}
    for r in range(repeats):
        for f, (tr_idx, te_idx) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=100 + r).split(allidx, y)):
            yt = y[te_idx]
            esc = np.full(len(yt), 2)
            rec["escalate_all"].append({"acc": np.nan, **{s: {"cost": float(cost(yt, esc, sp).mean()), "st": 0.0, "audit": 0.0,
                                                              "esc": 1.0, "wa": 0, "wrongful_audit_A": 0, "wrongful_audit_E": 0}
                                                          for s, sp in SPECS.items()}})
            for k in kinds:
                Pin = oof(k, tr_idx, seed=r)
                P = fit_predict(k, tr_idx, te_idx)
                d = {"acc": float((P.argmax(1) == yt).mean())}
                for s, sp in SPECS.items():
                    tc = pick(Pin, y[tr_idx], sp)
                    g = rs.gate(P, tc)
                    d[s] = {"cost": float(cost(yt, g, sp).mean()), "tc": float(tc),
                            "st": float(np.mean(g == 0)), "audit": float(np.mean(g == 1)), "esc": float(np.mean(g == 2)),
                            "wa": int(np.sum((g == 0) & (yt != 0))),
                            "wrongful_audit_A": int(np.sum((g == 1) & (yt == 0))),
                            "wrongful_audit_E": int(np.sum((g == 1) & (yt == 2)))}
                rec[k].append(d)
            print(f"  CV repeat {r} fold {f} done", flush=True)
    summ = {}
    for k, v in rec.items():
        summ[k] = {"acc": {"mean": float(np.nanmean([d["acc"] for d in v])), "sd": float(np.nanstd([d["acc"] for d in v]))}}
        for s in SPECS:
            summ[k][s] = {m: float(np.mean([d[s][m] for d in v])) for m in ("cost", "st", "audit", "esc", "wa",
                                                                           "wrongful_audit_A", "wrongful_audit_E")}
            summ[k][s]["cost_sd"] = float(np.std([d[s]["cost"] for d in v]))
    diffs = {}
    pairs = [("v1", "B4"), ("v1_C100", "B4"), ("b5fair", "v1"), ("v2", "v1"), ("v2", "B4"), ("nonum", "v1"),
             ("nonum", "B4"), ("v1", "B1"), ("v1", "B3")]
    for a, b in pairs:
        if a in rec and b in rec:
            diffs[f"{a} - {b} (acc)"] = corrected_t([x["acc"] - z["acc"] for x, z in zip(rec[a], rec[b])])
            for s in SPECS:
                diffs[f"{a} - {b} ({s})"] = corrected_t([x[s]["cost"] - z[s]["cost"] for x, z in zip(rec[a], rec[b])])
    for k in kinds:
        for s in SPECS:
            diffs[f"{k} - escalate_all ({s})"] = corrected_t([x[s]["cost"] - z[s]["cost"]
                                                              for x, z in zip(rec[k], rec["escalate_all"])])
    return summ, diffs


# ---------------------------------------------------------------- B. learning curves with rule-aware comparators
SEV = ["trivial damage", "minor damage", "major damage", "total loss"]
TXT = np.array([[m[0] == s for s in SEV] + ["theft" in m[1], m[2] == "unrecorded", m[2] == "unavailable"] for m in meta], float)
QIMG = np.log1p(rs.image_quality_features())


def text_rule_pred():
    out = []
    for sev, inc, pol in meta:
        out.append(2 if ("theft" in inc or pol == "unrecorded") else 1 if sev == "total loss" else 0)
    return np.array(out)


def fit_rule_lr(X, tr_idx, te_idx):
    sc = StandardScaler().fit(X[tr_idx])
    m = LogisticRegression(C=10, max_iter=20000, random_state=rs.SEED).fit(sc.transform(X[tr_idx]), y[tr_idx])
    return m.predict_proba(sc.transform(X[te_idx]))


def lc_fit(kind, tr_idx, te_idx):
    if kind == "rule_lr_text":
        return fit_rule_lr(TXT, tr_idx, te_idx)
    if kind == "rule_lr_text_img":
        return fit_rule_lr(np.hstack([TXT, QIMG]), tr_idx, te_idx)
    return fit_predict(kind, tr_idx, te_idx)


def lc_oof(kind, idx, seed):
    inner = int(min(5, np.bincount(y[idx]).min()))
    out = np.zeros((len(idx), 3))
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(idx, y[idx]):
        out[b] = lc_fit(kind, idx[a], idx[b])
    return out


def learning_curves(kinds, sizes=(25, 50, 100, 200, 400, 700), draws=10):
    tr_all = np.where(split == "train")[0]
    ev = np.where(split != "train")[0]
    yev = y[ev]
    ar = yev != 2
    spec, hspec = SPECS[DEFAULT], SPECS["handling H1 (audit 1, escalation 1, wrong approval 10)"]
    out = {k: {} for k in kinds}
    correct = {k: {} for k in kinds}
    for n in sizes:
        for k in kinds:
            res = {"acc": [], "acc_AR": [], "cost": [], "cost_H1": [], "cov": []}
            correct[k][n] = []
            for dsd in range(draws if n < 700 else 1):
                r = np.random.default_rng(1000 + dsd)
                sub = np.concatenate([r.choice(tr_all[y[tr_all] == c], max(2, round(n * np.mean(y[tr_all] == c))), replace=False)
                                      for c in range(3)])
                P = lc_fit(k, sub, ev)
                Pin = lc_oof(k, sub, seed=dsd)
                g = rs.gate(P, pick(Pin, y[sub], spec))
                gh = rs.gate(P, pick(Pin, y[sub], hspec))
                c = P.argmax(1) == yev
                correct[k][n].append(c)
                res["acc"].append(c.mean())
                res["acc_AR"].append(c[ar].mean())
                res["cost"].append(cost(yev, g, spec).mean())
                res["cost_H1"].append(cost(yev, gh, hspec).mean())
                res["cov"].append(np.mean(g != 2))
            out[k][n] = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for m, v in res.items()}
        print(f"  learning curve n={n} done", flush=True)
    tr_pred = text_rule_pred()
    out["text_rule (0 labels)"] = {"acc": float((tr_pred[ev] == yev).mean()), "acc_AR": float((tr_pred[ev][ar] == yev[ar]).mean()),
                                   "cost": float(cost(yev, tr_pred[ev], spec).mean()),
                                   "cost_H1": float(cost(yev, tr_pred[ev], hspec).mean())}
    out["escalate_all"] = {"cost": float(cost(yev, np.full(len(ev), 2), spec).mean()),
                           "cost_H1": float(cost(yev, np.full(len(ev), 2), hspec).mean())}
    # two-way bootstrap (draws x claims) for accuracy differences at n = 25 and n = 50
    rng = np.random.default_rng(7)
    boot = {}
    for n in (25, 50):
        for a, b in [("v1", "B4"), ("v1", "rule_lr_text_img"), ("v1", "rule_lr_text"), ("v1", "b5fair")]:
            if a in correct and b in correct:
                D = np.array(correct[a][n], float) - np.array(correct[b][n], float)
                bs = [D[rng.integers(0, D.shape[0], D.shape[0])][:, rng.integers(0, D.shape[1], D.shape[1])].mean()
                      for _ in range(5000)]
                boot[f"n={n}: {a} - {b}"] = {"mean": float(D.mean()), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
    out["bootstrap_differences"] = boot
    return out


# ---------------------------------------------------------------- C. fixed-split extras
def fixed_split_extras():
    P = rs.fixed_split_probs()
    va, te = split == "val", split == "test"
    yv, yt = y[va], y[te]
    res = {"route_rates": {}, "handling_cost_test": {}}
    for name in ["v1 combiner", "no-numbers combiner", "v2 combiner (post-hoc)", "B4 early fusion (XGBoost)",
                 "B5-fair + combiner", "B1 text-only (TF-IDF + LR)"]:
        if name not in P:
            continue
        pv, pt, _ = P[name]
        g = rs.gate(pt, rs.pick_tc(pv, yv))
        res["route_rates"][name] = {
            "straight_through": int(np.sum(g == 0)), "audit": int(np.sum(g == 1)), "escalate": int(np.sum(g == 2)),
            "wrong_approvals": int(np.sum((g == 0) & (yt != 0))),
            "wrong_approval_rate_among_approvals": float(np.mean(yt[g == 0] != 0)) if np.any(g == 0) else None,
            "wrongful_audits_true_A": int(np.sum((g == 1) & (yt == 0))),
            "wrongful_audits_true_E": int(np.sum((g == 1) & (yt == 2))),
            "untouched_share": float(np.mean(g == 0))}
        res["handling_cost_test"][name] = {}
        for s, sp in SPECS.items():
            if sp[0] == "h":
                gh = rs.gate(pt, pick(pv, yv, sp))
                res["handling_cost_test"][name][s] = {"cost": float(cost(yt, gh, sp).mean()), "straight_through": float(np.mean(gh == 0)),
                                                      "escalate_all": float(cost(yt, np.full(len(yt), 2), sp).mean())}
        # confusion before gating
        cm = np.zeros((3, 3), int)
        for a, b in zip(yt, pt.argmax(1)):
            cm[a, b] += 1
        res.setdefault("confusion_test", {})[name] = cm.tolist()
    # frozen combiner weights (C = 100, trained on train)
    tr = split == "train"
    m = rs.lr(100).fit(rs.D1[tr], y[tr])
    res["combiner_weights_C100"] = {"features": rs.F7, "classes": rs.LABELS,
                                    "coef": np.round(m.coef_, 3).tolist(), "intercept": np.round(m.intercept_, 3).tolist(),
                                    "feature_sd_train": np.round(rs.D1[tr].std(0), 3).tolist()}
    # matched subset on test only, plus discordant counts on the pooled subset
    vt = split != "train"
    ds_mask = (y != 2) & np.array([mm[0] == "major damage" and mm[2] in ("available", "unavailable") for mm in meta])
    order = np.concatenate([np.where(va)[0], np.where(te)[0]])
    preds = {}
    for name in ["v1 combiner", "no-numbers combiner", "B4 early fusion (XGBoost)", "B3 image-only (CLIP + LR)",
                 "B1 text-only (TF-IDF + LR)", "B5-fair + combiner"]:
        pv, pt, _ = P[name]
        allp = np.full(N, -1)
        allp[order] = np.vstack([pv, pt]).argmax(1)
        preds[name] = allp
    res["matched_subset"] = {}
    for scope, mask in [("val+test", vt & ds_mask), ("test only", te & ds_mask)]:
        res["matched_subset"][scope] = {"n": int(mask.sum()),
                                        "acc": {k: float((p[mask] == y[mask]).mean()) for k, p in preds.items()}}
    pooled = vt & ds_mask
    for a, b in [("v1 combiner", "B4 early fusion (XGBoost)"), ("B3 image-only (CLIP + LR)", "v1 combiner")]:
        ca, cb = preds[a][pooled] == y[pooled], preds[b][pooled] == y[pooled]
        bb, cc, p = rs.mcnemar(ca, cb)
        res["matched_subset"][f"{a} vs {b}"] = {"a_only": bb, "b_only": cc, "exact_p": p}
    # McNemar no-numbers vs v1 on test
    pv, pt, _ = P["no-numbers combiner"]
    pv1, pt1, _ = P["v1 combiner"]
    bb, cc, p = rs.mcnemar(pt.argmax(1) == yt, pt1.argmax(1) == yt)
    res["mcnemar_nonum_vs_v1_test"] = {"nonum_only": bb, "v1_only": cc, "exact_p": p}
    return res


# ---------------------------------------------------------------- D. efficiency decomposition
def efficiency():
    def rd(name):
        return [json.loads(l) for l in (RES / name).read_text().splitlines()]
    d1, b5f, b5 = rd("decision1_test.jsonl"), rd("b5fair_test.jsonl"), rd("b5_gemini-3.8-flash_test.jsonl")
    lat_d1 = np.median([o["latency_s"] for o in d1])
    in_d1 = np.mean([o["input_tokens"] for o in d1])
    cost_d1 = in_d1 * 0.10 / 1e6 * 1000
    th = np.array([o.get("thinking_tokens", 0) or 0 for o in b5f])
    out_ans = np.array([o["output_tokens"] for o in b5f])
    inp = np.array([o["input_tokens"] for o in b5f])
    lat = np.array([o["latency_s"] for o in b5f])
    no_th = th == 0
    c_full = np.mean(inp * 0.75 / 1e6 + (out_ans + th) * 3.75 / 1e6) * 1000
    c_ans = np.mean(inp * 0.75 / 1e6 + out_ans * 3.75 / 1e6) * 1000
    c_in = np.mean(inp * 0.75 / 1e6) * 1000
    labels_no_th = [o["label"] for o, z in zip(b5f, no_th) if z]
    return {
        "decision1": {"median_latency_s": float(lat_d1), "mean_input_tokens": float(in_d1), "usd_per_1k": float(cost_d1)},
        "b5fair": {"median_latency_s": float(np.median(lat)), "n_claims": len(b5f), "n_with_thinking": int((~no_th).sum()),
                   "mean_thinking_tokens": float(th.mean()), "mean_answer_tokens": float(out_ans.mean()),
                   "median_latency_no_thinking_s": float(np.median(lat[no_th])) if no_th.any() else None,
                   "labels_of_no_thinking_claims": {l: labels_no_th.count(l) for l in set(labels_no_th)},
                   "usd_per_1k_full": float(c_full), "usd_per_1k_answer_tokens_only": float(c_ans),
                   "usd_per_1k_input_only": float(c_in)},
        "b5_default": {"median_latency_s": float(np.median([o["latency_s"] for o in b5]))},
        "ratios": {"latency_vs_b5fair": float(np.median(lat) / lat_d1),
                   "latency_vs_b5_default": float(np.median([o["latency_s"] for o in b5]) / lat_d1),
                   "latency_vs_b5fair_no_thinking_subset": float(np.median(lat[no_th]) / lat_d1) if no_th.any() else None,
                   "cost_full": float(c_full / cost_d1), "cost_answer_tokens_only": float(c_ans / cost_d1),
                   "cost_input_only": float(c_in / cost_d1)},
    }


# ---------------------------------------------------------------- E. stability details
def stability():
    test_rows = [r for r in rows if r["split"] == "test"][:100]
    o_files = [ROOT / "experiments/cache/test" / f"{r['sample_id']}.json" for r in test_rows]
    n_files = [ROOT / "experiments/cache/retest_test" / f"{r['sample_id']}.json" for r in test_rows]
    o = [json.loads(f.read_text()) for f in o_files]
    n = [json.loads(f.read_text()) for f in n_files]
    q1o = [x["response"]["answers"]["q1_route"]["choice"] for x in o]
    q1n = [x["response"]["answers"]["q1_route"]["choice"] for x in n]
    ts_o = sorted(x.get("timestamp") for x in o if x.get("timestamp"))
    mt = sorted(f.stat().st_mtime for f in n_files)
    orig, new = np.load(ROOT / "experiments/cache/retest_features.npy")
    tr, va = split == "train", split == "val"
    m = rs.lr(100).fit(rs.D1[tr], y[tr])
    tc = rs.pick_tc(m.predict_proba(rs.D1[va]), y[va])
    po, pn = m.predict_proba(orig), m.predict_proba(new)
    go, gn = rs.gate(po, tc), rs.gate(pn, tc)
    flips = {}
    for a, b in zip(go, gn):
        if a != b:
            k = f"{'ARE'[a]}->{'ARE'[b]}"
            flips[k] = flips.get(k, 0) + 1
    return {"selection": "first 100 test claims in file order",
            "original_window": [ts_o[0], ts_o[-1]] if ts_o else None,
            "retest_window_file_mtime": [time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt[0])),
                                         time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt[-1]))],
            "q1_choice_counts_original": {c: q1o.count(c) for c in set(q1o)},
            "q1_choice_counts_retest": {c: q1n.count(c) for c in set(q1n)},
            "q1_choice_agreement": float(np.mean(np.array(q1o) == np.array(q1n))),
            "combiner_route_changes": int(np.sum(po.argmax(1) != pn.argmax(1))),
            "gated_decision_changes": int(np.sum(go != gn)), "gated_flip_directions": flips,
            "automated_decisions_original": int(np.sum(go != 2)),
            "max_abs_combiner_prob_change": float(np.abs(po - pn).max())}


# ---------------------------------------------------------------- F. label-rule precedence
def rule_precedence():
    lab = np.array([r["decision_label"] for r in rows])
    pol = np.array([m[2] for m in meta])
    theft = np.array(["theft" in m[1] for m in meta])
    sev = np.array([m[0] for m in meta])
    reason = [r["pairing_reason"] for r in rows]
    cnt = lambda mask: {l: int(np.sum(mask & (lab == l))) for l in rs.LABELS}
    e_mask = lab == "Escalate_Human"
    return {"police_unrecorded": cnt(pol == "unrecorded"), "police_unavailable": cnt(pol == "unavailable"),
            "theft": cnt(theft), "theft_and_unrecorded": cnt(theft & (pol == "unrecorded")),
            "major_or_total_loss": cnt(np.isin(sev, ["major damage", "total loss"])),
            "E_reason_mentions": {"police unrecorded": int(sum("Police-report availability is unrecorded" in reason[i] for i in np.where(e_mask)[0])),
                                  "theft": int(sum("theft" in reason[i] for i in np.where(e_mask)[0])),
                                  "scope review": int(sum("needs review" in reason[i] for i in np.where(e_mask)[0]))},
            "R_with_unrecorded_police_reason_starts": int(sum(reason[i].startswith("Severity mismatch")
                                                              for i in np.where((lab == "Reject_Fraud_Audit") & (pol == "unrecorded"))[0]))}


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    print("F. rule precedence"); res["rule_precedence"] = rule_precedence()
    print("D. efficiency"); res["efficiency"] = efficiency()
    print("E. stability"); res["stability"] = stability()
    print("C. fixed split"); res["fixed_split"] = fixed_split_extras()
    OUT.write_text(json.dumps(res, indent=2))
    print("B. learning curves")
    res["learning_curve"] = learning_curves(["v1", "B4", "b5fair", "rule_lr_text", "rule_lr_text_img", "B1", "B3"])
    OUT.write_text(json.dumps(res, indent=2))
    print("A. cross-validation")
    res["cv"], res["cv_diffs"] = cv_fold_level(["v1", "v1_C100", "nonum", "v2", "B1", "B3", "B4", "b5fair"])
    OUT.write_text(json.dumps(res, indent=2))
    print("done")


if __name__ == "__main__":
    main()
