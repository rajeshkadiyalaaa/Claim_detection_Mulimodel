"""Revision analyses requested by the review panel (no API calls; uses cached scores).

1. Exact McNemar tests + Holm on a declared confirmatory family     (R1 W1)
2. ECE with 5 and 10 bins, bootstrap CIs, perfect-calibration null (R1 W3)
3. Paired cost difference vs escalate-all for each method           (DA-2)
4. Repeated stratified cross-validation over all 1,000 claims with
   cross-fitted threshold selection                                  (R1 W4, DA-2)
5. Label-efficiency learning curves                                  (R1 W6, DA-4)
6. Shortcut controls: text-rule baseline, image-quality detector,
   A/R-only accuracy, de-shortcut subset, deduplicated test          (DA-1, R2 W1/W3)
7. Risk-coverage restricted to automatable (A/R) predictions          (DA-11)
8. Extended regularisation grid for the combiner                     (R1 W9)
9. Combiner routing agreement under the test-retest scores           (R1 W11)
10. Fair generative baseline (B5-fair) with and without combiner     (JFR W1, R1 W5, DA-3)
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import binomtest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
LABELS = ["Auto-Approve", "Reject_Fraud_Audit", "Escalate_Human"]
F7 = ["p_A", "p_R", "p_E", "s2", "s3", "s4", "s5"]
V2F = ["p_A", "p_R", "p_E", "q2_damage_match", "q3_photo_clear", "q4_police_recorded", "q5_theft", "q7_amounts_match"]
NUM = ["total_claim_amount", "vehicle_claim", "injury_claim", "property_claim", "policy_deductable",
       "policy_annual_premium", "age", "months_as_customer", "auto_year", "bodily_injuries", "witnesses"]
W = (10, 4, 1)
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- helpers
def cost_arr(y, g, w=W):
    y, g = np.asarray(y), np.asarray(g)
    c = np.zeros(len(y))
    wrong = y != g
    c[wrong & (g == 0)] = w[0]
    c[wrong & (g == 1)] = w[1]
    c[wrong & (g == 2)] = w[2]
    return c


def gate(P, tc):
    pred, conf = P.argmax(1), P.max(1)
    return np.where((pred != 2) & (conf >= tc), pred, 2)


def pick_tc(P, y, w=W):
    best = None
    for tc in GRID:
        g = gate(P, tc)
        key = (cost_arr(y, g, w).mean(), -np.mean(g != 2))
        if best is None or key < best[0]:
            best = (key, tc)
    return best[1]


def ece(conf, correct, bins):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def mcnemar(c1, c2):
    b = int(np.sum(c1 & ~c2)); c = int(np.sum(~c1 & c2))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def lr(C=10):
    return LogisticRegression(C=C, max_iter=20000, random_state=SEED)


def xgb(d=6, n=200):
    return XGBClassifier(n_estimators=n, max_depth=d, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                         random_state=SEED, eval_metric="mlogloss", n_jobs=4)


def load(name):
    f = RES / f"{name}.jsonl"
    return {o["sample_id"]: o for o in map(json.loads, f.read_text().splitlines())} if f.exists() else None


# ---------------------------------------------------------------- data
rows = [json.loads(l) for l in (ROOT / "data/multimodal_claims_benchmark.jsonl").read_text().splitlines()]
ids = np.array([r["sample_id"] for r in rows])
y = np.array([LABELS.index(r["decision_label"]) for r in rows])
split = np.array([r["split"] for r in rows])
text = [r["claim_description"] for r in rows]
nums = np.array([[r[k] for k in NUM] for r in rows], float)
clip = np.load(ROOT / "experiments/cache/clip_vitb32.npz")
cdict = dict(zip(clip["ids"], clip["emb"]))
IMG = np.array([cdict[i] for i in ids])


def feats_from(prefix, fields):
    out = np.full((len(rows), len(fields)), np.nan)
    for sp in ("train", "val", "test"):
        d = load(f"{prefix}_{sp}" if prefix != "decision1" else f"decision1_{sp}")
        if d is None:
            continue
        for i, s in enumerate(ids):
            if s in d:
                out[i] = [d[s][f] for f in fields]
    return out


D1 = feats_from("decision1", F7)
D1nn = np.full_like(D1, np.nan)
for sp in ("train", "val", "test"):
    d = load(f"decision1_{sp}_nonumbers")
    for i, s in enumerate(ids):
        if s in d:
            D1nn[i] = [d[s][f] for f in F7]
V2 = feats_from("v2", V2F)
B5F = feats_from("b5fair", F7)


def parse(t):
    sev = re.search(r"reported severity was ([a-z ]+)\.", t).group(1)
    inc = re.search(r"incident type was ([a-z\- ]+)\.", t).group(1)
    pol = re.search(r"police report status was ([a-z]+)\.", t).group(1)
    return sev, inc, pol


meta = [parse(t) for t in text]


# ---------------------------------------------------------------- 1-3, 7-8: fixed-split test analyses
def fixed_split_probs():
    tr, va, te = split == "train", split == "val", split == "test"
    out = {}

    def comb(X, name, Cgrid=(0.01, 0.1, 1, 10, 100)):
        best = None
        for C in Cgrid:
            m = lr(C).fit(X[tr], y[tr])
            ll = log_loss(y[va], m.predict_proba(X[va]), labels=[0, 1, 2])
            if best is None or ll < best[0]:
                best = (ll, C, m)
        out[name] = (best[2].predict_proba(X[va]), best[2].predict_proba(X[te]), best[1])

    comb(D1, "v1 combiner")
    comb(D1nn, "no-numbers combiner")
    comb(V2[:, :], "v2 combiner (post-hoc)")
    comb(D1, "v1 combiner, extended C grid", (0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5))
    if not np.isnan(B5F).any():
        comb(B5F, "B5-fair + combiner")
    bl = json.loads((RES / "baselines_val_test.json").read_text())
    for k, e in bl["models"].items():
        out[k] = (np.array(e["val_proba"]), np.array(e["test_proba"]), None)
    return out


def test_analyses(P):
    te = split == "test"
    yt = y[te]
    res = {"methods": {}}
    cor = {}
    for name, (pv, pt, C) in P.items():
        tc = pick_tc(pv, y[split == "val"])
        g = gate(pt, tc)
        pred = pt.argmax(1)
        correct = pred == yt
        cor[name] = correct
        # ECE + bootstrap
        e5 = [ece(pt.max(1)[i], correct[i], 5) for i in [rng.integers(0, len(yt), len(yt)) for _ in range(2000)]]
        e10 = [ece(pt.max(1)[i], correct[i], 10) for i in [rng.integers(0, len(yt), len(yt)) for _ in range(2000)]]
        # perfect-calibration null: labels drawn from the model's own probabilities
        null10 = []
        for _ in range(1000):
            ysim = np.array([rng.choice(3, p=p / p.sum()) for p in pt])
            null10.append(ece(pt.max(1), pred == ysim, 10))
        # cost vs escalate-all, paired bootstrap
        d = cost_arr(yt, g) - cost_arr(yt, np.full(len(yt), 2))
        boots = [d[i].mean() for i in [rng.integers(0, len(yt), len(yt)) for _ in range(10000)]]
        # risk-coverage over automatable predictions only (A/R), gate-consistent
        auto = pred != 2
        order = np.argsort(-pt.max(1)[auto])
        err = (~correct[auto])[order].astype(float)
        k = np.arange(1, len(err) + 1)
        rc_ar = (np.cumsum(err) / k)
        res["methods"][name] = {
            "C": C, "tau_c": float(tc), "accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(yt, pred, average="macro")),
            "ece5": float(ece(pt.max(1), correct, 5)), "ece5_ci": np.percentile(e5, [2.5, 97.5]).tolist(),
            "ece10": float(ece(pt.max(1), correct, 10)), "ece10_ci": np.percentile(e10, [2.5, 97.5]).tolist(),
            "ece10_null_perfect_calibration_median": float(np.median(null10)),
            "ece10_null_95": float(np.percentile(null10, 95)),
            "cost_gated": float(cost_arr(yt, g).mean()),
            "cost_minus_escalate_all": float(d.mean()), "cost_minus_escalate_all_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "coverage": float(np.mean(g != 2)), "wrong_approvals": int(np.sum((g == 0) & (yt != 0))),
            "aurc_automatable_only": float(rc_ar.mean()) if len(rc_ar) else None,
            "rc_automatable_only": rc_ar.tolist(),
        }
    # McNemar
    pairs = [("v1 combiner", "B4 early fusion (XGBoost)"), ("v2 combiner (post-hoc)", "v1 combiner"),
             ("v2 combiner (post-hoc)", "B4 early fusion (XGBoost)"), ("no-numbers combiner", "B4 early fusion (XGBoost)"),
             ("v1 combiner", "B1 text-only (TF-IDF + LR)"), ("v1 combiner", "B3 image-only (CLIP + LR)")]
    if "B5-fair + combiner" in cor:
        pairs.append(("v1 combiner", "B5-fair + combiner"))
    mc = []
    for a, b in pairs:
        bb, cc, p = mcnemar(cor[a], cor[b])
        mc.append({"a": a, "b": b, "a_only_correct": bb, "b_only_correct": cc, "exact_p": p})
    # confirmatory family: v1 vs B4, v1 vs B1, v1 vs B3 (the paper's stated comparisons with its pre-registered method)
    fam = [i for i, m in enumerate(mc) if m["a"] == "v1 combiner" and m["b"] in
           ("B4 early fusion (XGBoost)", "B1 text-only (TF-IDF + LR)", "B3 image-only (CLIP + LR)")]
    adj = holm(np.array([mc[i]["exact_p"] for i in fam]))
    for i, a in zip(fam, adj):
        mc[i]["holm_p_confirmatory_family"] = float(a)
    res["mcnemar"] = mc
    # dedup: test claims whose description also appears in train
    trtext = set(t for t, s in zip(text, split) if s == "train")
    keep = np.array([t not in trtext for t, s in zip(text, split) if s == "test"])
    res["dedup"] = {"n_overlap": int((~keep).sum()),
                    "acc_without_overlap": {k: float(c[keep].mean()) for k, c in cor.items()}}
    return res


# ---------------------------------------------------------------- 4: repeated CV over all 1,000 claims
def build_baseline_X(train_idx, test_idx, kind):
    if kind == "B1":
        tf = TfidfVectorizer(ngram_range=(1, 2)).fit([text[i] for i in train_idx])
        return tf.transform([text[i] for i in train_idx]), tf.transform([text[i] for i in test_idx])
    if kind == "B3":
        return IMG[train_idx], IMG[test_idx]
    if kind == "B4":
        tf = TfidfVectorizer(ngram_range=(1, 2)).fit([text[i] for i in train_idx])
        svd = TruncatedSVD(64, random_state=SEED).fit(tf.transform([text[i] for i in train_idx]))
        sc = StandardScaler().fit(nums[train_idx])
        f = lambda ix: np.hstack([IMG[ix], svd.transform(tf.transform([text[i] for i in ix])), sc.transform(nums[ix])])
        return f(train_idx), f(test_idx)
    raise ValueError(kind)


def fit_predict(kind, tr_idx, te_idx):
    if kind in ("v1", "nonum", "v2", "b5fair"):
        X = {"v1": D1, "nonum": D1nn, "v2": V2, "b5fair": B5F}[kind]
        return lr(10).fit(X[tr_idx], y[tr_idx]).predict_proba(X[te_idx])
    Xtr, Xte = build_baseline_X(tr_idx, te_idx, kind)
    m = xgb() if kind == "B4" else lr(10)
    return m.fit(Xtr, y[tr_idx]).predict_proba(Xte)


def cross_fitted(kind, idx, inner=5, seed=0):
    """Out-of-fold probabilities inside idx (for threshold selection without touching the outer test fold)."""
    oof = np.zeros((len(idx), 3))
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(idx, y[idx]):
        oof[b] = fit_predict(kind, idx[a], idx[b])
    return oof


def repeated_cv(kinds, repeats=5, folds=5):
    allidx = np.arange(len(rows))
    rec = {k: [] for k in kinds + ["escalate_all"]}
    perclaim = {k: [] for k in kinds}
    for r in range(repeats):
        for tr_idx, te_idx in StratifiedKFold(folds, shuffle=True, random_state=100 + r).split(allidx, y):
            yt = y[te_idx]
            rec["escalate_all"].append({"acc": np.nan, "cost": cost_arr(yt, np.full(len(yt), 2)).mean(), "cov": 0.0, "wa": 0})
            for k in kinds:
                tc = pick_tc(cross_fitted(k, tr_idx, seed=r), y[tr_idx])
                P = fit_predict(k, tr_idx, te_idx)
                g = gate(P, tc)
                rec[k].append({"acc": float((P.argmax(1) == yt).mean()), "f1": float(f1_score(yt, P.argmax(1), average="macro")),
                               "cost": float(cost_arr(yt, g).mean()), "cov": float(np.mean(g != 2)),
                               "wa": int(np.sum((g == 0) & (yt != 0))), "tc": float(tc)})
                perclaim[k].append((r, te_idx, P.argmax(1) == yt, cost_arr(yt, g)))
    summ = {}
    for k, v in rec.items():
        a = {m: np.array([d[m] for d in v], float) for m in v[0]}
        summ[k] = {m: {"mean": float(np.nanmean(x)), "sd_over_folds": float(np.nanstd(x))} for m, x in a.items()}
    # paired differences per repeat (pool folds within a repeat), CI over repeats via per-claim bootstrap
    def pooled(k, field):
        by_r = {}
        for r, te_idx, corr, c in perclaim[k]:
            by_r.setdefault(r, np.zeros(len(rows)))
            by_r[r][te_idx] = corr if field == "acc" else c
        return np.mean([by_r[r] for r in sorted(by_r)], axis=0)  # per-claim average over repeats
    esc = cost_arr(y, np.full(len(y), 2))
    diffs = {}
    for a, b in [("v1", "B4"), ("v2", "v1"), ("v2", "B4"), ("nonum", "B4"), ("v1", "B1"), ("b5fair", "v1")]:
        if a in kinds and b in kinds:
            for field in ("acc", "cost"):
                d = pooled(a, field) - pooled(b, field)
                bs = [d[i].mean() for i in [rng.integers(0, len(d), len(d)) for _ in range(5000)]]
                diffs[f"{a} - {b} ({field})"] = {"mean": float(d.mean()), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
    for k in kinds:
        d = pooled(k, "cost") - esc
        bs = [d[i].mean() for i in [rng.integers(0, len(d), len(d)) for _ in range(5000)]]
        diffs[f"{k} - escalate_all (cost)"] = {"mean": float(d.mean()), "ci": np.percentile(bs, [2.5, 97.5]).tolist()}
    return summ, diffs


# ---------------------------------------------------------------- 5: learning curves
def learning_curve(kinds, sizes=(25, 50, 100, 200, 400, 700), draws=10):
    tr_all = np.where(split == "train")[0]
    ev = np.where(split != "train")[0]          # val + test (300) as evaluation pool, no tuning on it
    out = {k: {} for k in kinds}
    for n in sizes:
        for k in kinds:
            accs, lls = [], []
            for dsd in range(draws if n < 700 else 1):
                r = np.random.default_rng(1000 + dsd)
                # stratified subsample
                sub = np.concatenate([r.choice(tr_all[y[tr_all] == c], max(2, round(n * np.mean(y[tr_all] == c))), replace=False)
                                      for c in range(3)])
                P = fit_predict(k, sub, ev)
                accs.append((P.argmax(1) == y[ev]).mean())
                lls.append(log_loss(y[ev], P, labels=[0, 1, 2]))
            out[k][n] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                         "logloss_mean": float(np.mean(lls))}
    return out


# ---------------------------------------------------------------- 6: shortcut controls
def image_quality_features():
    feats = []
    for r in rows:
        im = Image.open(ROOT / r["image_path"]).convert("L")
        w, h = im.size
        small = im.resize((max(1, w * 512 // max(w, h)), max(1, h * 512 // max(w, h))))
        a = np.asarray(small, float)
        lap = np.asarray(small.filter(ImageFilter.FIND_EDGES), float)
        feats.append([lap.var(), a.mean(), a.std(), w * h, w / h])
    return np.array(feats)


def shortcut_controls(Pfixed):
    out = {}
    te = split == "test"
    # (a) no-learning text rule: escalate if theft or police unrecorded; audit if total loss; approve otherwise
    def text_rule(i):
        sev, inc, pol = meta[i]
        if "theft" in inc or pol == "unrecorded":
            return 2
        if sev == "total loss":
            return 1
        return 0
    tr_pred = np.array([text_rule(i) for i in range(len(rows))])
    out["text_rule_test_acc"] = float((tr_pred[te] == y[te]).mean())
    out["text_rule_all_acc"] = float((tr_pred == y).mean())
    # (b) image-quality detector for Escalate vs rest (train on train, evaluate on test)
    Q = np.log1p(image_quality_features())
    sc = StandardScaler().fit(Q[split == "train"])
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Q[split == "train"]), (y[split == "train"] == 2))
    pq = clf.predict_proba(sc.transform(Q[te]))[:, 1]
    out["image_quality_detector_auc_E_vs_rest_test"] = float(roc_auc_score(y[te] == 2, pq))
    out["s3_auc_E_vs_rest_test"] = float(roc_auc_score(y[te] == 2, 1 - D1[te][:, 4]))
    # (c) rule + image detector (no API): text rule, but escalate also if detector > 0.5
    comb_pred = tr_pred.copy()
    comb_pred[te] = np.where(pq > 0.5, 2, tr_pred[te])
    out["text_rule_plus_quality_detector_test_acc"] = float((comb_pred[te] == y[te]).mean())
    # (d) A/R-only accuracy and binary A-vs-R accuracy on test
    ar = te & (y != 2)
    out["ar_only"] = {}
    for name, (pv, pt, _) in Pfixed.items():
        pred = pt.argmax(1)
        mask = (y[te] != 2)
        out["ar_only"][name] = {"three_way_acc_on_AR_claims": float((pred[mask] == y[te][mask]).mean()),
                                "binary_A_vs_R_acc": float((np.where(pt[mask][:, 0] >= pt[mask][:, 1], 0, 1) == y[te][mask]).mean())}
    # (e) de-shortcut subset: A/R claims (val+test) reporting major damage with police status recorded,
    #     where neither severity text nor police status separates the classes
    vt = split != "train"
    ds = vt & (y != 2) & np.array([m[0] == "major damage" and m[2] in ("available", "unavailable") for m in meta])
    out["deshortcut_subset"] = {"n": int(ds.sum()), "class_counts": {LABELS[c]: int(np.sum(y[ds] == c)) for c in range(3)}}
    vt_idx = np.where(vt)[0]
    pos = {i: j for j, i in enumerate(vt_idx)}
    sel = [pos[i] for i in np.where(ds)[0]]
    out["deshortcut_subset"]["acc"] = {}
    for name, (pv, pt, _) in Pfixed.items():
        Pvt = np.vstack([pv, pt])  # val rows then test rows, both in file order
        order = np.concatenate([np.where(split == "val")[0], np.where(split == "test")[0]])
        P_by_idx = {i: Pvt[k] for k, i in enumerate(order)}
        pred = np.array([P_by_idx[i].argmax() for i in np.where(ds)[0]])
        out["deshortcut_subset"]["acc"][name] = float((pred == y[ds]).mean())
    out["deshortcut_subset"]["text_rule_acc"] = float((tr_pred[ds] == y[ds]).mean())
    return out


# ---------------------------------------------------------------- 9: retest agreement
def retest_agreement():
    f = ROOT / "experiments/cache/retest_features.npy"
    if not f.exists():
        return None
    orig, new = np.load(f)
    tr = split == "train"
    best = None
    for C in (0.01, 0.1, 1, 10, 100):
        m = lr(C).fit(D1[tr], y[tr])
        ll = log_loss(y[split == "val"], m.predict_proba(D1[split == "val"]), labels=[0, 1, 2])
        if best is None or ll < best[0]:
            best = (ll, m)
    m = best[1]
    po, pn = m.predict_proba(orig), m.predict_proba(new)
    vt = split == "val"
    tc = pick_tc(m.predict_proba(D1[vt]), y[vt])
    return {"route_agreement": float(np.mean(po.argmax(1) == pn.argmax(1))),
            "gated_decision_agreement": float(np.mean(gate(po, tc) == gate(pn, tc))),
            "max_abs_prob_change": float(np.abs(po - pn).max())}


def main():
    P = fixed_split_probs()
    res = {"test": test_analyses(P)}
    res["shortcuts"] = shortcut_controls(P)
    res["retest"] = retest_agreement()
    kinds = ["v1", "nonum", "v2", "B1", "B3", "B4"] + (["b5fair"] if not np.isnan(B5F).any() else [])
    res["cv"], res["cv_diffs"] = repeated_cv(kinds)
    res["learning_curve"] = learning_curve(["v1", "B1", "B3", "B4"] + (["b5fair"] if not np.isnan(B5F).any() else []))
    (RES / "revision_stats.json").write_text(json.dumps(res, indent=2))

    t = res["test"]
    print("== Fixed split (test, n=150)")
    for k, v in t["methods"].items():
        print(f"{k[:34]:34} acc {v['accuracy']:.3f} ECE10 {v['ece10']:.3f} [{v['ece10_ci'][0]:.3f},{v['ece10_ci'][1]:.3f}] "
              f"null95 {v['ece10_null_95']:.3f} ECE5 {v['ece5']:.3f} cost {v['cost_gated']:.3f} "
              f"vs esc {v['cost_minus_escalate_all']:+.3f} [{v['cost_minus_escalate_all_ci'][0]:+.3f},{v['cost_minus_escalate_all_ci'][1]:+.3f}] "
              f"AURC(A/R) {v['aurc_automatable_only'] if v['aurc_automatable_only'] is None else round(v['aurc_automatable_only'],3)} C={v['C']}")
    print("\n== McNemar")
    for m in t["mcnemar"]:
        print(f"{m['a'][:26]:26} vs {m['b'][:26]:26} {m['a_only_correct']:3d}/{m['b_only_correct']:3d} p={m['exact_p']:.3f}"
              + (f" holm={m['holm_p_confirmatory_family']:.3f}" if "holm_p_confirmatory_family" in m else ""))
    print("\n== Dedup:", t["dedup"]["n_overlap"], {k[:20]: round(v, 3) for k, v in t["dedup"]["acc_without_overlap"].items()})
    print("\n== Shortcuts:", json.dumps({k: v for k, v in res["shortcuts"].items() if k not in ("ar_only", "deshortcut_subset")}, indent=1))
    print("A/R only:", json.dumps({k[:24]: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res["shortcuts"]["ar_only"].items()}, indent=1))
    print("De-shortcut subset:", json.dumps(res["shortcuts"]["deshortcut_subset"], indent=1))
    print("\n== Retest:", res["retest"])
    print("\n== Repeated 5x5 CV (all 1,000 claims)")
    for k, v in res["cv"].items():
        print(f"{k:12} " + "  ".join(f"{m} {s['mean']:.3f}±{s['sd_over_folds']:.3f}" for m, s in v.items()))
    print("CV paired differences:")
    for k, v in res["cv_diffs"].items():
        print(f"  {k:34} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    print("\n== Learning curve (eval on val+test, 300 claims)")
    for k, v in res["learning_curve"].items():
        print(f"{k:8} " + "  ".join(f"n={n}: {s['acc_mean']:.3f}±{s['acc_sd']:.3f}" for n, s in v.items()))


if __name__ == "__main__":
    main()
