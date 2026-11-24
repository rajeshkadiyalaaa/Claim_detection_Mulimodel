"""Decoupled-degradation control (round-2 review: DA-1 residual, R2 W1).

In the benchmark every escalation photograph was degraded and no other photograph
was. To measure how much of the escalation route is detection of that artefact,
this script degrades the clear photographs of the 103 test approvals and audits
with the three documented operations, re-sends the otherwise identical v1 request
to Decision 1, and passes the answers through the frozen combiner (C = 100,
trained on the training split) and the frozen gate (tau_c = 0.85).

Degradations follow the `pairing_reason` field: Gaussian blur (radius 17 px, the
median of the documented radii), simulated low light at 24% brightness, and a
corner crop to about 18% of the original area. Each claim gets one degradation,
assigned in rotation after a seeded shuffle, so each operation covers about a third.
Degraded images are written to experiments/cache/ (not redistributed).

Cost: 103 requests of about 1,100 input tokens at US$0.10 per million, about US$0.01.
"""
import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression

import decision1 as d1

LABELS = ["Auto-Approve", "Reject_Fraud_Audit", "Escalate_Human"]
F7 = ["p_A", "p_R", "p_E", "s2", "s3", "s4", "s5"]
OPS = ["blur", "low_light", "corner_crop"]
TAU_C = 0.85
C_FROZEN = 100


def degrade(img, op, rng):
    img = img.convert("RGB")
    if op == "blur":
        return img.filter(ImageFilter.GaussianBlur(17))
    if op == "low_light":
        return ImageEnhance.Brightness(img).enhance(0.24)
    w, h = img.size
    cw, ch = round(w * 0.42), round(h * 0.42)          # 0.42^2 = 17.6% of the area
    corner = rng.integers(4)
    x0 = 0 if corner in (0, 2) else w - cw
    y0 = 0 if corner in (0, 1) else h - ch
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def encode(img):
    scale = d1.MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def feats(resp):
    a = resp["answers"]
    p = a["q1_route"]["probabilities"]
    return [p["auto_approve"], p["fraud_audit"], p["escalate_human"],
            a["q2_damage_match"]["noul"], a["q3_photo_clear"]["noul"],
            a["q4_details_complete"]["noul"], a["q5_amounts_match"]["noul"]]


def main():
    key = d1.KEY_FILE.read_text().strip()
    rows = [json.loads(l) for l in d1.DATA.read_text().splitlines()]
    test = [r for r in rows if r["split"] == "test" and r["decision_label"] != "Escalate_Human"]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(test))
    op_of = {test[j]["sample_id"]: OPS[k % 3] for k, j in enumerate(order)}

    img_dir = d1.ROOT / "experiments/cache/degraded_images"
    cache = d1.ROOT / "experiments/cache/degraded_test"
    img_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        f = cache / f"{row['sample_id']}.json"
        if f.exists():
            return json.loads(f.read_text())
        crop_rng = np.random.default_rng(int(row["sample_id"].split("-")[1]))
        with Image.open(d1.ROOT / row["image_path"]) as im:
            deg = degrade(im, op_of[row["sample_id"]], crop_rng)
        deg.save(img_dir / f"{row['sample_id']}.jpg", quality=90)
        blocks = d1.build_input(row)
        for b in blocks:
            if b["type"] == "image":
                b["data"] = encode(deg)
        payload = {"model": d1.MODEL, "input": blocks, "questions": d1.QUESTIONS}
        res, lat = d1.call_api(payload, key)
        rec = {"sample_id": row["sample_id"], "op": op_of[row["sample_id"]], "latency_s": lat, "response": res}
        f.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(2) as ex:
        recs = list(ex.map(fetch, test))

    # frozen combiner and gate
    def load(sp):
        return {o["sample_id"]: o for o in map(json.loads, (d1.ROOT / f"experiments/results/decision1_{sp}.jsonl").read_text().splitlines())}
    tr = load("train")
    te = load("test")
    Xtr = np.array([[o[k] for k in F7] for o in tr.values()])
    ytr = np.array([LABELS.index(o["label"]) for o in tr.values()])
    m = LogisticRegression(C=C_FROZEN, max_iter=20000, random_state=42).fit(Xtr, ytr)

    def gate(P):
        pred, conf = P.argmax(1), P.max(1)
        return np.where((pred != 2) & (conf >= TAU_C), pred, 2)

    y = np.array([LABELS.index(r["decision_label"]) for r in test])
    Xc = np.array([[te[r["sample_id"]][k] for k in F7] for r in test])
    Xd = np.array([feats(rec["response"]) for rec in recs])
    Pc, Pd = m.predict_proba(Xc), m.predict_proba(Xd)
    ops = np.array([rec["op"] for rec in recs])

    def summ(P, mask):
        g = gate(P[mask])
        return {"n": int(mask.sum()),
                "route_E_before_gate": float(np.mean(P[mask].argmax(1) == 2)),
                "gated_escalated": float(np.mean(g == 2)),
                "gated_auto_approve": float(np.mean(g == 0)),
                "gated_audit": float(np.mean(g == 1)),
                "mean_s3": float(np.mean(Xd[mask][:, 4] if P is Pd else Xc[mask][:, 4]))}

    out = {"n": len(test), "ops": {o: int(np.sum(ops == o)) for o in OPS},
           "clear": summ(Pc, np.ones(len(test), bool)),
           "degraded": summ(Pd, np.ones(len(test), bool)),
           "degraded_by_op": {o: summ(Pd, ops == o) for o in OPS},
           "true_escalations_reference": None}
    # reference: what the router does on the real (degraded) test escalations
    E = [r for r in rows if r["split"] == "test" and r["decision_label"] == "Escalate_Human"]
    XE = np.array([[te[r["sample_id"]][k] for k in F7] for r in E])
    PE = m.predict_proba(XE)
    out["true_escalations_reference"] = {"n": len(E), "gated_escalated": float(np.mean(gate(PE) == 2)),
                                         "route_E_before_gate": float(np.mean(PE.argmax(1) == 2)),
                                         "mean_s3": float(XE[:, 4].mean())}
    (d1.ROOT / "experiments/results/degradation_control.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
