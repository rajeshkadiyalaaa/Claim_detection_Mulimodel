"""Test-retest stability of Gemini Decision 1 (revision; R1 W11, R3 W6).

Re-sends the identical v1 request for the first 100 test claims (in file order)
and compares the seven returned probabilities and the combiner's routing decision
with the original call cached in experiments/cache/test/.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import decision1 as d1

N = 100


def main():
    key = d1.KEY_FILE.read_text().strip()
    rows = [json.loads(l) for l in d1.DATA.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "test"][:N]
    cache = d1.ROOT / "experiments" / "cache" / "retest_test"
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        f = cache / f"{row['sample_id']}.json"
        if f.exists():
            return json.loads(f.read_text())
        payload = {"model": d1.MODEL, "input": d1.build_input(row), "questions": d1.QUESTIONS}
        res, lat = d1.call_api(payload, key)
        rec = {"sample_id": row["sample_id"], "latency_s": lat, "response": res}
        f.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(2) as ex:
        recs = list(ex.map(fetch, rows))

    def feats(resp):
        a = resp["answers"]
        p = a["q1_route"]["probabilities"]
        return [p["auto_approve"], p["fraud_audit"], p["escalate_human"],
                a["q2_damage_match"]["noul"], a["q3_photo_clear"]["noul"],
                a["q4_details_complete"]["noul"], a["q5_amounts_match"]["noul"]]

    orig = np.array([feats(json.loads((d1.ROOT / "experiments/cache/test" / f"{r['sample_id']}.json").read_text())["response"]) for r in rows])
    new = np.array([feats(r["response"]) for r in recs])
    names = ["p_A", "p_R", "p_E", "s2", "s3", "s4", "s5"]
    out = {"n": N,
           "max_abs_diff": dict(zip(names, np.abs(orig - new).max(0).round(6).tolist())),
           "mean_abs_diff": dict(zip(names, np.abs(orig - new).mean(0).round(6).tolist())),
           "pearson_r": dict(zip(names, [float(np.corrcoef(orig[:, j], new[:, j])[0, 1]) for j in range(7)])),
           "q1_choice_agreement": float(np.mean(orig[:, :3].argmax(1) == new[:, :3].argmax(1))),
           "identical_responses": int(np.sum(np.all(np.isclose(orig, new, atol=1e-9), axis=1)))}
    np.save(d1.ROOT / "experiments/cache/retest_features.npy", np.stack([orig, new]))
    (d1.ROOT / "experiments/results/stability_retest.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
