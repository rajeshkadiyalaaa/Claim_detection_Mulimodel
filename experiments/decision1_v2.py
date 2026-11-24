"""Level-1 revision (post-hoc, decided after the v1 test run; the error analysis is summarised in the paper, Section 4.6).

Same model, same inputs (photo + description + numbers), revised request:
  * the instruction states the insurer's handling policy (the benchmark's documented routing rules);
  * q1 options are defined by visible evidence and the policy, without the word "inflated";
  * new checks: police-report status recorded, theft claim, vehicle matches stated make/model;
  * the numeric figures are described as coming from the policy record.

Tuning is done on TRAIN/VAL only. Any test score from this version is reported as a
post-hoc analysis, never as the main result.

Usage: .venv/bin/python experiments/decision1_v2.py --split val [--limit 1] [--workers 3]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import decision1 as d1

d1.INSTRUCTION = (
    "You are assessing a motor insurance claim under this handling policy:\n"
    "- Escalate to a human reviewer when the police-report status is unrecorded, when the claim is a theft, "
    "or when the photograph is too blurred, dark or tightly cropped to judge the damage.\n"
    "- Send to fraud audit when the photograph is clear but the visible damage is much smaller than the severity "
    "reported in the claim description.\n"
    "- Approve automatically when the photograph is clear, the details are recorded and the visible damage is "
    "compatible with the reported severity.\n"
    "The photograph comes from the claimant. The claim figures come from the policy record and are context only.\n"
)

QUESTIONS = {
    "q1_route": {
        "type": "choice",
        "instructions": "Under the handling policy, which route does this claim require?",
        "criteria": {
            "auto_approve": "Clear photograph, details recorded, visible damage compatible with the reported severity.",
            "fraud_audit": "Clear photograph, but the visible damage is much smaller than the reported severity.",
            "escalate_human": "Police-report status unrecorded, a theft claim, or a photograph too poor to judge.",
        },
    },
    "q2_damage_match": {"type": "noul", "instructions":
        "Is the damage visible in the photograph compatible with the severity reported in the claim description?"},
    "q3_photo_clear": {"type": "noul", "instructions":
        "Is the photograph clear, well lit and complete enough to judge the extent of the damage?"},
    "q4_police_recorded": {"type": "noul", "instructions":
        "Is the police-report status recorded (available or not available), rather than unrecorded?"},
    "q5_theft": {"type": "noul", "instructions": "Is this a theft claim?"},
    "q6_vehicle_match": {"type": "noul", "instructions":
        "Does the vehicle in the photograph plausibly match the make and model stated in the claim description?"},
    "q7_amounts_match": {"type": "noul", "instructions":
        "Are the claimed amounts consistent with the reported severity and incident type?"},
}
CHECK_KEYS = [k for k in QUESTIONS if k != "q1_route"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    key = d1.KEY_FILE.read_text().strip()
    rows = [json.loads(l) for l in d1.DATA.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == a.split][: a.limit]
    cache = d1.ROOT / "experiments" / "cache" / f"v2_{a.split}"
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        f = cache / f"{row['sample_id']}.json"
        if f.exists():
            return json.loads(f.read_text())
        payload = {"model": d1.MODEL, "input": d1.build_input(row), "questions": QUESTIONS}
        try:
            res, lat = d1.call_api(payload, key)
        except Exception as exc:
            print(row["sample_id"], "FAILED", exc)
            return None
        rec = {"sample_id": row["sample_id"], "latency_s": lat, "workers": a.workers,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": res}
        f.write_text(json.dumps(rec))
        time.sleep(0.35)
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(fetch, rows))

    out = []
    for row, rec in zip(rows, recs):
        if rec is None:
            continue
        ans = rec["response"]["answers"]
        pr = ans["q1_route"]["probabilities"]
        o = {"sample_id": row["sample_id"], "label": row["decision_label"],
             "pred": d1.ROUTE_TO_LABEL[ans["q1_route"]["choice"]],
             "p_A": pr["auto_approve"], "p_R": pr["fraud_audit"], "p_E": pr["escalate_human"],
             "conf": ans["q1_route"]["confidence"], "latency_s": rec["latency_s"],
             "input_tokens": rec["response"].get("usage", {}).get("total_input_tokens")}
        o.update({k: ans[k]["noul"] for k in CHECK_KEYS})
        out.append(o)
    path = d1.ROOT / "experiments" / "results" / f"v2_{a.split}.jsonl"
    path.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {len(out)} rows to {path.relative_to(d1.ROOT)}")


if __name__ == "__main__":
    main()
