"""Score benchmark claims with Gemini Decision 1 (Decisions API, REST v1beta).

One request per claim carries the instruction, the photograph, the categorical
description and the serialised numerical fields, plus five closed-set questions
(q1 routing choice, q2-q5 yes/no checks). Raw API responses are cached under
experiments/cache/ so that reruns do not re-bill the API.

Usage:
  .venv/bin/python experiments/decision1.py --split val [--limit 1] [--drop image|text|numbers]
"""
import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multimodal_claims_benchmark.jsonl"
KEY_FILE = ROOT / "geminiapi.txt"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
MODEL = "gemini-decision-1"
MAX_SIDE = 1024
IMAGE_RESOLUTION = "medium"

NUMERIC_FIELDS = [
    ("total_claim_amount", "Total claim amount"),
    ("vehicle_claim", "Vehicle damage claim"),
    ("injury_claim", "Injury claim"),
    ("property_claim", "Property damage claim"),
    ("policy_deductable", "Policy deductible"),
    ("policy_annual_premium", "Annual policy premium"),
    ("age", "Policyholder age (years)"),
    ("months_as_customer", "Months as customer"),
    ("auto_year", "Vehicle model year"),
    ("bodily_injuries", "Reported bodily injuries"),
    ("witnesses", "Reported witnesses"),
]

INSTRUCTION = (
    "You are assessing a motor insurance claim. The photograph shows the claimed "
    "vehicle. The claim description and claim figures follow. Answer each question "
    "using only this evidence."
)

QUESTIONS = {
    "q1_route": {
        "type": "choice",
        "instructions": "Which handling route does this claim require?",
        "criteria": {
            "auto_approve": "Auto-Approve: the evidence is clear, complete and consistent.",
            "fraud_audit": "Reject/Fraud-Audit: the photograph and the claim disagree, or the claim appears inflated.",
            "escalate_human": "Escalate-Human: the evidence is too unclear or incomplete to decide.",
        },
    },
    "q2_damage_match": {
        "type": "noul",
        "instructions": "Is the damage visible in the photograph consistent with the severity reported in the claim description?",
    },
    "q3_photo_clear": {
        "type": "noul",
        "instructions": "Is the photograph clear, well lit and complete enough to judge the extent of the damage?",
    },
    "q4_details_complete": {
        "type": "noul",
        "instructions": "Are the claim details complete and unambiguous?",
    },
    "q5_amounts_match": {
        "type": "noul",
        "instructions": "Are the claimed amounts consistent with the reported severity and incident type?",
    },
}

ROUTE_TO_LABEL = {
    "auto_approve": "Auto-Approve",
    "fraud_audit": "Reject_Fraud_Audit",
    "escalate_human": "Escalate_Human",
}


def serialise_numbers(row):
    lines = []
    for key, label in NUMERIC_FIELDS:
        value = row[key]
        lines.append(f"{label}: {value:,.2f}" if isinstance(value, float) else f"{label}: {value:,}")
    return "\n".join(lines)


def encode_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        scale = MAX_SIDE / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_input(row, drop=None):
    blocks = [{"type": "text", "text": INSTRUCTION + "\n"}]
    if drop != "image":
        blocks.append({
            "type": "image",
            "data": encode_image(ROOT / row["image_path"]),
            "mime_type": "image/jpeg",
            "resolution": IMAGE_RESOLUTION,
        })
    text = ""
    if drop != "text":
        text += "\nClaim description: " + row["claim_description"] + "\n"
    if drop != "numbers":
        text += "\nClaim figures:\n" + serialise_numbers(row) + "\n"
    if text:
        blocks.append({"type": "text", "text": text})
    return blocks


def call_api(payload, key, retries=5):
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.load(resp)
            return result, time.perf_counter() - start
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            if err.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"HTTP {err.code}: {detail}") from None
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drop", choices=["image", "text", "numbers"], default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    key = KEY_FILE.read_text().strip()
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == args.split][: args.limit]

    tag = f"{args.split}" + (f"_no{args.drop}" if args.drop else "")
    cache = ROOT / "experiments" / "cache" / tag
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        cfile = cache / f"{row['sample_id']}.json"
        if cfile.exists():
            return json.loads(cfile.read_text())
        payload = {"model": MODEL, "input": build_input(row, args.drop), "questions": QUESTIONS}
        try:
            result, latency = call_api(payload, key)
        except Exception as exc:  # record the failure and continue
            print(f"{row['sample_id']} FAILED: {exc}")
            return None
        rec = {"sample_id": row["sample_id"], "latency_s": latency, "workers": args.workers,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": result}
        cfile.write_text(json.dumps(rec))
        time.sleep(0.35)  # stay well under the free-tier request rate
        return rec

    # Latency is reported only from sequential runs (workers=1); parallel runs are for ablations.
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(fetch, rows))

    out = []
    for i, (row, rec) in enumerate(zip(rows, recs), 1):
        if rec is None:
            continue
        ans = rec["response"]["answers"]
        usage = rec["response"].get("usage", {})
        by_mod = {m.get("modality"): m.get("tokens") for m in usage.get("input_tokens_by_modality", [])}
        probs = ans["q1_route"]["probabilities"]
        out.append({
            "sample_id": row["sample_id"],
            "label": row["decision_label"],
            "pred": ROUTE_TO_LABEL[ans["q1_route"]["choice"]],
            "p_A": probs.get("auto_approve"), "p_R": probs.get("fraud_audit"), "p_E": probs.get("escalate_human"),
            "conf": ans["q1_route"]["confidence"],
            "s2": ans["q2_damage_match"]["noul"], "s3": ans["q3_photo_clear"]["noul"],
            "s4": ans["q4_details_complete"]["noul"], "s5": ans["q5_amounts_match"]["noul"],
            "latency_s": rec["latency_s"],
            "input_tokens": usage.get("total_input_tokens"),
            "image_tokens": by_mod.get("image"), "text_tokens": by_mod.get("text"),
            "output_tokens": usage.get("total_output_tokens"),
        })
        if i % 10 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] done")

    res_dir = ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / f"decision1_{tag}.jsonl"
    out_file.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {len(out)} rows to {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
