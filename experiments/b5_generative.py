"""Baseline B5: generative Gemini model on the same request content.

The model receives the same instruction, photograph, description and numerical
block as Decision 1 and must return JSON {"route": ..., "confidence": ...}.
Latency and token usage (including thinking tokens, if any) are recorded.

Usage:
  .venv/bin/python experiments/b5_generative.py --split val [--limit 1] [--model gemini-3.8-flash]
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from decision1 import DATA, KEY_FILE, ROOT, encode_image, serialise_numbers, INSTRUCTION

ROUTES = {"auto_approve": "Auto-Approve", "fraud_audit": "Reject_Fraud_Audit", "escalate_human": "Escalate_Human"}
TASK = (
    "\nDecide which handling route this claim requires. Options:\n"
    "- auto_approve: the evidence is clear, complete and consistent.\n"
    "- fraud_audit: the photograph and the claim disagree, or the claim appears inflated.\n"
    "- escalate_human: the evidence is too unclear or incomplete to decide.\n"
    "Return JSON with the chosen route and your confidence between 0 and 1 that it is correct."
)
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "route": {"type": "STRING", "enum": list(ROUTES)},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["route", "confidence"],
}


def payload(row):
    text = ("\nClaim description: " + row["claim_description"] + "\n"
            "\nClaim figures:\n" + serialise_numbers(row) + "\n" + TASK)
    return {
        "contents": [{"role": "user", "parts": [
            {"text": INSTRUCTION + "\n"},
            {"inline_data": {"mime_type": "image/jpeg", "data": encode_image(ROOT / row["image_path"])}},
            {"text": text},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA, "temperature": 0},
    }


def call(model, body, key, retries=5):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:400]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    key = KEY_FILE.read_text().strip()
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == a.split][: a.limit]
    cache = ROOT / "experiments" / "cache" / f"b5_{a.model}_{a.split}"
    cache.mkdir(parents=True, exist_ok=True)

    def work(row):
        f = cache / f"{row['sample_id']}.json"
        if f.exists():
            return json.loads(f.read_text())
        try:
            resp, lat = call(a.model, payload(row), key)
        except Exception as exc:
            print(row["sample_id"], "FAILED", exc)
            return None
        rec = {"sample_id": row["sample_id"], "latency_s": lat, "response": resp,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        f.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(work, rows))

    out = []
    for row, rec in zip(rows, recs):
        if rec is None:
            continue
        resp = rec["response"]
        um = resp.get("usageMetadata", {})
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            ans = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
            route, conf = ROUTES[ans["route"]], float(ans["confidence"])
        except Exception:
            route, conf = "Escalate_Human", 0.0  # unparseable output is treated as an escalation
        out.append({"sample_id": row["sample_id"], "label": row["decision_label"], "pred": route,
                    "conf": min(max(conf, 0.0), 1.0), "latency_s": rec["latency_s"],
                    "input_tokens": um.get("promptTokenCount"), "output_tokens": um.get("candidatesTokenCount"),
                    "thinking_tokens": um.get("thoughtsTokenCount", 0), "model_version": resp.get("modelVersion")})
    res = ROOT / "experiments" / "results" / f"b5_{a.model}_{a.split}.jsonl"
    res.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {len(out)} rows to {res.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
