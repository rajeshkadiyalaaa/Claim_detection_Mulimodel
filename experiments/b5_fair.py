"""Fair generative baseline (B5-fair), added in revision after reviewer comments
(JFR W1, R1 W5, DA-3). Differences from the original B5 are removed:

  * thinking budget set to 0 (lowest setting this model accepts),
  * same image preprocessing and MEDIUM media resolution as Decision 1,
  * the same five questions as Decision 1 (route + four yes/no checks),
  * its answers can be fed through the same combiner (see revision_stats.py).

Each answer is returned as JSON with a probability in [0, 1] for every option,
so the combiner gets the same seven features as for Decision 1.

Usage: .venv/bin/python experiments/b5_fair.py --split val [--limit 1] [--workers 1]
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from decision1 import DATA, KEY_FILE, ROOT, INSTRUCTION, QUESTIONS, encode_image, serialise_numbers

MODEL = "gemini-3.8-flash"
TASK = (
    "\nAnswer the five questions below about this claim. For each question give a probability "
    "between 0 and 1 for every option; the probabilities of one question's options must sum to 1.\n"
    f"q1. {QUESTIONS['q1_route']['instructions']} Options: "
    + "; ".join(f"{k} = {v}" for k, v in QUESTIONS["q1_route"]["criteria"].items()) + "\n"
    f"q2. {QUESTIONS['q2_damage_match']['instructions']} (yes/no)\n"
    f"q3. {QUESTIONS['q3_photo_clear']['instructions']} (yes/no)\n"
    f"q4. {QUESTIONS['q4_details_complete']['instructions']} (yes/no)\n"
    f"q5. {QUESTIONS['q5_amounts_match']['instructions']} (yes/no)\n"
    "Return JSON only."
)
NUM = {"type": "NUMBER"}
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "q1_auto_approve": NUM, "q1_fraud_audit": NUM, "q1_escalate_human": NUM,
        "q2_yes": NUM, "q3_yes": NUM, "q4_yes": NUM, "q5_yes": NUM,
    },
    "required": ["q1_auto_approve", "q1_fraud_audit", "q1_escalate_human", "q2_yes", "q3_yes", "q4_yes", "q5_yes"],
}
LABELS = {"auto_approve": "Auto-Approve", "fraud_audit": "Reject_Fraud_Audit", "escalate_human": "Escalate_Human"}


def payload(row, thinking):
    text = ("\nClaim description: " + row["claim_description"] + "\n"
            "\nClaim figures:\n" + serialise_numbers(row) + "\n" + TASK)
    gen = {"responseMimeType": "application/json", "responseSchema": SCHEMA, "temperature": 0,
           "mediaResolution": "MEDIA_RESOLUTION_MEDIUM"}
    gen["thinkingConfig"] = thinking
    return {"contents": [{"role": "user", "parts": [
        {"text": INSTRUCTION + "\n"},
        {"inline_data": {"mime_type": "image/jpeg", "data": encode_image(ROOT / row["image_path"])}},
        {"text": text}]}], "generationConfig": gen}


def call(body, key, retries=5):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
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
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--thinking", default='{"thinkingBudget": 0}')
    a = ap.parse_args()
    thinking = json.loads(a.thinking)
    key = KEY_FILE.read_text().strip()
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == a.split][: a.limit]
    cache = ROOT / "experiments" / "cache" / f"b5fair_{a.split}"
    cache.mkdir(parents=True, exist_ok=True)

    def work(row):
        f = cache / f"{row['sample_id']}.json"
        if f.exists():
            return json.loads(f.read_text())
        try:
            resp, lat = call(payload(row, thinking), key)
        except Exception as exc:
            print(row["sample_id"], "FAILED", exc)
            return None
        rec = {"sample_id": row["sample_id"], "latency_s": lat, "workers": a.workers, "thinking": thinking,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": resp}
        f.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(work, rows))

    out, bad = [], 0
    for row, rec in zip(rows, recs):
        if rec is None:
            continue
        resp = rec["response"]
        um = resp.get("usageMetadata", {})
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            ans = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
            pA, pR, pE = (max(float(ans[k]), 0.0) for k in ("q1_auto_approve", "q1_fraud_audit", "q1_escalate_human"))
            s = pA + pR + pE
            pA, pR, pE = (pA / s, pR / s, pE / s) if s > 0 else (1 / 3, 1 / 3, 1 / 3)
            checks = [min(max(float(ans[k]), 0.0), 1.0) for k in ("q2_yes", "q3_yes", "q4_yes", "q5_yes")]
        except Exception:
            bad += 1
            pA, pR, pE, checks = 0.0, 0.0, 1.0, [0.5] * 4
        probs = {"Auto-Approve": pA, "Reject_Fraud_Audit": pR, "Escalate_Human": pE}
        pred = max(probs, key=probs.get)
        out.append({"sample_id": row["sample_id"], "label": row["decision_label"], "pred": pred,
                    "p_A": pA, "p_R": pR, "p_E": pE, "conf": probs[pred],
                    "s2": checks[0], "s3": checks[1], "s4": checks[2], "s5": checks[3],
                    "latency_s": rec["latency_s"], "input_tokens": um.get("promptTokenCount"),
                    "output_tokens": um.get("candidatesTokenCount"), "thinking_tokens": um.get("thoughtsTokenCount", 0)})
    path = ROOT / "experiments" / "results" / f"b5fair_{a.split}.jsonl"
    path.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {len(out)} rows ({bad} unparseable) to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
