"""Frozen CLIP ViT-B/32 image embeddings for all 1,000 benchmark photographs.

Images are resized the same way as for Decision 1 (longer side <= 1,024 px)
and then passed through the CLIP processor (224 x 224). Embeddings are
L2-normalised and saved to experiments/cache/clip_vitb32.npz (git-ignored).
"""
import json

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from decision1 import DATA, ROOT

MODEL_ID = "openai/clip-vit-base-patch32"


def main():
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    embs = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = [Image.open(ROOT / r["image_path"]).convert("RGB") for r in rows[i:i + 32]]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            feats = out if torch.is_tensor(out) else out.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embs.append(feats.cpu().numpy())
    emb = np.concatenate(embs)
    out = ROOT / "experiments" / "cache" / "clip_vitb32.npz"
    np.savez(out, ids=np.array([r["sample_id"] for r in rows]), emb=emb)
    print("saved", emb.shape, "to", out.relative_to(ROOT), "device", device)


if __name__ == "__main__":
    main()
