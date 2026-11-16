"""
Usage (from the backend folder, venv active):
    python debug_detect.py path\\to\\eye_photo.jpg
    python debug_detect.py path\\to\\eye_photo.jpg --ultra C:\\Users\\LOQ\\eye_seg_model.pt   (optional, needs: pip install ultralytics)

Prints what the fused model detects under two resize methods and saves
debug_letterbox.jpg / debug_squash.jpg with the best box drawn on them.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from build_unified_model import HerHealthUnifiedModel

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--model", default=os.getenv("HERHEALTH_MODEL_PATH", "herhealth_unified.pt"))
ap.add_argument("--ultra", default=None, help="path to the ORIGINAL eye_seg_model.pt for a side-by-side check")
args = ap.parse_args()

img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
print(f"image size: {img.size}")

# conf_thresh=0 so we always see the best candidate, even a weak one
m = HerHealthUnifiedModel(conf_thresh=0.0, min_mask_frac=0.0)
m.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True)["state_dict"], strict=True)
m.eval()


def letterbox(im, size=640, pad=114):
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    canvas.paste(im.resize((nw, nh), Image.Resampling.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return canvas


variants = {"letterbox": letterbox(img), "squash": img.resize((640, 640), Image.Resampling.BILINEAR)}

for name, im in variants.items():
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        _, scores, _, _ = m.segmenter(x)
        out = m(x)
    top = scores[0, 0].topk(5).values.tolist()
    print(f"\n[{name}] top-5 detection scores: {[round(s, 3) for s in top]}  (server threshold is 0.45)")
    print(f"[{name}] best box (640px space): {out['roi_box']}  mask_fraction: {out['mask_fraction']:.2f}")
    print(f"[{name}] stage-2 (only meaningful if score >= 0.45): prob={out['anemia_prob']:.3f}  Hb={out['hb_g_dl']:.1f}")
    dbg = im.copy()
    ImageDraw.Draw(dbg).rectangle(out["roi_box"], outline=(0, 255, 0), width=3)
    dbg.save(f"debug_{name}.jpg")

if args.ultra:
    from ultralytics import YOLO
    res = YOLO(args.ultra)(args.image, conf=0.01, verbose=False)[0]
    confs = [] if res.boxes is None else [round(float(c), 3) for c in res.boxes.conf]
    print(f"\n[ORIGINAL ultralytics model] detection confidences at conf>=0.01: {confs or 'none'}")
