import io
import os
import threading

import numpy as np
import pillow_heif
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from app.database import execute_lastrowid, query, row_to_dict

from build_unified_model import HerHealthUnifiedModel
from app.ml.nail_predictor import predict_nail_anemia

pillow_heif.register_heif_opener()

router = APIRouter(prefix="/api/screenings", tags=["Anemia Screening"])

# ---------------------------------------------------------------------
# Config: tune these on a validation set (see notes in the review).
# ---------------------------------------------------------------------
MODEL_PATH = os.getenv("HERHEALTH_MODEL_PATH", "herhealth_unified.pt")
INPUT_SIZE = 640
SEG_CONF_THRESH = 0.20
ANEMIA_PROB_CUTOFF = 0.50
HB_ANEMIA_CUTOFF = 11.8        # g/dL  (WHO: 12.0 non-pregnant women, 11.0 pregnant)
HIGH_RISK_PROB = 0.80
HIGH_RISK_HB = 9.5             # g/dL
HB_PLAUSIBLE_RANGE = (3.0, 20.0)
MAX_UPLOAD_SIZE_MB = 12
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_PIXELS = 40_000_000
DISCLAIMER = ("Educational wellness prototype only. A clinical CBC blood test is "
              "required for medical diagnosis.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Model is loaded once at import time.
# ---------------------------------------------------------------------
unified_model = HerHealthUnifiedModel(conf_thresh=SEG_CONF_THRESH).to(DEVICE)
_ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
unified_model.load_state_dict(_ckpt["state_dict"], strict=True)
unified_model.eval().requires_grad_(False)
MODEL_NAME = _ckpt.get("model_name", "HerHealthUnifiedAnemiaModel")

# One forward pass at a time: avoids CPU/GPU oversubscription and memory spikes.
_infer_lock = threading.Lock()


def _letterbox(img: Image.Image, size: int = INPUT_SIZE, pad: int = 114):
    """Aspect-preserving resize + gray padding (how YOLO models are trained/served)."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img, (left, top))
    return canvas, r, left, top


def _load_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image resolution is too large.")
        img = ImageOps.exif_transpose(img)          # honour phone-camera rotation
        return img.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Could not read the file as an image.")


def _validate_image_quality(img: Image.Image, body_part: str) -> None:
    pixels = np.asarray(img, dtype=np.uint8)
    channel_std = pixels.std(axis=(0, 1))
    if float(pixels.mean()) < 12 or float(channel_std.max()) < 4:
        raise HTTPException(status_code=422, detail={
            "code": "image_blank_or_too_dark",
            "message": f"This image appears blank or too dark. Please upload a clear photo of your {body_part}.",
        })


def _risk(prob: float, hb: float):
    is_anemic = prob >= ANEMIA_PROB_CUTOFF or hb < HB_ANEMIA_CUTOFF
    if hb < HIGH_RISK_HB or prob >= HIGH_RISK_PROB:
        tier = "High"
    elif is_anemic:
        tier = "Moderate"
    else:
        tier = "Low"
    return is_anemic, tier


# NOTE: plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# blocking forward pass doesn't freeze the event loop, and inference_mode really applies.
@router.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than {MAX_UPLOAD_SIZE_MB} MB.",
        )

    orig = _load_image(raw)
    _validate_image_quality(orig, "inner lower eyelid")
    orig_w, orig_h = orig.size
    boxed, scale, left, top = _letterbox(orig)

    x = (torch.from_numpy(np.asarray(boxed, dtype=np.float32) / 255.0)
         .permute(2, 0, 1).unsqueeze(0).to(DEVICE))

    with _infer_lock, torch.inference_mode():
        out = unified_model(x, allow_center_fallback=False)

    if out["roi_source"] == "none":
        raise HTTPException(status_code=422, detail={
            "code": "conjunctiva_not_found",
            "message": ("This doesn't look like a clear inner lower-eyelid photo. Please upload a close-up "
                        "of your inner lower eyelid, use bright even light, and keep the eye in focus."),
            "roi_confidence": round(out["seg_confidence"], 3),
        })

    prob = out["anemia_prob"]
    hb = min(max(out["hb_g_dl"], HB_PLAUSIBLE_RANGE[0]), HB_PLAUSIBLE_RANGE[1])
    is_anemic, tier = _risk(prob, hb)

    # ROI box mapped back to the ORIGINAL image's pixel coordinates (handy for a UI overlay).
    x1, y1, x2, y2 = out["roi_box"]
    box_orig = [
        round(max(0.0, min(orig_w, (x1 - left) / scale)), 1),
        round(max(0.0, min(orig_h, (y1 - top) / scale)), 1),
        round(max(0.0, min(orig_w, (x2 - left) / scale)), 1),
        round(max(0.0, min(orig_h, (y2 - top) / scale)), 1),
    ]

    record_id = execute_lastrowid(
        """
        INSERT INTO anemia_screenings
            (modality, model_name, anemia_probability, estimated_hemoglobin, risk_tier, roi_confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("eyelid", MODEL_NAME, round(prob, 4), round(hb, 1), tier, round(out["seg_confidence"], 3)),
    )

    return {
        "model": MODEL_NAME,
        "record_id": record_id,
        "roi_detected": out["roi_source"] == "conjunctiva",
        "roi_label": "conjunctiva" if out["roi_source"] == "conjunctiva" else "center_fallback",
        "roi_confidence": round(out["seg_confidence"], 3),
        "roi_box_xyxy": box_orig,
        "anemia_probability": round(prob, 4),
        "estimated_hemoglobin_g_dl": round(hb, 1),
        "is_anemic_flag": is_anemic,
        "heads_agree": (prob >= ANEMIA_PROB_CUTOFF) == (hb < HB_ANEMIA_CUTOFF),
        "risk_tier": tier,
        "disclaimer": DISCLAIMER,
    }


@router.post("/analyze-nail")
def analyze_nail_image(file: UploadFile = File(...), high_sensitivity: bool = False):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than {MAX_UPLOAD_SIZE_MB} MB.",
        )

    img = _load_image(raw)
    _validate_image_quality(img, "fingernail")

    try:
        res = predict_nail_anemia(raw, high_sensitivity=high_sensitivity)
        prob = res["nail_anemia_probability"]
        tier = "High" if prob >= 0.80 else ("Moderate" if res["nail_is_anemic"] else "Low")

        record_id = execute_lastrowid(
            """
            INSERT INTO anemia_screenings
                (modality, model_name, anemia_probability, estimated_hemoglobin, risk_tier, roi_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("nail", res["nail_model"], prob, None, tier, None),
        )
        res["risk_tier"] = tier
        res["record_id"] = record_id
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nail inference failed: {str(e)}")


@router.get("/history")
def get_screening_history(limit: int = 10):
    rows = query(
        """
        SELECT id, modality, model_name, anemia_probability, estimated_hemoglobin, risk_tier, roi_confidence, created_at
        FROM anemia_screenings
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return {"items": [row_to_dict(r) for r in rows]}
