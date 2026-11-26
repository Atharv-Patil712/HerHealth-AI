"""
Anemia screening routes.

Design decisions
----------------
- Both routes are plain `def` (not async): FastAPI runs them in a threadpool
  worker so the blocking forward pass does not block the event loop.
- _infer_semaphore serialises full screening requests, including image decode,
  to prevent concurrent uploads from multiplying image memory.
- Upload cap: MAX_UPLOAD_BYTES (5 MB). Caller gets HTTP 413 beyond that.
- Images retain their decoded pixels before the models' existing resizes;
  adding another resize measurably changes nail probabilities.
- Quality validation uses a 64×64 sample so it does not allocate an array
  proportional to the uploaded image resolution.
- Each temporary Image is closed in a finally block.
"""

import gc
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH        = os.getenv("HERHEALTH_MODEL_PATH", "herhealth_unified.pt")
INPUT_SIZE        = 640
# Cap upload bytes and decoded pixels to bound per-request memory.
MAX_UPLOAD_SIZE_MB = 5
MAX_UPLOAD_BYTES  = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_PIXELS        = 12_000_000
SEG_CONF_THRESH   = 0.20
ANEMIA_PROB_CUTOFF = 0.50
HB_ANEMIA_CUTOFF  = 11.8
HIGH_RISK_PROB    = 0.80
HIGH_RISK_HB      = 9.5
HB_PLAUSIBLE_RANGE = (3.0, 20.0)
DISCLAIMER = (
    "Educational wellness prototype only. A clinical CBC blood test is "
    "required for medical diagnosis."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Optional RSS debug logging (set MEM_DEBUG=1 in env to enable) -----------
_MEM_DEBUG = os.getenv("MEM_DEBUG", "0") == "1"

def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except Exception:
        return -1.0

def _dbg(label: str) -> None:
    if _MEM_DEBUG:
        import logging
        logging.getLogger(__name__).debug("[MEM] %s: %.1f MB RSS", label, _rss_mb())
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
unified_model: HerHealthUnifiedModel | None = None
MODEL_NAME: str = "HerHealthUnifiedAnemiaModel"
_model_lock = threading.Lock()
_infer_semaphore = threading.Semaphore(1)


def _get_model() -> tuple[HerHealthUnifiedModel, str]:
    global unified_model, MODEL_NAME
    if unified_model is not None:
        return unified_model, MODEL_NAME
    with _model_lock:
        if unified_model is not None:
            return unified_model, MODEL_NAME
        _dbg("torch model load start")
        try:
            m = HerHealthUnifiedModel(conf_thresh=SEG_CONF_THRESH).to(DEVICE)
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            m.load_state_dict(ckpt["state_dict"], strict=True)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=f"Anemia screening model not found at {MODEL_PATH}. "
                       "Please ensure the model file is built/deployed. "
                       "Run: python build_unified_model.py (requires eye_seg_model.pt and model.pt)",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load anemia screening model: {str(e)}",
            )
        m.eval().requires_grad_(False)
        MODEL_NAME = ckpt.get("model_name", "HerHealthUnifiedAnemiaModel")
        unified_model = m
        gc.collect()
        _dbg("torch model load done")
        return unified_model, MODEL_NAME


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _letterbox(img: Image.Image, size: int = INPUT_SIZE, pad: int = 114):
    """Aspect-preserving resize + gray padding (YOLO convention)."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img, (left, top))
    return canvas, r, left, top


def _load_image(raw: bytes) -> Image.Image:
    """Open and EXIF-rotate the image without changing inference pixels."""
    opened = None
    transposed = None
    try:
        opened = Image.open(io.BytesIO(raw))
        if opened.width * opened.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image resolution is too large.")
        transposed = ImageOps.exif_transpose(opened)
        if transposed.mode == "RGB":
            result = transposed
            transposed = None
            if result is opened:
                opened = None
            return result
        return transposed.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Could not read the file as an image.")
    finally:
        if transposed is not None and transposed is not opened:
            transposed.close()
        if opened is not None:
            opened.close()


def _validate_image_quality(img: Image.Image, body_part: str) -> None:
    sample = img.resize((64, 64), Image.Resampling.BILINEAR)
    try:
        pixels = np.asarray(sample, dtype=np.uint8)
        channel_std = pixels.std(axis=(0, 1))
        if float(pixels.mean()) < 12 or float(channel_std.max()) < 4:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "image_blank_or_too_dark",
                    "message": (
                        f"This image appears blank or too dark. "
                        f"Please upload a clear photo of your {body_part}."
                    ),
                },
            )
    finally:
        sample.close()


def _risk(prob: float, hb: float):
    is_anemic = prob >= ANEMIA_PROB_CUTOFF or hb < HB_ANEMIA_CUTOFF
    if hb < HIGH_RISK_HB or prob >= HIGH_RISK_PROB:
        tier = "High"
    elif is_anemic:
        tier = "Moderate"
    else:
        tier = "Low"
    return is_anemic, tier


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    with _infer_semaphore:
        return _analyze_image(file)


def _analyze_image(file: UploadFile):
    # -- 1. Upload size cap (5 MB) -------------------------------------------
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than {MAX_UPLOAD_SIZE_MB} MB. Please compress the photo.",
        )

    _dbg("analyze_image: after read")

    orig = None
    boxed = None
    x = None
    try:
        # -- 2. Decode + cap resolution -------------------------------------
        _dbg("analyze_image: before decode")
        orig = _load_image(raw)
        del raw  # release compressed bytes ASAP
        gc.collect()

        _dbg("analyze_image: after decode")

        _validate_image_quality(orig, "inner lower eyelid")
        orig_w, orig_h = orig.size

        # -- 3. Letterbox to 640×640 ----------------------------------------
        boxed, scale, left, top = _letterbox(orig)
        orig.close()
        orig = None

        # -- 4. Tensor + inference ------------------------------------------
        x = (
            torch.from_numpy(np.asarray(boxed, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )
        boxed.close()
        boxed = None

        _dbg("analyze_image: before inference")

        model, model_name = _get_model()
        with torch.inference_mode():
            out = model(x, allow_center_fallback=False)
    finally:
        if orig is not None:
            orig.close()
        if boxed is not None:
            boxed.close()
        if x is not None:
            del x
        gc.collect()
        _dbg("analyze_image: after inference")

    if out["roi_source"] == "none":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "conjunctiva_not_found",
                "message": (
                    "This doesn't look like a clear inner lower-eyelid photo. "
                    "Please upload a close-up of your inner lower eyelid, "
                    "use bright even light, and keep the eye in focus."
                ),
                "roi_confidence": round(out["seg_confidence"], 3),
            },
        )

    prob = out["anemia_prob"]
    hb   = min(max(out["hb_g_dl"], HB_PLAUSIBLE_RANGE[0]), HB_PLAUSIBLE_RANGE[1])
    is_anemic, tier = _risk(prob, hb)

    x1, y1, x2, y2 = out["roi_box"]
    box_orig = [
        round(max(0.0, min(orig_w, (x1 - left) / scale)), 1),
        round(max(0.0, min(orig_h, (y1 - top)  / scale)), 1),
        round(max(0.0, min(orig_w, (x2 - left) / scale)), 1),
        round(max(0.0, min(orig_h, (y2 - top)  / scale)), 1),
    ]

    record_id = execute_lastrowid(
        """
        INSERT INTO anemia_screenings
            (modality, model_name, anemia_probability, estimated_hemoglobin, risk_tier, roi_confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("eyelid", model_name, round(prob, 4), round(hb, 1), tier, round(out["seg_confidence"], 3)),
    )

    return {
        "model":                    model_name,
        "record_id":                record_id,
        "roi_detected":             out["roi_source"] == "conjunctiva",
        "roi_label":                "conjunctiva" if out["roi_source"] == "conjunctiva" else "center_fallback",
        "roi_confidence":           round(out["seg_confidence"], 3),
        "roi_box_xyxy":             box_orig,
        "anemia_probability":       round(prob, 4),
        "estimated_hemoglobin_g_dl": round(hb, 1),
        "is_anemic_flag":           is_anemic,
        "heads_agree":              (prob >= ANEMIA_PROB_CUTOFF) == (hb < HB_ANEMIA_CUTOFF),
        "risk_tier":                tier,
        "disclaimer":               DISCLAIMER,
    }


@router.post("/analyze-nail")
def analyze_nail_image(file: UploadFile = File(...), high_sensitivity: bool = False):
    with _infer_semaphore:
        return _analyze_nail_image(file, high_sensitivity)


def _analyze_nail_image(file: UploadFile, high_sensitivity: bool):
    # -- 1. Upload size cap ---------------------------------------------------
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than {MAX_UPLOAD_SIZE_MB} MB. Please compress the photo.",
        )

    _dbg("analyze_nail: after read")

    # -- 2. Validate (decode + quality check) --------------------------------
    _dbg("analyze_nail: before decode")
    img = _load_image(raw)
    _dbg("analyze_nail: after decode")
    try:
        _validate_image_quality(img, "fingernail")
    finally:
        img.close()

    _dbg("analyze_nail: before predict")

    try:
        # predict_nail_anemia re-opens the bytes; extract_features resizes to
        # 224×224 internally, so raw bytes is the right thing to pass.
        res = predict_nail_anemia(raw, high_sensitivity=high_sensitivity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nail inference failed: {str(e)}")
    finally:
        del raw
        gc.collect()

    _dbg("analyze_nail: after predict")

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
    res["risk_tier"]  = tier
    res["record_id"]  = record_id
    return res


@router.get("/history")
def get_screening_history(limit: int = 10):
    rows = query(
        """
        SELECT id, modality, model_name, anemia_probability, estimated_hemoglobin,
               risk_tier, roi_confidence, created_at
        FROM anemia_screenings
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return {"items": [row_to_dict(r) for r in rows]}
