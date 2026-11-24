"""
Nail anemia predictor.

Models are vendored into backend/app/ml/ (inference.py, mlp_model.joblib,
feature_scaler.joblib) and loaded lazily on first request.  hf_hub_download
is kept only as a fallback in case the vendored files are absent (e.g. during
local development before the files were added).

Memory notes
------------
- mlp_model.joblib  : 151 KB on disk, ~1–2 MB resident after joblib.load
- feature_scaler.joblib : 1.3 KB on disk, negligible resident
- extract_features()  : always resizes to 224×224 uint8 → float64 internally
  (scipy uniform_filter works on float64), producing ~3 × 224 × 224 × 8 B ≈ 1.1 MB
  of temporaries per call.  All temporaries are stack-local and GC'd after return.
- No large intermediate kept across calls.
"""
import gc
import importlib.util
import io
import os
import threading
from pathlib import Path

import joblib
from PIL import Image

REPO_ID = "JetX-GT/nail-anemia-detector"

# Vendored copies live next to this file. hf_hub_download is the fallback.
_HERE = Path(__file__).parent
_VENDORED_INFERENCE = _HERE / "inference.py"
_VENDORED_MODEL    = _HERE / "mlp_model.joblib"
_VENDORED_SCALER   = _HERE / "feature_scaler.joblib"

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

# Lazy-loaded globals – initialised on first call to predict_nail_anemia().
_nail_lock      = threading.Lock()
_nail_inference = None   # the inference module (has extract_features)
_scaler         = None   # sklearn StandardScaler
_mlp_model      = None   # sklearn MLPClassifier


def _resolve_path(vendored: Path, filename: str) -> str:
    """Return vendored file path if it exists, else download from HuggingFace."""
    if vendored.exists():
        return str(vendored)
    # Fallback: download (network required, slow on cold start)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def _load_nail_models() -> None:
    global _nail_inference, _scaler, _mlp_model
    if _nail_inference is not None:
        return
    with _nail_lock:
        if _nail_inference is not None:
            return

        _dbg("nail model load start")

        inference_path = _resolve_path(_VENDORED_INFERENCE, "inference.py")
        model_path     = _resolve_path(_VENDORED_MODEL,    "mlp_model.joblib")
        scaler_path    = _resolve_path(_VENDORED_SCALER,   "feature_scaler.joblib")

        spec = importlib.util.spec_from_file_location("nail_inference", inference_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        _nail_inference = mod
        _scaler         = joblib.load(scaler_path)
        _mlp_model      = joblib.load(model_path)
        gc.collect()

        _dbg("nail model load done")


def predict_nail_anemia(image_bytes: bytes, high_sensitivity: bool = False) -> dict:
    _load_nail_models()

    _dbg("nail predict start")

    # Keep the original pixels here; extract_features() performs the model's
    # existing 224×224 resize, and another resize changes its predictions.
    opened = None
    img = None
    try:
        _dbg("nail before image decode")
        opened = Image.open(io.BytesIO(image_bytes))
        img = opened if opened.mode == "RGB" else opened.convert("RGB")
        _dbg("nail after image decode")
        _dbg("nail before extract_features")
        features = _nail_inference.extract_features(img)
        _dbg("nail after extract_features")
    finally:
        if img is not None:
            img.close()
        if opened is not None and opened is not img:
            opened.close()

    features_scaled = _scaler.transform(features.reshape(1, -1))
    _dbg("nail before predict_proba")
    prob = float(_mlp_model.predict_proba(features_scaled)[0, 1])
    cutoff = 0.255 if high_sensitivity else 0.10

    _dbg("nail predict done")

    return {
        "nail_model": REPO_ID,
        "nail_anemia_probability": round(prob, 4),
        "nail_is_anemic": prob >= cutoff,
    }
