# import importlib.util
# import io
# import os
# import tempfile
# from huggingface_hub import hf_hub_download
# from PIL import Image

# REPO_ID = "JetX-GT/nail-anemia-detector"

# inference_script_path = hf_hub_download(repo_id=REPO_ID, filename="inference.py")
# hf_hub_download(repo_id=REPO_ID, filename="mlp_model.joblib")
# hf_hub_download(repo_id=REPO_ID, filename="feature_scaler.joblib")
# hf_hub_download(repo_id=REPO_ID, filename="model_metadata.json")

# spec = importlib.util.spec_from_file_location("nail_inference", inference_script_path)
# nail_inference = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(nail_inference)

# CACHE_DIR = os.path.dirname(inference_script_path)
# nail_detector = (
#     nail_inference.NailAnemiaDetector(model_dir=CACHE_DIR)
#     if hasattr(nail_inference, "NailAnemiaDetector")
#     else None
# )

# def predict_nail_anemia(image_bytes: bytes, high_sensitivity: bool = False) -> dict:
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#         img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         img.save(tmp.name, format="JPEG")
#         tmp_path = tmp.name

#     try:
#         if nail_detector is not None:
#             res = nail_detector.predict(tmp_path)
#         elif hasattr(nail_inference, "predict"):
#             res = nail_inference.predict(tmp_path, model_dir=CACHE_DIR)
#         else:
#             raise RuntimeError("Check inference.py in Hugging Face cache for exact method name.")

#         prob = float(res.get("probability", res.get("anemia_probability", 0.0)))
#         cutoff = 0.255 if high_sensitivity else 0.50
#         return {
#             "nail_model": REPO_ID,
#             "nail_anemia_probability": round(prob, 4),
#             "nail_is_anemic": prob >= cutoff,
#             "raw_output": res,
#         }
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

import gc
import importlib.util
import io
import threading

import joblib
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "JetX-GT/nail-anemia-detector"

# Lazy-loaded globals — initialised on first call to predict_nail_anemia().
# This keeps them out of RAM during startup so the app doesn't OOM before it
# even receives a request (Render free tier: 512 MB).
_nail_lock = threading.Lock()
_nail_inference = None
_scaler = None
_mlp_model = None


def _load_nail_models():
    global _nail_inference, _scaler, _mlp_model
    if _nail_inference is not None:
        return
    with _nail_lock:
        if _nail_inference is not None:
            return
        inference_script_path = hf_hub_download(repo_id=REPO_ID, filename="inference.py")
        model_path = hf_hub_download(repo_id=REPO_ID, filename="mlp_model.joblib")
        scaler_path = hf_hub_download(repo_id=REPO_ID, filename="feature_scaler.joblib")

        spec = importlib.util.spec_from_file_location("nail_inference", inference_script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        _nail_inference = mod
        _scaler = joblib.load(scaler_path)
        _mlp_model = joblib.load(model_path)
        gc.collect()


def predict_nail_anemia(image_bytes: bytes, high_sensitivity: bool = False) -> dict:
    _load_nail_models()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    features = _nail_inference.extract_features(img)
    features_scaled = _scaler.transform(features.reshape(1, -1))

    prob = float(_mlp_model.predict_proba(features_scaled)[0, 1])
    cutoff = 0.255 if high_sensitivity else 0.10

    return {
        "nail_model": REPO_ID,
        "nail_anemia_probability": round(prob, 4),
        "nail_is_anemic": prob >= cutoff,
    }