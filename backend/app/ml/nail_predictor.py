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

import importlib.util
import io
import joblib
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "JetX-GT/nail-anemia-detector"

# Download inference script and joblib artifacts once at startup
inference_script_path = hf_hub_download(repo_id=REPO_ID, filename="inference.py")
model_path = hf_hub_download(repo_id=REPO_ID, filename="mlp_model.joblib")
scaler_path = hf_hub_download(repo_id=REPO_ID, filename="feature_scaler.joblib")

# Load extract_features from inference.py
spec = importlib.util.spec_from_file_location("nail_inference", inference_script_path)
nail_inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nail_inference)

# Load scikit-learn scaler and MLP model
scaler = joblib.load(scaler_path)
mlp_model = joblib.load(model_path)


def predict_nail_anemia(image_bytes: bytes, high_sensitivity: bool = False) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    features = nail_inference.extract_features(img)
    features_scaled = scaler.transform(features.reshape(1, -1))

    prob = float(mlp_model.predict_proba(features_scaled)[0, 1])
    cutoff = 0.255 if high_sensitivity else 0.10

    return {
        "nail_model": REPO_ID,
        "nail_anemia_probability": round(prob, 4),
        "nail_is_anemic": prob >= cutoff,
    }