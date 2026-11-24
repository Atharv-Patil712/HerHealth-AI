"""
Memory measurement & before/after output comparison script.

Usage (from backend/ directory, with .venv active):
    python measure_rss.py --mode before
    python measure_rss.py --mode after
    python measure_rss.py --mode compare

Generates a 12 MP (4000x3000) JPEG and a 12 MP HEIC in memory, plus 3 sample
images for before/after probability comparison.
"""
import argparse
import gc
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pillow_heif
import psutil
from PIL import Image, ImageDraw

pillow_heif.register_heif_opener()

proc = psutil.Process()


def rss_mb() -> float:
    return proc.memory_info().rss / 1_048_576


class PeakRSS:
    def __enter__(self):
        self.peak = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(0.002):
            self.peak = max(self.peak, rss_mb())

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss_mb())


def _make_12mp_pattern(first: tuple[int, int, int], second: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (4000, 3000), first)
    draw = ImageDraw.Draw(img)
    for y in range(0, 3000, 100):
        draw.rectangle((0, y, 3999, y + 49), fill=second if (y // 100) % 2 else first)
    return img


def _check_upload_size(data: bytes, kind: str) -> bytes:
    if len(data) > 5 * 1024 * 1024:
        raise RuntimeError(f"Generated 12 MP {kind} exceeds the 5 MB route limit.")
    return data


def make_12mp_jpeg() -> bytes:
    img = _make_12mp_pattern((205, 145, 150), (175, 115, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img.close()
    gc.collect()
    return _check_upload_size(buf.getvalue(), "JPEG")


def make_12mp_heic() -> bytes:
    img = _make_12mp_pattern((205, 145, 150), (170, 110, 115))
    buf = io.BytesIO()
    pillow_heif.from_pillow(img).save(buf, quality=85)
    img.close()
    gc.collect()
    return _check_upload_size(buf.getvalue(), "HEIC")


def make_sample_images() -> list[tuple[str, bytes]]:
    samples = []
    specs = [
        ("pink_nail_12mp", (215, 130, 135), (190, 100, 110), (4000, 3000)),
        ("pale_nail_12mp", (235, 220, 215), (210, 195, 190), (4000, 3000)),
        ("medium_nail_4mp", (200, 150, 145), (170, 120, 115), (2000, 2000)),
    ]
    for name, c1, c2, size in specs:
        img = Image.new("RGB", size, c1)
        ImageDraw.Draw(img).rectangle(
            (size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4),
            fill=c2,
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        img.close()
        samples.append((name, buf.getvalue()))
    return samples


def run_before():
    from app.ml.nail_predictor import _load_nail_models
    import app.ml.nail_predictor as np_mod

    print(f"[BEFORE] baseline RSS: {rss_mb():.1f} MB")
    _load_nail_models()
    print(f"[BEFORE] after nail model load RSS: {rss_mb():.1f} MB")

    jpeg_bytes = make_12mp_jpeg()
    gc.collect()
    print(f"[BEFORE] 12MP JPEG ({len(jpeg_bytes)/1024:.0f} KB) before decode RSS: {rss_mb():.1f} MB")
    with PeakRSS() as peak:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        print(f"[BEFORE] 12MP JPEG after full-res decode RSS: {rss_mb():.1f} MB")
        feats = np_mod._nail_inference.extract_features(img)
        print(f"[BEFORE] 12MP JPEG after extract_features RSS: {rss_mb():.1f} MB")
        prob = float(np_mod._mlp_model.predict_proba(np_mod._scaler.transform(feats.reshape(1, -1)))[0, 1])
        print(f"[BEFORE] 12MP JPEG after predict (prob={prob:.4f}) RSS: {rss_mb():.1f} MB")
        img.close()
    print(f"[BEFORE] 12MP JPEG sampled peak RSS: {peak.peak:.1f} MB")
    del jpeg_bytes, feats
    gc.collect()

    heic_bytes = make_12mp_heic()
    gc.collect()
    print(f"[BEFORE] 12MP HEIC ({len(heic_bytes)/1024:.0f} KB) before decode RSS: {rss_mb():.1f} MB")
    with PeakRSS() as peak:
        img = Image.open(io.BytesIO(heic_bytes)).convert("RGB")
        print(f"[BEFORE] 12MP HEIC after full-res decode RSS: {rss_mb():.1f} MB")
        feats = np_mod._nail_inference.extract_features(img)
        prob = float(np_mod._mlp_model.predict_proba(np_mod._scaler.transform(feats.reshape(1, -1)))[0, 1])
        print(f"[BEFORE] 12MP HEIC after predict (prob={prob:.4f}) RSS: {rss_mb():.1f} MB")
        img.close()
    print(f"[BEFORE] 12MP HEIC sampled peak RSS: {peak.peak:.1f} MB")


def run_after():
    from app.ml.nail_predictor import _load_nail_models, predict_nail_anemia

    print(f"[AFTER] baseline RSS: {rss_mb():.1f} MB")
    _load_nail_models()
    print(f"[AFTER] after nail model load RSS: {rss_mb():.1f} MB")

    jpeg_bytes = make_12mp_jpeg()
    gc.collect()
    print(f"[AFTER] 12MP JPEG ({len(jpeg_bytes)/1024:.0f} KB) before predict RSS: {rss_mb():.1f} MB")
    with PeakRSS() as peak:
        res_j = predict_nail_anemia(jpeg_bytes)
    print(f"[AFTER] 12MP JPEG after predict (prob={res_j['nail_anemia_probability']:.4f}) RSS: {rss_mb():.1f} MB")
    print(f"[AFTER] 12MP JPEG sampled peak RSS: {peak.peak:.1f} MB")
    del jpeg_bytes
    gc.collect()

    heic_bytes = make_12mp_heic()
    gc.collect()
    print(f"[AFTER] 12MP HEIC ({len(heic_bytes)/1024:.0f} KB) before predict RSS: {rss_mb():.1f} MB")
    with PeakRSS() as peak:
        res_h = predict_nail_anemia(heic_bytes)
    print(f"[AFTER] 12MP HEIC after predict (prob={res_h['nail_anemia_probability']:.4f}) RSS: {rss_mb():.1f} MB")
    print(f"[AFTER] 12MP HEIC sampled peak RSS: {peak.peak:.1f} MB")


def run_compare():
    from app.ml.nail_predictor import _load_nail_models, predict_nail_anemia
    import app.ml.nail_predictor as np_mod

    _load_nail_models()
    for name, raw in make_sample_images():
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        feats = np_mod._nail_inference.extract_features(img)
        prob_before = round(float(np_mod._mlp_model.predict_proba(np_mod._scaler.transform(feats.reshape(1, -1)))[0, 1]), 4)
        img.close()

        prob_after = predict_nail_anemia(raw)["nail_anemia_probability"]
        diff = abs(prob_after - prob_before)
        print(f"[COMPARE] {name:18s} | before={prob_before:.4f} | after={prob_after:.4f} | diff={diff:.4f}")


def run_fullstack():
    import torch
    from app.routers import screenings
    from app.ml.nail_predictor import _load_nail_models, predict_nail_anemia

    print(f"[FULL] RSS after API, PIL, and torch imports: {rss_mb():.1f} MB")
    with PeakRSS() as peak:
        model, _ = screenings._get_model()
    print(f"[FULL] after eyelid model load: {rss_mb():.1f} MB; sampled peak={peak.peak:.1f} MB")

    with PeakRSS() as peak:
        _load_nail_models()
    print(f"[FULL] after nail model load: {rss_mb():.1f} MB; sampled peak={peak.peak:.1f} MB")

    for image_name, make_image in (
        ("12MP JPEG", make_12mp_jpeg),
        ("12MP HEIC", make_12mp_heic),
    ):
        raw = make_image()
        print(f"[FULL] {image_name} input={len(raw) / 1024:.0f} KB; before inference RSS={rss_mb():.1f} MB")

        with PeakRSS() as peak:
            image = screenings._load_image(raw)
            boxed, _, _, _ = screenings._letterbox(image)
            image.close()
            x = (
                torch.from_numpy(np.asarray(boxed, dtype=np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(screenings.DEVICE)
            )
            boxed.close()
            with screenings._infer_semaphore, torch.inference_mode():
                eye_result = model(x, allow_center_fallback=False)
            del x
            gc.collect()
        print(
            f"[FULL] {image_name} eyelid probability={eye_result.get('anemia_prob')}; "
            f"sampled peak={peak.peak:.1f} MB"
        )

        with PeakRSS() as peak:
            nail_result = predict_nail_anemia(raw)
        print(
            f"[FULL] {image_name} nail probability={nail_result['nail_anemia_probability']:.4f}; "
            f"sampled peak={peak.peak:.1f} MB"
        )
        del raw
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["before", "after", "compare", "fullstack"],
        default="compare",
    )
    args = parser.parse_args()
    if args.mode == "before":
        run_before()
    elif args.mode == "after":
        run_after()
    elif args.mode == "fullstack":
        run_fullstack()
    else:
        run_compare()
