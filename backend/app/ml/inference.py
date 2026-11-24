#!/usr/bin/env python3
"""
Inference script for Nail Anemia Detector.

Usage:
    python inference.py --image path/to/nail.jpg
    python inference.py --image nail.jpg --threshold 0.255  # For 100% recall mode
"""

import argparse
import joblib
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from pathlib import Path


def extract_features(img):
    """Extract handcrafted color features from nail image."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img).astype(float)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = 0.299*r + 0.587*g + 0.114*b

    features = []

    # Brightness features
    features.extend([brightness.mean(), brightness.std()])
    features.extend([np.percentile(brightness, p) for p in [10, 25, 50, 75, 90]])

    # Redness features
    redness = r / (r + g + b + 1e-10)
    features.extend([redness.mean(), redness.std()])

    # Pallor features
    white_ratio = (brightness > 180).sum() / brightness.size
    pink_ratio = ((r > 150) & (g < 150) & (b < 150)).sum() / brightness.size
    features.extend([white_ratio, pink_ratio])

    # Channel statistics
    for ch in [r, g, b]:
        features.extend([ch.mean(), ch.std()])

    # Color ratios
    features.extend([
        (r.mean() + 1) / (g.mean() + 1),
        (r.mean() + 1) / (b.mean() + 1),
        (r.mean() - b.mean()) / 255,
    ])

    # Hemoglobin proxy
    hb = r / (g + b + 1)
    features.extend([hb.mean(), hb.std()])

    # Spatial features
    h, w = arr.shape[:2]
    top = brightness[:h//3, :].mean()
    bottom = brightness[2*h//3:, :].mean()
    features.append(top - bottom)

    center = brightness[h//4:3*h//4, w//4:3*w//4].mean()
    features.append(center - brightness.mean())

    # Gradient features
    gx = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, :1]))
    gy = np.abs(np.diff(brightness, axis=0, prepend=brightness[:1, :]))
    gradient = np.sqrt(gx**2 + gy**2)
    features.extend([gradient.mean(), gradient.std()])

    # Local variance
    local_mean = uniform_filter(brightness, size=7)
    local_var = uniform_filter((brightness - local_mean)**2, size=7)
    local_var = np.maximum(local_var, 0)
    features.extend([np.sqrt(local_var).mean(), np.sqrt(local_var).std()])

    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Nail Anemia Detection Inference')
    parser.add_argument('--image', '-i', required=True, help='Path to nail image')
    parser.add_argument('--threshold', '-t', type=float, default=0.10,
                        help='Prediction threshold (default: 0.10 for balanced mode, use 0.255 for 100%% recall)')
    parser.add_argument('--model', '-m', default='mlp_model.joblib', help='Path to model file')
    parser.add_argument('--scaler', '-s', default='feature_scaler.joblib', help='Path to scaler file')

    args = parser.parse_args()

    # Load model and scaler
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    # Load and process image
    img = Image.open(args.image)
    features = extract_features(img)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    probability = model.predict_proba(features_scaled)[0, 1]
    prediction = "anemia" if probability >= args.threshold else "healthy"

    print(f"\n{'='*50}")
    print("NAIL ANEMIA DETECTION RESULT")
    print(f"{'='*50}")
    print(f"Image: {args.image}")
    print(f"Probability: {probability:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Prediction: {prediction.upper()}")
    print(f"{'='*50}")

    if prediction == "anemia":
        print("\n⚠️  POSITIVE - Please confirm with blood test (hemoglobin/hematocrit)")
    else:
        print("\n✅ NEGATIVE - Low probability of anemia")


if __name__ == '__main__':
    main()
