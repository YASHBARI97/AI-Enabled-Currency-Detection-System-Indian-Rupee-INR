"""Classical baseline: HSV colour histograms + edge/texture stats -> SVM (scikit-learn).

Banknote colour is a strong denomination cue, so this is a sensible, torch-free baseline.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(img: Image.Image, size: int = 224, bins: int = 16) -> np.ndarray:
    arr = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    feats = []
    for ch, rng in zip(range(3), ([0, 180], [0, 256], [0, 256])):
        h = cv2.calcHist([hsv], [ch], None, [bins], rng).flatten()
        feats.append(h / (h.sum() + 1e-8))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # 4x4 grid of edge density keeps coarse layout information
    g = edges.reshape(4, size // 4, 4, size // 4).mean(axis=(1, 3)).flatten() / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    feats.append(g)
    feats.append(np.array([np.log1p(lap)]))
    return np.concatenate(feats).astype(np.float32)


def build_baseline(C: float = 10.0, seed: int = 42):
    return make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", probability=True, random_state=seed))
