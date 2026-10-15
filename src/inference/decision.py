"""Confidence/abstention logic and response schema. Torch-free."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

STATUS_OK = "prediction_available"
STATUS_UNCERTAIN = "low_confidence"
UNCERTAIN_MESSAGE = "Unable to classify confidently"
DISCLAIMER = ("Denomination estimate only. This does NOT verify authenticity or legal tender status. "
              "Poor lighting, wear, folds, cropping or unsupported note designs can cause errors. "
              "Verify manually for any consequential decision.")


@dataclass
class Prediction:
    status: str
    predicted_denomination: str | None
    confidence: float | None
    inference_ms: float | None
    model_version: str
    message: str | None = None
    top_k: list | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def decide(probs: Sequence[float], class_names: Sequence[str], threshold: float,
           model_version: str, inference_ms: float | None = None, top_k: int = 3) -> Prediction:
    """Apply abstention: below `threshold` max-probability => no label is forced."""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or len(p) != len(class_names):
        raise ValueError("probs length must match class_names")
    order = np.argsort(-p)
    best = int(order[0])
    conf = float(p[best])
    tk = [{"label": class_names[i], "confidence": round(float(p[i]), 4)} for i in order[:top_k]]
    if conf < threshold:
        return Prediction(STATUS_UNCERTAIN, None, round(conf, 4), inference_ms, model_version,
                          UNCERTAIN_MESSAGE, tk)
    return Prediction(STATUS_OK, class_names[best], round(conf, 4), inference_ms, model_version, None, tk)


def invalid_prediction(status: str, message: str, model_version: str) -> Prediction:
    return Prediction(status, None, None, None, model_version, message, None)


class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self._t) * 1000.0
