"""Unified predictor: validates input -> preprocesses -> predicts -> applies abstention."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.validate import InvalidInputError, validate_and_load
from src.inference.decision import Timer, decide, invalid_prediction, softmax, Prediction
from src.utils.logging_setup import get_logger

log = get_logger("predictor")


class ModelLoadError(RuntimeError):
    pass


class Predictor:
    """Backends: 'torch' (checkpoint .pt) or 'sklearn' (joblib baseline)."""

    def __init__(self, model, meta: dict, max_upload_mb: float = 5.0):
        self.model = model
        self.meta = meta
        self.class_names = meta["class_names"]
        self.threshold = float(meta["threshold"])
        self.version = str(meta.get("model_version", "0.0.0"))
        self.img_size = int(meta.get("img_size", 224))
        self.backend = meta.get("backend", "torch")
        self.max_upload_mb = max_upload_mb

    @classmethod
    def load(cls, model_path: str, meta_path: str, max_upload_mb: float = 5.0) -> "Predictor":
        try:
            meta = json.loads(Path(meta_path).read_text())
            backend = meta.get("backend", "torch")
            if backend == "sklearn":
                import joblib
                model = joblib.load(model_path)
            else:
                import torch
                from src.models.factory import build_model
                model = build_model(meta["arch"], len(meta["class_names"]), pretrained=False)
                state = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state["model"] if "model" in state else state)
                model.eval()
        except Exception as e:  # noqa: BLE001 - surface a safe message, log details
            log.error("Model load failed: %s", type(e).__name__)
            raise ModelLoadError("The model could not be loaded. Check artifacts and configuration.") from None
        log.info("Loaded model v%s (%s)", meta.get("model_version"), backend)
        return cls(model, meta, max_upload_mb)

    def _probs(self, img: Image.Image) -> np.ndarray:
        if self.backend == "sklearn":
            from src.models.baseline import extract_features
            return self.model.predict_proba(extract_features(img, self.img_size)[None])[0]
        import torch
        from src.preprocessing.transforms import to_normalised_array
        x = torch.from_numpy(to_normalised_array(img, self.img_size))[None]
        with torch.no_grad():
            return softmax(self.model(x)[0].numpy())

    def predict_image(self, img: Image.Image) -> Prediction:
        with Timer() as t:
            probs = self._probs(img)
        return decide(probs, self.class_names, self.threshold, self.version, round(t.ms, 2))

    def predict_bytes(self, data: bytes, filename: str | None = None) -> Prediction:
        try:
            img = validate_and_load(data, filename, self.max_upload_mb)
        except InvalidInputError as e:
            log.warning("Rejected input: %s", e.status)
            return invalid_prediction(e.status, e.message, self.version)
        try:
            return self.predict_image(img)
        except Exception as e:  # noqa: BLE001
            log.error("Inference failed: %s", type(e).__name__)
            return invalid_prediction("inference_error", "Prediction failed. Please try another image.", self.version)
