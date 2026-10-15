import json

import joblib
import numpy as np
import pytest
from PIL import Image

from src.inference.predictor import ModelLoadError, Predictor
from src.models.baseline import build_baseline, extract_features
from tests.conftest import make_png

COLORS = {"INR_10": (150, 100, 60), "INR_100": (160, 120, 200), "INR_500": (140, 150, 120)}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    rng = np.random.default_rng(0)
    X, y = [], []
    names = sorted(COLORS)
    for ci, n in enumerate(names):
        for _ in range(20):
            arr = np.clip(np.array(COLORS[n]) + rng.integers(-15, 15, (64, 128, 3)), 0, 255).astype("uint8")
            X.append(extract_features(Image.fromarray(arr))); y.append(ci)
    model = build_baseline().fit(np.stack(X), y)
    d = tmp_path_factory.mktemp("m")
    joblib.dump(model, d / "model.joblib")
    meta = {"backend": "sklearn", "class_names": names, "threshold": 0.6, "img_size": 224, "model_version": "t"}
    (d / "meta.json").write_text(json.dumps(meta))
    return Predictor.load(str(d / "model.joblib"), str(d / "meta.json"))


def test_predicts_known_class(trained):
    out = trained.predict_bytes(make_png(COLORS["INR_100"], (128, 64)), "x.png")
    assert out.status in ("prediction_available", "low_confidence")
    assert out.inference_ms is not None and out.model_version == "t"


def test_invalid_inputs_return_status_not_exception(trained):
    assert trained.predict_bytes(b"", "a.png").status == "empty_input"
    assert trained.predict_bytes(b"junk", "a.png").status == "unreadable_image"
    assert trained.predict_bytes(make_png(), "a.bmp").status == "unsupported_format"


def test_model_load_error_is_safe(tmp_path):
    with pytest.raises(ModelLoadError) as e:
        Predictor.load(str(tmp_path / "nope.joblib"), str(tmp_path / "nope.json"))
    assert str(tmp_path) not in str(e.value)  # no filesystem path leak


def test_api_endpoints(trained, monkeypatch):
    from fastapi.testclient import TestClient
    import app.api as api
    api.state["predictor"] = trained
    c = TestClient(api.app)
    assert c.post("/predict", files={"file": ("a.png", make_png(), "image/png")}).status_code == 200
    r = c.post("/predict", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code == 400 and r.json()["status"] == "empty_input"
    r = c.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert "Traceback" not in r.text
    api.state["predictor"] = None
    assert c.post("/predict", files={"file": ("a.png", make_png(), "image/png")}).status_code == 503


def test_health_reports_model_info(trained):
    from fastapi.testclient import TestClient
    import app.api as api
    api.state["predictor"] = trained
    j = TestClient(api.app).get("/health").json()
    assert j["status"] == "ok" and j["classes"] == trained.class_names and j["threshold"] == 0.6
    api.state["predictor"] = None
    assert TestClient(api.app).get("/health").json()["status"] == "model_unavailable"
