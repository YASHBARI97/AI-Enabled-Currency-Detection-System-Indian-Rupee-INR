import numpy as np
import pytest
from src.inference.decision import STATUS_OK, STATUS_UNCERTAIN, decide, softmax

C = ["INR_10", "INR_100", "INR_500"]


def test_confident_prediction():
    r = decide([0.05, 0.02, 0.93], C, 0.8, "1.0.0", 12.3)
    assert r.status == STATUS_OK and r.predicted_denomination == "INR_500" and r.confidence == 0.93


def test_abstains_below_threshold():
    r = decide([0.4, 0.35, 0.25], C, 0.8, "1.0.0")
    assert r.status == STATUS_UNCERTAIN and r.predicted_denomination is None
    assert r.message == "Unable to classify confidently"


def test_threshold_boundary_is_inclusive():
    assert decide([0.8, 0.1, 0.1], C, 0.8, "v").status == STATUS_OK


def test_schema_keys():
    d = decide([0.1, 0.1, 0.8], C, 0.5, "1.0.0", 5.0).to_dict()
    for k in ("predicted_denomination", "confidence", "status", "inference_ms", "model_version"):
        assert k in d


def test_length_mismatch():
    with pytest.raises(ValueError):
        decide([0.5, 0.5], C, 0.5, "v")


def test_softmax_sums_to_one():
    assert softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)
