import numpy as np
from src.evaluation.metrics import classification_summary, coverage_accuracy_curve, select_threshold


def test_macro_f1_exposes_minority_failure():
    y = [0] * 95 + [1] * 5
    pred = [0] * 100  # always majority class
    s = classification_summary(y, pred, ["a", "b"])
    assert s["accuracy"] == 0.95 and s["macro_f1"] < 0.55


def test_threshold_raises_accuracy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    probs = np.full((400, 3), 0.1)
    for i, t in enumerate(y):
        correct = rng.random() < 0.8
        c = t if correct else (t + 1) % 3
        probs[i, c] = 0.9 if correct else 0.5
    curve = coverage_accuracy_curve(probs, y, thresholds=[0.3, 0.8])
    assert curve[1]["accuracy"] > curve[0]["accuracy"] and curve[1]["coverage"] < curve[0]["coverage"]
    th = select_threshold(probs, y, target_accuracy=0.99, min_coverage=0.5)
    assert 0.5 < th <= 0.99


def test_select_threshold_fallback():
    probs = np.array([[0.5, 0.5]] * 4); y = np.array([0, 1, 0, 1])
    assert isinstance(select_threshold(probs, y), float)
