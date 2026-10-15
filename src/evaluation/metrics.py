"""Metrics, abstention-threshold selection, coverage/accuracy, latency. Torch-free."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)


def classification_summary(y_true, y_pred, class_names) -> dict:
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def coverage_accuracy_curve(probs: np.ndarray, y_true: np.ndarray, thresholds=None) -> list[dict]:
    """For each threshold: coverage (fraction answered) and accuracy on answered samples."""
    thresholds = np.linspace(0.3, 0.99, 24) if thresholds is None else thresholds
    conf = probs.max(1)
    pred = probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        cov = float(keep.mean())
        acc = float((pred[keep] == y_true[keep]).mean()) if keep.any() else float("nan")
        rows.append({"threshold": float(t), "coverage": cov, "accuracy": acc})
    return rows


def select_threshold(val_probs: np.ndarray, val_true: np.ndarray,
                     target_accuracy: float = 0.95, min_coverage: float = 0.7) -> float:
    """Pick the LOWEST threshold on VALIDATION data reaching target accuracy at >= min_coverage.

    Falls back to the threshold with the best accuracy subject to min_coverage.
    Never call this with test data.
    """
    curve = coverage_accuracy_curve(val_probs, val_true)
    ok = [r for r in curve if r["coverage"] >= min_coverage and not np.isnan(r["accuracy"])]
    if not ok:
        return 0.5
    hit = [r for r in ok if r["accuracy"] >= target_accuracy]
    return float(min(hit, key=lambda r: r["threshold"])["threshold"]) if hit else \
        float(max(ok, key=lambda r: (r["accuracy"], r["coverage"]))["threshold"])


def latency_stats(times_ms) -> dict:
    a = np.asarray(times_ms, dtype=float)
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)), "n": int(len(a))}
