import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.metrics import (classification_summary, coverage_accuracy_curve,
                                    latency_stats, select_threshold)
from src.inference.predictor import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--splits", default="data/splits/splits.csv")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--select-threshold", action="store_true", help="validation split only")
    ap.add_argument("--target-accuracy", type=float, default=0.95)
    ap.add_argument("--min-coverage", type=float, default=0.70)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    if a.select_threshold and a.split != "val":
        raise SystemExit("Refusing: threshold selection must use the validation split.")

    md = Path(a.model_dir)
    weights = md / ("model.joblib" if (md / "model.joblib").exists() else "best_model.pt")
    pred = Predictor.load(str(weights), str(md / "model_meta.json"))
    df = pd.read_csv(a.splits); df = df[df.split == a.split]
    cmap = {c: i for i, c in enumerate(pred.class_names)}
    y = df.label.map(cmap).values
    probs, times = [], []
    import time
    for p in df.path:
        with Image.open(p) as im:
            im = im.convert("RGB")
            t = time.perf_counter(); pr = pred._probs(im); times.append((time.perf_counter() - t) * 1000)
        probs.append(pr)
    probs = np.stack(probs)

    out = Path(a.out); out.mkdir(exist_ok=True)
    tag = f"{pred.meta.get('experiment', md.name)}_{a.split}"
    if a.select_threshold:
        th = select_threshold(probs, y, a.target_accuracy, a.min_coverage)
        meta = json.loads((md / "model_meta.json").read_text()); meta["threshold"] = th
        (md / "model_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Selected threshold (validation): {th:.3f}")
        pred.threshold = th

    summ = classification_summary(y, probs.argmax(1), pred.class_names)
    curve = coverage_accuracy_curve(probs, y)
    keep = probs.max(1) >= pred.threshold
    summ["abstention"] = {"threshold": pred.threshold, "coverage": float(keep.mean()),
                          "accuracy_on_answered": float((probs.argmax(1)[keep] == y[keep]).mean()) if keep.any() else None}
    summ["latency_includes_preprocessing"] = True
    summ["latency"] = latency_stats(times)
    summ["n_samples"] = int(len(y))
    (out / f"metrics_{tag}.json").write_text(json.dumps(summ, indent=2))
    pd.DataFrame(curve).to_csv(out / f"coverage_curve_{tag}.csv", index=False)

    cm = np.array(summ["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(pred.class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(pred.class_names)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix ({a.split})")
    fig.tight_layout(); fig.savefig(out / f"confusion_{tag}.png", dpi=150)

    wrong = df.assign(pred=[pred.class_names[i] for i in probs.argmax(1)], conf=probs.max(1))
    wrong[wrong.pred != wrong.label].to_csv(out / f"errors_{tag}.csv", index=False)
    print(f"[{a.split}] acc={summ['accuracy']:.4f} macro-F1={summ['macro_f1']:.4f} "
          f"coverage={summ['abstention']['coverage']:.2f} p95={summ['latency']['p95_ms']:.1f}ms")


if __name__ == "__main__":
    main()
