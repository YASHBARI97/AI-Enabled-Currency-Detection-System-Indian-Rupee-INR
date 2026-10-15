import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.metrics import classification_summary
from src.models.baseline import build_baseline, extract_features
from src.utils.config import load_config
from src.utils.seed import set_seed


def feats(paths, size):
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(extract_features(im, size))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_svm.yaml")
    ap.add_argument("--splits", default="data/splits/splits.csv")
    ap.add_argument("--out-dir", default="artifacts/baseline")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    df = pd.read_csv(a.splits)
    classes = sorted(df["label"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    tr, va = df[df.split == "train"], df[df.split == "val"]
    model = build_baseline(cfg["svm_C"], cfg["seed"]).fit(feats(tr.path, cfg["img_size"]), tr.label.map(cmap))
    pred = model.predict(feats(va.path, cfg["img_size"]))
    summ = classification_summary(va.label.map(cmap).values, pred, classes)  # VALIDATION only
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {"backend": "sklearn", "class_names": classes, "img_size": cfg["img_size"],
            "threshold": 0.5, "model_version": cfg["model_version"], "experiment": cfg["experiment"]}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "experiment_summary.json").write_text(json.dumps(
        {"config": cfg, "val_accuracy": summ["accuracy"], "val_macro_f1": summ["macro_f1"]}, indent=2))
    print(f"[baseline] val acc={summ['accuracy']:.4f} macro-F1={summ['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
