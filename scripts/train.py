import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import classification_summary
from src.models.factory import build_model
from src.preprocessing.transforms import build_eval_transform, build_train_transform
from src.utils.config import load_config
from src.utils.seed import set_seed


class SplitDataset(Dataset):
    def __init__(self, df, cmap, tfm):
        self.paths, self.y, self.tfm = df.path.tolist(), df.label.map(cmap).tolist(), tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tfm(im.convert("RGB")), self.y[i]


def run_epoch(model, loader, loss_fn, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, n, ys, ps = 0.0, 0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
            pr = out.argmax(1); correct += (pr == y).sum().item()
            ys += y.cpu().tolist(); ps += pr.cpu().tolist()
    return tot / n, correct / n, ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default="data/splits/splits.csv")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir or f"artifacts/{cfg['experiment']}"); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.splits)
    classes = sorted(df.label.unique()); cmap = {c: i for i, c in enumerate(classes)}
    size = cfg["img_size"]
    tr = SplitDataset(df[df.split == "train"], cmap, build_train_transform(size, cfg["augmentation"]))
    va = SplitDataset(df[df.split == "val"], cmap, build_eval_transform(size))   # NO augmentation
    g = torch.Generator().manual_seed(cfg["seed"])
    tl = DataLoader(tr, cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], generator=g)
    vl = DataLoader(va, cfg["batch_size"], num_workers=cfg["num_workers"])

    model = build_model(cfg["arch"], len(classes), cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    # class-weighted loss to counter imbalance
    counts = df[df.split == "train"].label.map(cmap).value_counts().sort_index().values
    w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best_f1, bad, hist = -1.0, 0, []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        trl, tra, _, _ = run_epoch(model, tl, loss_fn, device, opt)
        vl_loss, vacc, ys, ps = run_epoch(model, vl, loss_fn, device)
        sched.step()
        s = classification_summary(ys, ps, classes)
        hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "val_loss": vl_loss,
                     "val_acc": vacc, "val_macro_f1": s["macro_f1"]})
        print(f"ep{ep:02d} train_loss={trl:.3f} val_loss={vl_loss:.3f} val_acc={vacc:.3f} val_f1={s['macro_f1']:.3f}")
        if s["macro_f1"] > best_f1:
            best_f1, bad = s["macro_f1"], 0
            torch.save({"model": model.state_dict()}, out / "best_model.pt")
        else:
            bad += 1
            if bad >= cfg["early_stopping_patience"]:
                print("early stopping"); break

    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    meta = {"backend": "torch", "arch": cfg["arch"], "class_names": classes, "img_size": size,
            "threshold": 0.5,  # replaced by scripts.evaluate --select-threshold using VALIDATION data
            "model_version": cfg["model_version"], "experiment": cfg["experiment"]}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "experiment_summary.json").write_text(json.dumps({
        "config": cfg, "best_val_macro_f1": best_f1, "epochs_run": len(hist),
        "hardware": device, "train_seconds": round(time.time() - t0, 1),
        "torch_version": torch.__version__, "pretrained_weights": "torchvision DEFAULT (ImageNet-1k)"}, indent=2))
    print(f"best val macro-F1 = {best_f1:.4f}; saved to {out}")


if __name__ == "__main__":
    main()
