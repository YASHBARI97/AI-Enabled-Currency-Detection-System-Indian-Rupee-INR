"""Shared preprocessing. Training and inference MUST both import from here."""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_rgb(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.BILINEAR)


def to_normalised_array(img: Image.Image, size: int,
                        mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """PIL -> float32 CHW array, RGB order, ImageNet-normalised."""
    arr = np.asarray(resize_rgb(img, size), dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return arr.transpose(2, 0, 1)


def build_train_transform(size: int, aug_cfg: dict):
    """Training-only augmentation (torchvision). Val/test use eval transform."""
    from torchvision import transforms as T
    ops = [
        T.RandomResizedCrop(size, scale=tuple(aug_cfg.get("crop_scale", [0.8, 1.0])),
                            ratio=(0.9, 1.6)),
        T.RandomRotation(aug_cfg.get("rotation_deg", 15), expand=False),
        T.ColorJitter(brightness=aug_cfg.get("brightness", 0.3),
                      contrast=aug_cfg.get("contrast", 0.3),
                      saturation=aug_cfg.get("saturation", 0.1)),  # keep hue: colour encodes denomination
    ]
    if aug_cfg.get("blur", True):
        ops.append(T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2))
    # NOTE: no horizontal flip -- mirrored notes are unrealistic and corrupt text/numerals.
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def build_eval_transform(size: int):
    from torchvision import transforms as T
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
