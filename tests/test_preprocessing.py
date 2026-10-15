import numpy as np
from PIL import Image
from src.preprocessing.transforms import to_normalised_array


def test_shape_dtype_and_rgb_order():
    img = Image.new("RGB", (300, 100), (255, 0, 0))
    a = to_normalised_array(img, 224)
    assert a.shape == (3, 224, 224) and a.dtype == np.float32
    assert a[0].mean() > a[2].mean()  # red channel first


def test_handles_grayscale_and_rgba():
    for mode in ("L", "RGBA"):
        assert to_normalised_array(Image.new(mode, (50, 50)), 64).shape == (3, 64, 64)
