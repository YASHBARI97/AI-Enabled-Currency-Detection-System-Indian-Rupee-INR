import pytest
from src.data.validate import InvalidInputError, validate_and_load
from tests.conftest import make_png


def test_valid_png():
    assert validate_and_load(make_png(), "a.png").mode == "RGB"


def test_valid_jpeg():
    assert validate_and_load(make_png(fmt="JPEG"), "a.jpg").size == (120, 60)


@pytest.mark.parametrize("data,name,status", [
    (b"", "a.png", "empty_input"),
    (b"not an image", "a.png", "unreadable_image"),
    (make_png(), "a.gif", "unsupported_format"),
    (make_png(), "noext", "unsupported_format"),
])
def test_rejections(data, name, status):
    with pytest.raises(InvalidInputError) as e:
        validate_and_load(data, name)
    assert e.value.status == status


def test_too_large():
    with pytest.raises(InvalidInputError) as e:
        validate_and_load(make_png(size=(800, 800)), "a.png", max_mb=0.0001)
    assert e.value.status == "file_too_large"


def test_wrong_real_format_with_png_extension():
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (10, 10)).save(buf, format="GIF")
    with pytest.raises(InvalidInputError) as e:
        validate_and_load(buf.getvalue(), "fake.png")
    assert e.value.status == "unsupported_format"
