import pandas as pd
import pytest
from src.data.splits import assert_no_leakage, make_splits


def toy(n_groups=60):
    rows = []
    for g in range(n_groups):
        label = ["INR_10", "INR_20", "INR_50"][g % 3]
        for k in range(4):  # 4 near-duplicate frames per group
            rows.append({"path": f"{label}/g{g}_{k}.jpg", "label": label, "group": f"g{g}"})
    return pd.DataFrame(rows)


def test_no_group_leakage():
    df = make_splits(toy(), seed=1)
    assert_no_leakage(df)
    assert set(df.split) == {"train", "val", "test"}


def test_all_classes_in_each_split():
    df = make_splits(toy(), seed=1)
    for s in ("train", "val", "test"):
        assert df[df.split == s].label.nunique() == 3


def test_leakage_detected():
    df = make_splits(toy(), seed=1)
    df.loc[df.index[0], "split"] = "test" if df.loc[df.index[0], "split"] != "test" else "train"
    with pytest.raises(AssertionError):
        assert_no_leakage(df)


def test_requires_group_col():
    with pytest.raises(ValueError):
        make_splits(toy().drop(columns="group"))


def test_deterministic():
    a, b = make_splits(toy(), seed=7), make_splits(toy(), seed=7)
    assert a.split.tolist() == b.split.tolist()
