import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.dedupe import assign_groups
from src.data.splits import assert_no_leakage, make_splits, scan_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="data/splits/splits.csv")
    ap.add_argument("--groups-csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hamming", type=int, default=6)
    a = ap.parse_args()

    df = scan_folder(a.data_dir)
    bad = []
    for p in df["path"]:  # cleaning log: unreadable files
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            bad.append(p)
    df = df[~df["path"].isin(bad)].reset_index(drop=True)

    df = assign_groups(df, a.hamming)
    if a.groups_csv:
        # The importer manifest also contains label and source metadata. Only
        # path/group belong in this join, so the scanned class label stays intact.
        g = pd.read_csv(a.groups_csv, usecols=["path", "group"])
        df = df.drop(columns="group").merge(g, on="path", how="left")
        # Series.fillna does not accept an Index as its value argument. Build
        # unique fallback IDs only for rows without a manifest group.
        missing = df["group"].isna()
        df.loc[missing, "group"] = [f"solo_{i}" for i in df.index[missing]]
    df = make_splits(df, seed=a.seed)
    assert_no_leakage(df)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    stats = {
        "total_images": int(len(df)), "removed_unreadable": bad,
        "per_class": df["label"].value_counts().to_dict(),
        "per_split_class": df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "n_groups": int(df["group"].nunique()),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
