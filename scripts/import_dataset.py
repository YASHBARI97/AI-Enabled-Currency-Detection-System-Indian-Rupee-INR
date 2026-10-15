import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "test": "test", "testing": "test"}
KEEP_EXT = {".jpg", ".jpeg", ".png"}
CONVERT_EXT = {".webp", ".bmp", ".tif", ".tiff", ".jfif"}
IMG_EXT = KEEP_EXT | CONVERT_EXT
KNOWN = {10, 20, 50, 100, 200, 500, 2000}
FILE_COLS = ["filename", "file", "file_name", "image", "image_name", "path"]
LABEL_COLS = ["class", "label", "category", "denomination"]


def find_split_dirs(src: Path) -> dict:
    found = {}
    for d in [src] + [p for p in src.iterdir() if p.is_dir()]:
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name.lower() in SPLIT_NAMES:
                found.setdefault(SPLIT_NAMES[sub.name.lower()], sub)
        if found:
            break
    return found


def images_in(d: Path):
    return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXT]


def to_label(name, explicit: dict):
    """Raw class name -> INR_xx label (None if it cannot be inferred)."""
    name = str(name).strip()
    if name in explicit:
        return explicit[name]
    if re.fullmatch(r"INR_\d+", name):
        return name
    nums = re.findall(r"\d+", name)
    return f"INR_{int(nums[0])}" if len(nums) == 1 and int(nums[0]) in KNOWN else None


def label_from_filename(stem: str):
    hits = {int(m) for m in re.findall(r"(?<!\d)(10|20|50|100|200|500|2000)(?!\d)", stem)}
    return f"INR_{hits.pop()}" if len(hits) == 1 else None


# ---- strategies: each returns list of (image_path, raw_class_name_or_None) or None if not applicable ----
def strat_folders(sdir: Path):
    cdirs = [p for p in sorted(sdir.iterdir()) if p.is_dir() and p.name.lower() not in ("images", "labels")]
    cdirs = [c for c in cdirs if images_in(c)]
    if not cdirs:
        return None
    return [(f, c.name) for c in cdirs for f in images_in(c)]


def strat_csv(sdir: Path):
    csvs = sorted(sdir.glob("*.csv"))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0], skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    fcol = next((low[c] for c in FILE_COLS if c in low), df.columns[0])
    lcol = next((low[c] for c in LABEL_COLS if c in low), None)
    by_name = {p.name.lower(): p for p in images_in(sdir)}
    result, per_file = [], defaultdict(set)
    if lcol is not None:                         # filename,label  (also detection-style CSV with a class column)
        for fn, cl in zip(df[fcol].astype(str), df[lcol].astype(str)):
            per_file[Path(fn).name.lower()].add(cl.strip())
    else:                                         # one-hot columns
        classes = [c for c in df.columns if c != fcol]
        for _, row in df.iterrows():
            on = [c for c in classes if str(row[c]).strip() in ("1", "1.0", "True")]
            per_file[Path(str(row[fcol])).name.lower()].update(on if len(on) == 1 else [])
    for key, p in by_name.items():
        cl = per_file.get(key, set())
        result.append((p, next(iter(cl)) if len(cl) == 1 else None))   # ambiguous or missing -> None
    return result


def strat_yolo(sdir: Path, src: Path):
    imgd, lbld = sdir / "images", sdir / "labels"
    if not (imgd.is_dir() and lbld.is_dir()):
        return None
    ycands = sorted(list(src.glob("*.yaml")) + list(src.glob("*/*.yaml")))
    if not ycands:
        raise SystemExit("YOLO-style folders found but no data.yaml (class names) under the source folder.")
    names = yaml.safe_load(ycands[0].read_text(encoding="utf-8")).get("names", {})
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    out = []
    for p in images_in(imgd):
        lf = lbld / f"{p.stem}.txt"
        ids = set()
        if lf.exists():
            ids = {int(float(line.split()[0])) for line in lf.read_text().splitlines() if line.strip()}
        out.append((p, names.get(next(iter(ids))) if len(ids) == 1 else None))
    return out


def strat_filename(sdir: Path):
    imgs = images_in(sdir)
    if imgs and all(p.parent == sdir for p in imgs):
        found = [(p, label_from_filename(p.stem)) for p in imgs]
        if sum(1 for _, l in found if l) >= 0.5 * len(found):
            return found
    return None


def diagnostic(src: Path, splits: dict):
    print("\n=== COULD NOT RECOGNISE THE LAYOUT. Please paste everything below. ===")
    for sp, d in splits.items():
        files = [p for p in d.rglob("*") if p.is_file()]
        print(f"\n[{sp}] {d}")
        print("  top-level entries:", [p.name for p in sorted(d.iterdir())][:12])
        print("  file types:", dict(Counter(p.suffix.lower() or "(none)" for p in files)))
        print("  sample files:", [p.name for p in files][:8])
        for p in [q for q in files if q.suffix.lower() in (".csv", ".txt", ".xml", ".json")][:2]:
            try:
                print(f"  first lines of {p.name}:", p.read_text(encoding="utf-8", errors="replace").splitlines()[:3])
            except OSError:
                pass
    print("  yaml files under source:", [str(p) for p in src.rglob("*.yaml")][:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/raw")
    ap.add_argument("--map", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    rs, rd = src.resolve(), dst.resolve()
    if rs == rd or rd in rs.parents or rs in rd.parents:
        raise SystemExit("--src and --dst overlap. Keep the downloaded dataset in its own folder (data/original).")
    explicit = dict(kv.split("=", 1) for kv in a.map.split(",") if "=" in kv)

    splits = find_split_dirs(src)
    if not splits:
        raise SystemExit("No train/validation/test folders found under the source. Check --src.")
    plan, unmapped, unlabeled, used = [], Counter(), Counter(), {}
    for sp, sdir in splits.items():
        found, name = None, None
        for name, fn in (("class folders", lambda: strat_folders(sdir)), ("CSV labels", lambda: strat_csv(sdir)),
                         ("YOLO images/labels", lambda: strat_yolo(sdir, src)),
                         ("class in file name", lambda: strat_filename(sdir))):
            found = fn()
            if found:
                break
        if not found:
            diagnostic(src, splits)
            raise SystemExit("\nNothing was written.")
        used[sp] = (name, len(found))
        for p, raw in found:
            if raw is None:
                unlabeled[sp] += 1
                continue
            lab = label_from_filename(p.stem) if name == "class in file name" and not str(raw).startswith("INR_") \
                else to_label(raw, explicit)
            if lab is None:
                unmapped[str(raw)] += 1
                continue
            plan.append((sp, str(raw), lab, p, p.suffix.lower() in CONVERT_EXT))

    print("Detected per split:", {k: f"{v[0]} ({v[1]} images)" for k, v in used.items()})
    if unlabeled:
        print("Images without a single clear label (skipped):", dict(unlabeled))
    if unmapped:
        print("\nUNMAPPED class names (cannot infer a denomination):")
        for k, v in unmapped.items():
            print(f"  '{k}': {v} images")
        raise SystemExit('Add them with --map "Name=INR_10,..." and re-run. Nothing was written.')
    table, mapping = defaultdict(Counter), set()
    for sp, raw, lab, _, _ in plan:
        table[lab][sp] += 1
        mapping.add((raw, lab))
    print("\nRaw class name -> label:")
    for raw, lab in sorted(mapping):
        print(f"  '{raw}' -> {lab}")
    print(f"\n  {'class':10s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s}")
    for lab in sorted(table):
        c = table[lab]
        print(f"  {lab:10s} {c['train']:6d} {c['val']:6d} {c['test']:6d} {sum(c.values()):6d}")
    print(f"\nTotal usable images: {len(plan)}")
    if not plan:
        raise SystemExit("No usable labelled images were found.")
    if not a.apply:
        print("\nPREVIEW ONLY. Check the mapping and counts, then re-run with --apply.")
        return

    existing = sum(1 for p in dst.rglob("*") if p.suffix.lower() in KEEP_EXT) if dst.exists() else 0
    if existing and not a.force:
        raise SystemExit(f"{dst} already has {existing} images; running again would duplicate them. Empty it or use --force.")
    manifest, converted, failed = [], 0, []
    safe = lambda t: re.sub(r"[^A-Za-z0-9_-]+", "_", t)  # noqa: E731
    for sp, raw, lab, f, conv in plan:
        od = dst / lab
        od.mkdir(parents=True, exist_ok=True)
        ext = ".jpg" if conv else f.suffix.lower()
        out, n = od / f"{sp}_{safe(f.stem)}{ext}", 1
        while out.exists():
            out, n = od / f"{sp}_{safe(f.stem)}_{n}{ext}", n + 1
        try:
            if conv:
                with Image.open(f) as im:
                    im.convert("RGB").save(out, quality=95)
                converted += 1
            else:
                shutil.copy2(f, out)
            manifest.append({"new_path": str(out), "label": lab, "orig_split": sp, "orig_path": str(f)})
        except Exception as e:  # noqa: BLE001
            failed.append((str(f), type(e).__name__))
    Path("reports").mkdir(exist_ok=True)
    with open("reports/merge_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["new_path", "label", "orig_split", "orig_path"])
        w.writeheader(); w.writerows(manifest)
    on_disk = sum(1 for p in dst.rglob("*") if p.suffix.lower() in KEEP_EXT)
    print(f"\nCopied {len(manifest)} images to {dst} ({converted} converted, {len(failed)} failed).")
    if on_disk != existing + len(manifest):
        print(f"WARNING: expected {existing + len(manifest)} images but found {on_disk}. Inspect before continuing.")
    print("Next: python -m scripts.prepare_data --data-dir", dst)


if __name__ == "__main__":
    main()