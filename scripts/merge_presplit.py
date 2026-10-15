import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "test": "test", "testing": "test"}
KEEP_EXT = {".jpg", ".jpeg", ".png"}
CONVERT_EXT = {".webp", ".bmp", ".tif", ".tiff", ".jfif"}
KNOWN = {10, 20, 50, 100, 200, 500, 2000}


def find_split_dirs(src: Path) -> dict:
    found = {}
    for d in [src] + [p for p in src.iterdir() if p.is_dir()]:
        for sub in d.iterdir() if d.is_dir() else []:
            if sub.is_dir() and sub.name.lower() in SPLIT_NAMES:
                found.setdefault(SPLIT_NAMES[sub.name.lower()], sub)
        if found:
            break
    return found


def auto_label(folder_name: str, explicit: dict) -> str | None:
    if folder_name in explicit:
        return explicit[folder_name]
    nums = re.findall(r"\d+", folder_name)
    if len(nums) == 1 and int(nums[0]) in KNOWN:
        return f"INR_{int(nums[0])}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train/validation/test")
    ap.add_argument("--dst", default="data/raw")
    ap.add_argument("--map", default="", help='"FolderName=INR_10,Other=INR_20"')
    ap.add_argument("--apply", action="store_true", help="actually copy files (default: preview only)")
    ap.add_argument("--force", action="store_true", help="allow writing into a destination that already has images")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    rs, rd = src.resolve(), dst.resolve()
    if rs == rd or rd in rs.parents or rs in rd.parents:
        raise SystemExit(
            f"--src ({rs}) and the output folder --dst ({rd}) overlap.\n"
            "Keep the downloaded dataset in its own folder (e.g. data/original/) and "
            "let this script create data/raw/ for the merged result.")
    explicit = dict(kv.split("=", 1) for kv in a.map.split(",") if "=" in kv)

    splits = find_split_dirs(src)
    if not splits:
        raise SystemExit("Could not find train/validation/test folders under the source. "
                         "Check the path (it should contain those folders).")
    print("Found splits:", {k: str(v) for k, v in splits.items()})

    plan, unmapped, skipped = [], defaultdict(int), Counter()
    for split, sdir in splits.items():
        class_dirs = [p for p in sorted(sdir.iterdir()) if p.is_dir()]
        if not class_dirs:
            raise SystemExit(f"No class sub-folders inside {sdir}. If it contains 'images' and 'labels' folders "
                             "this is a YOLO-style dataset: tell the assistant and use a different converter.")
        for cdir in class_dirs:
            label = auto_label(cdir.name, explicit)
            for f in sorted(cdir.rglob("*")):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext not in KEEP_EXT and ext not in CONVERT_EXT:
                    skipped[ext or "(none)"] += 1
                    continue
                if label is None:
                    unmapped[cdir.name] += 1
                    continue
                plan.append((split, cdir.name, label, f, ext in CONVERT_EXT))

    if unmapped:
        print("\nUNMAPPED class folders (could not infer a denomination from the name):")
        for k, v in unmapped.items():
            print(f"  '{k}': {v} images")
        raise SystemExit('Add them with --map "FolderName=INR_10,..." and run again. Nothing was written.')

    table = defaultdict(Counter)
    folder_to_label = {}
    for split, cname, label, _, _ in plan:
        table[label][split] += 1
        folder_to_label[(split, cname)] = label
    print("\nFolder -> label mapping:")
    for (split, cname), lab in sorted(folder_to_label.items()):
        print(f"  {split:5s} '{cname}'  ->  {lab}")
    print("\nImages per class (original split):")
    print(f"  {'class':10s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s}")
    for lab in sorted(table):
        c = table[lab]
        print(f"  {lab:10s} {c['train']:6d} {c['val']:6d} {c['test']:6d} {sum(c.values()):6d}")
    if skipped:
        print("\nSkipped unsupported files:", dict(skipped))
    print(f"\nTotal usable images: {len(plan)}")

    if not a.apply:
        print("\nPREVIEW ONLY. Check the mapping above, then re-run with --apply.")
        return

    existing = sum(1 for p in dst.rglob("*") if p.suffix.lower() in KEEP_EXT) if dst.exists() else 0
    if existing and not a.force:
        raise SystemExit(f"{dst} already contains {existing} images; running again would duplicate them. "
                         "Delete/empty it first, or pass --force if you really want to add to it.")
    manifest, converted, failed = [], 0, []
    for split, cname, label, f, needs_convert in plan:
        out_dir = dst / label
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = lambda t: re.sub(r"[^A-Za-z0-9_-]+", "_", t)  # noqa: E731
        stem = f"{split}_{safe(cname)}_{safe(f.stem)}"
        try:
            ext_out = ".jpg" if needs_convert else f.suffix.lower()
            out, n = out_dir / f"{stem}{ext_out}", 1
            while out.exists():
                out, n = out_dir / f"{stem}_{n}{ext_out}", n + 1
            if needs_convert:
                with Image.open(f) as im:
                    im.convert("RGB").save(out, quality=95)
                converted += 1
            else:
                shutil.copy2(f, out)
            manifest.append({"new_path": str(out), "label": label, "orig_split": split, "orig_path": str(f)})
        except Exception as e:  # noqa: BLE001
            failed.append((str(f), type(e).__name__))
    Path("reports").mkdir(exist_ok=True)
    with open("reports/merge_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["new_path", "label", "orig_split", "orig_path"])
        w.writeheader(); w.writerows(manifest)
    on_disk = sum(1 for p in dst.rglob("*") if p.suffix.lower() in KEEP_EXT)
    print(f"\nCopied {len(manifest)} images to {dst} ({converted} converted to JPG, {len(failed)} failed).")
    if on_disk != existing + len(manifest):
        print(f"WARNING: expected {existing + len(manifest)} images in {dst} but found {on_disk}. Inspect before continuing.")
    for p, e in failed[:10]:
        print("  failed:", p, e)
    print("Manifest: reports/merge_manifest.csv\nNext: python -m scripts.prepare_data --data-dir", dst)


if __name__ == "__main__":
    main()