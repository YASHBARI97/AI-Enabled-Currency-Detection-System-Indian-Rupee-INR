import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "test": "test", "testing": "test"}
KNOWN = {10, 20, 50, 100, 200, 500, 2000}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def to_label(name, explicit):
    name = str(name).strip()
    if name in explicit:
        return explicit[name]
    if re.fullmatch(r"INR_\d+", name):
        return name
    nums = re.findall(r"\d+", name)
    return f"INR_{int(nums[0])}" if len(nums) == 1 and int(nums[0]) in KNOWN else None


def base_name(filename: str) -> str:
    """'100__102_jpg.rf.43d1...jpg' -> '100__102_jpg' (the original photo, before Roboflow's hash)."""
    return filename.split(".rf.")[0] if ".rf." in filename else Path(filename).stem


def prefix_label(filename: str):
    m = re.match(r"^(\d+)__", filename)
    return f"INR_{int(m.group(1))}" if m and int(m.group(1)) in KNOWN else None


def find_split_dirs(src: Path) -> dict:
    found = {}
    for d in [src] + [p for p in src.iterdir() if p.is_dir()]:
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name.lower() in SPLIT_NAMES:
                found.setdefault(SPLIT_NAMES[sub.name.lower()], sub)
        if found:
            break
    return found


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
        raise SystemExit("--src and --dst overlap. Keep the downloaded dataset in data/original.")
    explicit = dict(kv.split("=", 1) for kv in a.map.split(",") if "=" in kv)
    splits = find_split_dirs(src)
    if not splits:
        raise SystemExit("No train/valid/test folders found under --src.")

    plan, unmapped = [], Counter()
    skipped = Counter()
    cat_usage = Counter()
    ann_per_image = Counter()
    for sp, sdir in splits.items():
        jf = next(iter(sorted(sdir.glob("*.json"))), None)
        if jf is None:
            raise SystemExit(f"No COCO .json file found in {sdir}")
        coco = json.loads(jf.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        per_img = defaultdict(set)
        n_ann = Counter()
        for an in coco.get("annotations", []):
            per_img[an["image_id"]].add(cats.get(an["category_id"], str(an["category_id"])))
            n_ann[an["image_id"]] += 1
        for im in coco.get("images", []):
            fname = Path(im["file_name"]).name
            path = sdir / fname
            if not path.exists() or path.suffix.lower() not in IMG_EXT:
                skipped[f"{sp}: image file missing"] += 1
                continue
            names = per_img.get(im["id"], set())
            ann_per_image[min(n_ann.get(im["id"], 0), 3)] += 1
            coco_lab = to_label(next(iter(names)), explicit) if len(names) == 1 else None
            if len(names) == 1 and coco_lab is None:
                unmapped[next(iter(names))] += 1
                continue
            pre = prefix_label(fname)
            if len(names) > 1:
                skipped[f"{sp}: several different classes in one image"] += 1
                continue
            lab = coco_lab or pre
            if lab is None:
                skipped[f"{sp}: no annotation and no class in file name"] += 1
                continue
            if coco_lab and pre and coco_lab != pre:
                skipped[f"{sp}: annotation and file-name class DISAGREE"] += 1
                continue
            cat_usage[(sp, next(iter(names)) if names else "(from file name)")] += 1
            plan.append((sp, lab, path, base_name(fname)))

    if unmapped:
        print("UNMAPPED COCO class names (cannot infer a denomination):")
        for k, v in unmapped.items():
            print(f"  '{k}': {v} images")
        raise SystemExit('Add them with --map "Name=INR_10,..." and re-run. Nothing was written.')

    table = defaultdict(Counter)
    for sp, lab, _, _ in plan:
        table[lab][sp] += 1
    print("COCO categories used (split, name): images")
    for (sp, nm), n in sorted(cat_usage.items()):
        print(f"  {sp:5s} '{nm}': {n}")
    print("\nAnnotations per image (0,1,2,3+):", {k: ann_per_image[k] for k in sorted(ann_per_image)})
    print(f"\n  {'class':10s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s}")
    for lab in sorted(table):
        c = table[lab]
        print(f"  {lab:10s} {c['train']:6d} {c['val']:6d} {c['test']:6d} {sum(c.values()):6d}")
    groups = defaultdict(set)
    for sp, _, _, g in plan:
        groups[g].add(sp)
    cross = [g for g, s in groups.items() if len(s) > 1]
    print(f"\nTotal usable images: {len(plan)}   original photos (groups): {len(groups)}"
          f"   avg copies per photo: {len(plan) / max(1, len(groups)):.2f}")
    print(f"Groups that appear in MORE THAN ONE original split (leakage in the download): {len(cross)}")
    if skipped:
        print("\nSkipped:", dict(skipped))
    if not plan:
        raise SystemExit("No usable labelled images.")
    if not a.apply:
        print("\nPREVIEW ONLY. Check the numbers above, then re-run with --apply.")
        return

    existing = sum(1 for p in dst.rglob("*") if p.suffix.lower() in IMG_EXT) if dst.exists() else 0
    if existing and not a.force:
        raise SystemExit(f"{dst} already has {existing} images; running again would duplicate them. Empty it or use --force.")
    safe = lambda t: re.sub(r"[^A-Za-z0-9_-]+", "_", t)  # noqa: E731
    manifest = []
    for sp, lab, f, g in plan:
        od = dst / lab
        od.mkdir(parents=True, exist_ok=True)
        out, n = od / f"{sp}_{safe(f.stem)}{f.suffix.lower()}", 1
        while out.exists():
            out, n = od / f"{sp}_{safe(f.stem)}_{n}{f.suffix.lower()}", n + 1
        shutil.copy2(f, out)
        manifest.append({"path": str(out), "group": f"{lab}__{g}", "label": lab, "orig_split": sp, "orig_path": str(f)})
    Path("reports").mkdir(exist_ok=True)
    with open("reports/groups.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "group", "label", "orig_split", "orig_path"])
        w.writeheader(); w.writerows(manifest)
    on_disk = sum(1 for p in dst.rglob("*") if p.suffix.lower() in IMG_EXT)
    print(f"\nCopied {len(manifest)} images to {dst}. Group file: reports/groups.csv")
    if on_disk != existing + len(manifest):
        print(f"WARNING: expected {existing + len(manifest)} images but found {on_disk}.")
    print("Next: python -m scripts.prepare_data --data-dir data/raw --groups-csv reports/groups.csv")


if __name__ == "__main__":
    main()