"""Audit source overlap and optionally copy YOLO images into grouped splits."""
import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "valid": "validation", "val": "validation", "validation": "validation", "test": "test"}


def load_dataset_yaml(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("names"):
        raise ValueError("Dataset YAML must define a non-empty 'names' mapping/list")
    root = Path(config.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return path, config, root


def source_id(image_path):
    """Infer a source key only when filenames expose a video/frame relationship."""
    stem = image_path.stem
    stem = re.sub(r"\.rf\.[^.]+$", "", stem, flags=re.I)
    stem = re.sub(r"_jpg$", "", stem, flags=re.I)
    frame_match = re.match(r"^(.*?)(?:[_-](?:mp4|avi|mov|mkv|webm))[-_](\d+)$", stem, re.I)
    if frame_match:
        return frame_match.group(1).casefold() + "_video"
    if re.match(r"^(?:frame|img|image|snapshot|video_frame)[_-]?\d+$", stem, re.I):
        return None
    return stem.casefold()


def labels_for_image(image_path):
    label_path = image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing YOLO label file for {image_path}: {label_path}")
    counts = Counter()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            # Accept either detection boxes (class + xywh) or segmentation
            # polygons (class + at least three xy points), as present here.
            coordinate_count = len(parts) - 1
            if class_id < 0 or not (coordinate_count == 4 or
                                    (coordinate_count >= 6 and coordinate_count % 2 == 0)):
                raise ValueError
            coords = [float(value) for value in parts[1:]]
            if any(value < 0 or value > 1 for value in coords):
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from exc
        counts[class_id] += 1
    return label_path, counts


def collect_splits(config, root, source_map=None):
    source_rows = defaultdict(list)
    class_count = len(config["names"])
    for key, split in SPLIT_ALIASES.items():
        if key not in config:
            continue
        value = config[key]
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            image_dir = Path(entry)
            if not image_dir.is_absolute():
                image_dir = root / image_dir
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory for split '{key}' does not exist: {image_dir}")
            for image in sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
                label_path, counts = labels_for_image(image)
                if any(index >= class_count for index in counts):
                    raise ValueError(f"Label class ID exceeds YAML names in {label_path}")
                inferred = source_id(image)
                relative = str(image.resolve().relative_to(root.resolve())).replace("\\", "/")
                mapped = (source_map or {}).get(relative) or (source_map or {}).get(image.name)
                known = inferred is not None or bool(mapped)
                group = str(mapped).casefold() if mapped else inferred
                # Keep unknown generic frame names separate until a source map joins them.
                if group is None:
                    group = f"__untraceable__:{relative.casefold()}"
                source_rows[group].append({"image": image, "label": label_path, "split": split,
                                           "counts": counts, "source_known": known})
    if not source_rows:
        raise ValueError("No labelled images found in configured YOLO splits")
    return source_rows


def summarize_source_overlap(source_rows):
    split_sources = {name: set() for name in ("train", "validation", "test")}
    split_images = Counter()
    split_instances = {name: Counter() for name in split_sources}
    source_class_support = Counter()
    for source, rows in source_rows.items():
        seen_classes = {class_id for row in rows for class_id, count in row["counts"].items() if count > 0}
        source_class_support.update(seen_classes)
        for row in rows:
            split_sources[row["split"]].add(source)
            split_images[row["split"]] += 1
            split_instances[row["split"]].update(row["counts"])
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        values = sorted(split_sources[a] & split_sources[b])
        overlaps[f"{a}/{b}"] = {"count": len(values), "examples": values[:20]}
    untraceable = sum(1 for rows in source_rows.values() for row in rows if not row.get("source_known", True))
    untraceable_examples = [row["image"].name for rows in source_rows.values() for row in rows
                            if not row.get("source_known", True)][:20]
    return {"images": dict(split_images), "source_groups": {k: len(v) for k, v in split_sources.items()},
            "overlap": overlaps, "class_instances": {k: dict(v) for k, v in split_instances.items()},
            "source_groups_total": len(source_rows), "class_source_group_support": dict(source_class_support),
            "untraceable_video_frame_images": untraceable,
            "untraceable_examples": untraceable_examples}


def make_assignment(source_rows, seed=42):
    groups = list(source_rows)
    n = len(groups)
    if n < 3:
        raise ValueError("At least three inferred source groups are required for train/validation/test")
    ratios = {"train": .70, "validation": .15, "test": .15}
    rng = random.Random(seed)
    rng.shuffle(groups)
    histograms = {}
    for group in groups:
        histogram = Counter()
        for row in source_rows[group]:
            histogram.update(row["counts"])
        histograms[group] = histogram
    totals = Counter()
    for histogram in histograms.values():
        totals.update(histogram)
    # Place groups carrying the rarest labels first; score candidate placements
    # against per-class instance targets as well as the desired group counts.
    def rarity(group):
        return max((count / max(1, totals[class_id]) for class_id, count in histograms[group].items()), default=0)
    groups.sort(key=rarity, reverse=True)
    validation_n = max(1, round(n * ratios["validation"]))
    test_n = max(1, round(n * ratios["test"]))
    if validation_n + test_n >= n:
        raise ValueError("Not enough source groups for requested three-way split")
    target_groups = {"train": n - validation_n - test_n,
                     "validation": validation_n, "test": test_n}
    target_instances = {name: {class_id: count * ratio for class_id, count in totals.items()}
                        for name, ratio in ratios.items()}
    assigned_groups = Counter()
    assigned_instances = {name: Counter() for name in ratios}
    assignment = {}
    for group in groups:
        options = list(ratios)
        rng.shuffle(options)
        available = [split for split in options if assigned_groups[split] < target_groups[split]]
        def cost(split):
            class_cost = 0.0
            for class_id, target in target_instances[split].items():
                delta = assigned_instances[split][class_id] + histograms[group][class_id] - target
                class_cost += (delta * delta) / max(1.0, target)
            return class_cost
        selected = min(available, key=cost)
        assignment[group] = selected
        assigned_groups[selected] += 1
        assigned_instances[selected].update(histograms[group])
    return assignment


def write_grouped_dataset(source_rows, assignment, classes, output_dir, seed):
    untraceable = [row["image"] for rows in source_rows.values() for row in rows if not row.get("source_known", True)]
    if untraceable:
        raise ValueError(f"Cannot claim a source-disjoint split: {len(untraceable)} generic frame filenames lack source IDs. Supply --source-map entries before copying.")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"seed": seed, "strategy": "seeded rarity-first multi-label group assignment with exact 70/15/15 source-group quotas",
                "classes": classes, "splits": {name: [] for name in ("train", "validation", "test")}}
    try:
        for source, rows in source_rows.items():
            split = assignment[source]
            for row in rows:
                prefix = hashlib.sha256(str(row["image"].resolve()).encode("utf-8")).hexdigest()[:12]
                filename = f"{prefix}_{row['image'].name}"
                image_out = output_dir / split / "images" / filename
                label_out = output_dir / split / "labels" / Path(filename).with_suffix(".txt").name
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                if image_out.exists() or label_out.exists():
                    raise FileExistsError(f"Duplicate destination filename while preparing split: {image_out}")
                shutil.copy2(row["image"], image_out)
                shutil.copy2(row["label"], label_out)
                manifest["splits"][split].append({"image": str(image_out.relative_to(output_dir)),
                                                  "source_group": source})
        data_yaml = {"path": str(output_dir), "train": "train/images", "val": "validation/images",
                     "test": "test/images", "nc": len(classes), "names": classes}
        (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        (output_dir / "source_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        # Keep partial data visible for inspection; never recursively delete user data.
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory for copied source-disjoint data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-map", help="Optional JSON mapping relative image paths (or basenames) to source video IDs")
    parser.add_argument("--dry-run", action="store_true", help="Audit only; do not copy files")
    args = parser.parse_args()
    try:
        _, config, root = load_dataset_yaml(args.data_yaml)
        source_map = json.loads(Path(args.source_map).read_text(encoding="utf-8")) if args.source_map else None
        if source_map is not None and not isinstance(source_map, dict):
            raise ValueError("Source map must be a JSON object mapping image paths to source IDs")
        rows = collect_splits(config, root, source_map)
        report = summarize_source_overlap(rows)
        print(json.dumps(report, indent=2))
        if args.dry_run:
            assignment = make_assignment(rows, args.seed)
            proposed = {name: Counter() for name in ("train", "validation", "test")}
            for group, split in assignment.items():
                for row in rows[group]:
                    proposed[split].update(row["counts"])
            print("Proposed grouped split report:", json.dumps({
                "source_groups": dict(Counter(assignment.values())),
                "class_instances": {split: dict(counts) for split, counts in proposed.items()},
            }, indent=2))
            if report["untraceable_video_frame_images"]:
                print("WARNING: generic frame names without source IDs remain; provide a source map before preparing training data.")
            print("Dry run complete; no data files were copied.")
            return
        assignment = make_assignment(rows, args.seed)
        output = write_grouped_dataset(rows, assignment, config["names"], args.output_dir, args.seed)
        print(f"Source-disjoint dataset prepared at {output}")
        print("Review class counts and source_split_manifest.json before training.")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
