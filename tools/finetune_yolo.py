"""Fine-tune the preserved forensic YOLO checkpoint into a new run directory."""
import argparse
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Existing custom checkpoint used as initialization")
    parser.add_argument("--data-yaml", required=True, help="Prepared source-disjoint dataset data.yaml")
    parser.add_argument("--output-dir", required=True, help="New run directory; existing directories are rejected")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="Ultralytics device, such as 0 or cpu")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report settings; do not train")
    args = parser.parse_args()
    weights, data_yaml, output = map(lambda p: Path(p).resolve(), (args.weights, args.data_yaml, args.output_dir))
    if not weights.is_file():
        parser.error(f"Starting weights do not exist: {weights}")
    if not data_yaml.is_file():
        parser.error(f"Dataset YAML does not exist: {data_yaml}")
    if output.exists():
        parser.error(f"Output directory already exists; choose a unique path: {output}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("epochs and batch must be positive; imgsz must be at least 32")
    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        manifest_path = data_yaml.parent / "source_split_manifest.json"
        if not manifest_path.is_file():
            parser.error(f"Source split manifest is required beside data.yaml: {manifest_path}")
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = {split: {row["source_group"] for row in manifest["splits"][split]}
                  for split in ("train", "validation", "test")}
        overlap = (groups["train"] & groups["validation"] | groups["train"] & groups["test"] |
                   groups["validation"] & groups["test"])
        if overlap:
            parser.error(f"Source split manifest leaks groups across splits: {sorted(overlap)[:5]}")
        names = data["names"]
        name_to_id = {str(name).casefold(): int(index) for index, name in names.items()} if isinstance(names, dict) else {str(name).casefold(): i for i, name in enumerate(names)}
        dataset_root = Path(data.get("path", data_yaml.parent))
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()
        support = {}
        for split in ("train", "val", "test"):
            if split not in data:
                raise ValueError(f"Prepared dataset YAML is missing split '{split}'")
            entries = data[split] if isinstance(data[split], list) else [data[split]]
            counts = Counter()
            for entry in entries:
                image_dir = Path(entry)
                if not image_dir.is_absolute():
                    image_dir = dataset_root / image_dir
                label_dir = image_dir.parent / "labels"
                if not label_dir.is_dir():
                    raise FileNotFoundError(f"Labels directory is missing for split '{split}': {label_dir}")
                for label_file in label_dir.glob("*.txt"):
                    for line in label_file.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            counts[int(line.split()[0])] += 1
            support[split] = counts
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"Invalid YOLO dataset YAML: {exc}")
    if "hammer" not in name_to_id or "rifle" not in name_to_id:
        parser.error("Dataset YAML must contain both hammer and rifle classes")
    for class_name in ("hammer", "rifle"):
        class_id = name_to_id[class_name]
        absent = [split for split, counts in support.items() if counts[class_id] == 0]
        if absent:
            parser.error(f"The source-disjoint dataset needs {class_name} instances in train, val, and test; absent from: {', '.join(absent)}")
    print("Hammer/rifle labelled-instance support:", {
        split: {name: support[split][name_to_id[name]] for name in ("hammer", "rifle")}
        for split in ("train", "val", "test")
    })
    print(f"Starting weights: {weights}")
    print(f"Dataset: {data_yaml}")
    print(f"New output directory: {output}")
    print(f"Classes: {names}")
    print("Training configuration:", {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                                     "patience": args.patience, "seed": args.seed, "device": args.device})
    if args.dry_run:
        print("Dry run complete; no model was loaded for training and no files were written.")
        return
    model = YOLO(str(weights))
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, seed=args.seed, device=args.device, optimizer="AdamW",
                cos_lr=True, warmup_epochs=3, close_mosaic=10, project=str(output.parent),
                name=output.name, exist_ok=False, plots=True, save=True)
    print(f"Candidate best weights: {output / 'weights' / 'best.pt'}")
    print("The input checkpoint was only read and remains unchanged.")


if __name__ == "__main__":
    main()
