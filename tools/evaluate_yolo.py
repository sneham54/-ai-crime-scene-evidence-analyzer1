"""Evaluate YOLO weights on an explicitly selected dataset split."""
import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Optional new JSON path; existing files are rejected")
    args = parser.parse_args()
    weights, data_yaml = Path(args.weights).resolve(), Path(args.data_yaml).resolve()
    if not weights.is_file():
        parser.error(f"YOLO weights do not exist: {weights}")
    if not data_yaml.is_file():
        parser.error(f"Dataset YAML does not exist: {data_yaml}")
    try:
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if args.split not in config:
            parser.error(f"Dataset YAML has no '{args.split}' split")
        manifest_path = data_yaml.parent / "source_split_manifest.json"
        if not manifest_path.is_file():
            parser.error(f"Source split manifest is required beside data.yaml: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_groups = {name: {row["source_group"] for row in manifest["splits"][name]}
                        for name in ("train", "validation", "test")}
        overlaps = (("train", "validation"), ("train", "test"), ("validation", "test"))
        leaked = [f"{a}/{b}" for a, b in overlaps if split_groups[a] & split_groups[b]]
        if leaked:
            parser.error(f"Source split manifest has cross-split leakage: {', '.join(leaked)}")
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"Could not read dataset YAML: {exc}")
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz,
                        device=args.device, conf=0.001, iou=0.6, plots=True)
    names = model.names
    result = {"weights": str(weights), "data_yaml": str(data_yaml), "split": args.split,
              "source_split_manifest": str(manifest_path),
              "imgsz": args.imgsz, "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": {}}
    for class_id, name in names.items():
        if class_id < len(metrics.box.ap50):
            result["per_class"][str(name)] = {"ap50": float(metrics.box.ap50[class_id]),
                                               "ap50_95": float(metrics.box.ap[class_id])}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        if output.exists():
            parser.error(f"Evaluation output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
