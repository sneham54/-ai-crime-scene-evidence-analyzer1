"""Evaluate a classifier checkpoint against a saved source-group manifest."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.services.video_classification_model import CNNLSTM, VIDEO_CLASS_NAMES
from train_video_classifier import (LabeledVideos, classification_metrics,
                                    collect_grouped_videos, make_group_splits,
                                    write_split_manifest)


def read_evaluation_rows(dataset_root, manifest_path, split):
    root = Path(dataset_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("class_names") != list(VIDEO_CLASS_NAMES):
        raise ValueError("Manifest class ordering does not match the inference class ordering")
    records = manifest.get("splits", {}).get(split)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records for split '{split}'")
    rows = []
    all_splits = manifest.get("splits", {})
    selected_groups = {item.get("source_group") for item in records}
    for other_split, other_records in all_splits.items():
        if other_split != split:
            overlap = selected_groups & {item.get("source_group") for item in other_records}
            if overlap:
                raise ValueError(f"Source groups overlap between {split} and {other_split}: {sorted(overlap)[:3]}")
    for item in records:
        if item.get("class") not in VIDEO_CLASS_NAMES:
            raise ValueError(f"Unexpected class in manifest: {item.get('class')}")
        path = (root / item["video"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest video is missing: {path}")
        source = item.get("source_group")
        if not source:
            raise ValueError(f"Manifest row has no source_group: {item}")
        rows.append({"path": path, "label": VIDEO_CLASS_NAMES.index(item["class"]), "source_group": source})
    return rows


def load_model(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "classes", "num_frames", "img_size"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing inference keys: {', '.join(sorted(missing))}")
    if checkpoint["classes"] != list(VIDEO_CLASS_NAMES):
        raise ValueError("Checkpoint classes do not match required order: " + ", ".join(VIDEO_CLASS_NAMES))
    model = CNNLSTM(len(VIDEO_CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, int(checkpoint["num_frames"]), int(checkpoint["img_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest", help="Existing split manifest; generated reproducibly when omitted")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", help="Optional JSON result path; does not overwrite an existing file")
    parser.add_argument("--output-dir", help="Directory for JSON and CSV reports; defaults beside the checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        model, frames, size = load_model(args.checkpoint)
        output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.checkpoint).resolve().parent / "evaluation"
        if args.manifest:
            manifest_path = Path(args.manifest)
        else:
            manifest_path = output_dir / "video_split_manifest.json"
            if not manifest_path.exists():
                excluded_conflicts = []
                source_rows = collect_grouped_videos(args.dataset_dir, hash_duplicates=True,
                                                     exclude_cross_label_duplicates=True,
                                                     excluded_conflicts=excluded_conflicts)
                splits = make_group_splits(source_rows, seed=args.seed)
                output_dir.mkdir(parents=True, exist_ok=True)
                write_split_manifest(splits, args.dataset_dir, manifest_path, args.seed, excluded_conflicts)
        rows = read_evaluation_rows(args.dataset_dir, manifest_path, args.split)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    loader = DataLoader(LabeledVideos(rows, frames, size), batch_size=2, shuffle=False, num_workers=0)
    expected, predicted, probabilities = [], [], []
    with torch.inference_mode():
        for videos, labels in loader:
            logits = model(videos)
            expected.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).tolist())
            probabilities.extend(torch.softmax(logits, dim=1).tolist())
    metrics = classification_metrics(expected, predicted)
    prediction_rows = []
    for row, truth, guess, probs in zip(rows, expected, predicted, probabilities):
        prediction_rows.append({"video": str(row["path"]), "source_group": row["source_group"],
                                "true_class": VIDEO_CLASS_NAMES[truth], "predicted_class": VIDEO_CLASS_NAMES[guess],
                                "confidence": max(probs),
                                **{f"probability_{name}": probs[index] for index, name in enumerate(VIDEO_CLASS_NAMES)}})
    result = {"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()),
              "manifest": str(manifest_path.resolve()), "seed": args.seed,
              "samples": len(rows), "metrics": metrics, "predictions": prediction_rows}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = Path(args.output) if args.output else output_dir / f"{args.split}_evaluation.json"
    csv_output = output_dir / f"{args.split}_predictions.csv"
    metrics_output = output_dir / f"{args.split}_metrics.csv"
    confusion_output = output_dir / f"{args.split}_confusion_matrix.csv"
    targets = [json_output, csv_output, metrics_output, confusion_output]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        parser.error("Evaluation output already exists; choose a new output directory/path: " + ", ".join(existing))
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(rendered + "\n", encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with metrics_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name, values in metrics["per_class"].items():
            writer.writerow([name, values["precision"], values["recall"], values["f1"], values["support"]])
        writer.writerow(["overall", "", "", "", len(rows)])
        writer.writerow(["accuracy", metrics["accuracy"], "", "", len(rows)])
        writer.writerow(["macro_f1", metrics["macro_f1"], "", "", len(rows)])
        writer.writerow(["weighted_f1", metrics["weighted_f1"], "", "", len(rows)])
    with confusion_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *VIDEO_CLASS_NAMES])
        for name, values in zip(VIDEO_CLASS_NAMES, metrics["confusion_matrix"]):
            writer.writerow([name, *values])


if __name__ == "__main__":
    main()
