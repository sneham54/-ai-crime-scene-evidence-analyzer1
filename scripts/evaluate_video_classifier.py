#!/usr/bin/env python3
"""Evaluate the whole-video CNN-LSTM checkpoint on a labeled test set.

Accepts either a training split manifest (recommended) or a directory whose
immediate subdirectories are Assault, Robbery, Shooting, and Normal.
"""
import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.video_classification_model import (  # noqa: E402
    CNNLSTM,
    VIDEO_CLASS_NAMES,
    decode_video_sequence,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_rows(test_dir, manifest):
    """Return ordered records with absolute paths and integer labels."""
    rows = []
    if manifest:
        manifest_path = Path(manifest).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("class_names") != list(VIDEO_CLASS_NAMES):
            raise ValueError("Manifest class order must be: " + ", ".join(VIDEO_CLASS_NAMES))
        records = payload.get("splits", {}).get("test")
        if not isinstance(records, list) or not records:
            raise ValueError("Manifest has no test split records")
        root = Path(test_dir).resolve() if test_dir else manifest_path.parent
        for item in records:
            label = item.get("class")
            if label not in VIDEO_CLASS_NAMES:
                raise ValueError(f"Unexpected manifest class: {label!r}")
            path = (root / item["video"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Test video listed in manifest is missing: {path}")
            rows.append({"path": path, "true_label": label, "label_id": VIDEO_CLASS_NAMES.index(label)})
    else:
        if not test_dir:
            raise ValueError("Provide --manifest or --test-dir")
        root = Path(test_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Test directory does not exist: {root}")
        for label in VIDEO_CLASS_NAMES:
            class_dir = root / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected class directory is missing: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    rows.append({"path": path, "true_label": label,
                                 "label_id": VIDEO_CLASS_NAMES.index(label)})
        if not rows:
            raise ValueError(f"No supported videos found in class folders under {root}")
    return rows


def load_model(checkpoint_path, device):
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    # weights_only prevents arbitrary pickle object construction from untrusted checkpoints.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "classes", "num_frames", "img_size"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint must contain model_state_dict, classes, num_frames, and img_size")
    if checkpoint["classes"] != list(VIDEO_CLASS_NAMES):
        raise ValueError("Checkpoint class ordering differs from " + ", ".join(VIDEO_CLASS_NAMES))
    frames, size = int(checkpoint["num_frames"]), int(checkpoint["img_size"])
    if frames != 16 or size != 112:
        raise ValueError(f"Expected the specified preprocessing (16 frames, 112 px); checkpoint has {frames} frames, {size} px")
    model = CNNLSTM(len(VIDEO_CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, frames, size


def calculate_metrics(true_ids, predicted_ids):
    count = len(VIDEO_CLASS_NAMES)
    matrix = [[0 for _ in range(count)] for _ in range(count)]
    for true_id, predicted_id in zip(true_ids, predicted_ids):
        matrix[true_id][predicted_id] += 1
    per_class = {}
    for index, name in enumerate(VIDEO_CLASS_NAMES):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(count)) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1_score": f1,
                           "support": sum(matrix[index])}
    total = len(true_ids)
    return {"overall_accuracy": sum(matrix[i][i] for i in range(count)) / total if total else 0.0,
            "per_class": per_class,
            "confusion_matrix": {"labels": list(VIDEO_CLASS_NAMES),
                                 "rows_true_columns_predicted": matrix}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(BACKEND_ROOT / "models" / "video_classifier.pth"))
    parser.add_argument("--test-dir", help="Directory with one subdirectory per target class")
    parser.add_argument("--manifest", help="Training video_split_manifest.json; its test split is used")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "evaluation_results_cnnlstm.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    try:
        rows = collect_rows(args.test_dir, args.manifest)
        model, frame_count, image_size = load_model(args.checkpoint, device)
        predictions = []
        true_ids, predicted_ids = [], []
        with torch.inference_mode():
            for row in rows:
                try:
                    frames = decode_video_sequence(row["path"], frame_count, image_size)
                    logits = model(torch.from_numpy(frames).unsqueeze(0).to(device))[0]
                    probabilities = torch.softmax(logits, dim=0)
                except Exception as exc:
                    raise RuntimeError(f"Failed to evaluate {row['path']}: {exc}") from exc
                pred_id = int(probabilities.argmax().item())
                predicted_ids.append(pred_id)
                true_ids.append(row["label_id"])
                predictions.append({
                    "video_name": row["path"].name,
                    "video_path": str(row["path"]),
                    "true_label": row["true_label"],
                    "predicted_label": VIDEO_CLASS_NAMES[pred_id],
                    "confidence_score": float(probabilities[pred_id].item()),
                    "logits": [float(value) for value in logits.cpu().tolist()],
                    "softmax_scores": {name: float(probabilities[i].item())
                                       for i, name in enumerate(VIDEO_CLASS_NAMES)},
                })
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    metrics = calculate_metrics(true_ids, predicted_ids)
    result = {"checkpoint": str(Path(args.checkpoint).resolve()), "sample_count": len(rows),
              "class_order": list(VIDEO_CLASS_NAMES), "preprocessing": {
                  "uniform_frames": frame_count, "resize": [image_size, image_size],
                  "color": "RGB", "normalization": "ImageNet mean/std"},
              "metrics": metrics, "predictions": predictions}
    print("Per-video predictions:")
    print(f"{'Video Name':48} {'True Label':12} {'Predicted Label':16} Confidence")
    for item in predictions:
        print(f"{item['video_name'][:48]:48} {item['true_label']:12} "
              f"{item['predicted_label']:16} {item['confidence_score']:.4f}")
    print("\nOverall Accuracy: {:.4f}".format(metrics["overall_accuracy"]))
    print("Per-Class Precision / Recall / F1:")
    for name, scores in metrics["per_class"].items():
        print(f"  {name}: {scores['precision']:.4f} / {scores['recall']:.4f} / {scores['f1_score']:.4f}")
    print("Confusion Matrix (rows=true, columns=predicted; order: " + ", ".join(VIDEO_CLASS_NAMES) + "):")
    for label, values in zip(VIDEO_CLASS_NAMES, metrics["confusion_matrix"]["rows_true_columns_predicted"]):
        print(f"  {label:10} {values}")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved evaluation report: {output_path}")


if __name__ == "__main__":
    main()
