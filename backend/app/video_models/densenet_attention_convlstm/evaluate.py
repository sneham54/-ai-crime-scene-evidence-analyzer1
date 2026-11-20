"""Held-out four-class evaluation and optional same-video legacy baseline comparison."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from app.video_models.densenet_attention_convlstm.classifier import load_checkpoint
from app.video_models.densenet_attention_convlstm.config import CLASSES, PreprocessConfig
from app.video_models.densenet_attention_convlstm.metrics import classification_metrics
from app.video_models.densenet_attention_convlstm.preprocessing import iter_video_clips
from app.video_models.densenet_attention_convlstm.train import read_manifest
from app.services.video_classification_model import VideoClassificationModel


def evaluate(model, rows, device, preprocess, baseline=None):
    truth, predicted, baseline_predicted = [], [], []
    elapsed, baseline_elapsed = [], []
    for path, label in rows:
        probability_sum = None
        clip_count = 0
        for clip in iter_video_clips(path, preprocess):
            start = time.perf_counter()
            with torch.inference_mode():
                probabilities = torch.softmax(model(clip.tensor.unsqueeze(0).to(device)), 1)[0].cpu().numpy()
            probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
            clip_count += 1
            elapsed.append(time.perf_counter() - start)
        if clip_count == 0:
            raise ValueError(f"No clips decoded from held-out video: {path}")
        truth.append(CLASSES.index(label))
        predicted.append(int((probability_sum / clip_count).argmax()))
        if baseline:
            start = time.perf_counter()
            result = baseline.predict_path(path)
            baseline_elapsed.append(time.perf_counter() - start)
            baseline_predicted.append(CLASSES.index(result["top_class"]))
    metrics = classification_metrics(truth, predicted)
    metrics.update({"inference_seconds_per_clip": float(np.mean(elapsed)) if elapsed else None,
               "evaluated_videos": len(rows), "class_order": list(CLASSES)}
                  )
    if baseline:
        legacy = classification_metrics(truth, baseline_predicted)
        legacy.update({"model": "existing ResNet18 + LSTM",
                       "inference_seconds_per_video": float(np.mean(baseline_elapsed)) if baseline_elapsed else None})
        metrics["baseline_same_test_videos"] = legacy
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-checkpoint"); parser.add_argument("--baseline-classes")
    args = parser.parse_args()
    rows, counts = read_manifest(args.dataset_root, args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    raw = checkpoint.get("preprocess_config", {})
    raw["mean"] = tuple(raw.get("mean", (0.485, 0.456, 0.406))); raw["std"] = tuple(raw.get("std", (0.229, 0.224, 0.225)))
    preprocess = PreprocessConfig(**raw)
    baseline = VideoClassificationModel(args.baseline_checkpoint, args.baseline_classes) if args.baseline_checkpoint else None
    result = evaluate(model, rows["test"], device, preprocess, baseline)
    result.update({"checkpoint": str(Path(args.checkpoint).resolve()), "manifest": str(Path(args.manifest).resolve()),
                   "test_split_counts": counts["test"], "device": str(device),
                   "model_parameters": sum(p.numel() for p in model.parameters()),
                   "method_note": "Video-level labels only; metrics evaluate video classes, not event boundary accuracy."})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
