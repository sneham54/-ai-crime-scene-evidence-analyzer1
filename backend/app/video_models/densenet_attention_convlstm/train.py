"""Training CLI for verified video-level splits. Full training is opt-in."""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from app.video_models.densenet_attention_convlstm.classifier import (DenseNetAttentionConvLSTM,
    load_checkpoint, load_compatible_pretrained_weights)
from app.video_models.densenet_attention_convlstm.config import CLASSES, ModelConfig, PreprocessConfig
from app.video_models.densenet_attention_convlstm.metrics import classification_metrics
from app.video_models.densenet_attention_convlstm.preprocessing import iter_video_clips, probe_video


def read_manifest(dataset_root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("class_names") != list(("Assault", "Robbery", "Shooting", "Normal")):
        raise ValueError("Dataset manifest class labels do not match the existing project mapping")
    splits = manifest["splits"]
    seen = {}
    resolved = {}
    counts = {}
    for split_name in ("train", "validation", "test"):
        rows = splits.get(split_name) or []
        resolved[split_name] = []
        counts[split_name] = Counter()
        for row in rows:
            label, rel = row["class"], row["video"]
            group = row.get("source_group") or rel.casefold()
            if label not in {"Assault", "Robbery", "Shooting", "Normal"}:
                raise ValueError(f"Unknown class {label}")
            previous = seen.setdefault(group, split_name)
            if previous != split_name:
                raise ValueError(f"Source group leakage: {group} appears in {previous} and {split_name}")
            path = (Path(dataset_root) / rel).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved[split_name].append((path, label))
            counts[split_name][label] += 1
    return resolved, {key: dict(value) for key, value in counts.items()}


class VideoDataset(Dataset):
    def __init__(self, rows, preprocess, train=False, seed=42):
        self.rows, self.preprocess, self.train, self.seed = rows, preprocess, train, seed
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        path, label = self.rows[index]
        rng = random if self.train else random.Random(self.seed + index)
        metadata = probe_video(path)
        estimated_clips = max(1, (metadata["frame_count"] - self.preprocess.clip_length) // self.preprocess.clip_stride + 1)
        selected = rng.randrange(estimated_clips)
        clip = next((candidate for clip_index, candidate in enumerate(iter_video_clips(path, self.preprocess))
                     if clip_index == selected), None)
        if clip is None:
            clip = next(iter_video_clips(path, self.preprocess), None)
        if clip is None:
            raise ValueError(f"No clips decoded from {path}")
        tensor = clip.tensor
        if self.train and rng.random() < .5:
            tensor = tensor.flip(-1)
        return tensor, CLASSES.index(label), str(path)


def _run_epoch(model, loader, device, optimizer=None, class_weights=None, amp=False):
    training = optimizer is not None
    model.train(training)
    losses, labels, predictions = [], [], []
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and training and device.type == "cuda"))
    for clips, targets, _ in loader:
        clips, targets = clips.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                logits = model(clips)
                loss = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return {"loss": float(np.mean(losses)), "macro_f1": classification_metrics(labels, predictions)["macro_f1"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--pretrained", help="Optional local checkpoint with compatible DenseNet3D backbone tensors")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows, split_counts = read_manifest(args.dataset_root, args.manifest)
    print(json.dumps({"counts": split_counts, "leakage_check": "passed"}, indent=2))
    if args.dry_run:
        return
    out = Path(args.output_dir)
    if args.resume:
        if not out.is_dir():
            raise ValueError("Resume requires the existing run output directory")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty training output directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
    preprocess = PreprocessConfig(clip_length=args.clip_length, image_size=args.image_size)
    train_ds = VideoDataset(rows["train"], preprocess, True, args.seed)
    val_ds = VideoDataset(rows["validation"], preprocess, False, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()
    model = DenseNetAttentionConvLSTM(model_config).to(device)
    pretrained_report = load_compatible_pretrained_weights(model, args.pretrained) if args.pretrained else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    class_counts = split_counts["train"]
    weights = torch.tensor([sum(class_counts.values()) / (4 * class_counts.get(name, 1)) for name in CLASSES], dtype=torch.float32, device=device)
    start_epoch, best_f1, history = 0, -1.0, []
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_f1 = float(checkpoint.get("best_validation_macro_f1", -1))
        history = checkpoint.get("history", [])
    training_config = {"args": vars(args) | {"resume": args.resume}, "classes": CLASSES,
                       "model_config": model_config.to_dict(), "preprocess_config": preprocess.__dict__,
                       "split_counts": split_counts, "device": str(device),
                       "pretrained_weights": pretrained_report,
                       "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (out / "training_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs):
        amp = device.type == "cuda" and not args.no_amp
        train_metrics = _run_epoch(model, train_loader, device, optimizer, weights, amp)
        val_metrics = _run_epoch(model, val_loader, device, amp=amp)
        history.append({"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics})
        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "model_config": model_config.to_dict(), "preprocess_config": preprocess.__dict__,
                      "classes": ["Assault", "Shooting", "Robbery", "Normal"], "epoch": epoch + 1,
                      "best_validation_macro_f1": max(best_f1, val_metrics["macro_f1"]), "history": history}
        torch.save(checkpoint, out / "last_checkpoint.pth")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint, out / "best_checkpoint.pth")
        (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch + 1} train={train_metrics} validation={val_metrics} best_macro_f1={best_f1:.4f}")


if __name__ == "__main__": main()
