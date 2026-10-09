import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.services.video_classification_model import (
    CNNLSTMClassifier,
    IMAGE_MEAN,
    IMAGE_STD,
    VIDEO_CLASS_NAMES,
    decode_video_sequence,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FOLDER_CLASSES = {
    "assault": "Assault",
    "robbery": "Robbery",
    "shooting": "Shooting",
    "normal": "Normal",
    "normalvideos": "Normal",
}


class LabeledVideos(Dataset):
    def __init__(self, items, sequence_length, image_size):
        self.items = items
        self.sequence_length = sequence_length
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        video_path, label = self.items[index]
        sequence = decode_video_sequence(video_path, self.sequence_length, self.image_size)
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)


def collect_video_level_splits(data_dir, seed):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")
    videos_by_class = defaultdict(list)
    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        class_name = FOLDER_CLASSES.get(folder.name.casefold())
        if class_name is None:
            continue
        videos_by_class[class_name].extend(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )

    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    label_ids = {name: index for index, name in enumerate(VIDEO_CLASS_NAMES)}
    for class_name in VIDEO_CLASS_NAMES:
        videos = sorted(videos_by_class[class_name])
        if len(videos) < 3:
            raise ValueError(
                f"Class '{class_name}' needs at least 3 videos for video-level train/validation/test splits; found {len(videos)}."
            )
        rng.shuffle(videos)
        validation_count = max(1, round(len(videos) * 0.15))
        test_count = max(1, round(len(videos) * 0.15))
        while len(videos) - validation_count - test_count < 1:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        train_end = len(videos) - validation_count - test_count
        partitions = {
            "train": videos[:train_end],
            "validation": videos[train_end:len(videos) - test_count],
            "test": videos[len(videos) - test_count:],
        }
        for split_name, paths in partitions.items():
            splits[split_name].extend((path, label_ids[class_name]) for path in paths)

    for split in splits.values():
        rng.shuffle(split)
    return splits


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    sample_count = 0
    expected_labels = []
    predicted_labels = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            logits = model(videos)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            predicted = logits.argmax(dim=1)
            correct += int((predicted == labels).sum().item())
            sample_count += labels.size(0)
            expected_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted.cpu().tolist())
    return (
        total_loss / max(1, sample_count),
        correct / max(1, sample_count),
        expected_labels,
        predicted_labels,
    )


def calculate_metrics(expected, predicted, class_names):
    confusion = [[0 for _ in class_names] for _ in class_names]
    for expected_class, predicted_class in zip(expected, predicted):
        confusion[expected_class][predicted_class] += 1

    report = {}
    f1_values = []
    supports = []
    for class_id, class_name in enumerate(class_names):
        true_positive = confusion[class_id][class_id]
        false_positive = sum(confusion[row][class_id] for row in range(len(class_names))) - true_positive
        false_negative = sum(confusion[class_id]) - true_positive
        support = sum(confusion[class_id])
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        supports.append(support)

    total = sum(supports)
    accuracy = sum(confusion[index][index] for index in range(len(class_names))) / max(1, total)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / max(1, len(f1_values)),
        "weighted_f1": sum(f1 * support for f1, support in zip(f1_values, supports)) / max(1, total),
        "per_class": report,
        "confusion_matrix": confusion,
    }


def print_split_summary(splits):
    print("Dataset summary")
    for class_id, class_name in enumerate(VIDEO_CLASS_NAMES):
        class_count = sum(label == class_id for _, label in splits["train"] + splits["validation"] + splits["test"])
        print(f"{class_name}: {class_count} videos")
    for split_name in ("train", "validation", "test"):
        print(f"{split_name.title()}:")
        for class_id, class_name in enumerate(VIDEO_CLASS_NAMES):
            count = sum(label == class_id for _, label in splits[split_name])
            print(f"  {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Train a video-level Assault/Robbery/Shooting/Normal CNN-LSTM classifier.")
    parser.add_argument("--data-dir", required=True, help="Folder containing Assault, Robbery, Shooting, and NormalVideos subfolders.")
    parser.add_argument("--output", default="models/video_classifier.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = collect_video_level_splits(args.data_dir, args.seed)
    print_split_summary(splits)

    train_counts = [sum(label == index for _, label in splits["train"]) for index in range(len(VIDEO_CLASS_NAMES))]
    total_train = sum(train_counts)
    class_weights = torch.tensor([
        total_train / (len(VIDEO_CLASS_NAMES) * count) for count in train_counts
    ], dtype=torch.float32, device=device)

    datasets = {
        name: LabeledVideos(items, args.sequence_length, args.image_size)
        for name, items in splits.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, dataset in datasets.items()
    }

    model = CNNLSTMClassifier(
        num_classes=len(VIDEO_CLASS_NAMES),
        pretrained=args.pretrained,
        freeze_backbone=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    best_validation_f1 = -1.0
    epochs_without_improvement = 0
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_names_path = output_path.with_name("class_names.json")
    preprocessing_path = output_path.with_name("video_preprocessing.json")
    manifest_path = output_path.with_name("video_split_manifest.json")
    class_names_path.write_text(json.dumps(list(VIDEO_CLASS_NAMES), indent=2), encoding="utf-8")
    preprocessing_path.write_text(json.dumps({
        "sequence_length": args.sequence_length,
        "image_size": args.image_size,
        "mean": IMAGE_MEAN,
        "std": IMAGE_STD,
        "color_order": "RGB",
        "normalization": "ImageNet mean/std",
        "sampling": "uniform over each whole video",
    }, indent=2), encoding="utf-8")
    dataset_root = Path(args.data_dir).resolve()
    manifest_path.write_text(json.dumps({
        split_name: [
            {"video": str(video_path.resolve().relative_to(dataset_root)), "label": VIDEO_CLASS_NAMES[label]}
            for video_path, label in split_items
        ]
        for split_name, split_items in splits.items()
    }, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(model, loaders["train"], loss_function, device, optimizer)
        validation_loss, validation_accuracy, validation_expected, validation_predicted = run_epoch(
            model, loaders["validation"], loss_function, device
        )
        validation_metrics = calculate_metrics(validation_expected, validation_predicted, VIDEO_CLASS_NAMES)
        print(
            f"Epoch {epoch}/{args.epochs}\n"
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.3f}\n"
            f"Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.3f} "
            f"| Validation Macro F1: {validation_metrics['macro_f1']:.3f}"
        )
        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": VIDEO_CLASS_NAMES,
                "sequence_length": args.sequence_length,
                "image_size": args.image_size,
                "hidden_size": 96,
                "freeze_backbone": True,
                "validation_metrics": validation_metrics,
                "split_counts": {name: len(items) for name, items in splits.items()},
            }, output_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: validation Macro F1 did not improve for {args.patience} epochs.")
                break

    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, _, test_expected, test_predicted = run_epoch(model, loaders["test"], loss_function, device)
    test_metrics = calculate_metrics(test_expected, test_predicted, VIDEO_CLASS_NAMES)
    performance_warning = None
    if test_metrics["macro_f1"] < 0.60:
        performance_warning = "The trained model has limited held-out performance. Predictions should be treated as experimental."
    checkpoint["test_metrics"] = test_metrics
    checkpoint["performance_warning"] = performance_warning
    torch.save(checkpoint, output_path)
    evaluation_path = output_path.with_name("video_classifier_evaluation.json")
    evaluation_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.3f}")
    print("Classification Report:")
    for class_name, metrics in test_metrics["per_class"].items():
        print(
            f"{class_name:10} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} support={metrics['support']}"
        )
    print("Confusion Matrix (rows=true, columns=predicted):")
    print(test_metrics["confusion_matrix"])
    if performance_warning:
        print(f"Warning: {performance_warning}")
    print(f"Saved classifier: {output_path}")
    print(f"Saved class names: {class_names_path}")
    print(f"Saved preprocessing config: {preprocessing_path}")
    print(f"Saved video-level split manifest: {manifest_path}")
    print(f"Best validation Macro F1: {best_validation_f1:.3f}")
    print("Splits are by source video; frames from a video never cross splits.")


if __name__ == "__main__":
    main()
