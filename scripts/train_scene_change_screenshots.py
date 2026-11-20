from pathlib import Path
import random
import csv
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

PROJECT = Path(r"C:\Users\admin\Downloads\ai-crime-scene-evidence-analyzer_v2\ai-crime-scene-evidence-analyzer")
DATA_ROOT = PROJECT / "datasets" / "scene_change"

BEFORE_DIR = DATA_ROOT / "before"
AFTER_DIR = DATA_ROOT / "after"

OUTPUT_DIR = PROJECT / "models" / "scene_change_screenshot_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = OUTPUT_DIR / "best_before_after_classifier.pt"
HISTORY = OUTPUT_DIR / "training_history.csv"

EPOCHS = 15
BATCH_SIZE = 16
IMAGE_SIZE = 224
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if not BEFORE_DIR.exists() or not AFTER_DIR.exists():
    raise FileNotFoundError(
        f"Expected folders:\n{BEFORE_DIR}\n{AFTER_DIR}"
    )

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

def get_images(folder):
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())

before_images = get_images(BEFORE_DIR)
after_images = get_images(AFTER_DIR)

print("Before images:", len(before_images))
print("After images:", len(after_images))

if len(before_images) < 2 or len(after_images) < 2:
    raise RuntimeError("Not enough images to create training and validation splits.")

# Label 0 = BEFORE, Label 1 = AFTER.
random.shuffle(before_images)
random.shuffle(after_images)

def split_images(images):
    n_val = max(1, round(len(images) * 0.20))
    return images[n_val:], images[:n_val]

before_train, before_val = split_images(before_images)
after_train, after_val = split_images(after_images)

train_records = (
    [(p, 0) for p in before_train] +
    [(p, 1) for p in after_train]
)

val_records = (
    [(p, 0) for p in before_val] +
    [(p, 1) for p in after_val]
)

random.shuffle(train_records)
random.shuffle(val_records)

print("\nTRAIN images:", len(train_records))
print("VALIDATION images:", len(val_records))

class ScreenshotDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        try:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            return image, label
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_loader = DataLoader(
    ScreenshotDataset(train_records, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    ScreenshotDataset(val_records, val_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load ImageNet pretrained ResNet18; fall back to random initialization
# if pretrained weights cannot be downloaded.
try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    print("\nLoaded ImageNet pretrained ResNet18.")
except Exception as exc:
    print("\nPretrained weights unavailable:", exc)
    print("Using randomly initialized ResNet18.")
    model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Training device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0005
)

best_accuracy = -1.0
best_state = None
history = []

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    train_loss /= max(train_total, 1)
    val_loss /= max(val_total, 1)
    train_accuracy = train_correct / max(train_total, 1)
    val_accuracy = val_correct / max(val_total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}",
        flush=True
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "class_names": ["BEFORE", "AFTER"],
            "image_size": IMAGE_SIZE,
            "validation_accuracy": best_accuracy,
            "epoch": epoch + 1
        }, BEST_MODEL)

    with open(HISTORY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)

print("\n========== TRAINING FINISHED ==========")
print("Best validation accuracy:", round(best_accuracy, 4))
print("Best checkpoint:", BEST_MODEL)
print("Training history:", HISTORY)
print("=======================================")
print("This experiment classifies screenshots as BEFORE or AFTER.")
print("It does not train paired-frame scene-change localization.")
