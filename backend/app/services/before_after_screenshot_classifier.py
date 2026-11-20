"""
Experimental BEFORE/AFTER screenshot classifier.

Loads the ResNet18 checkpoint trained on the user's 337 screenshots.
It classifies each uploaded image independently as BEFORE or AFTER.
It does NOT classify scene-change events or localize changed objects.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path


class ScreenshotClassifierUnavailableError(RuntimeError):
    """The screenshot classifier checkpoint or dependencies are unavailable."""


class BeforeAfterScreenshotClassifier:
    _instance = None
    _lock = threading.Lock()

    CLASS_NAMES = ("BEFORE", "AFTER")
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(self, checkpoint_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[3]
        default_path = (
            project_root
            / "models"
            / "scene_change_screenshot_experiment"
            / "best_before_after_classifier.pt"
        )
        configured_path = checkpoint_path or os.getenv(
            "SCENE_CHANGE_SCREENSHOT_MODEL_PATH", str(default_path)
        )
        self.checkpoint_path = Path(configured_path).expanduser().resolve()

        if not self.checkpoint_path.is_file():
            raise ScreenshotClassifierUnavailableError(
                f"BEFORE/AFTER screenshot classifier checkpoint was not found: "
                f"{self.checkpoint_path}"
            )

        # Import ML dependencies lazily so a missing optional dependency
        # cannot prevent the rest of the Flask application from starting.
        try:
            import torch
            from PIL import Image
            from torch import nn
            from torchvision import models, transforms
        except Exception as error:
            raise ScreenshotClassifierUnavailableError(
                f"Screenshot-classifier dependencies are unavailable: {error}"
            ) from error

        self._torch = torch
        self._Image = Image

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise ScreenshotClassifierUnavailableError(
                f"Could not load screenshot classifier checkpoint: {error}"
            ) from error

        if not isinstance(checkpoint, dict):
            raise ScreenshotClassifierUnavailableError(
                "Screenshot classifier checkpoint has an unsupported format."
            )

        class_names = checkpoint.get("class_names")
        if tuple(class_names or ()) != self.CLASS_NAMES:
            raise ScreenshotClassifierUnavailableError(
                "Checkpoint classes must be ['BEFORE', 'AFTER'] in that order."
            )

        try:
            self.image_size = int(checkpoint["image_size"])
            state_dict = checkpoint["model_state_dict"]
            self.training_validation_accuracy = float(
                checkpoint.get("validation_accuracy", float("nan"))
            )
            self.best_epoch = int(checkpoint.get("epoch", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise ScreenshotClassifierUnavailableError(
                "Checkpoint is missing valid model metadata."
            ) from error

        if self.image_size <= 0:
            raise ScreenshotClassifierUnavailableError(
                "Checkpoint image_size must be positive."
            )

        self.device = torch.device("cpu")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.CLASS_NAMES))

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, TypeError) as error:
            raise ScreenshotClassifierUnavailableError(
                f"Checkpoint architecture does not match ResNet18: {error}"
            ) from error

        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD),
        ])

    @classmethod
    def get(cls, checkpoint_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[3]
        default_path = (
            project_root
            / "models"
            / "scene_change_screenshot_experiment"
            / "best_before_after_classifier.pt"
        )
        key = str(
            Path(
                checkpoint_path
                or os.getenv("SCENE_CHANGE_SCREENSHOT_MODEL_PATH", str(default_path))
            ).expanduser().resolve()
        )
        with cls._lock:
            if cls._instance is None or str(cls._instance.checkpoint_path) != key:
                cls._instance = cls(key)
            return cls._instance

    def predict_path(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        try:
            with self._Image.open(path) as source:
                image = source.convert("RGB")
                tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as error:
            raise ValueError(
                f"Could not read image for screenshot classification: {path}"
            ) from error

        with self._torch.inference_mode():
            logits = self.model(tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0]

        predicted_index = int(self._torch.argmax(probabilities).item())
        predicted_class = self.CLASS_NAMES[predicted_index]
        score = float(probabilities[predicted_index].item())

        return {
            "predicted_class": predicted_class,
            "softmax_score": round(score, 4),
            "class_scores": {
                name: round(float(probabilities[index].item()), 4)
                for index, name in enumerate(self.CLASS_NAMES)
            },
            "score_note": (
                "Softmax score is not independently calibrated accuracy "
                "or evidence that a scene change occurred."
            ),
        }
