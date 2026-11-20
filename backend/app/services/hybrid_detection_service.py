"""YOLO detection with optional, name-mapped ResNet18 crop classification."""
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import current_app

from app.services.yolo_service import YOLOService


class HybridDetectionService:
    _model = None
    _classes = None
    _device = None
    _load_error = None
    _lock = threading.RLock()
    SUPPORTED_CLASSES = {"gun", "knife", "person", "pistol", "rifle"}

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return
        with cls._lock:
            if cls._model is not None:
                return
            path = Path(current_app.config["CNN_MODEL_PATH"])
            if not path.is_file():
                cls._load_error = f"CNN checkpoint not found: {path}"
                return
            import torch
            from torchvision import models

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)
            classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not classes or len(classes) != 5:
                raise ValueError("CNN checkpoint must include its five class names")
            if {str(name).strip().lower() for name in classes} != cls.SUPPORTED_CLASSES:
                raise ValueError("CNN checkpoint classes must be gun, knife, person, pistol, rifle")
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(state)
            model.to(device).eval()
            cls._model, cls._classes, cls._device, cls._load_error = model, list(classes), device, None

    @classmethod
    def _classify(cls, crop):
        import torch
        from torchvision import transforms

        size = current_app.config["CNN_IMAGE_SIZE"]
        mean = current_app.config["CNN_MEAN"]
        std = current_app.config["CNN_STD"]
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(cls._device)
        with torch.inference_mode():
            probabilities = torch.softmax(cls._model(tensor), dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        return cls._classes[index], float(probabilities[index].item())

    @classmethod
    def analyze(cls, image_path, conf_threshold=0.25, cnn_threshold=0.0, compare=True):
        image_read_started = time.perf_counter()
        yolo = YOLOService()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Uploaded file is not a readable image")
        current_app.logger.info(
            "image_analysis_timing stage=image_preprocessing duration_ms=%.1f",
            (time.perf_counter() - image_read_started) * 1000,
        )

        # YOLO-only pass preserves the trained model's detections and coordinates.
        inference_started = time.perf_counter()
        yolo_detections, yolo_annotated_path = yolo.detect(image_path, conf_threshold=conf_threshold)
        current_app.logger.info(
            "image_analysis_timing stage=yolo_inference duration_ms=%.1f detections=%d",
            (time.perf_counter() - inference_started) * 1000,
            len(yolo_detections),
        )

        cnn_load_started = time.perf_counter()
        cls.load_model()
        current_app.logger.info(
            "image_analysis_timing stage=cnn_model_initialization duration_ms=%.1f available=%s",
            (time.perf_counter() - cnn_load_started) * 1000,
            cls._model is not None,
        )
        classification_started = time.perf_counter()
        hybrid = []
        if cls._model is not None:
            yolo_names = {str(name).strip().lower(): int(class_id)
                          for class_id, name in yolo.model_info["class_id_names"]}
            class_to_id = {name: yolo_names[name] for name in cls.SUPPORTED_CLASSES if name in yolo_names}
            for detection in yolo_detections:
                item = dict(detection)
                name = str(detection.get("class_name", detection.get("class", ""))).strip().lower()
                if name not in cls.SUPPORTED_CLASSES:
                    item.update({"yolo_class": detection.get("class_name", detection.get("class")),
                                 "yolo_confidence": detection["confidence"], "classification": "unsupported"})
                    hybrid.append(item)
                    continue
                x1, y1, x2, y2 = detection["bbox"]
                h, w = image.shape[:2]
                crop = image[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
                item.update({"yolo_class": detection.get("class_name", detection.get("class")),
                             "yolo_class_id": detection.get("class_id"),
                             "yolo_confidence": detection["confidence"], "classification": "yolo_only"})
                if crop.size:
                    cnn_label, cnn_conf = cls._classify(crop)
                    item.update({"cnn_class": cnn_label, "cnn_class_id": cls._classes.index(cnn_label),
                                 "cnn_confidence": round(cnn_conf, 4),
                                 "mapped_yolo_class_id": class_to_id.get(cnn_label)})
                    if cnn_conf >= cnn_threshold:
                        item["class"] = item["class_name"] = cnn_label
                        item["confidence"] = round(cnn_conf, 4)
                        item["category"] = yolo._map_to_forensic_category(cnn_label)
                        item["classification"] = "hybrid"
                hybrid.append(item)
        else:
            hybrid = [dict(d, yolo_class=d.get("class_name", d.get("class")),
                           yolo_confidence=d["confidence"], classification="yolo_only") for d in yolo_detections]

        current_app.logger.info(
            "image_analysis_timing stage=crop_classification duration_ms=%.1f",
            (time.perf_counter() - classification_started) * 1000,
        )

        annotated = image.copy()
        for detection in hybrid:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            label = f'{detection.get("class_name", "object")} {detection["confidence"]:.0%}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 210, 180), 2)
            cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, (40, 210, 180), 2)
        output_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "annotated"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"hybrid_{uuid.uuid4().hex[:12]}.jpg"
        if not cv2.imwrite(str(output), annotated):
            raise OSError("Could not save annotated image")
        if yolo_annotated_path and os.path.abspath(yolo_annotated_path) != os.path.abspath(output):
            try:
                os.remove(yolo_annotated_path)
            except OSError:
                pass
        return {
            "detections": hybrid,
            "yolo_detections": yolo_detections if compare else None,
            "annotated_path": str(output),
            "cnn_available": cls._model is not None,
            "cnn_error": cls._load_error,
            "preprocessing": {"image_size": current_app.config["CNN_IMAGE_SIZE"],
                              "mean": current_app.config["CNN_MEAN"], "std": current_app.config["CNN_STD"],
                              "source": current_app.config["CNN_PREPROCESSING_SOURCE"]},
        }
