"""
YOLO Detection Service - loads custom best.pt if present, else yolov8n.pt.
The heavyweight Ultralytics import is deferred until inference is requested so
the Flask API can start and serve authentication while the model initializes.
"""
import os
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from flask import current_app
from app.utils.image_utils import save_annotated_image


class YOLOService:
    _instance = None
    _model = None
    _model_path_used = None
    _load_error = None
    _load_lock = threading.RLock()
    _video_lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if YOLOService._model is None and YOLOService._load_error is None:
            with YOLOService._load_lock:
                if YOLOService._model is None and YOLOService._load_error is None:
                    self._load_model()

    def _load_model(self):
        from ultralytics import YOLO

        base = Path(current_app.root_path).parent  # backend/
        candidates = []
        configured = current_app.config.get("YOLO_MODEL_PATH")
        if configured:
            candidates.append(configured)
        candidates.extend([
            str(base / "models" / "best.pt"),
            r"C:\Users\admin\AI-Crime-Scene-Analyzer\backend\models\best.pt",
            r"C:\Users\admin\OneDrive\Documents\ai-crime-scene-evidence-analyzer\backend\models\best.pt",
        ])

        last_error = None
        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            if not os.path.isfile(path):
                continue
            try:
                print(f"Trying YOLO model: {path}")
                model = YOLO(path)
                YOLOService._model = model
                YOLOService._model_path_used = path
                print(f"✓ YOLO model loaded → {path}")
                try:
                    names = list(model.names.values())[:20]
                    print(f"  Classes: {names}")
                except Exception:
                    pass
                return
            except Exception as e:
                last_error = e
                print(f"  Failed {path}: {e}")

        YOLOService._load_error = (
            f"YOLO model not available. Last error: {last_error}. "
            "Place trained weights at backend/models/best.pt or allow internet so yolov8n.pt can download."
        )
        raise RuntimeError(YOLOService._load_error)

    def detect(self, image_path, conf_threshold=0.25):
        if YOLOService._model is None:
            self._load_model()
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        from app.services.yolo_accuracy_patch import enhanced_detect
        with YOLOService._video_lock:
            detections, annotated = enhanced_detect(
                YOLOService._model,
                image_path,
                imgsz=current_app.config["YOLO_IMAGE_SIZE"],
                base_conf=conf_threshold,
                category_fn=self._map_to_forensic_category,
            )
        annotated_name = f"annotated_{uuid.uuid4().hex[:12]}.jpg"
        annotated_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        annotated_path = os.path.join(annotated_dir, annotated_name)
        save_annotated_image(annotated, annotated_path)
        return detections, annotated_path

    def detect_frame(self, frame, conf_threshold=None):
        """Run the configured model on an OpenCV BGR frame and return its overlay."""
        if YOLOService._model is None:
            self._load_model()

        confidence = conf_threshold or current_app.config["YOLO_CONFIDENCE"]
        results = YOLOService._model(
            frame,
            conf=confidence,
            iou=0.45,
            imgsz=current_app.config["YOLO_IMAGE_SIZE"],
            device="cpu",
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = YOLOService._model.names.get(class_id, f"class_{class_id}")
                bbox = [round(float(value), 1) for value in box.xyxy[0].tolist()]
                detections.append({
                    "class_id": class_id,
                    "class": label,
                    "class_name": label,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": bbox,
                    "bbox_coords": dict(zip(("x1", "y1", "x2", "y2"), bbox)),
                    "category": self._map_to_forensic_category(label),
                })

        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        return detections, results[0].plot(line_width=2)

    @contextmanager
    def video_tracking_session(self):
        """Keep ByteTrack state isolated from concurrent image/video inference."""
        YOLOService._video_lock.acquire()
        try:
            self._reset_video_trackers()
            yield
        finally:
            self._reset_video_trackers()
            YOLOService._video_lock.release()

    @staticmethod
    def _reset_video_trackers():
        predictor = getattr(YOLOService._model, "predictor", None)
        trackers = getattr(predictor, "trackers", []) if predictor else []
        for tracker in trackers:
            tracker.reset()
        if predictor is not None and hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(trackers)

    def track_video_frame(self, frame, timestamp, frame_number):
        """Run ByteTrack on a sampled frame and draw only model-backed detections."""
        if YOLOService._model is None:
            self._load_model()

        import torch
        import cv2

        allowed_classes = set(current_app.config["VIDEO_ALLOWED_CLASSES"])
        model_classes = {
            name.lower() for name in YOLOService._model.names.values()
        }
        missing_classes = allowed_classes - model_classes
        if missing_classes:
            raise RuntimeError(
                "Loaded YOLO checkpoint is missing required video classes: "
                + ", ".join(sorted(missing_classes))
            )
        class_ids = [
            class_id
            for class_id, name in YOLOService._model.names.items()
            if name.lower() in allowed_classes
        ]
        model_classes = {name.lower() for name in YOLOService._model.names.values()}
        missing_classes = sorted(allowed_classes - model_classes)
        if missing_classes:
            raise RuntimeError(
                "The loaded YOLO checkpoint is missing required trained video classes: "
                + ", ".join(missing_classes)
            )

        device = 0 if torch.cuda.is_available() else "cpu"
        result = YOLOService._model.track(
            frame,
            persist=True,
            tracker=current_app.config["VIDEO_TRACKER"],
            classes=class_ids,
            conf=current_app.config["VIDEO_CONFIDENCE_THRESHOLD"],
            iou=0.45,
            imgsz=current_app.config["YOLO_IMAGE_SIZE"],
            device=device,
            verbose=False,
        )[0]

        detections = []
        annotated = frame.copy()
        boxes = result.boxes
        if boxes is not None:
            for index, box in enumerate(boxes):
                class_id = int(box.cls[0])
                label = YOLOService._model.names.get(class_id, f"class_{class_id}")
                if label.lower() not in allowed_classes:
                    continue
                track_id = int(boxes.id[index]) if boxes.id is not None else None
                confidence = round(float(box.conf[0]), 3)
                bbox = [round(float(value), 1) for value in box.xyxy[0].tolist()]
                detections.append({
                    "class": label,
                    "confidence": confidence,
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "bbox": bbox,
                    "track_id": track_id,
                    "category": self._map_to_forensic_category(label),
                })

                x1, y1, x2, y2 = [int(value) for value in bbox]
                color = (40, 210, 180)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                track_label = f" #{track_id}" if track_id is not None else ""
                text = f"{label.title()}{track_label} {confidence:.0%}"
                cv2.putText(annotated, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        minutes = int(timestamp // 60)
        seconds = timestamp - minutes * 60
        cv2.putText(
            annotated,
            f"Time: {minutes:02d}:{seconds:05.2f}",
            (12, max(24, annotated.shape[0] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return detections, annotated

    def _map_to_forensic_category(self, label: str) -> str:
        label = label.lower().strip()
        weapon_kw = (
            "gun", "handgun", "pistol", "revolver", "rifle", "shotgun", "firearm",
            "weapon", "knife", "blade", "dagger", "sword", "machete", "scissors",
            "axe", "bat", "crowbar", "hammer",
        )
        person_kw = ("person", "people", "man", "woman", "child", "suspect", "victim")
        vehicle_kw = ("car", "truck", "bus", "motorcycle", "bicycle", "van", "suv", "vehicle")
        if any(w in label for w in weapon_kw):
            return "Weapons"
        if any(p in label for p in person_kw):
            return "Persons"
        if any(v in label for v in vehicle_kw):
            return "Vehicles"
        return "Others"

    @property
    def model_info(self):
        return {
            "path": YOLOService._model_path_used,
            "classes": list(YOLOService._model.names.values()) if YOLOService._model else [],
            "error": YOLOService._load_error,
        }
