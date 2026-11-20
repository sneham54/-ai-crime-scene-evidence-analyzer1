"""
YOLO Detection Service - loads the project's nine-class backend/models/best.pt.
The heavyweight Ultralytics import is deferred until inference is requested so
the Flask API can start and serve authentication while the model initializes.
"""
import os
import uuid
import threading
from contextlib import contextmanager
from pathlib import Path
from flask import current_app
from app.config import Config as config
from app.utils.image_utils import save_annotated_image


class YOLOService:
    # Production detector class map. Embedded model names take precedence.
    DETECTOR_CLASS_NAMES = (
        "gun", "hammer", "helmet", "knife", "mask", "person", "pistol", "rifle", "weapon",
    )
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
        # Video/image evidence must use the project's trained nine-class model.
        # Do not silently fall through to another experiment or downloadable model.
        candidates = [str(base / "models" / "best.pt")]

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
                names = model.names
                if not isinstance(names, dict) or sorted(names) != list(range(9)):
                    raise RuntimeError("backend/models/best.pt must contain the existing nine classes with IDs 0 through 8")
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
            f"Nine-class YOLO model at backend/models/best.pt is not available or could not be loaded. Last error: {last_error}."
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

        import torch

        confidence = conf_threshold if conf_threshold is not None else current_app.config["YOLO_CONFIDENCE"]
        device = 0 if torch.cuda.is_available() else "cpu"
        with YOLOService._video_lock:
            results = YOLOService._model(
                frame,
                conf=confidence,
                iou=0.45,
                imgsz=current_app.config["YOLO_IMAGE_SIZE"],
                device=device,
                verbose=False,
            )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                # The model's global cutoff is applied during inference; the
                # per-class cutoff is applied once after full-frame and ROI
                # observations have been merged below.
                if float(box.conf[0]) < confidence:
                    continue
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
                    "source": "full-frame",
                })

        # Keep the full-frame pass as the primary result; ROI inference is opt-in.
        if current_app.config.get("VIDEO_PERSON_ROI_ENABLED", config.VIDEO_PERSON_ROI_ENABLED) is True:
            detections.extend(self._person_roi_detections(frame, detections))
        for detection in detections:
            detection.setdefault("source", "full-frame")
        detections = [
            detection for detection in detections
            if float(detection.get("confidence", 0.0)) >= _class_confidence_threshold(detection.get("class_id", -1))
        ]
        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        return detections, results[0].plot(line_width=2)

    def predict_frame_raw(self, frame, conf_threshold=0.001):
        """Return full-frame model predictions before class-specific filtering.

        This diagnostic method is intended for the regression helper. It uses
        the already loaded production checkpoint and does not include ROI
        detections, so raw and final results can be compared directly.
        """
        if YOLOService._model is None:
            self._load_model()
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        with YOLOService._video_lock:
            results = YOLOService._model(
                frame,
                conf=conf_threshold,
                iou=0.45,
                imgsz=current_app.config["YOLO_IMAGE_SIZE"],
                device=device,
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
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": bbox,
                    "source": "full-frame",
                })
        return detections

    def _person_roi_detections(self, frame, detections):
        """Supplement full-frame results with high-resolution lower-body ROIs.

        ROI results are additional visual observations and have no tracker identity.
        They are deduplicated against full-frame boxes to avoid inflating counts.
        """
        import cv2
        import torch
        height, width = frame.shape[:2]
        person_boxes = [item.get("bbox") for item in detections if item.get("class_id") == 5]
        results = []
        maximum = max(0, int(current_app.config.get("VIDEO_PERSON_ROI_MAX", 2)))
        for person_box in sorted(person_boxes, key=lambda box: max(0, box[2]-box[0]) * max(0, box[3]-box[1]), reverse=True)[:maximum]:
            x1, y1, x2, y2 = [int(value) for value in person_box]
            box_w, box_h = x2 - x1, y2 - y1
            if (box_w < 40 or box_h < 60 or x1 <= 0 or y1 <= 0
                    or x2 >= width or y2 >= height):
                # Reject tiny people and person boxes clipped by the image edge;
                # partial bodies create unreliable crops that can hallucinate weapons.
                continue
            pad_x = int(box_w * 0.12)
            rx1, rx2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
            # Hands/waist are usually below shoulders; include the full torso/arms.
            ry1, ry2 = max(0, y1 + int(box_h * 0.2)), min(height, y2)
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
                continue
            # The expanded ROI itself must not be clipped by any image border.
            if rx1 == 0 or ry1 == 0 or rx2 == width or ry2 == height:
                continue
            crop_max_dimension = max(crop.shape[:2])
            # Oversized crops are skipped rather than rescaled into a larger model input.
            if crop_max_dimension > 640:
                continue
            roi_conf = float(current_app.config.get("VIDEO_ROI_CONFIDENCE", 0.35))
            roi_limit = min(640, int(current_app.config.get("VIDEO_ROI_IMAGE_SIZE", 640)), crop_max_dimension)
            # Align downward to Ultralytics' common stride so its size normalization
            # cannot round a small crop upward beyond the source dimensions.
            roi_imgsz = max(32, (roi_limit // 32) * 32)
            # Defensive check: never allow the inference target to exceed 2x crop size.
            if roi_imgsz > crop_max_dimension * 2:
                continue
            # img size is no larger than the source crop, so Ultralytics will not upscale it.
            roi_results = YOLOService._model(crop, conf=roi_conf,
                                             imgsz=roi_imgsz,
                                             device=(0 if torch.cuda.is_available() else "cpu"), verbose=False)
            for result in roi_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id not in (0, 1, 3, 6, 7, 8):
                        continue
                    if float(box.conf[0]) < max(roi_conf, _class_confidence_threshold(class_id)):
                        continue
                    bbox = [round(float(box.xyxy[0][0]) + rx1, 1), round(float(box.xyxy[0][1]) + ry1, 1),
                            round(float(box.xyxy[0][2]) + rx1, 1), round(float(box.xyxy[0][3]) + ry1, 1)]
                    label = YOLOService._model.names.get(class_id, f"class_{class_id}")
                    # Ignore an ROI result already represented by a full-frame box.
                    if any(item.get("class_id") == class_id and _box_iou(item.get("bbox"), bbox) >= .5 for item in detections + results):
                        continue
                    results.append({"class_id": class_id, "class": label, "class_name": label,
                                    "confidence": round(float(box.conf[0]), 3), "bbox": bbox,
                                    "bbox_coords": dict(zip(("x1", "y1", "x2", "y2"), bbox)),
                                    "category": self._map_to_forensic_category(label),
                                    "source": "roi-crop", "detection_source": "person_roi", "track_id": None,
                                    "tracking_status": "untracked"})
        return results

    @contextmanager
    def video_tracking_session(self):
        """Keep ByteTrack state isolated from concurrent image/video inference."""
        YOLOService._video_lock.acquire()
        try:
            try:
                self._reset_video_trackers()
            except Exception as error:
                current_app.logger.warning("Could not reset ByteTrack state before video: %s", error)
            yield
        finally:
            try:
                self._reset_video_trackers()
            except Exception as error:
                current_app.logger.warning("Could not reset ByteTrack state after video: %s", error)
            finally:
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

        # Use every class from the configured checkpoint. Filtering by names can
        # silently discard trained classes or change the nine-class mapping.

        device = 0 if torch.cuda.is_available() else "cpu"
        global_confidence = float(current_app.config.get(
            "VIDEO_CONFIDENCE_THRESHOLD", current_app.config.get("YOLO_CONFIDENCE", 0.25)
        ))
        result = YOLOService._model.track(
            frame,
            persist=True,
            tracker=current_app.config.get("VIDEO_TRACKER", "bytetrack.yaml"),
            conf=global_confidence,
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
                # Apply only the global inference cutoff here. The per-class
                # threshold is applied once after optional ROI results merge.
                if float(box.conf[0]) < global_confidence:
                    continue
                label = YOLOService._model.names.get(class_id, f"class_{class_id}")
                track_id = int(boxes.id[index]) if boxes.id is not None else None
                confidence = round(float(box.conf[0]), 3)
                bbox = [round(float(value), 1) for value in box.xyxy[0].tolist()]
                detections.append({
                    "class_id": class_id,
                    "class": label,
                    "class_name": label,
                    "confidence": confidence,
                    "timestamp_sec": timestamp,
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "bbox": bbox,
                    "track_id": track_id,
                    "tracking_status": "tracked" if track_id is not None else "untracked",
                    "category": self._map_to_forensic_category(label),
                    "source": "full-frame",
                })

                x1, y1, x2, y2 = [int(value) for value in bbox]
                color = (40, 210, 180)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                track_label = f" #{track_id}" if track_id is not None else ""
                text = f"{label.title()}{track_label} {confidence:.0%}"
                cv2.putText(annotated, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if timestamp is None:
            time_label = "Time unavailable"
        else:
            minutes = int(timestamp // 60)
            seconds = timestamp - minutes * 60
            time_label = f"Time: {minutes:02d}:{seconds:05.2f}"
        cv2.putText(
            annotated,
            time_label,
            (12, max(24, annotated.shape[0] - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        if current_app.config.get("VIDEO_PERSON_ROI_ENABLED", config.VIDEO_PERSON_ROI_ENABLED) is True:
            for item in self._person_roi_detections(frame, detections):
                item.update({"timestamp_sec": timestamp, "timestamp": timestamp,
                             "frame_number": frame_number})
                detections.append(item)
                x1, y1, x2, y2 = map(int, item["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(annotated, f"{item['class'].title()} ROI {item['confidence']:.0%}",
                            (x1, max(18, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 140, 255), 2)
        for detection in detections:
            detection.setdefault("source", "full-frame")
        detections = [
            detection for detection in detections
            if float(detection.get("confidence", 0.0)) >= _class_confidence_threshold(detection.get("class_id", -1))
        ]
        return detections, annotated

    @staticmethod
    def _map_to_forensic_category(label: str) -> str:
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

    @classmethod
    def class_name_for_id(cls, class_id: int) -> str | None:
        """Resolve a detector ID from loaded model metadata or its known map."""
        names = getattr(YOLOService._model, "names", None)
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return cls.DETECTOR_CLASS_NAMES[class_id] if 0 <= class_id < len(cls.DETECTOR_CLASS_NAMES) else None

    @property
    def model_info(self):
        return {
            "path": YOLOService._model_path_used,
            "classes": list(YOLOService._model.names.values()) if YOLOService._model else [],
            "class_id_names": list(YOLOService._model.names.items()) if YOLOService._model else [],
            "error": YOLOService._load_error,
        }


def _box_iou(first, second):
    if not first or not second:
        return 0.0
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
    area_b = max(0, second[2]-second[0]) * max(0, second[3]-second[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _class_confidence_threshold(class_id):
    """Read per-class cutoffs without changing checkpoint class IDs."""
    defaults = {
        0: ("VIDEO_CLASS_CONFIDENCE_GUN", 0.30),
        1: ("VIDEO_CLASS_CONFIDENCE_HAMMER", 0.25),
        2: ("VIDEO_CLASS_CONFIDENCE_HELMET", 0.25),
        3: ("VIDEO_CLASS_CONFIDENCE_KNIFE", 0.25),
        4: ("VIDEO_CLASS_CONFIDENCE_MASK", 0.25),
        5: ("VIDEO_CLASS_CONFIDENCE_PERSON", 0.25),
        6: ("VIDEO_CLASS_CONFIDENCE_PISTOL", 0.25),
        7: ("VIDEO_CLASS_CONFIDENCE_RIFLE", 0.25),
        8: ("VIDEO_CLASS_CONFIDENCE_WEAPON", 0.25),
    }
    try:
        custom_thresholds = current_app.config.get("VIDEO_CLASS_CONFIDENCE_THRESHOLDS")
        if isinstance(custom_thresholds, dict) and int(class_id) in custom_thresholds:
            return float(custom_thresholds[int(class_id)])
        key, default = defaults.get(int(class_id), (None, 0.25))
        if key is None:
            return default
        return float(current_app.config.get(key, getattr(config, key, default)))
    except Exception:
        return defaults.get(int(class_id), (None, 0.25))[1]
