"""Bounded-memory video classification and sampled-frame YOLO evidence analysis."""
import cv2
import numpy as np
from flask import current_app

from app.services.video_classification_model import VideoClassificationModel
from app.services.yolo_service import YOLOService


class VideoService:
    def __init__(self):
        self.yolo = YOLOService()
        self.classifier = VideoClassificationModel(
            current_app.config["VIDEO_MODEL_PATH"],
            current_app.config["CLASS_NAMES_PATH"],
            current_app.config["VIDEO_MIN_CONFIDENCE"],
        )

    def analyze(self, video_path, max_frames=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Uploaded video is invalid or cannot be decoded")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if total_frames <= 0:
                raise ValueError("Video has no readable frames")
            valid_fps = np.isfinite(fps) and fps > 0
            duration = total_frames / fps if valid_fps else None
            sample_count = max(1, min(
                int(max_frames or current_app.config["VIDEO_MAX_SAMPLED_FRAMES"]),
                current_app.config["VIDEO_MAX_SAMPLED_FRAMES"],
                total_frames,
            ))
            indices = np.unique(np.linspace(0, total_frames - 1, sample_count).round().astype(int))
            frame_results = []
            for frame_number in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                ok, frame = cap.read()
                if not ok:
                    continue
                timestamp = float(frame_number / fps) if valid_fps else None
                detections, annotated = self.yolo.detect_frame(frame)
                for detection in detections:
                    detection["frame_number"] = int(frame_number)
                    detection["timestamp_sec"] = round(timestamp, 3) if timestamp is not None else None
                frame_results.append({
                    "frame_number": int(frame_number),
                    "timestamp_sec": round(timestamp, 3) if timestamp is not None else None,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "detections": detections,
                    "detection_count": len(detections),
                })
        finally:
            cap.release()

        classification = self.classifier.predict_path(video_path)
        all_detections = [d for frame in frame_results for d in frame["detections"]]
        timeline = [
            {"frame_number": d["frame_number"], "timestamp_sec": d["timestamp_sec"],
             "class": d["class"], "class_name": d["class"], "confidence": d["confidence"], "bbox": d["bbox"]}
            for d in all_detections
        ]
        unique_objects = sorted({d["class"] for d in all_detections})
        event_name = classification["label"]
        evidence = ", ".join(unique_objects) if unique_objects else "no objects"
        summary = (
            f"The video classifier labeled this video {event_name} "
            f"({classification['confidence']:.1%} softmax confidence). "
            f"YOLO visual evidence in sampled frames includes {evidence}. "
            "Object detections do not establish that a crime occurred."
        )
        return {
            "video_info": {
                "total_frames": total_frames,
                "fps": round(fps, 3) if valid_fps else None,
                "duration_sec": round(duration, 3) if duration is not None else None,
                "frames_analyzed": len(frame_results),
                "timestamps_available": bool(valid_fps),
            },
            "classification": classification,
            "event_classification": classification,
            "frame_results": frame_results,
            "detections": all_detections,
            "unique_objects": unique_objects,
            "total_detections": len(all_detections),
            "timeline": timeline,
            "event_timeline": timeline,
            "investigation_summary": summary,
            "disclaimer": "AI-generated results are intended to assist investigation and require verification by authorized personnel.",
            "blood_stain_supported": False,
        }
