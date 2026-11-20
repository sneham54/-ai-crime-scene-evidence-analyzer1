"""Inspect the model at the known 00:10 clip or an extracted still frame.

Place the supplied clip at ``samples/00_10_frame.mp4`` or an extracted frame
at ``samples/00_10_frame.jpg`` (also accepts .jpeg, .png, .bmp, .webp,
.avi, .mov, .mkv and .webm). The helper prints raw model predictions and the
final detections after the configured confidence filters.
"""
import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _bbox_xywh(bbox):
    """Format the service's xyxy coordinates as x,y,width,height."""
    if not bbox or len(bbox) < 4:
        return "unavailable"
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    return f"{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f}"


def _print_raw(raw_detections):
    if not raw_detections:
        print("RAW   | no predictions")
    for detection in raw_detections:
        print(
            f"RAW   | {detection.get('class', detection.get('class_name', 'unknown'))} "
            f"(id={detection.get('class_id')}) "
            f"conf={float(detection.get('confidence', 0.0)):.3f} "
            f"bbox=<{_bbox_xywh(detection.get('bbox'))}> source=full-frame"
        )


def _print_final(detections, timestamp):
    if not detections:
        print("FINAL | no detections")
    for detection in detections:
        ts = detection.get("timestamp_sec", detection.get("timestamp", timestamp))
        ts_text = f"{float(ts):.2f}" if ts is not None else "unavailable"
        print(
            f"FINAL | {detection.get('class', detection.get('class_name', 'unknown'))} "
            f"(id={detection.get('class_id')}) "
            f"conf={float(detection.get('confidence', 0.0)):.3f} "
            f"bbox=<{_bbox_xywh(detection.get('bbox'))}> ts={ts_text}s "
            f"source={detection.get('source', 'full-frame')}"
        )


def _raw_video_frames(video_path, frame_results, yolo, app):
    """Run the raw diagnostic pass on the exact frames selected by VideoService."""
    import cv2

    indexed = {
        int(result["frame_number"]): result
        for result in frame_results
        if result.get("frame_number") is not None
    }
    if not indexed:
        return {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to reopen video for raw predictions: {video_path}")
    raw_by_frame = {}
    try:
        frame_number = 0
        last_target = max(indexed)
        while frame_number <= last_target:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number in indexed:
                if app.config.get("VIDEO_CLAHE_ENABLED", True):
                    from app.services.video_service import _enhance_low_contrast
                    inference_frame = _enhance_low_contrast(frame)
                else:
                    inference_frame = frame
                raw_by_frame[frame_number] = yolo.predict_frame_raw(inference_frame)
            frame_number += 1
    finally:
        capture.release()
    return raw_by_frame


def _is_known_0010(path):
    stem = path.stem.lower().replace("-", "_").replace(" ", "_")
    return "00_10" in stem or "00:10" in path.stem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Image or video file to inspect")
    parser.add_argument("--roi", action="store_true", help="Enable person ROI for this run only")
    args = parser.parse_args(argv)

    if not args.input_path.is_file():
        print(f"[ERROR] File not found: {args.input_path}")
        return 1

    prior_roi_setting = os.environ.get("VIDEO_PERSON_ROI_ENABLED")
    if args.roi:
        os.environ["VIDEO_PERSON_ROI_ENABLED"] = "true"
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        import cv2
        from dotenv import load_dotenv
        from flask import Flask

        # Load root .env before Config is imported, matching the app factory.
        load_dotenv(PROJECT_ROOT / ".env")
        from app.config import Config
        from app.services.video_service import VideoService
        from app.services.yolo_service import YOLOService

        app = Flask("forensic_backend", root_path=str(BACKEND_DIR / "app"))
        app.config.from_object(Config)
        with app.app_context():
            yolo = YOLOService()
            print("=== REGRESSION REPORT ===")
            print(f"Checkpoint: {yolo.model_info.get('path')}")
            print(f"ROI enabled: {app.config['VIDEO_PERSON_ROI_ENABLED']}")

            suffix = args.input_path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(args.input_path))
                if frame is None:
                    raise ValueError(f"Unable to decode image: {args.input_path}")
                raw = yolo.predict_frame_raw(frame)
                final, _annotated = yolo.detect_frame(frame)
                timestamp = 10.0 if _is_known_0010(args.input_path) else 0.0
                frame_results = [{
                    "frame_number": 0,
                    "timestamp_sec": timestamp,
                    "raw_detections": raw,
                    "detections": final,
                }]
            elif suffix in VIDEO_EXTENSIONS:
                result = VideoService().analyze(str(args.input_path))
                frame_results = result.get("frame_results", [])
                raw_by_frame = _raw_video_frames(args.input_path, frame_results, yolo, app)
                for frame_result in frame_results:
                    frame_result["raw_detections"] = raw_by_frame.get(
                        int(frame_result.get("frame_number", -1)), []
                    )
            else:
                raise ValueError(f"Unsupported image/video extension: {suffix or '(none)'}")

            all_final = []
            for index, frame_result in enumerate(frame_results):
                timestamp = frame_result.get("timestamp_sec")
                timestamp_text = f"{timestamp:.2f}s" if timestamp is not None else "unavailable"
                print(f"--- Frame {frame_result.get('frame_number', index)} (timestamp={timestamp_text}) ---")
                _print_raw(frame_result.get("raw_detections", []))
                detections = frame_result.get("detections") or []
                _print_final(detections, timestamp)
                all_final.extend(detections)

            if _is_known_0010(args.input_path):
                gun_detections = [item for item in all_final if int(item.get("class_id", -1)) == 0]
                if not gun_detections:
                    print("No gun detection at 00:10")
                else:
                    for detection in gun_detections:
                        ts = detection.get("timestamp_sec", detection.get("timestamp"))
                        ts_text = f"{float(ts):.2f}" if ts is not None else "unavailable"
                        print(
                            "Gun detection at 00:10: "
                            f"FINAL | {detection.get('class', 'gun')} (id=0) "
                            f"conf={float(detection.get('confidence', 0.0)):.3f} "
                            f"bbox=<{_bbox_xywh(detection.get('bbox'))}> ts={ts_text}s "
                            f"source={detection.get('source', 'full-frame')}"
                        )
        return 0
    except Exception as error:
        print(f"[ERROR] Regression analysis failed: {type(error).__name__}: {error}")
        return 1
    finally:
        if args.roi:
            if prior_roi_setting is None:
                os.environ.pop("VIDEO_PERSON_ROI_ENABLED", None)
            else:
                os.environ["VIDEO_PERSON_ROI_ENABLED"] = prior_roi_setting


if __name__ == "__main__":
    raise SystemExit(main())
