# CCTV YOLO review and threshold evaluation

The video route loads the checkpoint configured as `YOLO_MODEL_PATH` (default: `backend/models/best.pt`). Its class IDs are checkpoint-backed: `0 gun`, `1 hammer`, `2 helmet`, `3 knife`, `4 mask`, `5 person`, `6 pistol`, `7 rifle`, `8 weapon`. The literal `weapon` label is a separate model class; code maps its label to the broad UI category `Weapons`, but does not rename it to `gun`, `pistol`, or `rifle`.

Video inference passes each sampled BGR frame to Ultralytics with `conf=YOLO_CONFIDENCE` (default `0.25`), `iou=0.45`, `imgsz=YOLO_IMAGE_SIZE` (default `832`), CPU inference, and the Ultralytics default class-aware NMS. Ultralytics performs its normal resize/letterbox and normalization internally. The returned `Results.plot()` image draws the checkpoint class name and confidence. The video service samples at `VIDEO_SAMPLE_FPS` (default one frame per second); it invokes ordinary prediction separately for each sampled frame, so the same object can be reported again at later timestamps. Image-only thresholds in `yolo_accuracy_patch.py` do not change this video path.

## Create a human-review bundle

Run from `backend/`:

```powershell
..\..\venv\Scripts\python.exe evaluate_yolo_cctv.py --review-video "C:\path\to\cctv.mp4" --review-output-dir "C:\path\to\review_bundle"
```

The new output directory contains annotated sampled frames and `review.csv`. Every sampled frame is listed, including frames with no predictions. Each row includes frame number, timestamp, class ID/name, confidence, bounding box, and blank reviewer fields with `review_status=pending`. Review each frame, then fill `reviewed_class` and `review_notes`, and change `review_status` to `confirmed`, `false_positive`, or `uncertain`. Do not treat a detector output as proof of a crime.

The existing `/api/video/analyze` route also stores annotated sampled frames and an `evidence_timeline.csv` with each detection's class, confidence, timestamp, and frame link.

## Evaluate thresholds on labeled frames

Prepare a CSV with pixel-coordinate boxes from the original images:

```csv
image_path,class,x1,y1,x2,y2
frames/frame001.jpg,weapon,120,85,188,210
frames/frame001.jpg,person,40,20,220,470
frames/negative001.jpg,,,,
```

Include one blank-label row for every negative image. Use the exact class names above. Images are resolved relative to the CSV file. For CCTV video evaluation, export labeled frames at chosen timestamps and list each frame as an image row.

Run the sweep from `backend/`:

```powershell
..\..\venv\Scripts\python.exe evaluate_yolo_cctv.py --labels-csv "C:\path\to\labels.csv" --report-json "C:\path\to\threshold_report.json"
```

The evaluator also accepts a standard YOLO dataset with `images/` and sibling `labels/` directories:

```powershell
..\..\venv\Scripts\python.exe evaluate_yolo_cctv.py --labels-yolo-images "C:\path\to\dataset\valid\images" --report-json "C:\path\to\threshold_report.json"
```

The default sweep includes the active `0.25` threshold plus `0.10`, `0.15`, `0.35`, `0.50`, `0.65`, and `0.75`. The JSON reports per-class and micro precision, recall, false positives, and missed detections at IoU 0.50. Candidate confidence thresholds filter the same predictions collected at the lowest threshold, using the video route's checkpoint, inference image size, IoU NMS setting, and class mapping. A threshold should be chosen only after reviewing the class-wise false-positive and missed-detection trade-off on representative labeled CCTV material; metrics do not generalize beyond that evaluation set.

For a quick, reproducible subset, add `--max-frames 100` (and optionally
`--sample-seed 42`). The full supplied split is evaluated by default. Reports
include the sampling details so subset metrics are not confused with full-split
results.

The currently configured external Roboflow test split contains 846 labeled
frames, but its annotations include no instances for checkpoint class ID 8
(`weapon`) or ID 6 (`pistol`). It therefore cannot measure false positives or
missed detections for those classes. It is also not a verified sample of the
CCTV footage where the reported `weapon 0.48` box appeared. Use the review
bundle and create representative, human-verified labels before selecting a
weapon confidence threshold.
