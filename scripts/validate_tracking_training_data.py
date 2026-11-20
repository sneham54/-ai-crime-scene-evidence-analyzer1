from pathlib import Path
import pandas as pd
import cv2
from ultralytics import YOLO

PROJECT = Path(r"C:\Users\admin\Downloads\ai-crime-scene-evidence-analyzer_v2\ai-crime-scene-evidence-analyzer")
DATASET = Path(r"C:\Users\admin\OneDrive\Documents\crime_video_dataset")
MODEL_PATH = PROJECT / "backend" / "models" / "best.pt"
INPUT = PROJECT / "reports" / "tracking_csv_validation.csv"
OUT = PROJECT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
READINESS_OUT = OUT / "tracking_training_readiness.csv"
CLASS_OUT = OUT / "tracking_candidate_class_counts.csv"

def finish(message):
    print(message)
    raise SystemExit(1)

if not INPUT.exists():
    finish(f"Inventory not found: {INPUT}")

if not MODEL_PATH.exists():
    finish(f"YOLO model not found: {MODEL_PATH}")

try:
    model = YOLO(str(MODEL_PATH))
    names = {int(k): str(v) for k, v in model.names.items()}
except Exception as exc:
    finish(f"Could not load YOLO class mapping: {exc}")

inventory = pd.read_csv(INPUT, keep_default_na=False)
required_inventory = {"Category", "Video", "CSVPath"}
missing = required_inventory - set(inventory.columns)
if missing:
    finish(f"Inventory is missing columns: {sorted(missing)}")

results = []
class_records = []

for _, item in inventory.iterrows():
    category = str(item["Category"]).strip()
    video_name = str(item["Video"]).strip()
    csv_path = Path(str(item["CSVPath"]).strip())
    source_path = DATASET / category / video_name

    result = {
        "Category": category,
        "Video": video_name,
        "SourcePath": str(source_path),
        "SourceExists": source_path.exists(),
        "CSVPath": str(csv_path),
        "CSVExists": csv_path.exists(),
        "VideoFrames": 0,
        "VideoFPS": 0.0,
        "VideoWidth": 0,
        "VideoHeight": 0,
        "CSVRows": 0,
        "ValidBoxesInsideFrame": 0,
        "InvalidOrOutOfBoundsBoxes": 0,
        "InvalidFrameNumbers": 0,
        "InvalidClassIDs": 0,
        "ClassNameMismatches": 0,
        "RowsWithTrackID": 0,
        "RowsWithoutTrackID": 0,
        "Status": "",
        "Error": "",
    }

    try:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not source_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_path}")

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open source video: {source_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        result.update({
            "VideoFrames": frame_count,
            "VideoFPS": round(fps, 4),
            "VideoWidth": width,
            "VideoHeight": height,
        })

        if frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Video metadata is invalid.")

        df = pd.read_csv(csv_path, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        result["CSVRows"] = len(df)

        # Your files use "class"; older exports may use "class_name".
        if "class" in df.columns:
            label_column = "class"
        elif "class_name" in df.columns:
            label_column = "class_name"
        else:
            label_column = None

        required = {"frame", "class_id", "x1", "y1", "x2", "y2"}
        missing_columns = required - set(df.columns)
        if label_column is None:
            missing_columns.add("class or class_name")

        if missing_columns:
            raise ValueError(f"Missing columns: {sorted(missing_columns)}")

        if "track_id" not in df.columns:
            df["track_id"] = ""

        if len(df) == 0:
            result["Status"] = "empty_csv_needs_investigation"
            result["Error"] = "CSV contains no detection rows."
            results.append(result)
            continue

        frame_values = pd.to_numeric(df["frame"], errors="coerce")
        class_ids = pd.to_numeric(df["class_id"], errors="coerce")
        x1 = pd.to_numeric(df["x1"], errors="coerce")
        y1 = pd.to_numeric(df["y1"], errors="coerce")
        x2 = pd.to_numeric(df["x2"], errors="coerce")
        y2 = pd.to_numeric(df["y2"], errors="coerce")

        # Permit either zero-based frames (0..N-1) or one-based frames (1..N).
        frame_ok = (
            frame_values.notna()
            & (frame_values % 1 == 0)
            & (frame_values >= 0)
            & (frame_values <= frame_count)
        )
        result["InvalidFrameNumbers"] = int((~frame_ok).sum())

        box_ok = (
            x1.notna() & y1.notna() & x2.notna() & y2.notna()
            & x1.map(lambda v: pd.notna(v) and abs(v) != float("inf"))
            & y1.map(lambda v: pd.notna(v) and abs(v) != float("inf"))
            & x2.map(lambda v: pd.notna(v) and abs(v) != float("inf"))
            & y2.map(lambda v: pd.notna(v) and abs(v) != float("inf"))
            & (x1 >= 0) & (y1 >= 0)
            & (x2 > x1) & (y2 > y1)
            & (x2 <= width) & (y2 <= height)
        )
        result["ValidBoxesInsideFrame"] = int(box_ok.sum())
        result["InvalidOrOutOfBoundsBoxes"] = int((~box_ok).sum())

        id_ok = class_ids.notna() & (class_ids % 1 == 0)
        known_class = id_ok & class_ids.isin(list(names.keys()))
        result["InvalidClassIDs"] = int((~known_class).sum())

        observed_names = df[label_column].astype(str).str.strip().str.lower()
        mismatches = 0
        for index in df.index[known_class]:
            expected = names[int(class_ids.loc[index])].strip().lower()
            observed = observed_names.loc[index]
            if observed and observed != expected:
                mismatches += 1
        result["ClassNameMismatches"] = mismatches

        track_ids = df["track_id"].astype(str).str.strip()
        has_track = track_ids.ne("") & track_ids.str.lower().ne("nan")
        result["RowsWithTrackID"] = int(has_track.sum())
        result["RowsWithoutTrackID"] = int((~has_track).sum())

        temp = pd.DataFrame({
            "class_id": class_ids,
            "class_name": observed_names,
        })
        counts = (
            temp.groupby(["class_id", "class_name"], dropna=False)
            .size()
            .reset_index(name="DetectionRows")
        )
        for _, c in counts.iterrows():
            class_records.append({
                "Category": category,
                "Video": video_name,
                "class_id": c["class_id"],
                "class_name": c["class_name"],
                "DetectionRows": int(c["DetectionRows"]),
            })

        problems = []
        if result["InvalidOrOutOfBoundsBoxes"]:
            problems.append("invalid/out-of-bounds boxes")
        if result["InvalidFrameNumbers"]:
            problems.append("invalid frame numbers")
        if result["InvalidClassIDs"]:
            problems.append("invalid class IDs")
        if result["ClassNameMismatches"]:
            problems.append("class name mismatches")

        if problems:
            result["Status"] = "requires_correction"
            result["Error"] = "; ".join(problems)
        else:
            result["Status"] = "structurally_valid_needs_human_review"
            result["Error"] = ""

    except Exception as exc:
        result["Status"] = "audit_error"
        result["Error"] = str(exc)

    results.append(result)

report = pd.DataFrame(results)
report.to_csv(READINESS_OUT, index=False)

classes = pd.DataFrame(class_records)
classes.to_csv(CLASS_OUT, index=False)

print("\nYOLO class mapping:", names)

print("\n=== VALIDATION SUMMARY BY CATEGORY ===")
if not report.empty:
    print(
        report.groupby(["Category", "Status"])
        .size()
        .reset_index(name="CSVCount")
        .to_string(index=False)
    )

print("\n=== BOX / FRAME / CLASS ISSUES ===")
if not report.empty:
    columns = [
        "CSVRows", "ValidBoxesInsideFrame",
        "InvalidOrOutOfBoundsBoxes", "InvalidFrameNumbers",
        "InvalidClassIDs", "ClassNameMismatches",
        "RowsWithTrackID", "RowsWithoutTrackID"
    ]
    print(report.groupby("Category")[columns].sum().to_string())

print(f"\nDetailed report: {READINESS_OUT}")
print(f"Class report: {CLASS_OUT}")
print("\nStructurally valid annotations still require visual review.")
