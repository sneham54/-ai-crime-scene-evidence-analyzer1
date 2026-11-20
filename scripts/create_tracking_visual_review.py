from pathlib import Path
import re
import cv2
import pandas as pd

PROJECT = Path(
    r"C:\Users\admin\Downloads\ai-crime-scene-evidence-analyzer_v2\ai-crime-scene-evidence-analyzer"
)
DATASET = Path(r"C:\Users\admin\OneDrive\Documents\crime_video_dataset")
INVENTORY = PROJECT / "reports" / "tracking_csv_validation.csv"
OUT = PROJECT / "reports" / "tracking_visual_review"
MAX_FRAMES_PER_VIDEO = 5

OUT.mkdir(parents=True, exist_ok=True)

def safe_name(s):
    return re.sub(r'[<>:"/\\|?*]', "_", str(s))

if not INVENTORY.exists():
    raise FileNotFoundError(f"Inventory not found: {INVENTORY}")

inventory = pd.read_csv(INVENTORY, keep_default_na=False)
index_rows = []
errors = []

for _, item in inventory.iterrows():
    category = str(item["Category"]).strip()
    video_name = str(item["Video"]).strip()
    csv_path = Path(str(item["CSVPath"]).strip())
    video_path = DATASET / category / video_name

    try:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        df = pd.read_csv(csv_path, keep_default_na=False)
        if df.empty:
            print(f"SKIP empty CSV: {category}/{video_name}")
            continue

        required = {"frame", "class_id", "x1", "y1", "x2", "y2"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        label_col = "class" if "class" in df.columns else (
            "class_name" if "class_name" in df.columns else None
        )
        if label_col is None:
            raise ValueError("No class or class_name column")

        if "confidence" not in df.columns:
            df["confidence"] = ""
        if "track_id" not in df.columns:
            df["track_id"] = ""
        if "timestamp_seconds" not in df.columns:
            df["timestamp_seconds"] = ""

        df["_frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df["_timestamp"] = pd.to_numeric(
            df["timestamp_seconds"], errors="coerce"
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("OpenCV could not open the source video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        # Infer whether CSV frame numbers are zero-based or one-based.
        timing = df[
            df["_frame"].notna()
            & df["_timestamp"].notna()
            & (df["_frame"] % 1 == 0)
        ]

        offset = 0
        err_zero = None
        err_one = None

        if fps > 0 and not timing.empty:
            err_zero = (
                timing["_timestamp"] - timing["_frame"] / fps
            ).abs().median()
            err_one = (
                timing["_timestamp"] - (timing["_frame"] - 1) / fps
            ).abs().median()

            if err_one < err_zero:
                offset = 1

        valid_frames = sorted({
            int(v)
            for v in df["_frame"].dropna().tolist()
            if float(v).is_integer()
            and 0 <= int(v) - offset < frame_count
        })

        if not valid_frames:
            cap.release()
            print(f"SKIP no usable frame numbers: {category}/{video_name}")
            continue

        count = min(MAX_FRAMES_PER_VIDEO, len(valid_frames))
        if count == 1:
            selected_frames = [valid_frames[0]]
        else:
            positions = [
                round(i * (len(valid_frames) - 1) / (count - 1))
                for i in range(count)
            ]
            selected_frames = sorted({valid_frames[p] for p in positions})

        output_dir = OUT / safe_name(category) / safe_name(Path(video_name).stem)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for csv_frame in selected_frames:
            source_frame = int(csv_frame) - offset
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = cap.read()
            if not ok:
                print(f"Frame read failed: {video_name}, CSV frame {csv_frame}")
                continue

            rows = df[df["_frame"] == csv_frame]
            boxes_drawn = 0

            for _, row in rows.iterrows():
                try:
                    x1 = float(row["x1"])
                    y1 = float(row["y1"])
                    x2 = float(row["x2"])
                    y2 = float(row["y2"])
                    if not all(pd.notna(v) for v in (x1, y1, x2, y2)):
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue

                    p1 = (round(x1), round(y1))
                    p2 = (round(x2), round(y2))
                    class_id = int(float(row["class_id"]))
                    class_label = str(row[label_col]).strip() or str(class_id)

                    # Deterministic display color; this is not a class quality score.
                    color = (
                        (class_id * 67 + 40) % 255,
                        (class_id * 113 + 90) % 255,
                        (class_id * 157 + 130) % 255,
                    )

                    cv2.rectangle(image, p1, p2, color, 2)

                    conf = str(row["confidence"]).strip()
                    track_id = str(row["track_id"]).strip()
                    label = class_label
                    if conf:
                        label += f" {conf}"
                    if track_id and track_id.lower() != "nan":
                        label += f" ID:{track_id}"

                    text_y = max(18, p1[1] - 6)
                    cv2.putText(
                        image, label, (max(0, p1[0]), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        cv2.LINE_AA
                    )
                    boxes_drawn += 1
                except (ValueError, TypeError, OverflowError):
                    continue

            banner = (
                f"{category} | {video_name} | CSV frame {csv_frame}"
                f" | source frame {source_frame}"
            )
            cv2.rectangle(image, (0, 0), (image.shape[1], 30), (20, 20, 20), -1)
            cv2.putText(
                image, banner[:150], (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA
            )

            filename = f"frame_{csv_frame:07d}.jpg"
            image_path = output_dir / filename
            if cv2.imwrite(str(image_path), image):
                saved += 1
                index_rows.append({
                    "Category": category,
                    "Video": video_name,
                    "CSVFrame": csv_frame,
                    "SourceFrameZeroBased": source_frame,
                    "FrameNumberOffset": offset,
                    "FPS": fps,
                    "DetectionsDrawn": boxes_drawn,
                    "ImagePath": str(image_path),
                    "TimingErrorZeroBased": err_zero,
                    "TimingErrorOneBased": err_one,
                })

        cap.release()
        print(f"{category}/{video_name}: saved {saved} review images; frame offset={offset}")

    except Exception as exc:
        errors.append({
            "Category": category,
            "Video": video_name,
            "Error": str(exc),
        })
        print(f"ERROR {category}/{video_name}: {exc}")

index_path = OUT / "review_index.csv"
pd.DataFrame(index_rows).to_csv(index_path, index=False)

if errors:
    pd.DataFrame(errors).to_csv(OUT / "review_errors.csv", index=False)

print(f"\nReview images folder: {OUT}")
print(f"Review index: {index_path}")
print(f"Images generated: {len(index_rows)}")
print(f"Video-level errors: {len(errors)}")
