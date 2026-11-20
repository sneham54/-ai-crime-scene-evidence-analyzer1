from pathlib import Path
import random, re, shutil, zipfile
import cv2
import pandas as pd

PROJECT = Path(r"C:\Users\admin\Downloads\ai-crime-scene-evidence-analyzer_v2\ai-crime-scene-evidence-analyzer")
DATASET = Path(r"C:\Users\admin\OneDrive\Documents\crime_video_dataset")
INVENTORY = PROJECT / "reports" / "tracking_csv_validation.csv"
OUT = PROJECT / "yolo_training_candidate"
ZIP_OUT = PROJECT / "reports" / "yolo_training_candidate.zip"

NAMES = {
    0: "gun", 1: "hammer", 2: "helmet", 3: "knife", 4: "mask",
    5: "person", 6: "pistol", 7: "rifle", 8: "weapon"
}

if not INVENTORY.exists():
    raise FileNotFoundError(f"Inventory missing: {INVENTORY}")

inventory = pd.read_csv(INVENTORY, keep_default_na=False)
for col in ("Category", "Video", "CSVPath"):
    if col not in inventory.columns:
        raise ValueError(f"Inventory is missing column: {col}")

if OUT.exists():
    shutil.rmtree(OUT)

for split in ("train", "val", "test"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# Keep video variants from the same numbered clip in the same split.
def group_key(category, video):
    stem = Path(video).stem
    stem = re.sub(r"_x264A?$", "", stem, flags=re.I)
    return f"{category}/{stem}"

video_records = []
for _, r in inventory.iterrows():
    csv_path = Path(str(r["CSVPath"]).strip())
    video_path = DATASET / str(r["Category"]).strip() / str(r["Video"]).strip()
    if csv_path.exists() and video_path.exists():
        video_records.append({
            "category": str(r["Category"]).strip(),
            "video": str(r["Video"]).strip(),
            "csv": csv_path,
            "source": video_path,
            "group": group_key(str(r["Category"]).strip(), str(r["Video"]).strip())
        })

if not video_records:
    raise RuntimeError("No matching CSV/source-video pairs were found.")

# Assign entire video groups to train/validation/test, never individual frames.
random.seed(42)
group_split = {}
categories = sorted({r["category"] for r in video_records})

for category in categories:
    groups = sorted({r["group"] for r in video_records if r["category"] == category})
    random.shuffle(groups)
    n = len(groups)

    if n >= 5:
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
    elif n >= 3:
        n_test, n_val = 1, 1
    elif n == 2:
        n_test, n_val = 0, 1
    else:
        n_test, n_val = 0, 0

    # Keep at least one group in training.
    while n_test + n_val >= n and n_test + n_val > 0:
        if n_val:
            n_val -= 1
        else:
            n_test -= 1

    for i, group in enumerate(groups):
        if i < n_test:
            group_split[group] = "test"
        elif i < n_test + n_val:
            group_split[group] = "val"
        else:
            group_split[group] = "train"

counts = {s: {"videos": set(), "images": 0, "boxes": 0} for s in ("train", "val", "test")}
errors = []

for r in video_records:
    split = group_split[r["group"]]
    try:
        df = pd.read_csv(r["csv"], keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        label_col = "class" if "class" in df.columns else "class_name"

        required = {"frame", "class_id", "x1", "y1", "x2", "y2", label_col}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")

        if "timestamp_seconds" not in df.columns:
            df["timestamp_seconds"] = ""

        frame_num = pd.to_numeric(df["frame"], errors="coerce")
        timestamps = pd.to_numeric(df["timestamp_seconds"], errors="coerce")

        cap = cv2.VideoCapture(str(r["source"]))
        if not cap.isOpened():
            raise RuntimeError("Could not open source video")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if min(frame_count, width, height) <= 0:
            cap.release()
            raise RuntimeError("Invalid source video metadata")

        timing = pd.DataFrame({"frame": frame_num, "ts": timestamps}).dropna()
        timing = timing[timing["frame"] % 1 == 0]
        offset = 0
        if fps > 0 and not timing.empty:
            err0 = (timing["ts"] - timing["frame"] / fps).abs().median()
            err1 = (timing["ts"] - (timing["frame"] - 1) / fps).abs().median()
            if pd.notna(err1) and pd.notna(err0) and err1 < err0:
                offset = 1

        df["_frame"] = frame_num
        df = df[df["_frame"].notna() & (df["_frame"] % 1 == 0)]

        video_tag = re.sub(r"[^A-Za-z0-9_-]", "_", r["category"] + "_" + Path(r["video"]).stem)
        video_tag += f"_{abs(hash(r['group'])) % 100000:05d}"
        frame_groups = df.groupby("_frame", sort=True)

        for csv_frame, rows in frame_groups:
            source_frame = int(csv_frame) - offset
            if not (0 <= source_frame < frame_count):
                continue

            boxes = []
            for _, row in rows.iterrows():
                try:
                    cid = int(float(row["class_id"]))
                    x1, y1, x2, y2 = [float(row[c]) for c in ("x1", "y1", "x2", "y2")]
                    if cid not in NAMES:
                        continue
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        continue

                    xc = ((x1 + x2) / 2) / width
                    yc = ((y1 + y2) / 2) / height
                    bw = (x2 - x1) / width
                    bh = (y2 - y1) / height

                    # Ensure the name agrees with the class mapping.
                    observed = str(row[label_col]).strip().lower()
                    if observed and observed != NAMES[cid]:
                        continue

                    boxes.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                except (ValueError, TypeError, OverflowError):
                    continue

            # Skip frames with no usable annotations rather than assuming they
            # are genuine negative frames.
            if not boxes:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = cap.read()
            if not ok:
                continue

            file_stem = f"{video_tag}_f{source_frame:07d}"
            image_path = OUT / "images" / split / f"{file_stem}.jpg"
            label_path = OUT / "labels" / split / f"{file_stem}.txt"

            if not cv2.imwrite(str(image_path), image):
                continue

            label_path.write_text("\n".join(boxes) + "\n", encoding="utf-8")
            counts[split]["images"] += 1
            counts[split]["boxes"] += len(boxes)
            counts[split]["videos"].add(r["group"])

        cap.release()
        print(f"Prepared {r['category']}/{r['video']} -> {split}")

    except Exception as e:
        errors.append(f"{r['category']}/{r['video']}: {e}")
        print(f"ERROR: {r['category']}/{r['video']}: {e}")

yaml_lines = [
    "path: .",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "names:"
]
for cid, name in NAMES.items():
    yaml_lines.append(f"  {cid}: {name}")
(OUT / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

(OUT / "README.txt").write_text(
    "EXPERIMENTAL TRAINING CANDIDATE DATASET\n"
    "Labels were converted from model-generated tracking CSVs.\n"
    "They have NOT been confirmed as ground truth.\n"
    "Review bounding boxes visually before relying on training results.\n"
    "Splits are grouped by video family to reduce frame leakage.\n",
    encoding="utf-8"
)

ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
if ZIP_OUT.exists():
    ZIP_OUT.unlink()
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for file in OUT.rglob("*"):
        if file.is_file():
            z.write(file, file.relative_to(OUT))

print("\n=== DATASET SUMMARY ===")
for split, c in counts.items():
    print(f"{split}: {len(c['videos'])} video groups, {c['images']} images, {c['boxes']} boxes")

print(f"\nDataset folder: {OUT}")
print(f"ZIP ready for Colab: {ZIP_OUT}")
if errors:
    error_path = PROJECT / "reports" / "yolo_dataset_preparation_errors.txt"
    error_path.write_text("\n".join(errors), encoding="utf-8")
    print(f"Some videos were skipped. See: {error_path}")
