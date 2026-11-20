"""Package existing Robbery VideoMAE outputs and related evaluation artifacts from Colab Drive.

Run this file as one Colab cell. It only reads Drive and creates a downloadable
ZIP; it never runs a model or changes source artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

from google.colab import drive, files


DRIVE_ROOT = Path("/content/gdrive/MyDrive")
DATASET_ROOT = DRIVE_ROOT / "robbery_activity_dataset"
VIDEOMAE_ROOT = DATASET_ROOT / "robbery_videomae"
RELATED_TERMS = (
    "manifest", "predict", "evaluation", "metric", "report", "config",
    "class_name", "preprocess", "checkpoint", "result", "split",
)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not DRIVE_ROOT.exists():
        drive.mount("/content/gdrive")
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Expected Drive folder was not found: {DATASET_ROOT}")
    if not VIDEOMAE_ROOT.is_dir():
        raise FileNotFoundError(f"Expected VideoMAE folder was not found: {VIDEOMAE_ROOT}")

    selected: dict[Path, str] = {}
    for path in VIDEOMAE_ROOT.rglob("*"):
        if path.is_file():
            selected[path] = "videomae_folder_contents"

    # Include relevant run metadata/results stored elsewhere in the dataset,
    # while excluding raw video corpora from this targeted export.
    for current, dirs, names in os.walk(DATASET_ROOT):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        folder = Path(current)
        for name in names:
            path = folder / name
            if path in selected or path.suffix.lower() in VIDEO_SUFFIXES:
                continue
            relative = path.relative_to(DATASET_ROOT).as_posix().lower()
            if any(term in relative for term in RELATED_TERMS):
                selected[path] = "related_evaluation_or_prediction_artifact"

    if not selected:
        raise RuntimeError("No files were found to export; Drive data has not been modified.")

    temp_root = Path(tempfile.mkdtemp(prefix="robbery_activity_recovery_", dir="/content"))
    zip_path = Path("/content/robbery_activity_recovery_export.zip")
    inventory_rows = []
    for source, kind in sorted(selected.items()):
        relative = source.relative_to(DRIVE_ROOT).as_posix()
        inventory_rows.append({
            "drive_path": str(source),
            "archive_path": relative,
            "artifact_group": kind,
            "size_bytes": source.stat().st_size,
            "sha256": sha256(source),
        })
    inventory_path = temp_root / "robbery_export_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for source in sorted(selected):
            archive.write(source, source.relative_to(DRIVE_ROOT).as_posix())
        archive.write(inventory_path, "robbery_export_inventory.csv")

    print(f"Files packaged: {len(selected)}")
    print(f"VideoMAE folder: {VIDEOMAE_ROOT}")
    print(f"ZIP: {zip_path}")
    files.download(str(zip_path))


main()
