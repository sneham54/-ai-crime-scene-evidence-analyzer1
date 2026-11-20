"""Colab-only export/inventory helper for unresolved Assault Drive artifacts.

Run this file in a Google Colab cell. It does not run inference or alter Drive.
It asks for assault_combined_reconciliation.csv, inventories matching outputs
throughout MyDrive, and downloads a ZIP containing only relevant artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from google.colab import drive, files


DRIVE_ROOT = Path("/content/gdrive/MyDrive")
OUTPUT_NAMES = {
    "assault_ground_truth_review_template.csv",
    "assault_predicted_event_intervals.csv",
    "assault_final_evaluation_report.csv",
    "assault_detection_evaluation_status.csv",
    "assault_video_coverage_audit.csv",
    "assault_videos_to_check.csv",
    "assault_detection_class_summary.csv",
    "assault_weapon_detections_integrated.csv",
}
FINAL_MARKERS = ("tracked", "annotated", "final", "recovered")
PARTIAL_MARKERS = ("partial", "incomplete", ".part")
OUTPUT_DIR_MARKERS = ("tracked_results", "weapon_detection", "assault", "recovered", "annotation", "output", "result")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_drive_path(raw: str) -> Path | None:
    if not raw:
        return None
    raw = raw.strip().replace(";", "")
    marker = "/MyDrive/"
    if marker in raw:
        rel = raw.split(marker, 1)[1]
        candidate = DRIVE_ROOT / rel
    else:
        return None
    try:
        candidate.resolve().relative_to(DRIVE_ROOT.resolve())
    except ValueError:
        return None
    return candidate


def artifact_kind(path: Path, target_keys: set[str]) -> tuple[str, str] | None:
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    matched_key = next((key for key in target_keys if key.lower() in name or key.lower() in parts), None)
    if not matched_key:
        return None
    path_marked = any(marker in "/".join(parts) for marker in OUTPUT_DIR_MARKERS)
    if path.name.lower() in OUTPUT_NAMES:
        return "review_or_summary", matched_key
    if path.suffix.lower() == ".csv":
        if (name == "detections.csv" and matched_key.lower() in parts) or "detection" in name or "potential_event" in name:
            return "detection_or_event_csv", matched_key
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        if any(marker in name for marker in FINAL_MARKERS):
            return "final_or_annotated_video", matched_key
        if any(marker in name for marker in PARTIAL_MARKERS):
            return "partial_video", matched_key
        if path_marked and path.parent.name.lower() in {matched_key.lower(), matched_key.lower() + "_tracked"}:
            return "video_in_output_folder", matched_key
    return None


def main():
    if not DRIVE_ROOT.exists():
        drive.mount("/content/gdrive")
    if not DRIVE_ROOT.exists():
        raise RuntimeError("Google Drive did not mount at /content/gdrive")

    print("Upload assault_combined_reconciliation.csv when prompted.")
    uploaded = files.upload()
    manifest_path = next((Path("/content") / name for name in uploaded
                          if Path(name).name == "assault_combined_reconciliation.csv"), None)
    if manifest_path is None:
        raise ValueError("Please upload assault_combined_reconciliation.csv")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reconciliation = list(csv.DictReader(stream))
    if not reconciliation or "source_key" not in reconciliation[0]:
        raise ValueError("The uploaded file is not the expected combined Assault reconciliation")

    unresolved_statuses = {
        "CSV_ONLY_NO_FINAL_MP4_ANYWHERE",
        "PARTIAL_ONLY_NEEDS_REVIEW",
        "FINAL_MP4_NO_CSV_ANYWHERE",
        "NO_MATCHING_OUTPUT_IN_AUDITED_LOCATIONS",
    }
    target_rows = [row for row in reconciliation if row.get("reconciliation_status") in unresolved_statuses]
    keys = {row["source_key"] for row in reconciliation}
    export_keys = {row["source_key"] for row in target_rows}
    temp_root = Path(tempfile.mkdtemp(prefix="assault_drive_export_", dir="/content"))
    artifacts_root = temp_root / "artifacts"
    artifacts_root.mkdir()
    inventory = []
    copied = set()

    # Include the exact saved locations reported by the existing Drive manifest.
    for row in target_rows:
        key = row["source_key"]
        for column in ("drive_mp4_paths", "drive_csv_paths"):
            for raw in (row.get(column) or "").split(";"):
                candidate = parse_drive_path(raw)
                if candidate is None:
                    continue
                if candidate.is_file():
                    rel = candidate.relative_to(DRIVE_ROOT)
                    target = artifacts_root / key / "exact_manifest_path" / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.resolve().is_relative_to(artifacts_root.resolve()) and str(candidate) not in copied:
                        shutil.copy2(candidate, target)
                        copied.add(str(candidate))
                    inventory.append({"source_key": key, "artifact_type": column,
                                      "drive_path": str(candidate), "exists": True,
                                      "copied_to_export": True, "sha256": sha256(candidate)})
                else:
                    inventory.append({"source_key": key, "artifact_type": column,
                                      "drive_path": str(candidate), "exists": False,
                                      "copied_to_export": False, "sha256": ""})

    # Inspect all of MyDrive for alternate finished outputs for the unresolved
    # source IDs. Source videos are excluded unless stored in an output-like tree.
    total_dirs = 0
    for current, dirs, filenames in os.walk(DRIVE_ROOT):
        dirs[:] = [name for name in dirs if not name.startswith(".") and name not in {"__pycache__", ".ipynb_checkpoints"}]
        total_dirs += 1
        if total_dirs % 5000 == 0:
            print(f"Scanned {total_dirs} Drive folders…")
        folder = Path(current)
        for filename in filenames:
            path = folder / filename
            kind = artifact_kind(path, keys)
            if kind is not None:
                artifact_type, key = kind
                copied_now = False
                if key in export_keys and str(path) not in copied:
                    rel = path.relative_to(DRIVE_ROOT)
                    dest = artifacts_root / key / "alternate_drive_search" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    copied.add(str(path))
                    copied_now = True
                inventory.append({"source_key": key, "artifact_type": artifact_type,
                                  "drive_path": str(path), "exists": True,
                                  "copied_to_export": copied_now, "sha256": sha256(path)})
            if filename.lower() in OUTPUT_NAMES:
                path = folder / filename
                rel = path.relative_to(DRIVE_ROOT)
                dest = temp_root / "review_and_evaluation" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                inventory.append({"source_key": "", "artifact_type": "review_or_summary",
                                  "drive_path": str(path), "exists": True,
                                  "copied_to_export": True, "sha256": sha256(path)})

    inventory_path = temp_root / "drive_search_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["source_key", "artifact_type", "drive_path", "exists", "copied_to_export", "sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    shutil.copy2(manifest_path, temp_root / "assault_combined_reconciliation.csv")

    status_path = temp_root / "drive_recovery_status.csv"
    with status_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_key", "reconciliation_status", "drive_candidates_found", "drive_artifacts_copied"])
        writer.writeheader()
        for row in reconciliation:
            key = row["source_key"]
            hits = [entry for entry in inventory if entry["source_key"] == key]
            writer.writerow({"source_key": key, "reconciliation_status": row["reconciliation_status"],
                             "drive_candidates_found": len(hits),
                             "drive_artifacts_copied": sum(bool(entry["copied_to_export"]) for entry in hits)})

    zip_path = Path("/content/assault_drive_recovery_export.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in temp_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(temp_root).as_posix())
    print(f"Relevant MyDrive matches: {len(inventory)}")
    print(f"Export ZIP: {zip_path}; copied Drive objects: {len(copied)}")
    files.download(str(zip_path))


main()
