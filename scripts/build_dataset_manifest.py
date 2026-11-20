#!/usr/bin/env python3
"""Build a source-video inventory without changing existing processing artifacts.

The manifest intentionally does not guess which tracking CSV belongs to a
video. Reuse must be verified separately against a recorded source hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

CATEGORIES = ("Shooting", "Assault", "Robbery", "Normal")
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
FIELDS = ("category", "source_path", "source_sha256", "status", "reason")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset_root: Path) -> list[dict[str, str]]:
    rows = []
    for category in CATEGORIES:
        category_root = dataset_root / category
        if not category_root.is_dir():
            continue
        for path in sorted(category_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            rows.append({
                "category": category,
                "source_path": str(path.resolve()),
                "source_sha256": sha256_file(path),
                "status": "pending_source_available",
                "reason": "Source hash recorded; processing/reuse requires an exact matching artifact manifest.",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_processing_manifest.csv"))
    args = parser.parse_args()
    rows = build_manifest(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} source-video rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
