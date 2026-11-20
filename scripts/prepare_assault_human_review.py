#!/usr/bin/env python3
"""Create a blank human-review worklist from saved Assault event candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_FIELDS = {
    "human_event_present": "",
    "human_event_type": "",
    "human_start_timestamp_seconds": "",
    "human_end_timestamp_seconds": "",
    "human_uncertainty": "",
    "review_status": "not_reviewed",
    "reviewer_id": "",
    "review_notes": "",
    "reviewed_at": "",
}


def prepare(input_path: Path, output_path: Path, *, force: bool = False) -> int:
    if not input_path.is_file():
        raise FileNotFoundError(f"saved event candidates or review template not found: {input_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing human-review file: {output_path}")
    with input_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not original_fields:
        raise ValueError("input CSV has no header")
    fields = original_fields + [name for name in REVIEW_FIELDS if name not in original_fields]
    for row in rows:
        for name, default in REVIEW_FIELDS.items():
            # Existing human labels are preserved exactly, including blanks.
            if name not in row:
                row[name] = default
        if not row.get("review_status"):
            row["review_status"] = "not_reviewed"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    with output_path.open(mode, newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="existing Assault event review template or predicted intervals CSV")
    parser.add_argument("--output", type=Path, default=Path("reports/assault_human_review_worklist.csv"))
    parser.add_argument("--force", action="store_true", help="replace only the specified output file")
    args = parser.parse_args()
    count = prepare(args.input, args.output, force=args.force)
    print(f"rows={count}; all new human labels remain blank; output={args.output}")


if __name__ == "__main__":
    main()
