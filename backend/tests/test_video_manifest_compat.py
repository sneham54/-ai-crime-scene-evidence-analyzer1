import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_video_eval_manifest import CLASSES, SPLITS, split_records, validate_manifest


class VideoManifestCompatibilityTests(unittest.TestCase):
    def test_evaluator_fields_paths_classes_and_source_groups_are_valid(self):
        records = []
        for class_name in CLASSES:
            for index in range(10):
                content = f"{class_name}-{index}".encode()
                alias = f"{class_name.casefold()}:clip{index:02d}"
                records.append({"path": f"{class_name}/clip{index:02d}_x264.mp4",
                                "label": class_name,
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "source_group": alias})
        # Identical bytes in different classes become one component.
        records[10]["sha256"] = records[0]["sha256"]

        assignments = split_records(records, seed=42)
        manifest = {"class_names": list(CLASSES), "splits": {
            split: [{"class": row["label"], "video": row["path"],
                     "source_group": row["source_group"]} for row in assignments[split]]
            for split in SPLITS}}
        with patch.object(Path, "is_file", return_value=True):
            counts = validate_manifest(manifest, Path("/fixture"), expected_total=40)
        self.assertEqual(sum(sum(values.values()) for values in counts.values()), 40)
        for split in SPLITS:
            self.assertEqual(set(counts[split]), set(CLASSES))
            for row in manifest["splits"][split]:
                self.assertEqual(set(row), {"class", "video", "source_group"})

        owner = {}
        for split, rows in manifest["splits"].items():
            for row in rows:
                self.assertEqual(owner.setdefault(row["source_group"], split), split)

    def test_split_is_deterministic(self):
        records = []
        for class_name in CLASSES:
            for index in range(10):
                records.append({"path": f"{class_name}/{index}.mp4", "label": class_name,
                                "sha256": f"{class_name}-{index}",
                                "source_group": f"{class_name.casefold()}:{index}"})
        first = split_records(records, seed=91)
        second = split_records(records, seed=91)
        self.assertEqual([[r["path"] for r in first[split]] for split in SPLITS],
                         [[r["path"] for r in second[split]] for split in SPLITS])


if __name__ == "__main__":
    unittest.main()
