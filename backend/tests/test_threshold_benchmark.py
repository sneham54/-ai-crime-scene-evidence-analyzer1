import subprocess
import sys
from pathlib import Path
import unittest
import uuid


class ThresholdBenchmarkTests(unittest.TestCase):
    def test_empty_folder_writes_empty_reports(self):
        project_root = Path(__file__).resolve().parents[2]
        output_dir = Path(__file__).resolve().parent / f"empty_benchmark_{uuid.uuid4().hex}"
        output_dir.mkdir()
        try:
            result = subprocess.run(
                [sys.executable, str(project_root / "threshold_benchmark.py"), str(output_dir)],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("Benchmark completed", result.stdout)
            self.assertIn("no supported video files", result.stdout)
            self.assertTrue((output_dir / "benchmark_report.csv").is_file())
            self.assertTrue((output_dir / "benchmark_report.json").is_file())
            with (output_dir / "benchmark_report.csv").open(encoding="utf-8") as csv_file:
                self.assertEqual(
                    csv_file.readline().strip(),
                    "video,class_id,class_name,detections,avg_conf,min_conf,max_conf,roi_detections,time_sec,frames_analyzed,false_positives,missed_detections",
                )
            self.assertEqual((output_dir / "benchmark_report.json").read_text(encoding="utf-8"), "[]")
        finally:
            for report_name in ("benchmark_report.csv", "benchmark_report.json"):
                (output_dir / report_name).unlink(missing_ok=True)
            output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
