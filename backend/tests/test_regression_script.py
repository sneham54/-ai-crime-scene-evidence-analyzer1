import os
import subprocess
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np


class RegressionScriptTests(unittest.TestCase):
    def test_runs_on_dummy_image(self):
        """The regression helper accepts a valid image and prints its report."""
        fixture_dir = Path(__file__).resolve().parent / f"regression_fixture_{os.getpid()}"
        fixture_dir.mkdir()
        img_path = fixture_dir / "blank.jpg"
        try:
            self.assertTrue(cv2.imwrite(str(img_path), 255 * np.ones((10, 10, 3), dtype=np.uint8)))

            project_root = Path(__file__).resolve().parents[2]
            script = project_root / "run_regression.py"
            result = subprocess.run(
                [sys.executable, str(script), str(img_path)],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("=== REGRESSION REPORT ===", result.stdout)
            self.assertIn("RAW   |", result.stdout)
            self.assertIn("FINAL |", result.stdout)
        finally:
            img_path.unlink(missing_ok=True)
            fixture_dir.rmdir()
