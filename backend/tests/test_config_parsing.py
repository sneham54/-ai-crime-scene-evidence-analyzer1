"""Environment parsing tests for the forensic video configuration."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

from app.config import _bool, _float


class ConfigParsingTests(unittest.TestCase):
    def test_roi_and_class_threshold_defaults(self):
        backend_dir = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHON_DOTENV_DISABLED"] = "1"
        config_names = ["VIDEO_PERSON_ROI_ENABLED"] + [
            f"VIDEO_CLASS_CONFIDENCE_{name}"
            for name in (
                "GUN", "HAMMER", "HELMET", "KNIFE", "MASK", "PERSON", "PISTOL", "RIFLE", "WEAPON",
            )
        ]
        for name in config_names:
            environment.pop(name, None)
        check_defaults = (
            "from app.config import Config; "
            "assert Config.VIDEO_PERSON_ROI_ENABLED is False; "
            "assert Config.VIDEO_CLASS_CONFIDENCE_GUN == 0.30; "
            "assert all(getattr(Config, 'VIDEO_CLASS_CONFIDENCE_' + n) == 0.25 "
            "for n in 'HAMMER HELMET KNIFE MASK PERSON PISTOL RIFLE WEAPON'.split())"
        )
        result = subprocess.run(
            [sys.executable, "-c", check_defaults], cwd=backend_dir,
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_boolean_parsing(self):
        for value in ("1", " true ", "TRUE", "yes", "Yes"):
            with self.subTest(value=value):
                self.assertTrue(_bool(value, False))
        for value in ("0", "false", "no", "invalid"):
            with self.subTest(value=value):
                self.assertFalse(_bool(value, True))
        self.assertTrue(_bool(None, True))
        self.assertFalse(_bool(None, False))

    def test_float_parsing_and_clamping(self):
        self.assertEqual(_float("0.7", 0.25), 0.7)
        self.assertEqual(_float("2.0", 0.25), 1.0)
        self.assertEqual(_float("-0.5", 0.25), 0.0)
        self.assertEqual(_float("not-a-number", 0.25), 0.25)
        self.assertEqual(_float(None, 0.25), 0.25)

if __name__ == "__main__":
    unittest.main()
