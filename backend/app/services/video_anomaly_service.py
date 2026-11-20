"""Optional inference adapter for a calibrated normal-only frame autoencoder."""
import json
import hashlib
import threading
from pathlib import Path

import torch

from training.video_anomaly import ConvAutoencoder, preprocess_bgr_frame


class VideoAnomalyUnavailableError(FileNotFoundError):
    """The autoencoder or its calibrated threshold metadata is unavailable."""


class VideoAnomalyScorer:
    _cache = {}
    _lock = threading.Lock()

    def __init__(self, checkpoint_path, config_path):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config_path = Path(config_path).resolve()
        if not self.checkpoint_path.is_file():
            raise VideoAnomalyUnavailableError(f"Autoencoder has not been trained: {self.checkpoint_path}")
        if not self.config_path.is_file():
            raise VideoAnomalyUnavailableError(f"Calibrated threshold config is missing: {self.config_path}")
        try:
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise VideoAnomalyUnavailableError(f"Autoencoder artifacts could not be loaded: {error}") from error
        if checkpoint.get("architecture") != "conv_autoencoder_v1":
            raise VideoAnomalyUnavailableError("Autoencoder checkpoint architecture is unsupported")
        if self.config.get("status") not in {"trained_thresholds_calibrated_test_not_evaluated", "evaluated"}:
            raise VideoAnomalyUnavailableError("Autoencoder thresholds have not been calibrated")
        configured_digest = self.config.get("checkpoint_sha256")
        if configured_digest:
            digest = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
            if digest != configured_digest:
                raise VideoAnomalyUnavailableError("Autoencoder checkpoint does not match the calibrated threshold config")
        self.image_size = int(checkpoint.get("image_size", 0))
        self.frame_threshold = float(self.config["frame_anomaly_threshold"])
        self.video_threshold = float(self.config["video_anomaly_threshold"])
        self.video_score_frame_cap = int(self.config.get("sampled_frames_per_video_max", 16))
        if self.image_size < 32 or self.frame_threshold < 0 or self.video_threshold < 0 or self.video_score_frame_cap < 1:
            raise VideoAnomalyUnavailableError("Autoencoder metadata contains invalid image size or thresholds")
        self.model = ConvAutoencoder()
        try:
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        except (KeyError, RuntimeError, TypeError) as error:
            raise VideoAnomalyUnavailableError(f"Autoencoder checkpoint does not match its architecture: {error}") from error
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @classmethod
    def get(cls, checkpoint_path, config_path):
        key = (str(Path(checkpoint_path).resolve()), str(Path(config_path).resolve()))
        with cls._lock:
            if key not in cls._cache:
                cls._cache[key] = cls(*key)
            return cls._cache[key]

    def score_frame(self, bgr_frame):
        frame = preprocess_bgr_frame(bgr_frame, self.image_size)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            reconstruction = self.model(tensor)
            score = float((reconstruction - tensor).pow(2).mean().item())
        return {
            "anomaly_score": score,
            "anomaly_threshold": self.frame_threshold,
            "potential_anomaly": score > self.frame_threshold,
            "score_type": "frame_mean_squared_reconstruction_error",
            "verification_status": "unverified",
        }
