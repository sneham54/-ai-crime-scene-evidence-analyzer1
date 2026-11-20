"""Experimental inference for the saved BiGRU/MIL checkpoint.

The original training class and R3D-18 preprocessing were not included with the
project. This candidate model matches the checkpoint's parameterized layers,
but is not claimed to reproduce the original training implementation.
"""
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn


SEGMENT_COUNT = 32
FEATURE_DIM = 512
PROJECTION_DIM = 256
HIDDEN_DIM = 128


class ExperimentalModelUnavailableError(FileNotFoundError):
    """The checkpoint cannot be safely loaded or does not match the candidate model."""


class InvalidFeatureInputError(ValueError):
    """The supplied feature matrix is not valid model input."""


class BigruMILCandidate(nn.Module):
    """Checkpoint-compatible candidate; ReLU choices are unverified assumptions."""

    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, PROJECTION_DIM), nn.ReLU())
        self.gru = nn.GRU(
            input_size=PROJECTION_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        projected = self.projection(features)
        sequence, _ = self.gru(projected)
        # Keep these as raw per-segment logits. No sigmoid or MIL pooling is known.
        return self.scorer(sequence).squeeze(-1)


class BigruMILModel:
    """Safe CPU checkpoint loader and validated experimental feature scorer."""

    _cache = {}
    _cache_lock = threading.Lock()

    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise ExperimentalModelUnavailableError(
                f"BiGRU/MIL checkpoint not found: {self.checkpoint_path}"
            )
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError as error:
            raise ExperimentalModelUnavailableError(
                "This PyTorch version does not support safe weights_only checkpoint loading."
            ) from error
        except Exception as error:
            raise ExperimentalModelUnavailableError(
                f"BiGRU/MIL checkpoint could not be safely loaded: {error}"
            ) from error

        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ExperimentalModelUnavailableError("Checkpoint is missing model_state_dict.")
        if checkpoint.get("input_dim") != FEATURE_DIM or checkpoint.get("hidden_dim") != HIDDEN_DIM:
            raise ExperimentalModelUnavailableError(
                "Checkpoint input_dim/hidden_dim metadata does not match the expected 512/128 values."
            )

        state = checkpoint["model_state_dict"]
        expected_shapes = {
            "projection.0.weight": (PROJECTION_DIM, FEATURE_DIM),
            "projection.0.bias": (PROJECTION_DIM,),
            "gru.weight_ih_l0": (3 * HIDDEN_DIM, PROJECTION_DIM),
            "gru.weight_hh_l0": (3 * HIDDEN_DIM, HIDDEN_DIM),
            "gru.bias_ih_l0": (3 * HIDDEN_DIM,),
            "gru.bias_hh_l0": (3 * HIDDEN_DIM,),
            "gru.weight_ih_l0_reverse": (3 * HIDDEN_DIM, PROJECTION_DIM),
            "gru.weight_hh_l0_reverse": (3 * HIDDEN_DIM, HIDDEN_DIM),
            "gru.bias_ih_l0_reverse": (3 * HIDDEN_DIM,),
            "gru.bias_hh_l0_reverse": (3 * HIDDEN_DIM,),
            "scorer.0.weight": (128, 2 * HIDDEN_DIM),
            "scorer.0.bias": (128,),
            "scorer.3.weight": (1, 128),
            "scorer.3.bias": (1,),
        }
        if set(state) != set(expected_shapes):
            raise ExperimentalModelUnavailableError("Checkpoint state-dict keys do not match the candidate model.")
        for name, expected_shape in expected_shapes.items():
            tensor = state[name]
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape:
                raise ExperimentalModelUnavailableError(
                    f"Checkpoint tensor {name} does not have expected shape {expected_shape}."
                )
            if not torch.isfinite(tensor).all():
                raise ExperimentalModelUnavailableError(f"Checkpoint tensor {name} contains non-finite values.")

        self.model = BigruMILCandidate()
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ExperimentalModelUnavailableError(
                f"Checkpoint does not strictly load into the candidate architecture: {error}"
            ) from error
        self.model.cpu().eval()

    @classmethod
    def get(cls, checkpoint_path):
        key = str(Path(checkpoint_path).resolve())
        with cls._cache_lock:
            if key not in cls._cache:
                cls._cache[key] = cls(key)
            return cls._cache[key]

    @staticmethod
    def validate_features(features):
        try:
            values = np.asarray(features)
        except (TypeError, ValueError) as error:
            raise InvalidFeatureInputError("features must be a numeric array with shape (32, 512).") from error
        if values.shape != (SEGMENT_COUNT, FEATURE_DIM):
            raise InvalidFeatureInputError("features must have shape (32, 512).")
        if values.dtype.kind not in "fiu":
            raise InvalidFeatureInputError("features must contain numeric values, not strings or booleans.")
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise InvalidFeatureInputError("features must contain only finite values.")
        return torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0)

    def infer(self, features):
        tensor = self.validate_features(features)
        with torch.inference_mode():
            scores = self.model(tensor)
        if tuple(scores.shape) != (1, SEGMENT_COUNT):
            raise RuntimeError("Candidate model returned an unexpected score shape.")
        scores = scores[0].cpu()
        if not torch.isfinite(scores).all():
            raise RuntimeError("Candidate model produced non-finite raw logits.")
        return [float(value) for value in scores.tolist()]
