import io
import sys
import unittest
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
TEMP_ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from app.video_models.densenet_attention_convlstm.classifier import DenseNetAttentionConvLSTM, load_checkpoint
from app.video_models.densenet_attention_convlstm.config import ModelConfig, PreprocessConfig
from app.video_models.densenet_attention_convlstm.event_extraction import extract_event_keyframes, merge_prediction_events
from app.video_models.densenet_attention_convlstm.preprocessing import iter_video_clips, probe_video
from app.video_models.densenet_attention_convlstm.metrics import classification_metrics
from app.video_models.densenet_attention_convlstm.service_adapter import DenseNetVideoClassifier


def small_model():
    return DenseNetAttentionConvLSTM(ModelConfig(growth_rate=4, block_layers=(1, 1),
        initial_features=8, attention_heads=2, convlstm_hidden=4, dropout=0))


class DenseNetVideoPipelineTests(unittest.TestCase):
    def test_real_3d_densenet_attention_convlstm_dimensions_and_checkpoint(self):
        torch.set_num_threads(1)
        model = small_model().eval()
        video = torch.randn(2, 4, 3, 32, 32)
        features = model.backbone(video.permute(0, 2, 1, 3, 4))
        self.assertEqual(tuple(features.shape[:3]), (2, 12, 4))
        attention, _ = model.attention(torch.randn(2, 4, 12))
        self.assertEqual(tuple(attention.shape), (2, 4, 12))
        convlstm = model.convlstm(torch.randn(2, 4, 12, 8, 8))
        self.assertEqual(tuple(convlstm.shape), (2, 4, 8, 8, 8))
        self.assertEqual(tuple(model(video).shape), (2, 4))
        checkpoint = io.BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "model_config": model.config.to_dict(),
            "classes": ["Assault", "Shooting", "Robbery", "Normal"]}, checkpoint)
        checkpoint.seek(0)
        loaded, _ = load_checkpoint(checkpoint)
        self.assertEqual(tuple(loaded(video).shape), (2, 4))

    def test_stream_preprocessing_timestamps_and_event_keyframes(self):
        path = BACKEND / "real_data_pilots" / "shooting" / "Shooting002_x264_20261110_193054" / "events" / "860a5892c1bc1eecebf8e10e" / "original_clip.mp4"
        info = probe_video(path)
        self.assertEqual(info["frame_count"], 306)
        cfg = PreprocessConfig(clip_length=4, image_size=16, clip_stride=2)
        clips = list(iter_video_clips(path, cfg))
        self.assertEqual(clips[0].frame_indices, [0, 1, 2, 3])
        self.assertEqual(clips[1].frame_indices, [2, 3, 4, 5])
        self.assertEqual(clips[-1].frame_indices[-1], 305)
        self.assertAlmostEqual(clips[1].start_timestamp_sec, 2 / info["fps"])
        self.assertEqual(tuple(clips[0].tensor.shape), (4, 3, 16, 16))
        predictions = [
            {"predicted_class": "Assault", "confidence": .8, "start_timestamp_sec": 1.0,
             "end_timestamp_sec": 1.4, "representative_frame_index": 30},
            {"predicted_class": "Assault", "confidence": .9, "start_timestamp_sec": 1.5,
             "end_timestamp_sec": 1.9, "representative_frame_index": 45},
            {"predicted_class": "Normal", "confidence": .99, "start_timestamp_sec": 2.0,
             "end_timestamp_sec": 2.2, "representative_frame_index": 60},
        ]
        events = merge_prediction_events(predictions, "vid-1", path, .7, .2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_timestamp"], 1.0)
        self.assertAlmostEqual(events[0]["end_timestamp"], 1.9)
        self.assertEqual(events[0]["verification_status"], "pending_review")
        events = extract_event_keyframes(path, events, predictions, TEMP_ROOT)
        self.assertTrue(events[0]["keyframes"])
        self.assertEqual(events[0]["keyframes"][0]["frame_index"], 45)
        keyframe = events[0]["keyframes"][0]
        self.assertAlmostEqual(keyframe["timestamp_sec"], 45 / info["fps"], places=3)
        self.assertEqual(keyframe["timestamp_mapping_status"], info["timestamp_mapping_status"])
        self.assertEqual(keyframe["timestamps_verified"], info["timestamps_verified"])
        self.assertTrue(Path(events[0]["keyframes"][0]["path"]).is_file())
        Path(events[0]["keyframes"][0]["path"]).unlink(missing_ok=True)

    def test_invalid_video_is_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_video_clips(TEMP_ROOT / "missing_video_pipeline_smoke.mp4",
                                  PreprocessConfig(clip_length=4, image_size=16)))

    def test_metrics_and_existing_service_event_contract(self):
        scores = classification_metrics([0, 1, 2, 3], [0, 1, 0, 3])
        self.assertAlmostEqual(scores["accuracy"], .75)
        self.assertEqual(len(scores["confusion_matrix"]), 4)
        payload = DenseNetVideoClassifier._to_contract({
            "classification": {"label": "Assault", "confidence": .8, "probabilities": {"Assault": .8},
                               "classification_mode": "experimental_densenet3d_attention_biconvlstm"},
            "clip_predictions": [{"predicted_class": "Assault", "confidence": .8,
                "probabilities": {"Assault": .8}, "start_timestamp_sec": 1.0, "end_timestamp_sec": 2.0,
                "representative_frame_index": 45, "representative_timestamp_sec": 1.5}],
            "events": [{"event_id": "id", "video_id": "a" * 32,
                "source_video": "private/path/video.mp4", "predicted_class": "Assault",
                "confidence": .8, "start_timestamp": 1.0, "end_timestamp": 2.0,
                "representative_frame": "C:/private/original_frame_000045_a.jpg",
                "verification_status": "pending_review", "keyframes": [{"frame_index": 45,
                    "timestamp_sec": 1.5, "path": "C:/private/original_frame_000045_a.jpg"}]}],
            "video_metadata": {"frame_count": 100, "fps": 30, "duration_sec": 3.3},
            "sampling_coverage": {"sampled_frame_count": 16, "covered_entire_video": True},
            "inference_seconds_per_clip": .01,
        })
        event = payload["activity_events"][0]
        self.assertEqual(event["category"], "Assault")
        self.assertEqual(event["start_timestamp_sec"], 1.0)
        self.assertEqual(event["verification_status"], "pending_review")
        self.assertEqual(event["source_video"], "video.mp4")
        self.assertEqual(event["keyframes"][0]["url"], f"/api/video/evidence/{'a' * 32}/original_frame_000045_a.jpg")
        self.assertEqual(event["representative_frame"], f"/api/video/evidence/{'a' * 32}/original_frame_000045_a.jpg")


if __name__ == "__main__":
    unittest.main()
