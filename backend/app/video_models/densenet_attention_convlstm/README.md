# Experimental CCTV classifier

This module implements a four-class **DenseNet-3D → temporal multi-head self-attention → bidirectional ConvLSTM → classifier** pipeline. It is opt-in. The deployed default remains the existing ResNet18 + LSTM checkpoint (`backend/models/video_classifier.pth`), and Flask falls back to it if an explicitly selected DenseNet checkpoint cannot load. YOLOv8 still detects objects and ByteTrack still tracks them separately.

The new class order is `Assault, Shooting, Robbery, Normal` (the order in the requested architecture). The legacy checkpoint order remains `Assault, Robbery, Shooting, Normal`; evaluation maps by class name when comparing both models. The new architecture is initialized from scratch because no compatible DenseNet-3D pretrained weights are present. Existing `best.pt`, `video_classifier.pth`, `video_classifier_v2.pth`, and `bigru_mil_best.pt` were preserved.

## Dataset verified in this workspace

The configured source directory `C:\Users\admin\OneDrive\Documents\crime_video_dataset` currently contains 100 Assault, 100 Shooting, 150 Robbery, and 50 Normal videos. The two additional non-video files are category metadata, not clips. The existing `reports/video_split_manifest_corrected.json` resolves all 400 videos as 276 train, 62 validation, and 62 test entries. Its source-group isolation check passes. These are video-level labels: they do not provide action start/end ground truth or keyframe labels.

YOLO's production class map remains `gun, hammer, helmet, knife, mask, person, pistol, rifle, weapon`. The existing Flask video service already validates and reuses matching tracking artifacts, associates overlapping detections with activity candidates, stores analysis on evidence records, and writes custody events. The React timeline accepts both the legacy event shape and the new `predicted_class/start_timestamp/end_timestamp/keyframes` shape.

## Smoke checks

From the repository root in PowerShell:

```powershell
backend\venv\Scripts\python.exe -m unittest backend.tests.test_densenet_video_pipeline -v
backend\venv\Scripts\python.exe -m compileall -q backend\app\video_models\densenet_attention_convlstm
node --input-type=module -e "import assert from 'node:assert/strict'; import { normalizeVideoEvents } from './frontend/src/utils/videoEvents.js'; const [e] = normalizeVideoEvents({events:[{predicted_class:'Assault',start_timestamp:1,end_timestamp:2}]}); assert.equal(e.start_timestamp_sec,1); assert.equal(e.category,'Assault')"
```

The tests use a small real forward pass and the repository's short event clip. The four logits are raw scores; inference returns softmax probabilities. Candidate intervals are merged by class and time gap, and keyframes preserve source frame indices and FPS-derived timestamps. Flask writes selected source-frame JPEGs inside the per-analysis evidence directory and serves their review URLs through the existing safe evidence endpoint. They remain `pending_review` candidates, not claims that a crime occurred.

## Dataset validation and training

First run the read-only split/path validation (no video decode, model write, or training):

```powershell
backend\venv\Scripts\python.exe -m app.video_models.densenet_attention_convlstm.train `
  --dataset-root 'C:\Users\admin\OneDrive\Documents\crime_video_dataset' `
  --manifest reports\video_split_manifest_corrected.json `
  --output-dir backend\training\densenet_attention_convlstm_run01 --dry-run
```

After reviewing the validation output, launch a new run in a previously unused directory:

```powershell
backend\venv\Scripts\python.exe -m app.video_models.densenet_attention_convlstm.train `
  --dataset-root 'C:\Users\admin\OneDrive\Documents\crime_video_dataset' `
  --manifest reports\video_split_manifest_corrected.json `
  --output-dir backend\training\densenet_attention_convlstm_run01 `
  --epochs 30 --batch-size 2 --learning-rate 0.0001 --workers 0
```

The script uses class-weighted cross-entropy, training-only horizontal flips, mixed precision on CUDA, a fixed source-video split, and validation macro F1 to choose `best_checkpoint.pth`. It writes a training config, per-epoch history, last checkpoint, and best checkpoint. It refuses to overwrite a non-empty new-run directory. To resume, point `--resume` to that run's `last_checkpoint.pth` and keep the same `--output-dir`.

On a Colab Tesla T4 (16 GB), start conservatively with 112-pixel frames, 16 frames per clip, batch size 2, CUDA AMP, and 2 data workers; use gradient accumulation externally only after establishing a stable run. This compact 3D model should fit the T4 at those settings, but exact memory depends on PyTorch/CUDA versions and video dimensions. Monitor the first batch; reduce batch size to 1 or image size to 96 if needed. No full training was started as part of implementation.

## Held-out evaluation and baseline comparison

After training, evaluate only the same manifest test entries and compare the existing model on those exact paths:

```powershell
backend\venv\Scripts\python.exe -m app.video_models.densenet_attention_convlstm.evaluate `
  --dataset-root 'C:\Users\admin\OneDrive\Documents\crime_video_dataset' `
  --manifest reports\video_split_manifest_corrected.json `
  --checkpoint backend\training\densenet_attention_convlstm_run01\best_checkpoint.pth `
  --baseline-checkpoint backend\models\video_classifier.pth `
  --baseline-classes backend\models\class_names.json `
  --output backend\training\densenet_attention_convlstm_run01\test_evaluation.json
```

The report includes accuracy, macro precision/recall/F1, per-class scores, confusion matrices, classification reports, inference time, and parameter count. No improvement is claimed until this command is run on the held-out set.

## Enable through Flask only after training

The existing video analyze endpoint, its authorization checks, and MongoDB evidence persistence are reused. Do not enable this setting until the new checkpoint has passed evaluation and review. In `.env`, set:

```dotenv
VIDEO_CLASSIFIER_BACKEND=densenet_attention_convlstm
VIDEO_DENSENET_MODEL_PATH=training/densenet_attention_convlstm_run01/best_checkpoint.pth
```

The default is `VIDEO_CLASSIFIER_BACKEND=legacy`. When selected, the existing API response still carries `classification`, `activity_events`, YOLO detections, ByteTrack information, evidence IDs, processing state, and human-review status. If the experimental checkpoint is absent or incompatible, the service logs the reason and uses the existing classifier. The checkpoint path can point directly to the reviewed run under `backend/training`; no production model copy or overwrite is needed.
