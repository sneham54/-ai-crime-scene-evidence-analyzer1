# Assault recovery and Robbery verification

## Assault

`backend/tracked_results_400/Assault` contains the 30 protected complete MP4/CSV pairs. Recovery code must never overwrite those pairs. The existing `scripts/recover_assault_videos_from_csv.py` renders a recovered MP4 from an existing detection CSV and source frames; it does not load YOLO. It now searches the configured results/export folders and the project's other result, report, run, upload, and scratch folders before treating a CSV referenced by Drive as unavailable. Use `--search-root PATH` more than once for additional local output locations.

The source Assault videos are locally available. The existing combined reconciliation references these 20 CSVs in Drive, under `/content/gdrive/MyDrive/assault_weapon_detection/recovered_20_videos/per_video_csv/`:

```text
Assault042_x264_detections.csv        Assault042_x264A_detections.csv
Assault044_x264_detections.csv        Assault044_x264A_detections.csv
Assault045_x264_detections.csv        Assault045_x264A_detections.csv
Assault046_x264_detections.csv        Assault046_x264A_detections.csv
Assault047_x264_detections.csv        Assault047_x264A_detections.csv
Assault048_x264_detections.csv        Assault048_x264A_detections.csv
Assault049_x264_detections.csv        Assault049_x264A_detections.csv
Assault050_x264_detections.csv        Assault050_x264A_detections.csv
Assault051_x264_detections.csv        Assault051_x264A_detections.csv
Assault052_x264_detections.csv        Assault052_x264A_detections.csv
```

Export them with the existing Colab helper `scripts/export_assault_drive_artifacts_colab.py`. In Colab, upload that script to `/content/`, run `!python /content/export_assault_drive_artifacts_colab.py`, and when `files.upload()` prompts, choose `C:\Users\admin\Downloads\assault_combined_reconciliation.csv`. Approve the Drive mount if prompted, then download the resulting `assault_drive_recovery_export.zip` from the Colab prompt to Windows. From the project root, extract it with `Expand-Archive -LiteralPath .\assault_drive_recovery_export.zip -DestinationPath .\backend\assault_drive_recovery_export`. That ZIP also carries the Drive search inventory and exact manifest paths, allowing files found elsewhere in Drive to supersede the preliminary categories. No Assault inference has been started.

The local orphan `Assault010_x264A_tracked.mp4` fully decodes (3,927 frames at 320x240 and 30 FPS), but has no matching local CSV. `Assault016_x264A_tracked.partial.mp4` decodes to 1,775 frames while its source has 6,613; it has no local CSV. `Assault017_x264_tracked.partial.mp4` cannot be opened by OpenCV (`moov atom not found`); its source has 2,450 frames and there is no local CSV. Keep these artifacts intact until the Drive export is inspected. A partial tracking run cannot be safely continued without its saved detections/tracker state and matching run configuration.

Saved Assault CSVs contain a `track_id` column, and `track_all_videos_corrected.py` calls `model.track` with `persist=True` but does not specify a tracker configuration. The historical files therefore support reporting saved IDs, but they do not verify ByteTrack provenance. Reconstructed video labels IDs only when the CSV contains them and does not assign new IDs.

## Robbery

The configured classifier is a ResNet-18 frame encoder followed by an LSTM and a four-class head (`Assault`, `Robbery`, `Shooting`, `Normal`). `backend/models/video_classifier.pth` is its whole-video activity-classification checkpoint. `backend/models/best.pt` is the nine-class YOLO object detector selected by the current service. `candidate.pt` is not referenced by the current service and its intended run is not documented in the accessible source. `bigru_mil_best.pt` belongs to experimental BiGRU/MIL temporal scoring; its original training/preprocessing contract is not verified and it is not the Robbery CNN-LSTM classifier.

The saved historical report has 62 test samples, with supports 16 Assault, 22 Robbery, 16 Shooting, and 8 Normal. Those counts match `reports/video_split_manifest_corrected.json`'s test split. The report's confusion matrix sums to 62 and has the same class supports, so its aggregate fields are internally consistent with that manifest. The API also compares the reported checkpoint file with the configured local classifier by SHA-256; the existing API test confirms the files match. The report contains no per-video prediction rows, so it cannot be reconciled against individual manifest entries. This is not a fresh evaluation and the displayed accuracy remains historical/unverified.

To verify the Drive VideoMAE run locally, export the complete contents of Google Drive's `robbery_activity_dataset/robbery_videomae/` directory, including its checkpoint file(s), per-video prediction CSV/JSON, evaluation report, run configuration, and any class mapping or preprocessing metadata. Also export the actual held-out test manifest used by the run. If predictions are stored outside that subfolder, include the prediction file(s) and report from `robbery_activity_dataset/` as well. Preserve directory names and provide a SHA-256 listing; do not rerun either classifier. No annotated Robbery video was found in accessible local result folders. The configured local recovery folder `backend/robbery_drive_recovery_export` does not exist yet; the Colab archive will be extracted there for the API to index.

Run `scripts/export_robbery_drive_artifacts_colab.py` in Colab to package the Drive files without inference. Upload the script to `/content/`, run `!python /content/export_robbery_drive_artifacts_colab.py`, approve the Drive mount, and download `robbery_activity_recovery_export.zip`. From the project root, extract it to a separate folder with `Expand-Archive -LiteralPath .\robbery_activity_recovery_export.zip -DestinationPath .\backend\robbery_drive_recovery_export`. The ZIP includes the full VideoMAE subtree, relevant named report/manifest/prediction files elsewhere in the dataset, and a per-file size/SHA-256 inventory.

The authenticated Evidence Library exposes saved activity predictions separately from YOLO object detections and track IDs. Event candidates remain human-review items. Ground-truth event accuracy is unverified until human-reviewed event annotations are available.

If Drive review confirms there are no reusable annotated Robbery videos, the existing YOLO service can create object-detection video output with tracking. Its current implementation explicitly passes the configured `VIDEO_TRACKER` (default `bytetrack.yaml`); any such new run must be recorded as YOLO object detection plus ByteTrack tracking. That output would remain separate from Robbery activity classification and would not establish robbery.

Blank worklists are in `reports/assault_event_human_review_template.csv` (100 source videos), `reports/shooting_event_human_review_template.csv` (100 source videos), and `reports/robbery_test_activity_review_template.csv` (62 held-out manifest entries). All rows are pending. Robbery manifest labels are explicitly marked `dataset_label_unverified`; no human label or prediction was filled in. Duplicate a video row when recording multiple events. No ground-truth event or activity annotation has been completed.
