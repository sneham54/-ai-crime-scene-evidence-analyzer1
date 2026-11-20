# API Reference – AI Crime Scene Evidence Analyzer

Base URL: `http://127.0.0.1:5000`

## Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/login` | Login (demo: investigator / demo123) |
| POST | `/api/auth/biometric` | Simulated biometric gate |

## Upload
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload/image` | Upload crime scene image |
| POST | `/api/upload/video` | Upload surveillance video |

## Detection
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/detection/status` | Check if YOLO model is loaded |
| POST | `/api/detection/analyze` | Run object detection on image |

## Scene Change
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/scene-change/compare` | Compare two scene views (SSIM + alignment) |

## Video
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/video/analyze` | Frame extraction + detection |
| POST | `/api/video/analyze-experimental` | Experimental R3D-18 features + BiGRU/MIL raw logits from an uploaded video filepath |
| POST | `/api/video/analyze-features` | Experimental BiGRU/MIL raw logits from precomputed `(32, 512)` features |

The experimental video endpoint divides the actual video duration into 32 equal segments, samples up to 16 frames per segment, and reports each segment midpoint as `timestamp_sec`. The timestamps are references only, not incident boundaries. It requires pretrained torchvision R3D-18 Kinetics-400 weights; it returns an error if the weights cannot be loaded and never falls back to random initialization. Preprocessing and BiGRU/MIL checkpoint fidelity are unverified because the original training setup is unavailable. Scores are raw logits, not probabilities or confirmed findings.

## Reports
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/report/generate` | Generate PDF forensic report |
| GET | `/api/report/download/<name>` | Download PDF |
| GET | `/api/report/list` | List recent reports |

## Other
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chatbot/query` | Case-grounded AI chatbot |
| GET | `/api/dashboard/summary` | Dashboard stats |
# Saved category evidence catalog

All endpoints below require a valid bearer token. They read saved files only and do not start model inference.

- `GET /api/evidence/categories` — artifact availability and counts for Assault, Robbery, and Shooting.
- `GET /api/evidence/<category>/videos` — source-video inventory, result availability, and saved processing status.
- `GET /api/evidence/<category>/videos/<video_id>` — saved result sets, normalized timestamps, confidence, bounding boxes, class labels, and track IDs where present.
- `GET /api/evidence/<category>/videos/<video_id>/media?kind=source|annotated&artifact_id=<id>` — stream a verified source or annotated video.
- `GET /api/evidence/<category>/status` — status records as stored; missing status remains missing.
- `GET /api/evidence/<category>/reports` — report/evaluation artifact inventory and any configured historical Robbery classifier metrics.
- `GET /api/evidence/<category>/reports/<report_id>` — download an allowlisted saved report file from that category's configured results directory.
- `GET /api/evidence/<category>/videos/<video_id>/report-payload?artifact_id=<id>` — a human-review-marked report payload for use with the existing `POST /api/report/generate-json` endpoint.

Categories are `assault`, `robbery`, and `shooting`. Evidence directories and status manifests can be supplied through `EVIDENCE_SOURCE_DIR`, `ASSAULT_RESULTS_DIR`, `ASSAULT_TRACKING_STATUS_PATH`, `ROBBERY_RESULTS_DIR`, `ROBBERY_TRACKING_STATUS_PATH`, `SHOOTING_RESULTS_DIR`, and `SHOOTING_TRACKING_STATUS_PATH` in `.env`. Relative paths resolve from the repository root. The evidence catalog exposes saved source videos, detections, annotated media, processing status, and investigator reports. Model evaluation is not part of the application UI or API; offline evaluation scripts remain developer tooling.

Event evidence can be extracted for an uploaded analysis with `POST /api/video/event-evidence/<analysis_id>` and `start_timestamp_sec` / `end_timestamp_sec`. Tracking is reused from category artifacts and never rerun by this endpoint. Packages remain `unverified_candidate` pending review and are listed at `GET /api/evidence/event-evidence/<category>`; the original Shooting aliases remain supported.
