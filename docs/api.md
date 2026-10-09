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
