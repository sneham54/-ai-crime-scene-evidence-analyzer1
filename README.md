# AI Crime Scene Evidence Analyzer

**Don Bosco Institute of Technology, Bengaluru**  
CSE (AI & ML) · Major Project (BCI785) · Group 09 · 2026-27 (ODD)

Matches official synopsis. **Fingerprints removed.**

## Features
- YOLOv8 object detection (gun, knife, weapon, person, …)
- Bounding boxes + risk score
- Scene Change Detection (alignment, SSIM, heatmap)
- Video analysis with timestamps
- Chain-of-custody logging
- AI forensic chatbot
- PDF investigation reports

## 1. Place your trained model

Copy your weights to:

```
backend/models/best.pt
```

Video analysis requires this custom checkpoint and validates its trained class names; it will not substitute a COCO detector for video evidence.

Four-class video analysis also requires a trained video-classifier checkpoint and its generated metadata. The video classifier does not modify the YOLO weights.

### Select the video-classifier checkpoint

`backend/.env` reads `VIDEO_MODEL_PATH`; the default remains `models/video_classifier.pth`. To opt in to the candidate, set `VIDEO_MODEL_PATH=models/video_classifier_v2.pth` and restart the backend. To roll back, restore `models/video_classifier.pth`. The repo-root form `backend/models/video_classifier_v2.pth` is also accepted. Both checkpoints use the existing strict CNNLSTM loader and the class order `Assault`, `Robbery`, `Shooting`, `Normal`.

The saved same-test-set evaluation (62 samples, seed 42, shared split manifest) scores the original at 85.48% accuracy, 0.8389 macro F1, and 0.9143 Shooting F1; v2 scores 83.87%, 0.8252, and 0.8485 respectively. Keep the original selected unless a later evaluation justifies promoting the candidate.

## 2. Run Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

Check model:

```
http://127.0.0.1:5000/api/detection/status
```

Must show `"status": "ready"`.

## 3. Run Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open: **http://127.0.0.1:5173**

Login as investigator: `investigator` / `demo123` → Dashboard  
Login as officer: `officer` / `demo123` → Dashboard

Both accounts use the same sign-in screen; the account's role is returned by the backend.

## 4. Train the video event classifier

Video analysis requires a trained MobileNetV3 feature extractor + LSTM classifier. Without these artifacts, `/api/video/analyze` returns `503 video_classifier_not_trained`; it does not fall back to an untrained heuristic. Arrange labeled videos in class folders (the normal folder may be named `NormalVideos`):

```
crime_video_dataset/
	Assault/
	Robbery/
	Shooting/
	NormalVideos/
```

Train from the backend directory:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python train_video_classifier.py --data-dir "D:\datasets\crime_video_dataset" --epochs 25
```

The script splits source videos per class before extracting frames, prints each class count in all splits, uses class-weighted loss, selects the best checkpoint by validation Macro F1, early-stops, and reports held-out accuracy, per-class precision/recall/F1, Macro F1, Weighted F1, and a confusion matrix. It writes `backend/models/video_classifier.pth`, `class_names.json`, `video_preprocessing.json`, `video_split_manifest.json`, and `video_classifier_evaluation.json`. The model samples 16 frames per video at 224x224; the MobileNetV3 backbone uses ImageNet pretrained weights by default. Use `--no-pretrained` if downloading pretrained weights is unavailable.

The API returns all four softmax probabilities and model confidence. The UI reports `Uncertain` when top-class confidence is below `VIDEO_CLASSIFIER_MIN_CONFIDENCE` (default `0.50`), while retaining the top class separately. Softmax values are not calibrated; review held-out test results before relying on predictions.

## 5. Ollama Chatbot

Install [Ollama](https://ollama.com/) and download the default model:

```powershell
ollama pull llama3.2
```

Keep Ollama running while using the chatbot. Set `OLLAMA_HOST` or `OLLAMA_MODEL` in `backend/.env` to use a different local Ollama server or model. If Ollama is unavailable, the chatbot falls back to its built-in forensic summaries.

## 6. If blank page

1. F12 → Application → Local Storage → Clear
2. Ctrl+Shift+R
3. Or open http://127.0.0.1:5173/login

## 7. If Analysis failed

1. Backend terminal must show: `✓ YOLO model loaded`
2. Open http://127.0.0.1:5000/api/detection/status
3. Put the trained YOLO weights at `backend/models/best.pt` and train the video classifier using the labeled video folders above.
4. Restart: `python run.py`

## 8. Scene Change tab

Upload View A + View B → Run Scene Change Detection → change score, SSIM, heatmap.

## 9. Docker Compose

```powershell
docker compose config
docker compose up --build -d
docker compose ps
```

Open the dashboard at **http://localhost:5173**. The frontend proxies `/api` requests to Flask inside Compose, and Flask connects to `mongodb:27017` using the existing `crime_evidence_db` database. MongoDB data and backend uploads use persistent named volumes, so restarting containers does not reset them.

To inspect service output:

```powershell
docker compose logs -f frontend backend mongodb
```

Set `MONGO_DB_NAME`, `SECRET_KEY`, and `JWT_SECRET_KEY` in a local, untracked `.env` file when overriding defaults. Copy `.env.example` as a starting point. Never put production credentials in the Compose file. The dashboard shows MongoDB-backed counts as unavailable while the database cannot be reached; it does not substitute training catalogs or local artifacts for application records.

The optional read-only dataset and saved tracking-result mounts in Compose expose local `datasets/` and `backend/tracked_results_*` content to the Evidence Library catalog. These inventory counts remain separate from persisted application evidence and analysis counts.

Group: Jahnavi V · Rakshitha J M · Sneha M  
Guide: Prof. Ashfaq Ahmed R
# Dataset catalog and Ollama connection (Windows Docker Desktop)

The backend catalog reads the existing source dataset through a read-only bind mount. Set `SOURCE_DATASET_PATH` in the repository `.env` to the host folder containing `Assault`, `Shooting`, `Robbery`, and `Normal`; the current local dataset is `C:/Users/admin/OneDrive/Documents/crime_video_dataset`. Docker Compose mounts it at `/data/source-videos:ro`. Do not copy the dataset into this repository. A missing mount is reported as unavailable, not as an empty inventory.

Ollama stays on its configured host and uses the existing chatbot API. Native Flask defaults to `OLLAMA_BASE_URL=http://127.0.0.1:11434`; Docker Compose defaults to `http://host.docker.internal:11434`. `OLLAMA_MODEL` selects the already installed model, and `/api/chatbot/health` reports reachability and model availability without pulling anything. Install a missing model explicitly on the Ollama host only if desired. On Windows, if Ollama only binds to loopback and Docker cannot reach it, set `OLLAMA_HOST=0.0.0.0:11434` in Ollama's environment and restart Ollama. Restrict Windows Firewall access to trusted/local Docker traffic; do not expose port 11434 to untrusted networks.
