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

## 9. Optional MongoDB

```powershell
docker-compose up -d
```

Demo mode works without MongoDB.

Group: Jahnavi V · Rakshitha J M · Sneha M  
Guide: Prof. Ashfaq Ahmed R
