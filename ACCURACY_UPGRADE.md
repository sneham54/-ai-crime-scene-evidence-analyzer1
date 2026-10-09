# Detection accuracy upgrade (YOLOv8 - same method)

1. Retrain: `python train_weapon_model.py "<dataset path>"` (yolov8s, 832px, 150 epochs, tuned augmentation).
   Copy the new best.pt to backend/models/best.pt.
2. Copy yolo_accuracy_patch.py to backend/app/services/ and call enhanced_detect() inside YOLOService.detect().
3. In .env set: YOLO_IMAGE_SIZE=832 and YOLO_CONFIDENCE=0.25 (was 0.15 -> too many false boxes).
4. Tune CLASS_CONF per class using the per-class AP printed after training.

Dataset tips (biggest gain): add 300+ images per class, include hard negatives
(images with no weapons, phones/remotes/tools that look like guns/knives), and
fix wrong labels - check runs/detect/weapon_forensic_v2/confusion_matrix.png.

NOTE: Model weights (*.pt) and the datasets folder were left out of this ZIP to keep it small.
Copy your existing backend/models/best.pt (or retrain) back into place before running.
