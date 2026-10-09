"""
Improved YOLOv8 training for crime-scene evidence (gun, hammer, helmet, knife, ...).
Same method as before (Ultralytics YOLOv8) - only stronger settings.

Run:  python train_weapon_model.py "C:\\path\\to\\weapon-detection.yolov8"
Then copy runs/detect/weapon_forensic_v2/weights/best.pt -> backend/models/best.pt
"""
import sys
from pathlib import Path
from ultralytics import YOLO

DATASET_ROOT = r"C:\Users\admin\Documents\ai crime\weapon-detection.yolov8"
BASE_WEIGHTS = "yolov8s.pt"   # n -> s gives a big accuracy jump; use yolov8m.pt if you have a GPU
IMG_SIZE = 832                # bigger input = small knives/guns detected better


def find_data_yaml(root):
    root = Path(root)
    for p in [root / "data.yaml", root / "dataset.yaml", *root.rglob("data.yaml")]:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No data.yaml in {root}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DATASET_ROOT
    data_yaml = find_data_yaml(root)
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=data_yaml,
        epochs=150,
        patience=30,
        imgsz=IMG_SIZE,
        batch=-1,              # auto batch size for your GPU memory
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.0005,
        # augmentation tuned for indoor/outdoor crime scenes
        mosaic=1.0,
        close_mosaic=15,       # last 15 epochs on real-looking images
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,   # lighting variation (dark scenes)
        label_smoothing=0.0,
        cache=True,
        amp=True,
        project="runs/detect",
        name="weapon_forensic_v2",
        exist_ok=True,
        plots=True,
    )
    # Validate with test-time augmentation and print per-class mAP
    best = YOLO("runs/detect/weapon_forensic_v2/weights/best.pt")
    m = best.val(data=data_yaml, imgsz=IMG_SIZE, augment=True, conf=0.001, iou=0.6)
    print("mAP50:", round(m.box.map50, 4), "mAP50-95:", round(m.box.map, 4))
    for i, name in best.names.items():
        try:
            print(f"  {name:12s} AP50={m.box.ap50[i]:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
