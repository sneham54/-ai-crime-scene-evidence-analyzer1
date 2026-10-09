"""
Drop-in accuracy upgrade for backend/app/services/yolo_service.py  (still YOLOv8 only).

Adds:
 1. Test-time augmentation (augment=True)       -> fewer missed objects
 2. Tiled (sliced) inference for large photos   -> small knives/guns found
 3. Weighted box fusion of full-image + tiles   -> cleaner, more accurate boxes
 4. Per-class confidence thresholds             -> fewer false alarms
 5. Min box size filter                          -> removes noise boxes

HOW TO USE: in YOLOService.detect(), replace the `results = YOLOService._model(...)`
block and the detections loop with:

    from app.services.yolo_accuracy_patch import enhanced_detect
    detections, annotated = enhanced_detect(
        YOLOService._model, image_path,
        imgsz=current_app.config["YOLO_IMAGE_SIZE"],
        base_conf=conf_threshold,
        category_fn=self._map_to_forensic_category,
    )
    # then save `annotated` exactly like before with save_annotated_image(...)
"""
import cv2
import numpy as np

# Raise for classes that give false positives, lower for ones that get missed.
CLASS_CONF = {"gun": 0.35, "knife": 0.30, "hammer": 0.30, "helmet": 0.40}
MIN_BOX_AREA_RATIO = 0.0004   # ignore boxes smaller than 0.04% of image
TILE = 640
OVERLAP = 0.25


def _predict(model, img, imgsz, conf, offset=(0, 0)):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.5,
                      augment=True, agnostic_nms=False, verbose=False)[0]
    out = []
    if r.boxes is None:
        return out
    ox, oy = offset
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(b.conf[0]), int(b.cls[0])])
    return out


def _tiles(h, w):
    step = int(TILE * (1 - OVERLAP))
    for y in range(0, max(h - TILE, 0) + 1, step):
        for x in range(0, max(w - TILE, 0) + 1, step):
            yield x, y
    # cover right / bottom edges
    if w > TILE:
        for y in range(0, max(h - TILE, 0) + 1, step):
            yield w - TILE, y
    if h > TILE:
        for x in range(0, max(w - TILE, 0) + 1, step):
            yield x, h - TILE


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def _fuse(boxes, iou_thr=0.55):
    """Weighted box fusion per class: merges duplicates from full image + tiles."""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    clusters = []
    for b in boxes:
        for c in clusters:
            if c[0][5] == b[5] and _iou(c[0], b) > iou_thr:
                c.append(b)
                break
        else:
            clusters.append([b])
    fused = []
    for c in clusters:
        w = np.array([x[4] for x in c])
        coords = np.array([x[:4] for x in c])
        xyxy = (coords * w[:, None]).sum(0) / w.sum()
        conf = float(min(1.0, w.max() * (1 + 0.05 * (len(c) - 1))))  # agreement bonus
        fused.append([*xyxy.tolist(), conf, c[0][5]])
    return fused


def enhanced_detect(model, image_path, imgsz=640, base_conf=0.15, category_fn=None):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    low = min(base_conf, 0.1)  # collect candidates low, filter per-class later

    boxes = _predict(model, img, max(imgsz, 832), low)
    if max(h, w) > TILE * 1.5:   # only tile big photos
        for x, y in _tiles(h, w):
            boxes += _predict(model, img[y:y+TILE, x:x+TILE], TILE, low, (x, y))

    fused = _fuse(boxes)
    min_area = MIN_BOX_AREA_RATIO * h * w
    detections = []
    for x1, y1, x2, y2, conf, cls in fused:
        label = model.names.get(cls, f"class_{cls}")
        if conf < CLASS_CONF.get(label.lower(), max(base_conf, 0.25)):
            continue
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        bbox = [round(v, 1) for v in (x1, y1, x2, y2)]
        detections.append({
            "class_id": cls, "class": label, "class_name": label,
            "confidence": round(conf, 3), "bbox": bbox,
            "bbox_coords": dict(zip(("x1", "y1", "x2", "y2"), bbox)),
            "category": category_fn(label) if category_fn else "evidence",
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    annotated = img.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 210, 180), 2)
        cv2.putText(annotated, f'{d["class"]} {d["confidence"]:.0%}', (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 210, 180), 2)
    return detections, annotated
