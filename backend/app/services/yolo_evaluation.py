"""Pure helpers for repeatable, class-aware CCTV detector evaluation."""
import math


def map_class_id(class_names, class_id):
    """Map the checkpoint's numeric class ID directly to its trained class name."""
    try:
        name = class_names[int(class_id)]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Unknown YOLO class id: {class_id}") from error
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid YOLO class name for class id {class_id}")
    return name


def filter_predictions(predictions, confidence_threshold):
    threshold = float(confidence_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be finite and between 0 and 1.")
    return [prediction for prediction in predictions if float(prediction["confidence"]) >= threshold]


def box_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_frame_detections(ground_truth, predictions, iou_threshold=0.5):
    """Greedy one-to-one class-aware matching; duplicate/extra boxes become FPs."""
    if not 0 < float(iou_threshold) <= 1:
        raise ValueError("IoU threshold must be greater than 0 and at most 1.")
    available = set(range(len(ground_truth)))
    matches, false_positives = [], []
    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        candidates = [
            (box_iou(prediction["bbox"], ground_truth[index]["bbox"]), index)
            for index in available
            if prediction["class"] == ground_truth[index]["class"]
        ]
        candidates = [item for item in candidates if item[0] >= iou_threshold]
        if candidates:
            overlap, matched_index = max(candidates)
            available.remove(matched_index)
            matches.append((prediction, ground_truth[matched_index], overlap))
        else:
            false_positives.append(prediction)
    false_negatives = [ground_truth[index] for index in sorted(available)]
    return matches, false_positives, false_negatives


def empty_counts():
    return {"true_positives": 0, "false_positives": 0, "missed_detections": 0}


def finalize_counts(counts):
    result = dict(counts)
    tp, fp, fn = (counts[key] for key in ("true_positives", "false_positives", "missed_detections"))
    result["precision"] = tp / (tp + fp) if tp + fp else 0.0
    result["recall"] = tp / (tp + fn) if tp + fn else 0.0
    return result
