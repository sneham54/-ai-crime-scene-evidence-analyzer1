"""Small dependency-free metrics for a fixed four-class classifier."""
import numpy as np

from .config import CLASSES


def classification_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=np.int64), np.asarray(predicted, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("truth and predicted must be equal-length one-dimensional labels")
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        if not 0 <= actual < len(CLASSES) or not 0 <= guess < len(CLASSES):
            raise ValueError("labels must be class indices in [0, 3]")
        matrix[actual, guess] += 1
    per_class = {}
    for index, name in enumerate(CLASSES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        precision = tp / max(1, int(matrix[:, index].sum()))
        recall = tp / max(1, support)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    macro = {key: float(np.mean([row[key] for row in per_class.values()])) for key in ("precision", "recall", "f1-score")}
    report = {**per_class, "accuracy": float(np.mean(truth == predicted)) if len(truth) else 0.0,
              "macro avg": {**macro, "support": int(len(truth))},
              "weighted avg": {key: (sum(row[key] * row["support"] for row in per_class.values()) / max(1, len(truth)))
                               for key in ("precision", "recall", "f1-score")} | {"support": int(len(truth))}}
    return {"accuracy": report["accuracy"], "macro_precision": macro["precision"],
            "macro_recall": macro["recall"], "macro_f1": macro["f1-score"],
            "per_class": per_class, "classification_report": report,
            "confusion_matrix": matrix.tolist()}
