"""Compare temporal candidate/refined intervals with manually verified labels.

Input CSV columns: category, video, reference_start_sec, reference_end_sec,
predicted_start_sec, predicted_end_sec, refined_start_sec, refined_end_sec.
Optional predicted_supporting_detection_count reports evidence coverage.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def interval_iou(left, right):
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _interval(row, start, end):
    try:
        a, b = float(row[start]), float(row[end])
        return (a, b) if 0 <= a < b else None
    except (KeyError, TypeError, ValueError):
        return None


def evaluate(rows, iou_threshold=.5):
    references, predictions = [], []
    start_errors_original, end_errors_original = [], []
    start_errors_refined, end_errors_refined = [], []
    supporting = total_predicted = 0
    for row in rows:
        key = (str(row.get("category", "")).casefold(), Path(str(row.get("video", ""))).name.casefold())
        reference = _interval(row, "reference_start_sec", "reference_end_sec")
        predicted = _interval(row, "predicted_start_sec", "predicted_end_sec")
        refined = _interval(row, "refined_start_sec", "refined_end_sec")
        if reference:
            references.append((key, reference))
        for interval, starts, ends in ((predicted, start_errors_original, end_errors_original),
                                       (refined, start_errors_refined, end_errors_refined)):
            if reference and interval:
                starts.append(abs(interval[0] - reference[0]))
                ends.append(abs(interval[1] - reference[1]))
        if predicted:
            predictions.append((key, predicted, "original"))
        if refined:
            predictions.append((key, refined, "refined"))
            total_predicted += 1
            supporting += int(float(row.get("predicted_supporting_detection_count") or 0) > 0)

    def metrics(kind):
        preds = [(key, interval) for key, interval, label in predictions if label == kind]
        candidates = []
        for pi, (pkey, pinterval) in enumerate(preds):
            for ri, (rkey, rinterval) in enumerate(references):
                if pkey == rkey:
                    candidates.append((interval_iou(pinterval, rinterval), pi, ri))
        matched_p, matched_r, matched_pairs = set(), set(), []
        for score, pi, ri in sorted(candidates, reverse=True):
            if score >= iou_threshold and pi not in matched_p and ri not in matched_r:
                matched_p.add(pi); matched_r.add(ri)
                matched_pairs.append((pi, ri, score))
        tp = len(matched_p); fp = len(preds) - tp; fn = len(references) - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if precision is not None and recall is not None else None)
        ious = [score for _, _, score in matched_pairs]
        return {"event_precision": precision, "event_recall": recall, "event_f1": f1,
                "mean_temporal_iou": sum(ious) / len(ious) if ious else None,
                "matched_events": tp, "predicted_events": len(preds), "reference_events": len(references)}

    def mean(values): return sum(values) / len(values) if values else None
    return {"status": "measurable" if references else "not_estimable_no_verified_reference_intervals",
        "iou_match_threshold": iou_threshold,
        "original_classifier_intervals": {**metrics("original"),
            "start_time_mae_sec": mean(start_errors_original), "end_time_mae_sec": mean(end_errors_original)},
        "refined_intervals": {**metrics("refined"),
            "start_time_mae_sec": mean(start_errors_refined), "end_time_mae_sec": mean(end_errors_refined),
            "supporting_detection_coverage": supporting / total_predicted if total_predicted else None},
        "accuracy_claim": "No improvement claim is made; use verified manual reference intervals."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path, help="CSV with verified reference and prediction intervals")
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    parser.add_argument("--iou-threshold", type=float, default=.5)
    args = parser.parse_args()
    with args.references.open(newline="", encoding="utf-8-sig") as stream:
        result = evaluate(list(csv.DictReader(stream)), args.iou_threshold)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
