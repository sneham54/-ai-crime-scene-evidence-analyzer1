# Temporal event association

`TemporalEventAssociationService` wraps the category-aware saved-artifact selector and event extractor in `shooting_event_evidence_service.py`. It does not run YOLO or ByteTrack. A manifest must identify one completed CSV/render pair; Assault folders use their exact source stem. CSV rows are filtered to the selected source video, then mapped using source PTS when optional PyAV can read them. Otherwise CSV timestamps are checked against frame/FPS mapping and that fallback is recorded in the event JSON. Frame offsets of zero or one are evaluated. Unmatched timing is rejected.

## Scoring and refinement

Classifier category probabilities are averaged over overlapping windows in configurable time bins and adjusted by neighboring-window consistency. YOLO evidence uses relevant class confidence and repeated observations. Tracking evidence uses per-video/class/track-ID continuity, observed duration, and gaps. Scores are normalized over available modalities, and a bin is eligible only when classifier probability meets its configured minimum. Therefore object or tracking evidence cannot independently refine an event. Optional action/audio scores are accepted only with `validated: true`; no such models are assumed to exist.

Defaults in the service are uncalibrated operational settings, not validated weights or an accuracy claim. Override them with `TEMPORAL_EVENT_WEIGHTS`, a JSON object keyed by `shooting`, `assault`, and `robbery`, each containing classifier/yolo/tracking weights. The classifier weight must be the largest. `TEMPORAL_EVENT_BIN_SEC`, `TEMPORAL_EVENT_MIN_SCORE`, `TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY`, `TEMPORAL_EVENT_MIN_SUPPORTED_BINS`, `TEMPORAL_EVENT_BOUNDARY_TOLERANCE_SEC`, and `TEMPORAL_EVENT_MAX_TRACK_GAP_SEC` configure binning and support rules. Calibrate category weights and thresholds against held-out, manually verified event intervals before drawing accuracy conclusions.

Refinement can only shrink or retain the original classifier candidate. The original interval is always preserved. If classifier support is missing or insufficient, the original timestamps remain the clip interval. Context before/after the selected interval affects clip extraction only. Every output remains `unverified_candidate` until independently reviewed.

## Evaluation

Use `backend/scripts/evaluate_temporal_intervals.py` with a CSV containing category, video, reference interval, original candidate interval, and refined interval columns. It reports original/refined start and end absolute error, temporal IoU, event precision/recall/F1, and supporting-detection coverage. The current repository has no verified event-boundary reference set, so timestamp accuracy and any improvement from refinement cannot yet be independently established.
