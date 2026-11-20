"""Create provenance-preserving event packages from saved tracking artifacts.

This service never runs YOLO or ByteTrack. Source frame offsets and available
presentation timestamps are supplied by validation in the association layer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from app.services.video_timing import probe_video_timeline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


class EventEvidenceService:
    CATEGORIES = {"shooting": "Shooting", "assault": "Assault", "robbery": "Robbery"}

    def __init__(self, config):
        self.config = config

    def _roots(self, category):
        label = self.CATEGORIES[category]
        if category == "shooting":
            results = Path(self.config["SHOOTING_RESULTS_DIR"])
            status = Path(self.config["SHOOTING_TRACKING_STATUS_PATH"])
            tracked = status.parent / "videos"
        elif category == "assault":
            results = Path(self.config["ASSAULT_RESULTS_DIR"])
            status = Path(self.config.get("ASSAULT_TRACKING_STATUS_PATH") or results / "tracking_status.csv")
            tracked = results
        else:
            results = Path(self.config["ROBBERY_RESULTS_DIR"])
            status = Path(self.config.get("ROBBERY_TRACKING_STATUS_PATH") or results / "tracking_status.csv")
            tracked = results
        return label, results.resolve(), status.resolve(), tracked.resolve()

    @staticmethod
    def _video_field(row):
        lowered = {str(key).strip().casefold(): value for key, value in row.items() if key is not None}
        return next((lowered[key] for key in ("video", "video_name", "filename", "source_video") if lowered.get(key)), None)

    @staticmethod
    def _named_files(root: Path, filename: str):
        if not filename or not root.is_dir():
            return []
        return [path.resolve() for path in root.rglob("*") if path.is_file() and path.name.casefold() == filename.casefold()]

    @classmethod
    def _manifest_file(cls, root: Path, reference: str, source_name: str):
        """Resolve one manifest path, scoping basename fallbacks to its video folder."""
        filename = Path(reference or "").name
        if not filename:
            return []
        supplied = Path(reference)
        if supplied.is_file():
            resolved = supplied.resolve()
            try:
                resolved.relative_to(root.resolve())
                return [resolved]
            except ValueError:
                return []
        candidates = cls._named_files(root, filename)
        if len(candidates) <= 1:
            return candidates
        stem = Path(source_name).stem.casefold()
        scoped = [path for path in candidates if any(parent.name.casefold() == stem for parent in path.parents)]
        return scoped

    def _source_verified(self, original: Path, category: str, source_video_name: str | None):
        source_root = self.config.get("EVIDENCE_SOURCE_DIR")
        if not source_root:
            return {"verified": None, "source_path": None, "source_sha256": sha256_file(original)}
        category_root = Path(source_root) / self.CATEGORIES[category]
        name = Path(source_video_name or original.name).name.casefold()
        candidates = [path for path in category_root.iterdir()
                      if path.is_file() and path.name.casefold() == name] if category_root.is_dir() else []
        if len(candidates) != 1:
            raise ValueError(f"{self.CATEGORIES[category]} original is missing or ambiguous in the configured source catalog.")
        source = candidates[0].resolve()
        digest = sha256_file(original)
        if digest != sha256_file(source):
            raise ValueError("Uploaded original does not match the configured category source video; saved tracking was not associated.")
        return {"verified": True, "source_path": str(source), "source_sha256": digest}

    def resolve_artifacts(self, original_path: Path, category="Shooting", source_video_name=None):
        """Select the category's recorded CSV/render without guessing between runs."""
        key = str(category or "").strip().casefold()
        if key not in self.CATEGORIES:
            raise ValueError("Event evidence supports Shooting, Assault, and Robbery only.")
        original = Path(original_path).resolve()
        label, results_root, status_path, tracked_root = self._roots(key)
        self._source_verified(original, key, source_video_name)
        name = Path(source_video_name or original.name).name
        matches = []
        row = {}

        if status_path.is_file():
            with status_path.open("r", newline="", encoding="utf-8-sig") as stream:
                status_rows = list(csv.DictReader(stream))
            matches = [candidate for candidate in status_rows
                       if Path(self._video_field(candidate) or "").name.casefold() == name.casefold()]
            if len(matches) != 1:
                raise ValueError(f"{label} tracking manifest has no unique record for {name}.")
            row = matches[0]
            values = {str(k).strip().casefold(): v for k, v in row.items() if k is not None}
            status = str(values.get("status") or "").strip().casefold()
            if status not in {"complete", "completed", "success", "successful"}:
                raise ValueError(f"{label} tracking status is {status or 'missing'}; incomplete results are excluded.")
            csv_ref = values.get("output_csv") or values.get("detections_csv") or ""
            video_ref = values.get("output_video") or values.get("annotated_video") or ""
            csv_name = Path(csv_ref).name
            tracked_name = Path(video_ref).name
            csv_candidates = self._manifest_file(results_root, csv_ref, name)
            tracked_candidates = self._manifest_file(tracked_root, video_ref, name)
            if key == "shooting" and tracked_name and not tracked_candidates:
                tracked_candidates = self._manifest_file(status_path.parent / "videos", video_ref, name)
            csv_candidates = list(dict.fromkeys(csv_candidates))
            tracked_candidates = list(dict.fromkeys(tracked_candidates))
            if len(csv_candidates) != 1:
                raise FileNotFoundError(f"{label} manifest-referenced detection CSV is missing or ambiguous: {csv_name or '(empty)'}.")
            if tracked_name and len(tracked_candidates) > 1:
                raise ValueError(f"{label} manifest-referenced annotated video is ambiguous: {tracked_name}.")
            tracked_path = tracked_candidates[0] if tracked_candidates else None
            row = {**row, "tracking_status_path": str(status_path), "artifact_selection": "unique_manifest_record",
                   "resolved_status": status, "resolved_csv": str(csv_candidates[0]),
                   "resolved_video": str(tracked_path) if tracked_path else None}
            return row, csv_candidates[0], tracked_path

        # Assault exports have one exact source-stem folder per tracking run,
        # with detections.csv and a sibling *_tracked.mp4. No later file version
        # is preferred: duplicate folders or renders make the match ambiguous.
        if key == "assault":
            stem = Path(name).stem.casefold()
            dirs = [path for path in results_root.iterdir()
                    if path.is_dir() and path.name.casefold() == stem] if results_root.is_dir() else []
            if len(dirs) != 1:
                raise ValueError(f"Assault tracking folder is missing or ambiguous for {name}.")
            csv_path = dirs[0] / "detections.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Assault detections.csv is missing for {name}.")
            renders = [path.resolve() for path in dirs[0].glob("*.mp4") if path.is_file() and
                       path.stem.casefold() in {f"{stem}_tracked", "annotated", "annotated_video"}]
            if len(renders) > 1:
                raise ValueError(f"Assault annotated render is ambiguous for {name}.")
            return {"video": name, "status": "folder_csv_pair", "artifact_selection": "exact_source_stem_folder",
                    "tracking_status_path": None, "output_csv": str(csv_path),
                    "output_video": str(renders[0]) if renders else None}, csv_path.resolve(), renders[0] if renders else None

        raise ValueError(f"{label} tracking manifest is missing; outputs were not selected by filename alone.")

    @staticmethod
    def _metadata(path):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for event clip extraction.") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not decode video: {path.name}")
        info = {"frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        capture.release()
        if not math.isfinite(info["fps"]) or info["fps"] <= 0 or info["frames"] <= 0:
            raise ValueError(f"Video metadata is invalid: {path.name}")
        return info

    def create_event(self, *, original_path, output_root, start_sec, end_sec,
                     category="Shooting", source_video_name=None, video_info=None,
                     confidence=None, verification_status="unverified",
                     classifier_metadata=None, context_before_sec=2.0, context_after_sec=2.0):
        original = Path(original_path).resolve()
        start, end = _num(start_sec), _num(end_sec)
        before, after = _num(context_before_sec, 2), _num(context_after_sec, 2)
        if start is None or end is None or start < 0 or end <= start or before < 0 or after < 0:
            raise ValueError("Event interval must have finite non-negative start and a later end.")
        category_key = str(category or "").strip().casefold()
        if category_key not in self.CATEGORIES:
            raise ValueError("Event evidence supports Shooting, Assault, and Robbery only.")
        category_label = self.CATEGORIES[category_key]
        if not isinstance(verification_status, str) or not verification_status.strip():
            raise ValueError("Verification status must be a non-empty string.")
        source_info = self._metadata(original)
        source_timeline = probe_video_timeline(original, source_info["frames"], source_info["fps"])
        pts = ({index: timestamp for index, timestamp in enumerate(source_timeline.timestamps_sec)}
               if source_timeline.mapping_status == "verified_vfr_presentation_timestamps" else {})
        duration = source_timeline.duration_sec or source_info["frames"] / source_info["fps"]
        if end > duration + 1 / source_info["fps"]:
            raise ValueError("Event interval extends beyond the source video.")
        source_metadata = self._source_verified(original, category_key, source_video_name)
        manifest, csv_path, tracked_path = self.resolve_artifacts(original, category_label, source_video_name)
        frame_offset = 0
        if video_info and source_metadata.get("verified"):
            # Reuse the existing CSV/source/timeline validator once a category
            # artifact has been selected deterministically from its manifest.
            from app.services.evidence_catalog_service import EvidenceCatalogService
            metadata = {**video_info, **source_info, "total_frames": source_info["frames"],
                        "fps": source_info["fps"], "duration_sec": duration}
            checked = EvidenceCatalogService(self.config).match_tracking_csv(
                category_key, Path(source_video_name or original.name).name,
                str(original), metadata, csv_path_override=csv_path,
            )
            if checked.get("status") != "verified_match":
                raise ValueError(checked.get("reason") or f"{category_label} detections do not align with the original video.")
            frame_offset = int(checked.get("frame_offset", 0))
        tracked_info = self._metadata(tracked_path) if tracked_path else None
        aligned = bool(tracked_info and tracked_info["frames"] == source_info["frames"] and
                       abs(tracked_info["fps"] - source_info["fps"]) <= max(.01, source_info["fps"] * .001))
        if tracked_path and not aligned:
            tracked_path = None  # Explicitly omit misaligned render; never silently substitute it.

        clip_start, clip_end = max(0.0, start - before), min(duration, end + after)
        if pts:
            first_frame = source_timeline.frame_at_or_after(clip_start)
            last_frame = source_timeline.frame_at_or_after(clip_end)
            first_frame = min(source_info["frames"] - 1, first_frame if first_frame is not None else source_info["frames"] - 1)
            last_frame = min(source_info["frames"] - 1, last_frame if last_frame is not None else source_info["frames"] - 1)
        else:
            first_frame = max(0, int(math.floor(clip_start * source_info["fps"])) )
            last_frame = min(source_info["frames"] - 1, int(math.ceil(clip_end * source_info["fps"])))
        rows = []
        source_name = Path(source_video_name or original.name).name
        with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
            for raw in csv.DictReader(stream):
                if Path(raw.get("video", source_name)).name.casefold() not in {source_name.casefold(), original.name.casefold()}:
                    continue
                frame = _num(raw.get("frame", raw.get("frame_number")))
                timestamp = _num(raw.get("timestamp_seconds"))
                if frame is None:
                    continue
                csv_frame = int(frame)
                frame = csv_frame - frame_offset
                if frame < 0 or frame >= source_info["frames"]:
                    continue
                csv_timestamp = timestamp
                timestamp = source_timeline.timestamp(frame)
                if timestamp is None:
                    continue
                if first_frame <= frame <= last_frame and clip_start <= timestamp <= clip_end:
                    bbox = [_num(raw.get(key)) for key in ("x1", "y1", "x2", "y2")]
                    rows.append({"video": source_name,
                                 "frame_number": frame, "csv_frame_number": csv_frame,
                                 "timestamp_seconds": timestamp, "csv_timestamp_seconds": csv_timestamp,
                                 "timestamp_source": source_timeline.mapping_status,
                                 "class_id": int(_num(raw.get("class_id"), -1)),
                                 "class_name": raw.get("class_name") or raw.get("class") or "unknown",
                                 "confidence": _num(raw.get("confidence")),
                                 "track_id": raw.get("track_id") or None,
                                 "bbox": bbox if all(v is not None for v in bbox) else None})
        rows.sort(key=lambda item: (item["frame_number"], str(item["track_id"]), item["class_name"]))
        # Deterministic event key makes retries idempotent on disk and in Mongo.
        identity = "|".join((category_key, sha256_file(original), f"{start:.6f}", f"{end:.6f}",
                              f"{before:.6f}", f"{after:.6f}", str(csv_path.resolve())))
        event_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        out = Path(output_root).resolve() / event_id
        out.mkdir(parents=True, exist_ok=True)
        import cv2
        self._extract_clip(original, out / "original_clip.mp4", first_frame, last_frame, source_info)
        annotated_clip = None
        if tracked_path and aligned:
            self._extract_clip(tracked_path, out / "annotated_clip.mp4", first_frame, last_frame, tracked_info)
            annotated_clip = out / "annotated_clip.mp4"

        detections_by_frame = {}
        for item in rows:
            detections_by_frame.setdefault(item["frame_number"], []).append(item)
        keyframes, annotated_keyframes, keyframe_timestamps = [], [], []
        cap = cv2.VideoCapture(str(original))
        tracked_cap = cv2.VideoCapture(str(tracked_path)) if tracked_path and aligned else None
        try:
            frames_to_save = sorted(detections_by_frame)
            if not frames_to_save:
                frames_to_save = sorted(set([first_frame, (first_frame + last_frame) // 2, last_frame]))
            for frame_no in frames_to_save:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                if not ok:
                    continue
                keyframe_timestamps.append({"frame_index": frame_no,
                    "timestamp_sec": source_timeline.timestamp(frame_no),
                    "timestamp_mapping_status": source_timeline.mapping_status,
                    "timestamps_verified": source_timeline.timestamps_verified})
                name = f"frame_{frame_no:08d}.jpg"
                src_out = out / "keyframes" / name
                src_out.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(src_out), frame)
                keyframes.append(str(src_out))
                if detections_by_frame.get(frame_no):
                    canvas = frame.copy()
                    for item in detections_by_frame[frame_no]:
                        if item["bbox"]:
                            x1, y1, x2, y2 = [int(round(x)) for x in item["bbox"]]
                            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{item['class_name']} #{item['track_id'] or '?'} {item['confidence'] or 0:.2f}"
                            cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
                    ann_out = out / "annotated_keyframes" / name
                    ann_out.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(ann_out), canvas)
                    annotated_keyframes.append(str(ann_out))
        finally:
            cap.release()
            if tracked_cap:
                tracked_cap.release()
        grouped = {}
        for item in rows:
            if item["track_id"] not in (None, ""):
                grouped.setdefault(str(item["track_id"]), []).append(item)
        grouped_tracks = [{"track_id": tid, "detections": items} for tid, items in sorted(grouped.items())]
        csv_out = out / "detections.csv"
        with csv_out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video", "frame_number", "csv_frame_number", "timestamp_seconds", "csv_timestamp_seconds", "timestamp_source", "class_id", "class_name", "confidence", "track_id", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            for item in rows:
                writer.writerow({**{key: item.get(key) for key in ("video", "frame_number", "csv_frame_number", "timestamp_seconds", "csv_timestamp_seconds", "timestamp_source", "class_id", "class_name", "confidence", "track_id")},
                                 **dict(zip(("x1", "y1", "x2", "y2"), item["bbox"] or [None] * 4))})
        record = {"evidence_id": event_id, "original_video": str(original), "original_video_name": Path(source_video_name or original.name).name,
                  "original_video_sha256": sha256_file(original), "source_video_verification": source_metadata,
                  "category": category_label, "start_timestamp_sec": start, "end_timestamp_sec": end,
                  "context_start_sec": clip_start, "context_end_sec": clip_end,
                  "timestamp_mapping_status": source_timeline.mapping_status,
                  "timestamp_basis": source_timeline.timestamp_basis,
                  "timestamps_verified": source_timeline.timestamps_verified,
                  "verification_status": verification_status, "classifier_confidence": _num(confidence),
                  "classifier_metadata": classifier_metadata or {}, "source_fps": source_info["fps"], "source_frame_count": source_info["frames"],
                  "manifest_record": {"video": manifest.get("video"), "status": manifest.get("status"), "output_csv": manifest.get("output_csv"), "output_video": manifest.get("output_video")},
                  "detection_csv_source": str(csv_path), "source_frame_offset": frame_offset,
                  "tracked_video_alignment": {"verified": aligned, "tracked_metadata": tracked_info},
                  "detections": rows, "tracking_groups": grouped_tracks,
                  "keyframe_timestamps": keyframe_timestamps,
                  "artifacts": {"original_clip": str(out / "original_clip.mp4"), "annotated_clip": str(annotated_clip) if annotated_clip else None,
                                "keyframes": keyframes, "annotated_keyframes": annotated_keyframes,
                                "csv": str(csv_out), "json": str(out / "event.json")},
                  "created_at": datetime.now(timezone.utc).isoformat()}
        artifact_hashes = {}
        for key, value in record["artifacts"].items():
            paths = value if isinstance(value, list) else [value]
            for index, item_path in enumerate(paths):
                if item_path and Path(item_path).is_file():
                    suffix = f"_{index}" if isinstance(value, list) else ""
                    artifact_hashes[f"{key}{suffix}"] = sha256_file(Path(item_path))
        record["artifact_hashes"] = artifact_hashes
        json_path = out / "event.json"
        json_path.write_text(json.dumps(record, allow_nan=False, indent=2), encoding="utf-8")
        return record

    @staticmethod
    def _extract_clip(source, destination, first, last, info):
        import cv2
        cap = cv2.VideoCapture(str(source))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"]))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create video clip: {destination.name}")
        count = 0
        try:
            for _ in range(first, last + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                count += 1
        finally:
            cap.release()
            writer.release()
        if not count:
            raise ValueError(f"No decodable frames in requested interval for {source.name}.")


# Backwards-compatible name for existing Shooting integrations.
ShootingEventEvidenceService = EventEvidenceService
