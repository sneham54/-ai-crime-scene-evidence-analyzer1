"""Dashboard statistics derived from persisted evidence records."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from pymongo.errors import PyMongoError

from app.models.evidence import serialize_evidence
from app.models.case import create_case_doc, serialize_case
from app.services.custody_service import CustodyService
from app.services.risk_service import RiskService

bp = Blueprint("dashboard", __name__)
CATEGORIES = ("Assault", "Robbery", "Shooting", "Normal")
RISK_BANDS = ("low", "moderate", "high", "critical")


@bp.route("/health", methods=["GET"])
@jwt_required(optional=True)
def database_health():
    """Expose current MongoDB availability without leaking connection details."""
    # Docker and dashboard readiness checks must observe the live database,
    # rather than a recently cached successful connection state.
    if getattr(current_app, "mongo_client", None) is not None:
        from app import refresh_database_connection
        refresh_database_connection(current_app, force=True)
    state = current_app.extensions.get("mongo_connection_state", {})
    connected = current_app.db is not None
    response = {
        "status": "connected" if connected else "unavailable",
        "database": current_app.config.get("DB_NAME") if connected else None,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "message": None if connected else (state.get("error") or "MongoDB is unavailable; persisted case and risk statistics cannot be loaded."),
    }
    if not connected and state.get("error"):
        # Include a sanitized exception class for diagnostics, never URI/credentials.
        response["error_type"] = state.get("error_type") or "DatabaseConnectionError"
    return jsonify(response), (200 if connected else 503)


@bp.route("/cases", methods=["GET", "POST"])
@jwt_required(optional=True)
def cases():
    """List and register persisted investigations using the existing case schema."""
    db = current_app.db
    if db is None:
        state = current_app.extensions.get("mongo_connection_state", {})
        message = state.get("error") or "MongoDB is unavailable; case records cannot be read or saved."
        return jsonify({"status": "unavailable", "error": message, "error_type": state.get("error_type")}), 503
    if request.method == "GET":
        try:
            rows = list(db.cases.find({}).sort("created_at", -1).limit(200))
            return jsonify({"status": "success", "cases": [serialize_case(row) for row in rows]})
        except Exception as exc:
            from app import _mongo_connection_error
            message = _mongo_connection_error(exc, current_app.config["MONGO_URI"])
            current_app.logger.warning("Could not load persisted investigation cases: %s", message)
            current_app.db = None
            current_app.extensions.setdefault("mongo_connection_state", {}).update(
                connected=False, error=message, error_type=type(exc).__name__)
            return jsonify({"status": "unavailable", "error": message,
                            "error_type": type(exc).__name__}), 503
    data = request.get_json(silent=True) or {}
    title = str(data.get("title") or "").strip()
    description = str(data.get("description") or "").strip()
    if not title:
        return jsonify({"status": "error", "error": "A case title is required."}), 400
    doc = create_case_doc(title, description, get_jwt_identity() or "anonymous")
    try:
        result = db.cases.insert_one(doc)
    except Exception as exc:
        from app import _mongo_connection_error
        message = _mongo_connection_error(exc, current_app.config["MONGO_URI"])
        current_app.logger.warning("Could not persist new case: %s", message)
        current_app.db = None
        current_app.extensions.setdefault("mongo_connection_state", {}).update(
            connected=False, error=message, error_type=type(exc).__name__)
        return jsonify({"status": "unavailable", "error": message,
                        "error_type": type(exc).__name__}), 503
    doc["_id"] = result.inserted_id
    return jsonify({"status": "success", "case": serialize_case(doc)}), 201


def _risk_payload(record):
    payload = record.get("risk_score")
    return payload if isinstance(payload, dict) else ({"overall_risk": payload} if isinstance(payload, (int, float)) else None)


def _risk_factors(payload):
    factors = payload.get("contributing_factors", []) if isinstance(payload, dict) else []
    names = []
    for factor in factors if isinstance(factors, list) else []:
        if not isinstance(factor, dict):
            continue
        try:
            if float(factor.get("points", 0)) > 0 and factor.get("name"):
                names.append(factor["name"])
        except (TypeError, ValueError):
            continue
    return names[:3]


def _risk_level(payload, service):
    if not payload:
        return None
    saved = str(payload.get("risk_level") or payload.get("threat_level") or "").casefold()
    aliases = {"medium": "moderate", "elevated": "high", "medium-high": "high"}
    if saved in RISK_BANDS:
        return saved
    if saved in aliases:
        return aliases[saved]
    score = payload.get("overall_risk", payload.get("score"))
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else service.thresholds
    if "high" not in thresholds and "elevated" in thresholds:
        if score >= thresholds.get("critical", 75): return "critical"
        if score >= thresholds.get("elevated", 45): return "high"
        if score >= thresholds.get("moderate", 20): return "moderate"
        return "low"
    if score >= thresholds.get("critical", service.thresholds["critical"]): return "critical"
    if score >= thresholds.get("high", service.thresholds["high"]): return "high"
    if score >= thresholds.get("moderate", service.thresholds["moderate"]): return "moderate"
    return "low"


def _source_key(record):
    """Use the stable upload name because temporary upload paths change per run."""
    source = record.get("filename")
    if not source:
        info = (record.get("video_analysis") or {}).get("video_info") or {}
        source = info.get("source_video_identity") or (record.get("video_analysis") or {}).get("source_video_path")
    return str(source).replace("\\", "/").casefold() if source else None


def _unique_evidence_records(records):
    """Collapse repeated Mongo documents for one stable evidence ID."""
    unique = {}
    for record in records:
        key = str(record.get("evidence_id") or record.get("_id") or "")
        if not key:
            key = _source_key(record) or f"anonymous:{id(record)}"
        previous = unique.get(key)
        if previous is None or _record_timestamp(record) >= _record_timestamp(previous):
            unique[key] = record
    def upload_time(record):
        stamp = _parse_timestamp(record.get("uploaded_at") or record.get("analyzed_at"))
        return stamp.timestamp() if stamp else float("-inf")

    return sorted(unique.values(), key=upload_time, reverse=True)


def _record_timestamp(record):
    stamp = record.get("analyzed_at") or record.get("uploaded_at")
    if not isinstance(stamp, datetime):
        return float("-inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def _iso_utc(stamp):
    if not isinstance(stamp, datetime):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _latest_video_records(records):
    latest = {}
    for record in records:
        if (record.get("file_type") != "video" or
                str(record.get("status", "")).casefold() not in {"analyzed", "completed", "success"}):
            continue
        # An evidence ID is the upload identity. Filename-only de-duplication
        # incorrectly collapsed separate uploads that happened to share a name.
        key = record.get("evidence_id") or _source_key(record) or record.get("_id")
        if key is None:
            continue
        current = latest.get(key)
        if current is None or _record_timestamp(record) >= _record_timestamp(current):
            latest[key] = record
    return list(latest.values())


def _parse_timestamp(value):
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value.strip():
        try:
            stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def _artifact_detections(payload):
    detections = payload.get("detections")
    if isinstance(detections, list):
        return [item for item in detections if isinstance(item, dict)]
    frames = payload.get("frame_results")
    if not isinstance(frames, list):
        frames = payload.get("frames")
    return [detection for frame in frames if isinstance(frame, dict)
            for detection in (frame.get("detections") if isinstance(frame.get("detections"), list) else [])
            if isinstance(detection, dict)] if isinstance(frames, list) else []


def _valid_track_ids(detections):
    ids = set()
    for detection in detections:
        track_id = detection.get("track_id")
        if track_id is None or isinstance(track_id, bool):
            continue
        try:
            value = float(track_id)
            if value >= 0 and value.is_integer():
                ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return ids


def _resolved_prediction(classification):
    if not isinstance(classification, dict):
        return None
    status = str(classification.get("classification_status") or "").casefold()
    if status in {"uncertain_prediction_disagreement", "unresolved_classifier_disagreement", "unavailable"}:
        return None
    label = str(classification.get("label") or "").strip().title()
    return label if label in CATEGORIES else None


def _video_artifacts(upload_folder):
    """Read durable analysis summaries already written beside their artifacts."""
    root = Path(upload_folder) / "evidence"
    if not root.is_dir():
        return []
    records = []
    for path in root.glob("*/analysis.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            analysis_id = str(payload.get("analysis_id") or path.parent.name)
            if not analysis_id or analysis_id != path.parent.name:
                continue
            classification = payload.get("event_classification") or payload.get("classification")
            status = str(payload.get("status") or "completed").casefold()
            is_failed = status in {"failed", "failure", "error"}
            if not isinstance(classification, dict):
                classification = {}
            if not classification.get("label") and not is_failed:
                continue
            video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
            video_info = payload.get("video_info") if isinstance(payload.get("video_info"), dict) else {}
            tracking = payload.get("tracking") if isinstance(payload.get("tracking"), dict) else {}
            detections = _artifact_detections(payload)
            stamp = _parse_timestamp(payload.get("analyzed_at"))
            if stamp is None:
                stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            filename = (payload.get("original_filename") or payload.get("filename") or
                        payload.get("file_name") or video.get("original_filename") or
                        video.get("filename") or Path(payload.get("source_video_path") or "").name or
                        f"Analysis {analysis_id[:8]}")
            uploaded_at = _parse_timestamp(payload.get("uploaded_at"))
            record = {
                "_id": f"artifact:{analysis_id}", "analysis_id": analysis_id,
                "filename": filename, "file_type": "video",
                "status": status,
                "uploaded_at": uploaded_at, "analyzed_at": stamp,
                "detections": detections,
                "risk_score": payload.get("risk_score", payload.get("risk")),
                "event_classification": classification,
                "reference_category": payload.get("reference_category") or (payload.get("video_analysis") or {}).get("reference_category"),
                "video_analysis": {"analysis_id": analysis_id, "tracking": tracking,
                                   "video_info": video_info},
            }
            records.append(record)
        except (OSError, ValueError, TypeError):
            current_app.logger.warning("Skipping unreadable video analysis artifact %s", path, exc_info=True)
    return records


def _saved_report_video_records(upload_folder):
    """Recover video analysis snapshots and saved risk from generated JSON reports."""
    root = Path(upload_folder) / "reports"
    if not root.is_dir():
        return []
    records = {}
    for path in root.glob("*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            video = report.get("video_result") if isinstance(report, dict) else None
            if not isinstance(video, dict):
                continue
            analysis_id = str(video.get("analysis_id") or "").strip()
            risk = report.get("risk")
            if not analysis_id or not isinstance(risk, dict):
                continue
            if not any(key in risk for key in ("overall_risk", "score")):
                continue
            stamp = _parse_timestamp(video.get("analyzed_at") or report.get("analyzed_at"))
            if stamp is None:
                stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            records.setdefault(analysis_id, {
                "_id": f"report:{analysis_id}", "analysis_id": analysis_id,
                "filename": video.get("file_name") or report.get("file_name") or f"Analysis {analysis_id[:8]}",
                "file_type": "video", "status": "completed", "analyzed_at": stamp,
                "uploaded_at": _parse_timestamp(video.get("uploaded_at") or report.get("uploaded_at")),
                "detections": _artifact_detections(video),
                "risk_score": risk,
                "event_classification": video.get("event_classification") or video.get("classification") or {},
                "reference_category": video.get("reference_category"),
                "video_analysis": {"analysis_id": analysis_id,
                                   "tracking": video.get("tracking") if isinstance(video.get("tracking"), dict) else {},
                                   "video_info": {}},
            })
        except (OSError, ValueError, TypeError):
            current_app.logger.warning("Skipping unreadable saved video report %s", path, exc_info=True)
    return list(records.values())


def _saved_image_analysis_count(upload_folder):
    """Count successful image-analysis outputs when image evidence is offline.

    The image pipeline writes an annotated image only after inference succeeds.
    These outputs are durable proof of completed image analyses, unlike the
    uploaded originals, which may never have been analyzed.
    """
    root = Path(upload_folder) / "annotated"
    if not root.is_dir():
        return 0
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    try:
        return sum(1 for path in root.iterdir()
                   if path.is_file() and path.suffix.casefold() in image_extensions
                   and path.stat().st_size > 0)
    except OSError:
        current_app.logger.exception("Dashboard could not read saved image-analysis outputs")
        return None


def _merge_evidence_records(database_records, artifact_records):
    database_ids = set()
    database_sources = set()
    for record in database_records:
        video = record.get("video_analysis") or {}
        analysis_id = record.get("analysis_id") or video.get("analysis_id")
        if analysis_id:
            database_ids.add(str(analysis_id))
        if record.get("file_type") == "video":
            source = _source_key(record)
            if source:
                database_sources.add(source)
    merged = list(database_records)
    for record in artifact_records:
        analysis_id = str(record.get("analysis_id") or "")
        source = _source_key(record)
        if analysis_id in database_ids or (source and source in database_sources):
            continue
        merged.append(record)
    return merged


@bp.route("/summary", methods=["GET"])
@jwt_required(optional=True)
def summary():
    db = current_app.db
    database_available = db is not None
    try:
        fields = {"filename": 1, "file_type": 1, "status": 1, "uploaded_at": 1,
                  "detections": 1, "risk_score": 1, "event_classification": 1,
                  "video_analysis": 1, "case_id": 1, "analysis_id": 1, "analyzed_at": 1,
                  "evidence_id": 1, "reference_category": 1}
        database_records = _unique_evidence_records(
            list(db.evidence.find({}, fields).sort("uploaded_at", -1))
        ) if database_available else []
        records = list(database_records)
    except Exception:
        current_app.logger.exception("Dashboard could not read MongoDB evidence; loading saved video artifacts instead")
        database_available = False
        records = []
        database_records = []

    # Dashboard analytics describe application records. Local analysis files and
    # training catalogs are not a substitute for a reachable database.
    if not database_available:
        state = current_app.extensions.get("mongo_connection_state", {})
        message = state.get("error") or "MongoDB is unavailable; dashboard statistics and persisted activity cannot be loaded."
        return jsonify({
            "status": "unavailable", "message": message,
            "error_type": state.get("error_type"),
            "total_evidence": None, "total_image_analyses": None,
            "total_video_analyses": None, "total_cases": None,
            "high_risk_cases": None, "reports_generated": None,
            "images_analyzed": None, "videos_analyzed": None,
            "analysis_attempts": None, "completed_analyses": None,
            "failed_analyses": None, "total_detections": None,
            "object_distribution": None, "crime_types": None,
            "video_categories": None, "monthly_evidence": None,
            "risk_distribution": None, "risk_assessments": None,
            "average_risk_score": None, "recent_evidence": None,
            "recent_investigations": None, "recent_detections": None,
            "recent_scene_changes": None, "recent_logs": None,
            "reference_video_categories": None,
            "reference_category_unknown_count": None,
            "custody_status": "unavailable", "recent_custody": None,
        }), 503

    try:
        # Keep all dashboard counts, trends, and recent rows on the same MongoDB
        # evidence source; saved artifact files are not application records.
        completed = [record for record in records if str(record.get("status", "")).casefold() in {"analyzed", "completed", "success"}]
        video_records = [record for record in _latest_video_records(records)
                         if str(record.get("status", "")).casefold() in {"analyzed", "completed", "success"}]
        persisted_video_records = [record for record in _latest_video_records(database_records)
                                   if str(record.get("status", "")).casefold() in {"analyzed", "completed", "success"}]
        dashboard_video_records = persisted_video_records if database_available else video_records
        latest_video_ids = {record.get("_id") for record in video_records}
        analysis_records = [record for record in completed
                            if record.get("file_type") != "video" or record.get("_id") in latest_video_ids]
        case_records = list(db.cases.find({}, {"title": 1, "risk_level": 1, "created_at": 1})) if database_available and "cases" in db.list_collection_names() else []
        cases = len(case_records) if database_available else None
        case_risk_distribution = {"high": 0, "medium": 0, "low": 0}
        for case in case_records:
            level = str(case.get("risk_level") or "").casefold()
            if level in {"high", "critical", "elevated", "medium-high"}:
                case_risk_distribution["high"] += 1
            elif level in {"medium", "moderate"}:
                case_risk_distribution["medium"] += 1
            elif level == "low":
                case_risk_distribution["low"] += 1
        risk_service = RiskService()
        risk_distribution = {band: 0 for band in RISK_BANDS}
        scores = []
        assessed_records = []
        high_risk_cases = case_risk_distribution["high"]
        risk_base_records = database_records if database_available else records
        risk_sources = [record for record in risk_base_records
                        if str(record.get("status", "")).casefold() in {"analyzed", "completed", "success"}
                        and record.get("file_type") != "video"] + dashboard_video_records
        for record in risk_sources:
            risk = _risk_payload(record)
            if risk is None:
                continue
            band = _risk_level(risk, risk_service)
            if band:
                risk_distribution[band] += 1
                assessed_records.append(record)
            score = risk.get("overall_risk", risk.get("score"))
            try:
                score = float(score)
                scores.append(score)
            except (TypeError, ValueError):
                pass

        crime_types = {category: 0 for category in CATEGORIES}
        reference_video_categories = {category: 0 for category in CATEGORIES}
        reference_category_unknown_count = 0
        for record in dashboard_video_records:
            classification = record.get("event_classification")
            label = _resolved_prediction(classification)
            if label:
                crime_types[label] += 1
            reference = (record.get("reference_category") or
                         (record.get("video_analysis") or {}).get("reference_category"))
            reference = str(reference or "").strip().title()
            if reference in reference_video_categories:
                reference_video_categories[reference] += 1
            else:
                reference_category_unknown_count += 1

        total_detections = 0
        object_distribution = {}
        for record in analysis_records:
            detections = record.get("detections") if isinstance(record.get("detections"), list) else []
            total_detections += len(detections)
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                name = detection.get("class") or detection.get("category") or "Unclassified"
                object_distribution[name] = object_distribution.get(name, 0) + 1

        valid_tracking_ids = 0
        for record in dashboard_video_records:
            detections = record.get("detections") if isinstance(record.get("detections"), list) else []
            ids = _valid_track_ids(detections)
            if not ids and isinstance((record.get("video_analysis") or {}).get("tracking"), dict):
                try:
                    valid_tracking_ids += int((record["video_analysis"]["tracking"].get("tracking_id_count")) or 0)
                    continue
                except (TypeError, ValueError):
                    pass
            valid_tracking_ids += len(ids)

        now = datetime.now(timezone.utc)
        month_starts = []
        for offset in range(6, -1, -1):
            index = now.year * 12 + now.month - 1 - offset
            month_starts.append(datetime(index // 12, index % 12 + 1, 1))
        month_counts = {item.strftime("%Y-%m"): {"evidence": 0, "cases": 0} for item in month_starts}
        for record in records:
            uploaded = _parse_timestamp(record.get("uploaded_at"))
            if uploaded:
                key = uploaded.strftime("%Y-%m")
                if key in month_counts:
                    month_counts[key]["evidence"] += 1
        for case in case_records:
            created = _parse_timestamp(case.get("created_at"))
            if created:
                key = created.strftime("%Y-%m")
                if key in month_counts:
                    month_counts[key]["cases"] += 1
        has_upload_timestamps = any(_parse_timestamp(record.get("uploaded_at")) for record in records)
        has_case_timestamps = any(_parse_timestamp(case.get("created_at")) for case in case_records)
        monthly_evidence = ([{"month": item.strftime("%b"),
                              "evidence_count": month_counts[item.strftime("%Y-%m")]["evidence"],
                              "case_count": month_counts[item.strftime("%Y-%m")]["cases"],
                              "count": sum(month_counts[item.strftime("%Y-%m")].values())} for item in month_starts]
                            if has_upload_timestamps or has_case_timestamps or not records and not case_records else None)

        reports_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports")
        reports_generated = sum(1 for name in os.listdir(reports_dir) if name.lower().endswith(".pdf")) if os.path.isdir(reports_dir) else 0
        recent = [serialize_evidence(record) for record in records[:10]]
        recent_investigations = []
        for record in dashboard_video_records:
            risk = _risk_payload(record)
            video = record.get("video_analysis") or {}
            tracking = video.get("tracking") or {}
            classification = record.get("event_classification") or {}
            predicted_category = _resolved_prediction(classification)
            model_candidate = ((classification.get("whole_video_prediction") or {}).get("label")
                               if isinstance(classification, dict) else None)
            if not model_candidate and isinstance(classification, dict):
                model_candidate = classification.get("label")
            recent_investigations.append({
                "id": str(record.get("_id", "")), "filename": record.get("filename"),
                "category": predicted_category,
                "predicted_category": predicted_category,
                "model_candidate_category": model_candidate,
                "reference_category": record.get("reference_category") or video.get("reference_category"),
                "classification_status": (classification.get("classification_status") if isinstance(classification, dict) else None),
                "analyzed_at": _iso_utc(record.get("analyzed_at") or record.get("uploaded_at")),
                "risk_score": risk.get("overall_risk", risk.get("score")) if risk else None,
                "risk_level": risk.get("risk_level", risk.get("threat_level")) if risk else None,
                "risk_factors": _risk_factors(risk),
                "status": record.get("status") or "unknown",
                "tracking_status": tracking.get("status") or tracking.get("method") or ("ByteTrack used" if tracking.get("used") else "unavailable"),
            })
        recent_investigations.sort(key=lambda row: row["analyzed_at"] or "", reverse=True)
        recent_detections = []
        successful_records = [row for row in database_records
                              if str(row.get("status", "")).casefold() in {"analyzed", "completed", "success"}]
        for record in successful_records:
            detections = record.get("detections") if isinstance(record.get("detections"), list) else []
            if detections:
                recent_detections.append({
                    "id": str(record.get("_id", "")), "filename": record.get("filename"),
                    "file_type": record.get("file_type"), "detection_count": len(detections),
                    "objects": list(dict.fromkeys(str(item.get("class") or item.get("category") or "Unclassified")
                                                   for item in detections if isinstance(item, dict)))[:5],
                    "analyzed_at": _iso_utc(record.get("analyzed_at") or record.get("uploaded_at")),
                })
        recent_detections.sort(key=lambda row: row["analyzed_at"] or "", reverse=True)
        recent_logs = []
        recent_scene_changes = []
        custody_metrics = {
            "custody_status": "unavailable" if not database_available else "available",
            "total_custody_entries": None,
            "unique_evidence_with_custody": None,
            "evidence_with_complete_custody": None,
            "evidence_missing_custody": None,
            "recent_custody": [],
        }
        if database_available:
            try:
                custody_collection = db["custody_logs"]
                custody_metrics["total_custody_entries"] = custody_collection.count_documents({})
                unique_ids = set(str(value) for value in custody_collection.distinct("evidence_id") if value)
                custody_metrics["unique_evidence_with_custody"] = len(unique_ids)
                recent_logs = CustodyService(db).get_logs(limit=10)
                scene_logs = list(custody_collection.find(
                    {"action": {"$regex": "^scene_change"}}
                ).sort("created_at", -1).limit(5))
                recent_scene_changes = [{
                    "id": str(row.get("_id", "")), "action": row.get("action"),
                    "evidence_id": row.get("evidence_id"), "user_id": row.get("user_id"),
                    "timestamp": row.get("timestamp") or _iso_utc(row.get("created_at")),
                    "details": row.get("details") if isinstance(row.get("details"), dict) else {},
                } for row in scene_logs]

                evidence_by_id = {}
                case_titles = {str(case.get("_id")): case.get("title") for case in case_records}
                linked_evidence_ids = set()
                for evidence in records:
                    evidence_id = evidence.get("evidence_id")
                    if evidence_id:
                        evidence_id = str(evidence_id)
                        linked_evidence_ids.add(evidence_id)
                        evidence_by_id[evidence_id] = {
                            "filename": evidence.get("filename"),
                            "case_id": str(evidence.get("case_id") or "") or None,
                        }
                for log in recent_logs:
                    evidence_id = str(log.get("evidence_id") or "")
                    details = log.get("details") if isinstance(log.get("details"), dict) else {}
                    evidence = evidence_by_id.get(evidence_id) or {}
                    log["filename"] = (evidence.get("filename") or details.get("original_name") or
                                       details.get("filename"))
                    log["case_id"] = evidence.get("case_id")
                    log["case"] = case_titles.get(log["case_id"], log["case_id"])
                    log["previous_custodian"] = details.get("previous_custodian")
                    log["new_custodian"] = details.get("new_custodian")
                    log["record_status"] = "Integrity hash recorded" if log.get("integrity_hash") else "Recorded"
                custody_metrics["recent_custody"] = recent_logs

                # Older evidence docs do not carry the upload evidence_id, so only
                # report coverage for docs that can be joined to custody reliably.
                linked_ids = {str(row.get("evidence_id")) for row in records if row.get("evidence_id")}
                if linked_ids:
                    history_by_id = {evidence_id: [] for evidence_id in linked_ids}
                    for entry in custody_collection.find({"evidence_id": {"$in": list(linked_ids)}}):
                        key = str(entry.get("evidence_id") or "")
                        if key in history_by_id:
                            history_by_id[key].append(entry)
                    complete_ids = {
                        evidence_id for evidence_id, entries in history_by_id.items()
                        if entries
                        and any(str(entry.get("action", "")).startswith("upload_") for entry in entries)
                        and all(entry.get("timestamp") and entry.get("integrity_hash") for entry in entries)
                    }
                    custody_metrics["evidence_with_complete_custody"] = len(complete_ids)
                    custody_metrics["evidence_missing_custody"] = len(linked_ids - complete_ids)
            except Exception:
                current_app.logger.exception("Dashboard could not read chain-of-custody records")
                custody_metrics.update({
                    "custody_status": "error", "total_custody_entries": None,
                    "unique_evidence_with_custody": None,
                    "evidence_with_complete_custody": None,
                    "evidence_missing_custody": None, "recent_custody": [],
                })
        failed_records = [record for record in records if str(record.get("status", "")).casefold() in {"failed", "failure", "error"}]
        failed_analyses = len(failed_records) if database_available or failed_records else None
        response_status = "success" if database_available else "partial"
        return jsonify({
            "status": response_status,
            "message": None,
            # Counts that require the full evidence collection stay unknown in
            # partial mode. Local artifacts can recover useful results, but do
            # not prove that no other image/evidence records exist.
            "total_evidence": len(database_records) if database_available else None,
            "total_image_analyses": sum(1 for row in database_records if row.get("file_type") == "image" and str(row.get("status", "")).casefold() in {"analyzed", "completed", "success"}) if database_available else None,
            "total_video_analyses": len(persisted_video_records) if database_available else None,
            "total_cases": cases, "high_risk_cases": high_risk_cases if database_available else None,
            "case_risk_distribution": case_risk_distribution if database_available else None,
            "reports_generated": reports_generated,
            "images_analyzed": sum(1 for row in database_records if row.get("file_type") == "image" and str(row.get("status", "")).casefold() in {"analyzed", "completed", "success"}),
            "images_analyzed_source": "database",
            "videos_analyzed": len(persisted_video_records), "analysis_attempts": len(completed) + len(failed_records),
            "completed_analyses": len(completed), "failed_analyses": failed_analyses,
            "total_detections": total_detections, "object_distribution": object_distribution,
            "crime_types": crime_types, "video_categories": crime_types,
            "reference_video_categories": reference_video_categories,
            "reference_category_unknown_count": reference_category_unknown_count,
            "monthly_evidence": monthly_evidence,
            "risk_distribution": risk_distribution, "risk_assessments": len(assessed_records),
            "average_risk_score": round(sum(scores) / len(scores), 2) if scores else None,
            "risk_thresholds": risk_service.thresholds, "valid_tracking_ids": valid_tracking_ids,
            "recent_evidence": recent, "recent_investigations": recent_investigations[:10],
            "recent_detections": recent_detections[:5], "recent_scene_changes": recent_scene_changes,
            "recent_logs": recent_logs, **custody_metrics,
        })
    except PyMongoError as exc:
        from app import _mongo_connection_error, refresh_database_connection
        message = _mongo_connection_error(exc, current_app.config["MONGO_URI"])
        current_app.logger.warning("Dashboard summary lost MongoDB connectivity: %s", message)
        # Re-ping through the shared connection manager to mark the outage and
        # expose the same safe, actionable reason as the health endpoint.
        refresh_database_connection(current_app._get_current_object(), force=True)
        state = current_app.extensions.get("mongo_connection_state", {})
        state.setdefault("error", message)
        state.setdefault("error_type", type(exc).__name__)
        return jsonify({
            "status": "unavailable",
            "message": state.get("error") or "MongoDB is unavailable; persisted dashboard data cannot be loaded.",
            "error_type": state.get("error_type"),
            "total_evidence": None, "total_image_analyses": None,
            "total_video_analyses": None, "total_cases": None,
            "high_risk_cases": None, "reports_generated": None,
            "images_analyzed": None, "videos_analyzed": None,
            "analysis_attempts": None, "completed_analyses": None,
            "failed_analyses": None, "total_detections": None,
            "object_distribution": None, "crime_types": None,
            "video_categories": None, "monthly_evidence": None,
            "risk_distribution": None, "risk_assessments": None,
            "average_risk_score": None, "recent_evidence": None,
            "recent_investigations": None, "recent_detections": None,
            "recent_scene_changes": None, "recent_logs": None,
            "reference_video_categories": None,
            "reference_category_unknown_count": None,
            "custody_status": "unavailable", "recent_custody": None,
        }), 503
    except Exception:
        current_app.logger.exception("Dashboard summary aggregation failed")
        return jsonify({"status": "error", "error": "Dashboard statistics could not be loaded."}), 500


@bp.route("/history", methods=["GET"])
@jwt_required(optional=True)
def history():
    db = current_app.db
    if db is None:
        return jsonify({"status": "unavailable", "message": "MongoDB is disconnected; custody history is unavailable.", "logs": []}), 503
    evidence_id = request.args.get("evidence_id")
    return jsonify({"status": "success", "evidence_id": evidence_id,
                    "logs": CustodyService(db).get_logs(evidence_id=evidence_id, limit=100)})
