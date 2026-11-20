from flask import Blueprint, request, jsonify, current_app, send_file, abort
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.video_service import VideoService
from app.services.video_service import _json_safe
from app.services.evidence_catalog_service import EvidenceCatalogService
from app.services.video_classification_model import VideoModelUnavailableError
from app.services.bigru_mil_model import (
    BigruMILModel,
    ExperimentalModelUnavailableError,
    InvalidFeatureInputError,
    SEGMENT_COUNT,
)
from app.services.r3d18_feature_service import (
    InvalidExperimentalVideoError,
    MAX_FEATURE_FRAMES,
    R3D18FeatureService,
    R3D18UnavailableError,
)
from app.services.risk_service import RiskService
from app.services.custody_service import CustodyService
from app.models.evidence import create_evidence_doc
from app.services.temporal_event_association_service import TemporalEventAssociationService
from app.models.case import update_case_risk
import os
import math
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from app.utils.file_handler import allowed_file

bp = Blueprint("video", __name__)
SUPPORTED_EVENT_CATEGORIES = {"shooting", "assault", "robbery"}


def _persist_failed_video_analysis(video_path, original_filename, failure_code, evidence_id=None):
    """Record a failed run alongside its video, and in MongoDB when available."""
    analysis_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    try:
        video_path = Path(video_path).resolve()
        try:
            uploaded_at = datetime.fromtimestamp(video_path.stat().st_mtime, timezone.utc)
        except OSError:
            uploaded_at = None
        artifact_dir = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence" / analysis_id).resolve()
        artifact_root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
        artifact_dir.relative_to(artifact_root)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 2, "analysis_id": analysis_id,
            "status": "failed", "file_type": "video",
            "filename": original_filename or video_path.name,
            "original_filename": original_filename or video_path.name,
            "source_video_identity": video_path.name,
            "uploaded_at": uploaded_at.isoformat().replace("+00:00", "Z") if uploaded_at else None,
            "analyzed_at": now.isoformat().replace("+00:00", "Z"),
            "failure_code": failure_code,
        }
        artifact_path = artifact_dir / "analysis.json"
        temp_path = artifact_path.with_name(f"{artifact_path.name}.tmp")
        temp_path.write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
        os.replace(temp_path, artifact_path)
    except Exception:
        current_app.logger.exception("Could not persist failed video analysis artifact")
        return analysis_id

    if current_app.db is not None:
        try:
            doc = create_evidence_doc(None, original_filename or video_path.name, "video", [], None)
            doc.update({"status": "failed", "analysis_id": analysis_id,
                        "analyzed_at": now, "failure_code": failure_code,
                        "evidence_id": str(evidence_id) if evidence_id else None,
                        "video_analysis": {"analysis_id": analysis_id,
                                           "source_video_path": str(video_path),
                                           "source_video_identity": video_path.name,
                                           "failure_code": failure_code}})
            existing = current_app.db.evidence.find_one({"evidence_id": str(evidence_id)}) if evidence_id else None
            if existing:
                doc["uploaded_at"] = existing.get("uploaded_at", doc["uploaded_at"])
                doc["case_id"] = existing.get("case_id")
                current_app.db.evidence.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {key: value for key, value in doc.items() if key not in {"uploaded_at", "_id"}}},
                )
            else:
                current_app.db.evidence.insert_one(doc)
        except Exception:
            current_app.logger.exception("Failed video run was saved to disk but not MongoDB")
    return analysis_id


@bp.route("/tracking-inventory", methods=["GET"])
@jwt_required(optional=True)
def tracking_inventory():
    """Return a read-only inventory of locally available tracking artifacts."""
    return jsonify({"success": True, **EvidenceCatalogService(current_app.config).tracking_artifact_inventory()})


@bp.route("/analyze-experimental", methods=["POST"])
@jwt_required(optional=True)
def analyze_video_experimental():
    """Extract experimental R3D-18 features from an already uploaded video."""
    data = request.get_json(silent=True)
    filepath = data.get("filepath") if isinstance(data, dict) else None
    if not isinstance(filepath, str) or not filepath:
        return jsonify({"success": False, "code": "video_file_missing", "error": "JSON body must include the uploaded video filepath."}), 400
    path = Path(filepath).resolve()
    if not path.is_file():
        return jsonify({"success": False, "code": "video_file_missing", "error": "Uploaded video file was not found. Upload it again and retry."}), 400
    video_root = (Path(current_app.config["UPLOAD_FOLDER"]) / "videos").resolve()
    try:
        path.relative_to(video_root)
    except ValueError:
        return jsonify({"success": False, "code": "invalid_video_path", "error": "Experimental analysis accepts only videos uploaded through the video upload endpoint."}), 400
    if not allowed_file(path.name, current_app.config.get("ALLOWED_VIDEO_EXTENSIONS", {"mp4", "avi", "mov", "mkv", "webm"})):
        return jsonify({"success": False, "code": "unsupported_video_format", "error": "Unsupported video format. Use MP4, AVI, MOV, MKV, or WEBM."}), 415

    try:
        features, timestamps, duration, frame_count, fps = R3D18FeatureService().extract(
            path, current_app.config.get("R3D18_MAX_DURATION_SEC", current_app.config["VIDEO_MAX_DURATION_SEC"])
        )
        model = BigruMILModel.get(current_app.config["ANOMALY_MODEL_PATH"])
        raw_logits = model.infer(features)
    except InvalidExperimentalVideoError as error:
        return jsonify({"success": False, "code": "experimental_video_invalid", "error": str(error)}), 400
    except R3D18UnavailableError as error:
        current_app.logger.error("Experimental R3D-18 extractor unavailable: %s", error)
        return jsonify({"success": False, "code": "r3d18_unavailable", "error": str(error)}), 503
    except ExperimentalModelUnavailableError as error:
        current_app.logger.error("Experimental anomaly model unavailable: %s", error)
        return jsonify({"success": False, "code": "anomaly_model_unavailable", "error": str(error)}), 503
    except InvalidFeatureInputError as error:
        current_app.logger.exception("R3D-18 features failed BiGRU/MIL input validation")
        return jsonify({"success": False, "code": "invalid_extracted_features", "error": str(error)}), 500
    except Exception as error:
        current_app.logger.exception("Experimental video inference failed")
        return jsonify({"success": False, "code": "experimental_video_inference_failed", "error": f"Experimental video inference failed ({type(error).__name__})."}), 500

    return jsonify({
        "success": True,
        "analysis_mode": "experimental_r3d18_bigru_mil",
        "verification_status": "unverified",
        "requires_human_review": True,
        "score_type": "raw_logit_not_probability",
        "feature_shape": [SEGMENT_COUNT, 512],
        "video_info": {"duration_sec": duration, "total_frames": frame_count, "fps": fps,
                       "segments": SEGMENT_COUNT, "sampled_frames_per_segment": 16,
                       "feature_frames_max": MAX_FEATURE_FRAMES},
        "segment_mapping": "32 equal-duration segments; timestamp_sec is each segment midpoint computed from video duration (frame_count / FPS). It is a reference time, not an event boundary.",
        "segment_scores": [
            {"segment_index": index, "raw_logit": raw_logits[index], "timestamp_sec": timestamps[index]}
            for index in range(SEGMENT_COUNT)
        ],
        "limitations": [
            "Experimental R3D-18 Kinetics-400 preprocessing and equal-duration segmentation are not known to match BiGRU/MIL training.",
            "The original BiGRU/MIL implementation and training setup have not been recovered; checkpoint architecture fidelity is unverified.",
            "Raw logits are not probabilities, accuracy measurements, confirmed crime detections, or incident boundaries.",
        ],
    })


@bp.route("/analyze-features", methods=["POST"])
@jwt_required(optional=True)
def analyze_precomputed_features():
    """Experimental inference for precomputed R3D-18 feature sequences only."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "features" not in data:
        return jsonify({"success": False, "code": "invalid_features", "error": "JSON body must include a features array with shape (32, 512)."}), 400

    timestamps = data.get("timestamps_sec")
    if timestamps is not None:
        if not isinstance(timestamps, list) or len(timestamps) != SEGMENT_COUNT or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in timestamps
        ):
            return jsonify({"success": False, "code": "invalid_timestamps", "error": "timestamps_sec must contain 32 finite, non-negative timestamps."}), 400
        times = [float(value) for value in timestamps]
        if any(not math.isfinite(value) or value < 0 for value in times):
            return jsonify({"success": False, "code": "invalid_timestamps", "error": "timestamps_sec must contain 32 finite, non-negative timestamps."}), 400
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            return jsonify({"success": False, "code": "invalid_timestamps", "error": "timestamps_sec must be in non-decreasing video time order."}), 400
    else:
        times = [None] * SEGMENT_COUNT

    try:
        model = BigruMILModel.get(current_app.config["ANOMALY_MODEL_PATH"])
        raw_logits = model.infer(data["features"])
    except InvalidFeatureInputError as error:
        return jsonify({"success": False, "code": "invalid_features", "error": str(error)}), 400
    except ExperimentalModelUnavailableError as error:
        current_app.logger.error("Experimental anomaly model unavailable: %s", error)
        return jsonify({"success": False, "code": "anomaly_model_unavailable", "error": str(error)}), 503
    except Exception as error:
        current_app.logger.exception("Experimental feature inference failed")
        return jsonify({"success": False, "code": "feature_inference_failed", "error": f"Experimental inference failed ({type(error).__name__})."}), 500

    segments = [
        {"segment_index": index, "raw_logit": raw_logits[index], "timestamp_sec": times[index]}
        for index in range(SEGMENT_COUNT)
    ]
    return jsonify({
        "success": True,
        "analysis_mode": "experimental_precomputed_features",
        "verification_status": "unverified",
        "requires_human_review": True,
        "score_type": "raw_logit_not_probability",
        "candidate_architecture_fidelity": "unverified_relu_candidate",
        "feature_shape": [SEGMENT_COUNT, 512],
        "segment_scores": segments,
        "limitations": [
            "This candidate architecture matches checkpoint parameter shapes, but its ReLU choices and dropout setting are unverified because the original training class is unavailable.",
            "The R3D-18 feature extractor, frame preprocessing, and temporal segmentation details are not available in this checkout.",
            "Raw logits are not calibrated probabilities, crime findings, or confirmed incident labels.",
            "No incident boundaries are inferred by this experimental endpoint.",
        ],
    })


@bp.route("/evidence/<analysis_id>/<filename>", methods=["GET"])
def download_video_evidence(analysis_id, filename):
    """Serve only generated timeline assets from the evidence upload directory."""
    if not analysis_id.isalnum() or len(analysis_id) != 32:
        abort(404)
    if os.path.basename(filename) != filename or filename in {".", ".."}:
        abort(404)
    if filename not in {"evidence_timeline.csv", "evidence_events.csv", "tracking_summary.csv", "analysis.json", "annotated_tracking.mp4"} and not (
        filename.endswith(".jpg") and filename.startswith(("frame_", "original_frame_"))
    ):
        abort(404)
    root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
    target = (root / analysis_id / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    return send_file(target, as_attachment=filename.endswith(".csv"), download_name=filename,
                     mimetype="video/mp4" if filename.endswith(".mp4") else None)


@bp.route("/evidence/<analysis_id>/shooting/<event_id>/<path:filename>", methods=["GET"])
@jwt_required(optional=True)
def download_shooting_event_artifact(analysis_id, event_id, filename):
    """Serve generated Shooting event files beneath the evidence directory."""
    root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
    event_root = (root / analysis_id / "shooting-events" / event_id).resolve()
    target = (event_root / filename).resolve()
    try:
        event_root.relative_to(root)
        target.relative_to(event_root)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid event artifact path."}), 400
    if not target.is_file():
        return jsonify({"success": False, "error": "Shooting event artifact was not found."}), 404
    return send_file(target, conditional=True, as_attachment=target.suffix.lower() == ".csv",
                     download_name=target.name)


@bp.route("/evidence/<analysis_id>/events/<category>/<event_id>/<path:filename>", methods=["GET"])
@jwt_required(optional=True)
def download_event_evidence_artifact(analysis_id, category, event_id, filename):
    category = category.casefold()
    if (category not in SUPPORTED_EVENT_CATEGORIES or not analysis_id.isalnum() or len(analysis_id) != 32
            or len(event_id) != 24 or any(character not in "0123456789abcdef" for character in event_id.casefold())):
        abort(404)
    root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
    event_root = (root / analysis_id / "category-events" / category / event_id).resolve()
    target = (event_root / filename).resolve()
    try:
        event_root.relative_to(root)
        target.relative_to(event_root)
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    return send_file(target, conditional=True, as_attachment=target.suffix.lower() == ".csv",
                     download_name=target.name)


@bp.route("/shooting-events/<analysis_id>", methods=["POST"])
@bp.route("/event-evidence/<analysis_id>", methods=["POST"])
@jwt_required(optional=True)
def create_shooting_event_evidence(analysis_id):
    """Extract an event package from an existing analysis and saved tracking run."""
    data = request.get_json(silent=True) or {}
    evidence_root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
    artifact_path = (evidence_root / analysis_id / "analysis.json").resolve()
    try:
        artifact_path.relative_to(evidence_root)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid analysis ID."}), 400
    if not artifact_path.is_file():
        return jsonify({"success": False, "error": "Video analysis was not found."}), 404
    try:
        analysis = json.loads(artifact_path.read_text(encoding="utf-8"))
        category = str(analysis.get("category", "")).strip().casefold()
        if category not in SUPPORTED_EVENT_CATEGORIES:
            return jsonify({"success": False, "error": "Event evidence supports Assault, Robbery, and Shooting analyses."}), 400
        source = Path(analysis.get("source_video_path") or "").resolve()
        videos_root = (Path(current_app.config["UPLOAD_FOLDER"]) / "videos").resolve()
        source.relative_to(videos_root)
        if not source.is_file():
            raise FileNotFoundError("Original uploaded video is unavailable.")
        video_info = analysis.get("video_info") or {}
        start = data.get("start_timestamp_sec", data.get("start_sec"))
        end = data.get("end_timestamp_sec", data.get("end_sec"))
        classifier_metadata = analysis.get("classification") or analysis.get("event_classification") or {}
        try:
            start_value, end_value = float(start), float(end)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Candidate start and end timestamps are required."}), 400
        candidates = classifier_metadata.get("activity_events") or analysis.get("activity_events") or []
        matched_candidate = next((item for item in candidates
            if str(item.get("category", "")).casefold() == category
            and abs(float(item.get("start_timestamp_sec", -1)) - start_value) <= .01
            and abs(float(item.get("end_timestamp_sec", -1)) - end_value) <= .01), None)
        legacy_shooting_request = category == "shooting" and "/shooting-events/" in request.path
        if matched_candidate is None and not legacy_shooting_request:
            return jsonify({"success": False, "error": "Requested interval is not a classifier candidate from this analysis."}), 400
        if matched_candidate is not None:
            start, end = matched_candidate["start_timestamp_sec"], matched_candidate["end_timestamp_sec"]
        service = TemporalEventAssociationService(current_app.config)
        output_root = (evidence_root / analysis_id / "shooting-events" if category == "shooting"
                       else evidence_root / analysis_id / "category-events" / category)
        record = service.create_event(
            original_path=source,
            output_root=output_root,
            start_sec=start, end_sec=end,
            category=category, source_video_name=analysis.get("original_filename") or analysis.get("filename") or source.name,
            video_info=video_info, classifier_confidence=(matched_candidate or {}).get("confidence", classifier_metadata.get("confidence")),
            temporal_predictions=classifier_metadata.get("temporal_windows") or classifier_metadata.get("window_predictions") or [],
            classifier_probabilities=classifier_metadata.get("probabilities") or {},
            # Analysis metadata is server-persisted. Client-supplied detections,
            # action labels, or audio labels cannot influence refinement scores.
            yolo_detections=analysis.get("yolo_detections") or [],
            additional_evidence=analysis.get("validated_additional_evidence") or {},
            verification_status="unverified_candidate",
            context_before_sec=data.get("context_before_sec", current_app.config.get("EVENT_CONTEXT_BEFORE_SEC", current_app.config.get("SHOOTING_EVENT_CONTEXT_BEFORE_SEC", 2))),
            context_after_sec=data.get("context_after_sec", current_app.config.get("EVENT_CONTEXT_AFTER_SEC", current_app.config.get("SHOOTING_EVENT_CONTEXT_AFTER_SEC", 2))),
        )
        if current_app.db is not None:
            stored_record = {"evidence_id": record["evidence_id"], "filename": source.name,
                             "file_type": f"{category}_event", "status": "analyzed",
                             "event_evidence": record, "uploaded_at": datetime.now(timezone.utc),
                             "uploaded_by": get_jwt_identity() or "anonymous"}
            if category == "shooting":
                stored_record["shooting_event"] = record
            current_app.db.evidence.update_one(
                {"evidence_id": record["evidence_id"]},
                {"$setOnInsert": stored_record},
                upsert=True,
            )
        response = {"success": True, "event_evidence": record}
        if category == "shooting":
            response["shooting_event"] = record
        return jsonify(response), 200
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"success": False, "error": str(error)}), 400
    except Exception:
        current_app.logger.exception("Event evidence extraction failed")
        return jsonify({"success": False, "error": "Could not create event evidence."}), 500

@bp.route("/analyze", methods=["POST"])
@jwt_required(optional=True)
def analyze_video():
    data = request.get_json() or {}
    filepath = data.get("filepath")
    evidence_id = data.get("evidence_id")
    category = str(data.get("category") or "").strip().casefold()
    original_filename = str(data.get("original_filename") or "").strip()
    if category and category not in {"assault", "robbery", "shooting", "normal"}:
        return jsonify({"success": False, "code": "invalid_video_category",
                        "error": "Category must be Assault, Robbery, Shooting, or Normal."}), 400
    if original_filename and Path(original_filename).name != original_filename:
        return jsonify({"success": False, "code": "invalid_original_filename",
                        "error": "Original filename must be a filename, not a path."}), 400
    requested = data.get("max_frames")
    full_video = data.get("full_video") is True
    max_frames = None
    if requested is not None and not full_video:
        try:
            max_frames = max(1, min(int(requested), current_app.config["VIDEO_MAX_PROCESSED_FRAMES"]))
        except (TypeError, ValueError):
            max_frames = None

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"success": False, "code": "video_file_missing", "error": "Uploaded video file was not found. Upload the video again and retry."}), 400
    video_path = Path(filepath).resolve()
    video_root = (Path(current_app.config["UPLOAD_FOLDER"]) / "videos").resolve()
    try:
        video_path.relative_to(video_root)
    except ValueError:
        return jsonify({"success": False, "code": "invalid_video_path", "error": "Analyze only videos uploaded through the video upload endpoint."}), 400
    if not allowed_file(video_path.name, current_app.config.get("ALLOWED_VIDEO_EXTENSIONS", {"mp4", "avi", "mov", "mkv", "webm"})):
        return jsonify({"success": False, "code": "unsupported_video_format", "error": "Unsupported video format. Use MP4, AVI, MOV, MKV, or WEBM."}), 415

    try:
        service = VideoService()
        analysis_options = ({"category": category,
                            "original_filename": original_filename or video_path.name,
                            "create_annotated_video": full_video}
                           if category else {})
        result = service.analyze(str(video_path), max_frames=max_frames,
                                 full_video=full_video, **analysis_options)

        # Aggregate detections for risk
        all_dets = []
        for fr in result["frame_results"]:
            all_dets.extend(fr["detections"])

        risk = RiskService().calculate(all_dets, context=result)
        result["risk"] = risk

        # The video service first writes a compact artifact index. Enrich that
        # same file with the actual completed-run summary and Risk Service output
        # so successful runs survive even when MongoDB is offline.
        artifact_path = Path((result.get("artifact_paths") or {}).get("analysis_json") or "").resolve()
        artifact_root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
        analyzed_at = datetime.now(timezone.utc)
        try:
            artifact_path.relative_to(artifact_root)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            try:
                uploaded_at = datetime.fromtimestamp(video_path.stat().st_mtime, timezone.utc)
            except OSError:
                uploaded_at = None
            tracking = result.get("tracking") if isinstance(result.get("tracking"), dict) else {}
            valid_ids = set()
            for detection in all_dets:
                track_id = detection.get("track_id") if isinstance(detection, dict) else None
                if track_id is None or isinstance(track_id, bool):
                    continue
                try:
                    value = float(track_id)
                    if value >= 0 and value.is_integer():
                        valid_ids.add(int(value))
                except (TypeError, ValueError):
                    continue
            artifact.update({
                "schema_version": max(2, int(artifact.get("schema_version") or 1)),
                "status": "completed", "file_type": "video",
                "filename": original_filename or video_path.name,
                "original_filename": original_filename or video_path.name,
                "source_video_path": str(video_path),
                "source_video_identity": video_path.name,
                "uploaded_at": uploaded_at.isoformat().replace("+00:00", "Z") if uploaded_at else None,
                "analyzed_at": analyzed_at.isoformat().replace("+00:00", "Z"),
                "event_classification": result.get("classification"),
                "reference_category": result.get("category"),
                "risk": risk, "risk_score": risk,
                "total_detections": len(all_dets),
                "valid_tracking_ids": len(valid_ids),
                "tracking": tracking, "video_info": result.get("video_info") or {},
                "source_integrity": artifact.get("source_integrity"),
                "source_metadata": artifact.get("source_metadata"),
            })
            temp_path = artifact_path.with_name(f"{artifact_path.name}.tmp")
            temp_path.write_text(json.dumps(_json_safe(artifact), allow_nan=False), encoding="utf-8")
            os.replace(temp_path, artifact_path)
            result["persistence"] = {"artifact": "saved"}
        except Exception:
            current_app.logger.exception("Could not persist completed video analysis summary to its artifact directory")
            result["persistence"] = {"artifact": "incomplete"}

        user_id = get_jwt_identity() or "anonymous"
        if current_app.db is not None:
            try:
                doc = create_evidence_doc(None, original_filename or os.path.basename(filepath), "video", all_dets, risk, user_id)
                doc["evidence_id"] = str(evidence_id) if evidence_id else None
                doc["analysis_id"] = result.get("analysis_id")
                doc["analyzed_at"] = analyzed_at
                doc["event_classification"] = result.get("classification")
                doc["reference_category"] = result.get("category")
                doc["video_analysis"] = {
                    "analysis_id": result.get("analysis_id"),
                    "source_video_path": result.get("source_video_path"),
                    "source_video_identity": video_path.name,
                    "reference_category": result.get("category"),
                    "artifact_paths": result.get("artifact_paths"),
                    "video_info": result.get("video_info"),
                    "tracking": result.get("tracking"),
                    "tracking_data_match": result.get("tracking_data_match"),
                    "evidence_events": result.get("evidence_events", []),
                    "activity_events": result.get("activity_events", []),
                    "candidate_interval_status": result.get("candidate_interval_status"),
                    "verification_status": result.get("verification_status"),
                }
                existing = current_app.db.evidence.find_one({"evidence_id": str(evidence_id)}) if evidence_id else None
                if existing:
                    doc["uploaded_at"] = existing.get("uploaded_at", doc["uploaded_at"])
                    doc["case_id"] = existing.get("case_id")
                    current_app.db.evidence.update_one({"_id": existing["_id"]}, {"$set": {key: value for key, value in doc.items() if key != "uploaded_at"}})
                    result["evidence"] = {"id": str(existing["_id"])}
                else:
                    stored = current_app.db.evidence.insert_one(doc)
                    result["evidence"] = {"id": str(stored.inserted_id)}
                if doc.get("case_id"):
                    try:
                        update_case_risk(current_app.db, doc["case_id"], risk, analyzed_at)
                    except Exception:
                        current_app.logger.exception("Video analysis was saved but the linked case risk summary could not be updated")
                result["persistence"]["database"] = "saved"
                if evidence_id:
                    custody = CustodyService(current_app.db)
                    custody.log(evidence_id, "analyze_video", user_id, {
                        "analysis_id": result.get("analysis_id"),
                        "frames_analyzed": result["video_info"]["frames_analyzed"],
                        "total_detections": result["total_detections"]
                    })
            except Exception:
                current_app.logger.exception("Video analysis completed but MongoDB persistence failed")
                result["persistence"]["database"] = "unavailable"
        else:
            current_app.logger.warning("Video analysis %s completed; MongoDB is unavailable, retained in analysis artifact", result.get("analysis_id"))
            result["persistence"]["database"] = "unavailable"

        result["risk"] = risk
        result["evidence_id"] = evidence_id
        return jsonify({
            "status": "success",
            "success": True,
            **result
        })
    except VideoModelUnavailableError as e:
        failure_id = _persist_failed_video_analysis(video_path, original_filename or video_path.name, "video_classifier_unavailable", evidence_id)
        current_app.logger.error("Video classifier unavailable: %s", e)
        return jsonify({
            "success": False,
            "analysis_id": failure_id,
            "code": "video_classifier_unavailable",
            "error": f"{e} Video upload succeeded, but classification requires a compatible trained model. Upload and preview remain available.",
        }), 503
    except ValueError as e:
        failure_id = _persist_failed_video_analysis(video_path, original_filename or video_path.name, "video_decode_failed", evidence_id)
        current_app.logger.info("Video could not be decoded: %s", e)
        return jsonify({"success": False, "analysis_id": failure_id, "code": "video_decode_failed",
                        "error": f"Video could not be decoded: {e}. Try a valid MP4, AVI, MOV, MKV, or WEBM file."}), 400
    except Exception as e:
        failure_id = _persist_failed_video_analysis(video_path, original_filename or video_path.name, "video_inference_failed", evidence_id)
        current_app.logger.exception("Video analysis failed")
        return jsonify({"success": False, "analysis_id": failure_id, "code": "video_inference_failed",
                        "error": f"Video inference failed ({type(e).__name__}). Check the backend logs and confirm the model files and video codecs are valid."}), 500
