
from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.yolo_service import YOLOService
from app.services.risk_service import RiskService
from app.services.custody_service import CustodyService
from app.models.evidence import create_evidence_doc, serialize_evidence
from app.models.case import update_case_risk
from app.services.hybrid_detection_service import HybridDetectionService
from app.utils.file_handler import allowed_file

import os
import json
import time
from pathlib import Path
from PIL import Image

bp = Blueprint("detection", __name__)


@bp.route("/status", methods=["GET"])
def model_status():
    """Check if YOLO model is loaded and which path is used."""
    try:
        yolo = YOLOService()
        info = yolo.model_info

        return jsonify({
            "status": "ready",
            "model_name": os.path.basename(info["path"] or ""),
            "num_classes": len(info["classes"]),
            "sample_classes": info["classes"][:20]
        })

    except Exception:
        current_app.logger.exception("YOLO model status check failed")

        return jsonify({
            "status": "error",
            "message": "YOLO model could not be loaded.",
            "hint": (
                "Check YOLO_MODEL_PATH and ensure "
                "the trained weights exist."
            )
        }), 503


@bp.route("/analyze", methods=["POST"])
@jwt_required()
def analyze_image():
    """Run YOLO detection on an uploaded image."""

    request_started = time.perf_counter()

    def log_phase(name, started):
        current_app.logger.info(
            "image_analysis_timing stage=%s duration_ms=%.1f",
            name,
            (time.perf_counter() - started) * 1000,
        )

    current_app.logger.info(
        "image_analysis_request_received content_length=%s content_type=%s",
        request.content_length,
        request.mimetype or "unknown",
    )

    filepath = None
    original_filename = None
    evidence_id = None
    case_id = None

    upload_started = time.perf_counter()
    if request.content_type and "multipart/form-data" in request.content_type:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if not allowed_file(
            file.filename,
            current_app.config["ALLOWED_IMAGE_EXTENSIONS"]
        ):
            return jsonify({"error": "Unsupported image format"}), 400

        from app.utils.file_handler import save_uploaded_file

        filepath, original_filename = save_uploaded_file(file, "images")
        case_id = request.form.get("case_id") or None
        evidence_id = request.form.get("evidence_id")

    else:
        data = request.get_json(silent=True) or {}
        filepath = data.get("filepath")
        evidence_id = data.get("evidence_id")
        case_id = data.get("case_id") or None

    if request.mimetype == "multipart/form-data":
        log_phase("file_upload_complete", upload_started)

    requested_confidence = (
        request.form.get("conf") if request.form else None
    )

    compare_yolo = (
        request.form.get("compare", "true").lower() != "false"
        if request.form else True
    )

    requested_cnn_confidence = (
        request.form.get("cnn_conf") if request.form else None
    )

    if request.is_json:
        json_data = request.get_json(silent=True) or {}

        if requested_confidence is None:
            requested_confidence = json_data.get("conf")

        if requested_cnn_confidence is None:
            requested_cnn_confidence = json_data.get("cnn_conf")

        compare_yolo = json_data.get("compare", True) is not False

    try:
        confidence = (
            float(requested_confidence)
            if requested_confidence is not None
            else current_app.config["YOLO_CONFIDENCE"]
        )
    except (TypeError, ValueError):
        return jsonify({
            "error": "Confidence must be a number between 0 and 1"
        }), 400

    if not 0.01 <= confidence <= 1:
        return jsonify({
            "error": "Confidence must be between 0.01 and 1"
        }), 400

    try:
        cnn_confidence = (
            float(requested_cnn_confidence)
            if requested_cnn_confidence is not None
            else current_app.config["CNN_CONFIDENCE"]
        )
    except (TypeError, ValueError):
        return jsonify({
            "error": "CNN confidence must be a number between 0 and 1"
        }), 400

    if not 0 <= cnn_confidence <= 1:
        return jsonify({
            "error": "CNN confidence must be between 0 and 1"
        }), 400

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Valid image file required"}), 400

    validation_started = time.perf_counter()
    try:
        with Image.open(filepath) as image:
            image.verify()
    except Exception:
        return jsonify({
            "error": "Uploaded file is not a valid image"
        }), 400
    log_phase("image_validation", validation_started)

    try:
        model_started = time.perf_counter()
        yolo = YOLOService()
        log_phase("model_initialization", model_started)

        inference_started = time.perf_counter()
        hybrid = HybridDetectionService.analyze(
            filepath,
            conf_threshold=confidence,
            cnn_threshold=cnn_confidence,
            compare=compare_yolo
        )
        log_phase("analysis_pipeline", inference_started)

        detections = hybrid["detections"]
        annotated_path = hybrid["annotated_path"]

        risk_service = RiskService()
        risk = risk_service.calculate(detections)

        user_id = get_jwt_identity() or "anonymous"
        evidence_doc = None

        persistence_started = time.perf_counter()
        if current_app.db is not None:
            doc = create_evidence_doc(
                case_id=case_id,
                filename=original_filename or os.path.basename(filepath),
                file_type="image",
                detections=detections,
                risk_score=risk,
                user_id=user_id
            )
            doc["evidence_id"] = str(evidence_id) if evidence_id else None

            matched = current_app.db.evidence.find_one({"evidence_id": str(evidence_id)}) if evidence_id else None
            if matched:
                doc["case_id"] = case_id or matched.get("case_id")
                updates = {key: value for key, value in doc.items() if key not in {"uploaded_at", "_id"}}
                current_app.db.evidence.update_one({"_id": matched["_id"]}, {"$set": updates})
                doc.update(matched)
                doc.update(updates)
            else:
                result = current_app.db.evidence.insert_one(doc)
                doc["_id"] = result.inserted_id
            if doc.get("case_id"):
                try:
                    update_case_risk(current_app.db, doc["case_id"], risk)
                except Exception:
                    current_app.logger.exception("Image analysis was saved but the linked case risk summary could not be updated")
            evidence_doc = serialize_evidence(doc)

            custody = CustodyService(current_app.db)

            custody.log(
                evidence_id or str(result.inserted_id),
                "analyze_image",
                user_id,
                {
                    "detection_count": len(detections),
                    "risk_score": risk.get("overall_risk"),
                    "weapons_found": sum(
                        1 for d in detections
                        if d["category"] == "Weapons"
                    )
                }
            )
        log_phase("mongodb_persistence", persistence_started)

        annotated_url = (
            "/api/detection/annotated/"
            + Path(annotated_path).name
        )

        response_payload = {
            "success": True,
            "status": "success",
            "detections": detections,
            "detection_count": len(detections),
            "count": len(detections),
            "weapons_detected": [
                d for d in detections
                if d["category"] == "Weapons"
            ],
            "persons_detected": [
                d for d in detections
                if d["category"] == "Persons"
            ],
            "risk": risk,
            "persistence": {"database": "saved" if evidence_doc else "unavailable"},
            "annotated_image": annotated_url,
            "model_used": os.path.basename(
                yolo.model_info["path"] or ""
            ),
            "pipeline": (
                "hybrid" if hybrid["cnn_available"] else "yolo_only"
            ),
            "cnn_available": hybrid["cnn_available"],
            "cnn_message": hybrid["cnn_error"],
            "cnn_preprocessing": hybrid["preprocessing"],
            "yolo_detections": hybrid["yolo_detections"],
            "comparison_available": (
                hybrid["yolo_detections"] is not None
            ),
            "evidence": evidence_doc
        }
        serialization_started = time.perf_counter()
        response = jsonify(response_payload)
        log_phase("response_serialization", serialization_started)
        current_app.logger.info(
            "image_analysis_timing stage=total_backend_response duration_ms=%.1f status=200 detections=%d",
            (time.perf_counter() - request_started) * 1000,
            len(detections),
        )
        return response

    except Exception:
        current_app.logger.info(
            "image_analysis_timing stage=total_backend_response duration_ms=%.1f status=500",
            (time.perf_counter() - request_started) * 1000,
        )
        current_app.logger.exception("Image detection failed")

        return jsonify({
            "success": False,
            "error": (
                "Image analysis failed. Check that the image is valid "
                "and YOLO_MODEL_PATH points to the trained model."
            )
        }), 500


@bp.route("/annotated/<path:filename>", methods=["GET"])
def serve_annotated(filename):
    """Serve annotated result images."""

    path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        "annotated",
        os.path.basename(filename)
    )

    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")

    return jsonify({"error": "File not found"}), 404


# --------------------------------------------------
# ASSAULT VIDEO DETECTION TIMESTAMPS
# --------------------------------------------------

@bp.route("/assault/detections", methods=["GET"])
def get_assault_detections():
    """Return previously generated assault detection results."""

    json_path = (
        Path(__file__).resolve().parents[2]
        / "assault_77_video_detections.json"
    )

    if not json_path.is_file():
        return jsonify({
            "error": "Assault detection JSON file not found",
            "expected_path": str(json_path)
        }), 404

    try:
        with json_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

    except (OSError, json.JSONDecodeError):
        current_app.logger.exception(
            "Could not read assault detection JSON"
        )
        return jsonify({
            "error": "Could not read assault detection data"
        }), 500

    video_name = request.args.get("video")

    if video_name:
        record = next(
            (
                item for item in records
                if item.get("video_name", "").lower()
                == video_name.lower()
            ),
            None
        )

        if record is None:
            return jsonify({"error": "Video not found"}), 404

        return jsonify(record)

    return jsonify({
        "video_count": len(records),
        "videos": records
    })
