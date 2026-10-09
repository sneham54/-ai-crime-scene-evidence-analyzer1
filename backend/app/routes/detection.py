from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.yolo_service import YOLOService
from app.services.risk_service import RiskService
from app.services.custody_service import CustodyService
from app.models.evidence import create_evidence_doc, serialize_evidence
import os

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
    except Exception as e:
        current_app.logger.exception("YOLO model status check failed")
        return jsonify({
            "status": "error",
            "message": "YOLO model could not be loaded.",
            "hint": "Check YOLO_MODEL_PATH and ensure the trained weights exist."
        }), 503


@bp.route("/analyze", methods=["POST"])
@jwt_required(optional=True)
def analyze_image():
    """
    Run YOLO object detection on an uploaded image.
    Body JSON: { "filepath": "...", "evidence_id": "...", "case_id": "..." }
    or multipart with 'file' for direct upload+analyze.
    """
    # Support both JSON (filepath) and direct file upload
    filepath = None
    evidence_id = None
    case_id = "default-case"

    if request.content_type and "multipart/form-data" in request.content_type:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        from app.utils.file_handler import save_uploaded_file
        filepath, _ = save_uploaded_file(file, "images")
        case_id = request.form.get("case_id", "default-case")
        evidence_id = request.form.get("evidence_id")
    else:
        data = request.get_json() or {}
        filepath = data.get("filepath")
        evidence_id = data.get("evidence_id")
        case_id = data.get("case_id", "default-case")

    requested_confidence = (request.form.get("conf") if request.form else None)
    if not requested_confidence and request.is_json:
        requested_confidence = (request.get_json(silent=True) or {}).get("conf")
    try:
        confidence = float(requested_confidence) if requested_confidence is not None else current_app.config["YOLO_CONFIDENCE"]
    except (TypeError, ValueError):
        return jsonify({"error": "Confidence must be a number between 0 and 1"}), 400
    if not 0.01 <= confidence <= 1:
        return jsonify({"error": "Confidence must be between 0.01 and 1"}), 400

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": f"Valid image filepath required. Got: {filepath}"}), 400

    try:
        yolo = YOLOService()
        detections, annotated_path = yolo.detect(filepath, conf_threshold=confidence)

        risk_service = RiskService()
        risk = risk_service.calculate(detections)

        user_id = get_jwt_identity() or "anonymous"

        evidence_doc = None
        if current_app.db is not None:
            doc = create_evidence_doc(
                case_id=case_id,
                filename=os.path.basename(filepath),
                file_type="image",
                detections=detections,
                risk_score=risk,
                user_id=user_id
            )
            result = current_app.db.evidence.insert_one(doc)
            doc["_id"] = result.inserted_id
            evidence_doc = serialize_evidence(doc)

            custody = CustodyService(current_app.db)
            custody.log(
                evidence_id or str(result.inserted_id),
                "analyze_image",
                user_id,
                {
                    "detection_count": len(detections),
                    "risk_score": risk.get("overall_risk"),
                    "weapons_found": sum(1 for d in detections if d["category"] == "Weapons")
                }
            )

        # Make annotated path relative / web-friendly if needed
        annotated_url = annotated_path.replace("\\", "/")

        return jsonify({
            "success": True,
            "status": "success",
            "detections": detections,
            "detection_count": len(detections),
            "count": len(detections),
            "weapons_detected": [d for d in detections if d["category"] == "Weapons"],
            "persons_detected": [d for d in detections if d["category"] == "Persons"],
            "risk": risk,
            "annotated_image": annotated_url,
            "model_used": os.path.basename(yolo.model_info["path"] or ""),
            "evidence": evidence_doc
        })
    except Exception as e:
        current_app.logger.exception("Image detection failed")
        return jsonify({
            "success": False,
            "error": "Image analysis failed. Check that the image is valid and YOLO_MODEL_PATH points to the trained model."
        }), 500


@bp.route("/annotated/<path:filename>", methods=["GET"])
def serve_annotated(filename):
    """Serve annotated result images."""
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], "annotated", filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "File not found"}), 404
