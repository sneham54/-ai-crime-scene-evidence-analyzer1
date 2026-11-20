from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.utils.file_handler import save_uploaded_file, allowed_file
from app.services.custody_service import CustodyService
import uuid
import os
from datetime import datetime, timezone
from app.models.evidence import create_evidence_doc
from bson import ObjectId
from bson.errors import InvalidId

bp = Blueprint("upload", __name__)


def _case_id_from_form():
    case_id = (request.form.get("case_id") or "").strip()
    if not case_id:
        return None, None
    if current_app.db is None:
        return None, (jsonify({"success": False, "error": "MongoDB is unavailable, so the supplied case ID cannot be verified."}), 503)
    try:
        case = current_app.db.cases.find_one({"_id": ObjectId(case_id)})
    except (InvalidId, TypeError):
        case = None
    except Exception:
        current_app.logger.exception("Could not validate supplied investigation case ID")
        return None, (jsonify({"success": False, "error": "MongoDB could not verify the supplied case ID."}), 503)
    if not case:
        return None, (jsonify({"success": False, "error": "The supplied case ID does not match a saved investigation."}), 404)
    return case_id, None

@bp.route("/image", methods=["POST"])
@jwt_required()
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, current_app.config["ALLOWED_IMAGE_EXTENSIONS"]):
        return jsonify({"error": "Invalid image type"}), 400

    case_id, case_error = _case_id_from_form()
    if case_error:
        return case_error

    filepath, original_name = save_uploaded_file(file, "images")
    if not filepath or not os.path.getsize(filepath):
        if filepath and os.path.exists(filepath): os.remove(filepath)
        return jsonify({"success": False, "error": "Uploaded image is empty"}), 400
    user_id = get_jwt_identity() or "anonymous"
    evidence_id = str(uuid.uuid4())

    uploaded_at = datetime.now(timezone.utc)
    database_persistence = "unavailable"
    if current_app.db is not None:
        try:
            doc = create_evidence_doc(case_id, original_name, "image", user_id=user_id)
            doc.update(evidence_id=evidence_id, uploaded_at=uploaded_at, status="uploaded")
            current_app.db.evidence.insert_one(doc)
            database_persistence = "saved"
            if case_id:
                current_app.db.cases.update_one({"_id": ObjectId(case_id)}, {"$addToSet": {"evidence_ids": evidence_id}})
            CustodyService(current_app.db).log(evidence_id, "upload_image", user_id, {
                "original_name": original_name, "path": filepath
            })
        except Exception:
            current_app.logger.exception("Image upload persistence step did not complete")

    return jsonify({
        "success": True,
        "message": "Image uploaded successfully",
        "evidence_id": evidence_id,
        "filepath": filepath,
        "original_name": original_name,
        "uploaded_at": uploaded_at.isoformat().replace("+00:00", "Z"),
        "persistence": {"database": database_persistence}
    })


@bp.route("/video", methods=["POST"])
@jwt_required(optional=True)
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, current_app.config["ALLOWED_VIDEO_EXTENSIONS"]):
        return jsonify({"error": "Invalid video type"}), 400

    case_id, case_error = _case_id_from_form()
    if case_error:
        return case_error

    try:
        filepath, original_name = save_uploaded_file(file, "videos")
        if not filepath or not os.path.getsize(filepath):
            if filepath and os.path.exists(filepath): os.remove(filepath)
            return jsonify({"success": False, "code": "empty_upload", "error": "Uploaded video is empty."}), 400
    except OSError:
        current_app.logger.exception("Video upload could not be saved")
        return jsonify({"success": False, "code": "upload_failed", "error": "Video upload could not be saved. Check upload storage permissions and available disk space."}), 500
    user_id = get_jwt_identity() or "anonymous"
    evidence_id = str(uuid.uuid4())

    uploaded_at = datetime.now(timezone.utc)
    database_persistence = "unavailable"
    if current_app.db is not None:
        try:
            doc = create_evidence_doc(case_id, original_name, "video", user_id=user_id)
            doc.update(evidence_id=evidence_id, uploaded_at=uploaded_at, status="uploaded")
            current_app.db.evidence.insert_one(doc)
            database_persistence = "saved"
            if case_id:
                current_app.db.cases.update_one({"_id": ObjectId(case_id)}, {"$addToSet": {"evidence_ids": evidence_id}})
            CustodyService(current_app.db).log(evidence_id, "upload_video", user_id, {
                "original_name": original_name, "path": filepath
            })
        except Exception:
            current_app.logger.exception("Video upload persistence step did not complete")

    return jsonify({
        "success": True,
        "message": "Video uploaded successfully",
        "evidence_id": evidence_id,
        "filepath": filepath,
        "original_name": original_name,
        "uploaded_at": uploaded_at.isoformat().replace("+00:00", "Z"),
        "persistence": {"database": database_persistence}
    })
