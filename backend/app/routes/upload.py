from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.utils.file_handler import save_uploaded_file, allowed_file
from app.services.custody_service import CustodyService
import uuid
import os

bp = Blueprint("upload", __name__)

@bp.route("/image", methods=["POST"])
@jwt_required(optional=True)
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, current_app.config["ALLOWED_IMAGE_EXTENSIONS"]):
        return jsonify({"error": "Invalid image type"}), 400

    filepath, original_name = save_uploaded_file(file, "images")
    if not filepath or not os.path.getsize(filepath):
        if filepath and os.path.exists(filepath): os.remove(filepath)
        return jsonify({"success": False, "error": "Uploaded image is empty"}), 400
    user_id = get_jwt_identity() or "anonymous"
    evidence_id = str(uuid.uuid4())

    # Log custody
    if current_app.db is not None:
        custody = CustodyService(current_app.db)
        custody.log(evidence_id, "upload_image", user_id, {
            "original_name": original_name,
            "path": filepath
        })

    return jsonify({
        "success": True,
        "message": "Image uploaded successfully",
        "evidence_id": evidence_id,
        "filepath": filepath,
        "original_name": original_name
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

    filepath, original_name = save_uploaded_file(file, "videos")
    if not filepath or not os.path.getsize(filepath):
        if filepath and os.path.exists(filepath): os.remove(filepath)
        return jsonify({"success": False, "error": "Uploaded video is empty"}), 400
    user_id = get_jwt_identity() or "anonymous"
    evidence_id = str(uuid.uuid4())

    if current_app.db is not None:
        custody = CustodyService(current_app.db)
        custody.log(evidence_id, "upload_video", user_id, {
            "original_name": original_name,
            "path": filepath
        })

    return jsonify({
        "success": True,
        "message": "Video uploaded successfully",
        "evidence_id": evidence_id,
        "filepath": filepath,
        "original_name": original_name
    })
