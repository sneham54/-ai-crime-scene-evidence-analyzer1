from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.scene_change_service import SceneChangeService
from app.services.custody_service import CustodyService
from app.utils.file_handler import save_uploaded_file
import os

bp = Blueprint("scene_change", __name__)


@bp.route("/compare", methods=["POST"])
@jwt_required(optional=True)
def compare_scenes():
    """
    Compare two crime-scene images (before/after or different viewpoints).
    Accepts multipart: file_a, file_b
    or JSON: { "filepath_a": "...", "filepath_b": "..." }
    """
    filepath_a = None
    filepath_b = None

    if request.content_type and "multipart/form-data" in (request.content_type or ""):
        if "file_a" not in request.files or "file_b" not in request.files:
            return jsonify({"error": "Upload two images as file_a and file_b"}), 400
        filepath_a, _ = save_uploaded_file(request.files["file_a"], "images")
        filepath_b, _ = save_uploaded_file(request.files["file_b"], "images")
    else:
        data = request.get_json() or {}
        filepath_a = data.get("filepath_a")
        filepath_b = data.get("filepath_b")

    if not filepath_a or not filepath_b:
        return jsonify({"error": "Two image paths required (filepath_a, filepath_b)"}), 400
    if not os.path.isfile(filepath_a) or not os.path.isfile(filepath_b):
        return jsonify({"error": "One or both image files not found"}), 400

    try:
        service = SceneChangeService()
        result = service.compare(filepath_a, filepath_b, align=True)

        user_id = get_jwt_identity() or "anonymous"
        if current_app.db is not None:
            try:
                custody = CustodyService(current_app.db)
                custody.log(
                    f"scene-{os.path.basename(filepath_a)}",
                    "scene_change_compare",
                    str(user_id),
                    {"change_score": result["change_score"], "ssim": result["ssim_score"]},
                )
            except Exception:
                pass

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@bp.route("/image/<path:filename>", methods=["GET"])
def serve_scene_image(filename):
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], "scene_change", os.path.basename(filename))
    if os.path.isfile(path):
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "Not found"}), 404
