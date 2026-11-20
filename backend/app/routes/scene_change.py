from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.scene_change_service import SceneChangeService
from app.services.before_after_screenshot_classifier import (
    BeforeAfterScreenshotClassifier,
    ScreenshotClassifierUnavailableError,
)
from app.services.custody_service import CustodyService
from app.utils.file_handler import save_uploaded_file
import os

bp = Blueprint("scene_change", __name__)


@bp.route("/compare", methods=["POST"])
@jwt_required(optional=True)
def compare_scenes():
    """
    Compare two uploaded scene images using the existing SSIM/alignment
    pipeline and optionally classify each screenshot as BEFORE or AFTER.

    Multipart fields: file_a (before/reference), file_b (after/compare).
    JSON alternative: filepath_a, filepath_b.
    """
    filepath_a = None
    filepath_b = None

    if request.content_type and "multipart/form-data" in request.content_type:
        if "file_a" not in request.files or "file_b" not in request.files:
            return jsonify({"error": "Upload two images as file_a and file_b"}), 400

        filepath_a, _ = save_uploaded_file(request.files["file_a"], "images")
        filepath_b, _ = save_uploaded_file(request.files["file_b"], "images")
    else:
        data = request.get_json(silent=True) or {}
        filepath_a = data.get("filepath_a")
        filepath_b = data.get("filepath_b")

    if not filepath_a or not filepath_b:
        return jsonify({
            "error": "Two image paths required (filepath_a, filepath_b)"
        }), 400

    if not os.path.isfile(filepath_a) or not os.path.isfile(filepath_b):
        return jsonify({"error": "One or both image files not found"}), 400

    try:
        # Existing image alignment, SSIM, difference mask and heatmap logic.
        service = SceneChangeService()
        result = service.compare(filepath_a, filepath_b, align=True)

        # Optional experimental classifier. Its failure must not break
        # the existing comparison feature.
        try:
            classifier = BeforeAfterScreenshotClassifier.get()
            result["screenshot_classifier"] = {
                "status": "available",
                "task": "individual_before_after_screenshot_classification",
                "reference_image": classifier.predict_path(filepath_a),
                "comparison_image": classifier.predict_path(filepath_b),
                "checkpoint": str(classifier.checkpoint_path),
                "training_metadata": {
                    "best_epoch": classifier.best_epoch,
                    "validation_accuracy_on_screenshot_split": (
                        round(classifier.training_validation_accuracy, 4)
                        if classifier.training_validation_accuracy == classifier.training_validation_accuracy
                        else None
                    ),
                },
                "limitations": [
                    "This model classifies each screenshot independently as BEFORE or AFTER.",
                    "It does not predict PERSON_APPEARED, PERSON_DISAPPEARED, PERSON_MOVED, ADDED_OBJECT, or MOVED_OBJECT.",
                    "Its validation accuracy is from the BEFORE/AFTER screenshot split, not scene-change event accuracy.",
                    "The softmax score is not calibrated confidence or forensic proof.",
                ],
            }
        except ScreenshotClassifierUnavailableError as error:
            current_app.logger.warning(
                "Optional BEFORE/AFTER screenshot classifier unavailable: %s", error
            )
            result["screenshot_classifier"] = {
                "status": "unavailable",
                "task": "individual_before_after_screenshot_classification",
                "message": str(error),
                "limitations": [
                    "The existing SSIM and changed-region comparison still ran.",
                    "No trained screenshot-classifier prediction was produced.",
                ],
            }
        except Exception:
            current_app.logger.exception(
                "Optional BEFORE/AFTER screenshot classification failed"
            )
            result["screenshot_classifier"] = {
                "status": "error",
                "message": "Optional screenshot classification failed. Existing image comparison results are still available.",
            }

        user_id = get_jwt_identity() or "anonymous"
        if current_app.db is not None:
            try:
                custody = CustodyService(current_app.db)
                custody.log(
                    f"scene-{os.path.basename(filepath_a)}",
                    "scene_change_compare",
                    str(user_id),
                    {
                        "change_score": result["change_score"],
                        "ssim": result["ssim_score"],
                        "screenshot_classifier_status": result.get(
                            "screenshot_classifier", {}
                        ).get("status"),
                    },
                )
            except Exception:
                current_app.logger.exception(
                    "Could not write scene-change comparison to custody log"
                )

        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        current_app.logger.exception("Scene comparison failed")
        return jsonify({"error": f"Scene comparison failed ({type(error).__name__})."}), 500


@bp.route("/image/<path:filename>", methods=["GET"])
def serve_scene_image(filename):
    path = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        "scene_change",
        os.path.basename(filename),
    )
    if os.path.isfile(path):
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": "Not found"}), 404
