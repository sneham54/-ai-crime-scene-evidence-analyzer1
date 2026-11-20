import json
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from flask_jwt_extended import jwt_required

from app.services.evidence_catalog_service import CATEGORIES, EvidenceCatalogService


bp = Blueprint("evidence_catalog", __name__)


@bp.route("/shooting-events", methods=["GET"])
@jwt_required()
def shooting_events():
    """Legacy Shooting listing endpoint."""
    return _list_event_packages("shooting")


@bp.route("/event-evidence/<category>", methods=["GET"])
@jwt_required()
def category_event_evidence(category):
    if category.casefold() not in {"assault", "robbery", "shooting"}:
        return jsonify({"success": False, "error": "Unknown event evidence category."}), 404
    return _list_event_packages(category.casefold())


def _list_event_packages(category):
    """List durable event indexes for any supported video category."""
    root = (Path(current_app.config["UPLOAD_FOLDER"]) / "evidence").resolve()
    events = []
    if root.is_dir():
        paths = list(root.glob("*/category-events/*/*/event.json")) + list(root.glob("*/shooting-events/*/event.json"))
        for path in paths:
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if str(item.get("category", "")).casefold() != category:
                    continue
                item["analysis_id"] = path.parents[2].name if path.parents[1].name == "shooting-events" else path.parents[3].name
                events.append(item)
            except (OSError, json.JSONDecodeError):
                current_app.logger.warning("Skipping unreadable %s event record: %s", category, path)
    events.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return jsonify({"success": True, "events": events})


def _catalog():
    return EvidenceCatalogService(current_app.config)


def _category_or_404(category):
    normalized = (category or "").strip().lower()
    if normalized not in CATEGORIES:
        return None, (jsonify({"success": False, "error": "Unknown evidence category."}), 404)
    return normalized, None


@bp.route("/categories", methods=["GET"])
@jwt_required()
def categories():
    return jsonify({"success": True, "categories": _catalog().list_categories()})


@bp.route("/<category>/videos", methods=["GET"])
@jwt_required()
def list_videos(category):
    category, error = _category_or_404(category)
    if error:
        return error
    return jsonify({"success": True, **_catalog().list_videos(category)})


@bp.route("/<category>/videos/<video_id>", methods=["GET"])
@jwt_required()
def video_detail(category, video_id):
    category, error = _category_or_404(category)
    if error:
        return error
    result = _catalog().video_detail(category, video_id)
    if result is None:
        return jsonify({"success": False, "error": "Video was not found in this category."}), 404
    return jsonify({"success": True, **result})


@bp.route("/<category>/videos/<video_id>/media", methods=["GET"])
@jwt_required()
def video_media(category, video_id):
    category, error = _category_or_404(category)
    if error:
        return error
    kind = request.args.get("kind", "annotated").strip().lower()
    if kind not in {"source", "annotated"}:
        return jsonify({"success": False, "error": "Media kind must be source or annotated."}), 400
    path = _catalog().media_path(category, video_id, kind, request.args.get("artifact_id"))
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        return jsonify({"success": False, "error": f"{kind.title()} video is unavailable."}), 404
    return send_file(path, conditional=True, as_attachment=False, download_name=Path(path).name)


@bp.route("/<category>/status", methods=["GET"])
@jwt_required()
def processing_status(category):
    category, error = _category_or_404(category)
    if error:
        return error
    return jsonify({"success": True, **_catalog().status(category)})


@bp.route("/<category>/reports", methods=["GET"])
@jwt_required()
def saved_reports(category):
    category, error = _category_or_404(category)
    if error:
        return error
    return jsonify({"success": True, **_catalog().reports(category)})


@bp.route("/<category>/reports/<path:report_id>", methods=["GET"])
@jwt_required()
def download_saved_report(category, report_id):
    category, error = _category_or_404(category)
    if error:
        return error
    path = _catalog().report_file_path(category, report_id)
    if path is None:
        return jsonify({"success": False, "error": "Saved report is unavailable."}), 404
    return send_file(path, conditional=True, as_attachment=True, download_name=path.name)


@bp.route("/<category>/videos/<video_id>/report-payload", methods=["GET"])
@jwt_required()
def report_payload(category, video_id):
    category, error = _category_or_404(category)
    if error:
        return error
    artifact_id = request.args.get("artifact_id", "")
    result = _catalog().report_payload(category, video_id, artifact_id)
    if result is None:
        return jsonify({"success": False, "error": "Saved results were not found for this video."}), 404
    return jsonify({"success": True, "report": result})
