from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.video_service import VideoService
from app.services.risk_service import RiskService
from app.services.custody_service import CustodyService
from app.models.evidence import create_evidence_doc
import os

bp = Blueprint("video", __name__)

@bp.route("/analyze", methods=["POST"])
@jwt_required(optional=True)
def analyze_video():
    data = request.get_json() or {}
    filepath = data.get("filepath")
    evidence_id = data.get("evidence_id")
    requested = data.get("max_frames")
    max_frames = current_app.config["VIDEO_MAX_SAMPLED_FRAMES"]
    if requested is not None:
        try:
            max_frames = max(1, min(int(requested), current_app.config["VIDEO_MAX_SAMPLED_FRAMES"]))
        except (TypeError, ValueError):
            max_frames = 3

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Valid filepath required"}), 400

    try:
        service = VideoService()
        result = service.analyze(filepath, max_frames=max_frames)

        # Aggregate detections for risk
        all_dets = []
        for fr in result["frame_results"]:
            all_dets.extend(fr["detections"])

        risk = RiskService().calculate(all_dets)
        result["risk"] = risk

        user_id = get_jwt_identity() or "anonymous"
        if current_app.db is not None:
            doc = create_evidence_doc("default-case", os.path.basename(filepath), "video", all_dets, risk, user_id)
            doc["event_classification"] = result.get("classification")
            stored = current_app.db.evidence.insert_one(doc)
            result["evidence"] = {"id": str(stored.inserted_id)}
            if evidence_id:
                custody = CustodyService(current_app.db)
                custody.log(evidence_id, "analyze_video", user_id, {
                "frames_analyzed": result["video_info"]["frames_analyzed"],
                "total_detections": result["total_detections"]
                })

        result["risk"] = risk
        result["evidence_id"] = evidence_id
        return jsonify({
            "status": "success",
            "success": True,
            **result
        })
    except Exception as e:
        current_app.logger.exception("Video analysis failed")
        return jsonify({"success": False, "error": "Video analysis failed. Check model configuration and video format."}), 500
