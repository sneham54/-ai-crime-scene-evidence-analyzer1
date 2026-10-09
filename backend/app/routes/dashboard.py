from datetime import datetime
import os
from flask import Blueprint, jsonify, current_app
from flask_jwt_extended import jwt_required
from app.services.custody_service import CustodyService

bp = Blueprint("dashboard", __name__)

@bp.route("/summary", methods=["GET"])
@jwt_required(optional=True)
def summary():
    db = current_app.db
    if db is None:
        return jsonify({
            "status": "demo",
            "total_evidence": 0,
            "total_cases": 0,
            "high_risk_cases": 0,
            "reports_generated": 0,
            "total_detections": 0,
            "images_analyzed": 0,
            "videos_analyzed": 0,
            "crime_types": {"Assault": 0, "Robbery": 0, "Shooting": 0, "Normal": 0},
            "monthly_evidence": [],
            "object_distribution": {},
            "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "recent_logs": [],
            "message": "MongoDB not connected – showing demo data"
        })

    total_evidence = db.evidence.count_documents({})
    total_cases = db.cases.count_documents({}) if "cases" in db.list_collection_names() else 0
    images_analyzed = db.evidence.count_documents({"file_type": "image"})
    videos_analyzed = db.evidence.count_documents({"file_type": "video"})
    total_detections = 0
    object_distribution = {}
    risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for doc in db.evidence.find({}, {"detections": 1, "risk_score.overall_risk": 1}):
        detections = doc.get("detections", [])
        total_detections += len(detections)
        for detection in detections:
            name = detection.get("class") or detection.get("category") or "Other"
            object_distribution[name] = object_distribution.get(name, 0) + 1
        risk = (doc.get("risk_score") or {}).get("overall_risk", 0)
        if risk >= 90:
            risk_distribution["critical"] += 1
        elif risk >= 70:
            risk_distribution["high"] += 1
        elif risk >= 40:
            risk_distribution["medium"] += 1
        else:
            risk_distribution["low"] += 1

    high_risk_cases = db.evidence.count_documents({"risk_score.overall_risk": {"$gte": 70}})
    reports_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports")
    reports_generated = sum(1 for name in os.listdir(reports_dir) if name.lower().endswith(".pdf")) if os.path.isdir(reports_dir) else 0

    # Build a seven-month evidence trend from stored upload timestamps.
    month_starts = []
    year, month = datetime.utcnow().year, datetime.utcnow().month
    for offset in range(6, -1, -1):
        index = year * 12 + month - 1 - offset
        month_starts.append(datetime(index // 12, index % 12 + 1, 1))
    month_counts = {item.strftime("%Y-%m"): 0 for item in month_starts}
    for doc in db.evidence.find({"uploaded_at": {"$gte": month_starts[0]}}, {"uploaded_at": 1}):
        uploaded_at = doc.get("uploaded_at")
        if uploaded_at:
            key = uploaded_at.strftime("%Y-%m")
            if key in month_counts:
                month_counts[key] += 1
    monthly_evidence = [
        {"month": item.strftime("%b"), "count": month_counts[item.strftime("%Y-%m")]}
        for item in month_starts
    ]
    crime_types = {name: 0 for name in ("Assault", "Robbery", "Shooting", "Normal")}
    for doc in db.evidence.find({"file_type": "video"}, {"event_classification.label": 1}):
        label = (doc.get("event_classification") or {}).get("label")
        if label in crime_types:
            crime_types[label] += 1

    custody = CustodyService(db)
    recent_logs = custody.get_logs(limit=10)

    # Recent evidence
    recent_evidence = list(db.evidence.find().sort("uploaded_at", -1).limit(5))
    from app.models.evidence import serialize_evidence
    recent_evidence = [serialize_evidence(e) for e in recent_evidence]

    return jsonify({
        "status": "success",
        "total_evidence": total_evidence,
        "total_cases": total_cases,
        "high_risk_cases": high_risk_cases,
        "reports_generated": reports_generated,
        "images_analyzed": images_analyzed,
        "videos_analyzed": videos_analyzed,
        "total_detections": total_detections,
        "crime_types": crime_types,
        "monthly_evidence": monthly_evidence,
        "object_distribution": object_distribution,
        "risk_distribution": risk_distribution,
        "recent_evidence": recent_evidence,
        "recent_logs": recent_logs
    })


@bp.route("/history", methods=["GET"])
@jwt_required(optional=True)
def history():
    db = current_app.db
    if db is None:
        return jsonify({"logs": []})

    custody = CustodyService(db)
    logs = custody.get_logs(limit=100)
    return jsonify({"logs": logs})
