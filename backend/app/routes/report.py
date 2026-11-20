from flask import Blueprint, request, jsonify, current_app, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.report_service import ReportService
from app.services.custody_service import CustodyService
import os
import json
import uuid
from datetime import datetime, timezone

bp = Blueprint("report", __name__)


def _write_pdf_report_metadata(report_name, data):
    """Keep submitted investigation references beside the PDF that owns them."""
    reports_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports")
    metadata_path = os.path.join(reports_dir, f"{report_name}.meta.json")
    video = data.get("video_result") if isinstance(data.get("video_result"), dict) else {}
    classification = video.get("event_classification") or video.get("classification") or {}
    risk = data.get("risk") if isinstance(data.get("risk"), dict) else {}
    metadata = {
        "report_id": report_name[:-4], "report_name": report_name,
        "status": "generated", "generated_at": datetime.now(timezone.utc).isoformat(),
        "investigation_id": (data.get("case") or {}).get("case_id") if isinstance(data.get("case"), dict) else None,
        "evidence_id": data.get("evidence_id"),
        "evidence_filename": data.get("file_name") or video.get("file_name"),
        "analysis_id": video.get("analysis_id"),
        "category": classification.get("label") if isinstance(classification, dict) else None,
        "risk_score": risk.get("overall_risk", risk.get("score")),
    }
    temp_path = metadata_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, allow_nan=False)
    os.replace(temp_path, metadata_path)
    return metadata


@bp.route("/generate-json", methods=["POST"])
@jwt_required(optional=True)
def generate_json_report():
    """Save the submitted structured investigation report as a downloadable JSON file."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON report object is required."}), 400

    reports_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_name = f"investigation_report_{uuid.uuid4().hex[:12]}.json"
    report_path = os.path.join(reports_dir, report_name)
    report = {
        **data,
        "report_type": "AI-assisted investigation evidence",
        "verification_status": "unverified",
        "human_review_required": True,
        "limitations": "Model outputs are candidate investigative leads. Anomaly scores are not calibrated crime probabilities; object detections alone do not establish a crime.",
    }
    try:
        with open(report_path, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError):
        current_app.logger.exception("JSON investigation report could not be serialized")
        return jsonify({"success": False, "error": "Report contains values that cannot be represented as JSON."}), 400
    return jsonify({
        "success": True,
        "report_name": report_name,
        "download_url": f"/api/report/download-json/{report_name}",
    })


@bp.route("/download-json/<report_name>", methods=["GET"])
def download_json_report(report_name):
    """Download a generated JSON report by its safe basename."""
    if os.path.basename(report_name) != report_name or not report_name.endswith(".json"):
        return jsonify({"error": "Invalid report name"}), 400
    report_path = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports", report_name)
    if not os.path.isfile(report_path):
        return jsonify({"error": "Report not found"}), 404
    return send_file(report_path, as_attachment=True, download_name=report_name, mimetype="application/json")


@bp.route("/generate", methods=["POST"])
@jwt_required(optional=True)
def generate_report():
    """
    Generate a PDF forensic report from analysis results.
    Body JSON:
      {
        "case": { "case_id": "...", "description": "..." },
        "detections": [...],
        "risk": {...},
        "annotated_image": "/path/or/filename",
        "evidence_id": "..."
      }
    """
    data = request.get_json() or {}

    case_data = data.get("case") or {
        "case_id": f"CASE-{__import__('uuid').uuid4().hex[:8].upper()}",
        "description": "Automated crime scene evidence analysis",
    }
    detections = data.get("detections") or []
    risk = data.get("risk") or {}
    annotated_image = data.get("annotated_image")
    evidence_id = data.get("evidence_id")
    video_result = data.get("video_result")
    source_filename = data.get("file_name")
    analyzed_at = data.get("analyzed_at")

    # Resolve annotated image path if only a filename was sent
    if annotated_image and not os.path.isfile(annotated_image):
        # Try common locations
        base = current_app.config["UPLOAD_FOLDER"]
        candidates = [
            annotated_image,
            os.path.join(base, "annotated", os.path.basename(annotated_image)),
            os.path.join(base, annotated_image),
            os.path.normpath(annotated_image),
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                annotated_image = c
                break

    custody_logs = []
    if current_app.db is not None and evidence_id:
        try:
            custody = CustodyService(current_app.db)
            custody_logs = custody.get_logs(evidence_id) or []
        except Exception as e:
            print(f"Custody fetch warning: {e}")

    investigator = get_jwt_identity() or "investigator"

    try:
        service = ReportService()
        report_path, report_name = service.generate(
            case_data=case_data,
            detections=detections,
            risk=risk,
            custody_logs=custody_logs,
            annotated_image_path=annotated_image,
            investigator=str(investigator),
            video_result=video_result,
            source_filename=source_filename,
            analyzed_at=analyzed_at,
        )
        try:
            _write_pdf_report_metadata(report_name, data)
        except (OSError, TypeError, ValueError):
            current_app.logger.exception("PDF was generated but report metadata could not be saved")

        if current_app.db is not None and evidence_id:
            try:
                custody = CustodyService(current_app.db)
                custody.log(evidence_id, "generate_report", str(investigator), {
                    "report_name": report_name,
                })
            except Exception:
                pass

        return jsonify({
            "success": True,
            "status": "success",
            "report_name": report_name,
            "download_url": f"/api/report/download/{report_name}",
            "message": "Forensic report generated successfully",
        })
    except Exception as e:
        current_app.logger.exception("PDF report generation failed")
        return jsonify({"success": False, "error": "PDF report generation failed."}), 500


@bp.route("/download/<report_name>", methods=["GET"])
def download_report(report_name):
    """Download a previously generated PDF report."""
    # Security: only allow simple filenames
    report_name = os.path.basename(report_name)
    if not report_name.endswith(".pdf"):
        return jsonify({"error": "Invalid report name"}), 400

    report_path = os.path.join(
        current_app.config["UPLOAD_FOLDER"], "reports", report_name
    )
    if not os.path.isfile(report_path):
        return jsonify({"error": "Report not found"}), 404

    inline = request.args.get("inline") == "1"
    return send_file(
        report_path,
        as_attachment=not inline,
        download_name=report_name,
        mimetype="application/pdf",
    )


@bp.route("/list", methods=["GET"])
@jwt_required(optional=True)
def list_reports():
    """List generated reports (most recent first)."""
    reports_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports")
    os.makedirs(reports_dir, exist_ok=True)
    files = []
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    for name in os.listdir(reports_dir):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(reports_dir, name)
        try:
            stat = os.stat(path)
            generated_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        except OSError:
            continue
        metadata = {}
        metadata_path = path + ".meta.json"
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as source:
                    candidate = json.load(source)
                if isinstance(candidate, dict):
                    metadata = candidate
                    parsed = datetime.fromisoformat(str(candidate.get("generated_at", "")).replace("Z", "+00:00"))
                    generated_at = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            except (OSError, ValueError, TypeError):
                current_app.logger.warning("Ignoring invalid report metadata for %s", name)
        files.append({
            "id": metadata.get("report_id") or name[:-4], "name": name,
            "size_kb": round(stat.st_size / 1024, 1),
            "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
            "status": "generated", "evidence_id": metadata.get("evidence_id"),
            "evidence_filename": metadata.get("evidence_filename"),
            "investigation_id": metadata.get("investigation_id"),
            "analysis_id": metadata.get("analysis_id"), "category": metadata.get("category"),
            "risk_score": metadata.get("risk_score"),
            "view_url": f"/api/report/download/{name}?inline=1",
            "download_url": f"/api/report/download/{name}",
            "metadata_available": bool(metadata),
        })
    files.sort(key=lambda row: row["generated_at"], reverse=True)
    associated = [row for row in files if row["metadata_available"]]
    association_count = (sum(bool(row["investigation_id"] or row["evidence_id"] or row["analysis_id"]) for row in associated)
                         if len(associated) == len(files) else None)
    return jsonify({
        "reports": files[:50], "total_reports": len(files),
        "successful_reports": len(files), "failed_reports": None,
        "reports_associated_with_investigations": association_count,
        "reports_generated_this_month": sum(
            1 for row in files if datetime.fromisoformat(row["generated_at"].replace("Z", "+00:00")) >= month_start),
        "status": "success",
    })
