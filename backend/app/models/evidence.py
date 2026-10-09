from datetime import datetime
from bson import ObjectId

def create_evidence_doc(case_id, filename, file_type, detections=None, risk_score=None, user_id=None):
    return {
        "case_id": case_id,
        "filename": filename,
        "file_type": file_type,          # image / video / fingerprint
        "detections": detections or [],
        "risk_score": risk_score,
        "uploaded_by": user_id,
        "uploaded_at": datetime.utcnow(),
        "status": "analyzed"
    }

def serialize_evidence(doc):
    if not doc:
        return None
    return {
        "id": str(doc["_id"]),
        "case_id": doc.get("case_id"),
        "filename": doc.get("filename"),
        "file_type": doc.get("file_type"),
        "detections": doc.get("detections", []),
        "risk_score": doc.get("risk_score"),
        "uploaded_at": doc.get("uploaded_at").isoformat() if doc.get("uploaded_at") else None,
        "status": doc.get("status")
    }
