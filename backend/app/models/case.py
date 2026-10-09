from datetime import datetime

def create_case_doc(title, description, created_by):
    return {
        "title": title,
        "description": description,
        "created_by": created_by,
        "created_at": datetime.utcnow(),
        "status": "open",
        "evidence_ids": [],
        "risk_level": "unknown"
    }

def serialize_case(doc):
    if not doc:
        return None
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "status": doc.get("status"),
        "risk_level": doc.get("risk_level"),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "evidence_count": len(doc.get("evidence_ids", []))
    }
