from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId

def create_case_doc(title, description, created_by):
    return {
        "title": title,
        "description": description,
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc),
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
        "latest_risk_score": doc.get("latest_risk_score"),
        "risk_updated_at": doc.get("risk_updated_at").isoformat() if doc.get("risk_updated_at") else None,
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "evidence_count": len(doc.get("evidence_ids", []))
    }


def update_case_risk(db, case_id, risk_score, assessed_at=None):
    """Attach the existing evidence assessment to its explicitly linked case."""
    if db is None or not case_id or not isinstance(risk_score, dict):
        return False
    try:
        object_id = ObjectId(str(case_id))
    except (InvalidId, TypeError):
        return False
    score = risk_score.get("overall_risk", risk_score.get("score"))
    level = risk_score.get("risk_level", risk_score.get("threat_level"))
    result = db.cases.update_one(
        {"_id": object_id},
        {"$set": {"risk_level": level or "unknown", "latest_risk_score": score,
                  "risk_updated_at": assessed_at or datetime.now(timezone.utc)}},
    )
    return bool(result.matched_count)
