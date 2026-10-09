from datetime import datetime
import hashlib
import json

def create_custody_entry(evidence_id, action, user_id, details=None):
    """
    Creates a tamper-evident chain-of-custody log entry.
    """
    timestamp = datetime.utcnow()
    payload = {
        "evidence_id": str(evidence_id),
        "action": action,               # upload, analyze, report, view, etc.
        "user_id": str(user_id) if user_id else "system",
        "timestamp": timestamp.isoformat(),
        "details": details or {}
    }
    # Simple hash for integrity (in production use proper blockchain or digital signature)
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    integrity_hash = hashlib.sha256(payload_str.encode()).hexdigest()

    return {
        **payload,
        "integrity_hash": integrity_hash,
        "created_at": timestamp
    }

def serialize_custody(doc):
    if not doc:
        return None
    return {
        "id": str(doc["_id"]),
        "evidence_id": doc.get("evidence_id"),
        "action": doc.get("action"),
        "user_id": doc.get("user_id"),
        "timestamp": doc.get("timestamp"),
        "details": doc.get("details"),
        "integrity_hash": doc.get("integrity_hash")
    }
