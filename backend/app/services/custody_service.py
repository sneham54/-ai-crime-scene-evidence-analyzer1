from app.models.custody import create_custody_entry, serialize_custody
from datetime import datetime

class CustodyService:
    def __init__(self, db):
        self.db = db
        self.collection = db["custody_logs"] if db is not None else None

    def log(self, evidence_id, action, user_id=None, details=None):
        if self.collection is None:
            return None
        entry = create_custody_entry(evidence_id, action, user_id, details)
        result = self.collection.insert_one(entry)
        entry["_id"] = result.inserted_id
        return serialize_custody(entry)

    def get_logs(self, evidence_id=None, limit=50):
        if self.collection is None:
            return []
        query = {}
        if evidence_id:
            query["evidence_id"] = str(evidence_id)
        cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
        return [serialize_custody(doc) for doc in cursor]
