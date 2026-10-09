from datetime import datetime
from bson import ObjectId

def create_user_doc(username, password_hash, role="investigator"):
    return {
        "username": username,
        "password_hash": password_hash,
        "role": role,
        "created_at": datetime.utcnow(),
        "last_login": None
    }

def serialize_user(user):
    if not user:
        return None
    return {
        "id": str(user["_id"]),
        "username": user["username"],
        "role": user.get("role", "investigator")
    }
