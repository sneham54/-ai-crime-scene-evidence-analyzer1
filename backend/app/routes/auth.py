from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app.utils.security import hash_password, verify_password
from app.models.user import create_user_doc, serialize_user
from datetime import datetime, timedelta

bp = Blueprint("auth", __name__)
_DEMO_USERS = {}

@bp.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    db = current_app.db
    if db is None:
        # Keep local/demo accounts usable for the current server session.
        if username in _DEMO_USERS:
            return jsonify({"error": "Username already exists"}), 409
        _DEMO_USERS[username] = {
            "password_hash": hash_password(password),
            "role": "investigator",
        }
        token = create_access_token(identity=f"demo-{username}", expires_delta=timedelta(hours=12))
        return jsonify({
            "message": "Demo investigator registered",
            "access_token": token,
            "user": {"id": f"demo-{username}", "username": username, "role": "investigator"}
        }), 201

    if db.users.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 409

    user_doc = create_user_doc(username, hash_password(password))
    result = db.users.insert_one(user_doc)
    user_doc["_id"] = result.inserted_id

    return jsonify({
        "message": "User registered successfully",
        "user": serialize_user(user_doc)
    }), 201


@bp.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    default_users = {
        "investigator": "investigator",
        "officer": "officer",
    }

    if username in default_users and password == "demo123":
        role = default_users[username]
        token = create_access_token(identity=f"demo-{username}", expires_delta=timedelta(hours=12))
        return jsonify({
            "access_token": token,
            "user": {"id": f"demo-{username}", "username": username, "role": role}
        })

    db = current_app.db
    if db is None:
        # Demo mode without DB
        demo_user = _DEMO_USERS.get(username)
        if demo_user and verify_password(demo_user["password_hash"], password):
            token = create_access_token(identity=f"demo-{username}", expires_delta=timedelta(hours=12))
            return jsonify({
                "access_token": token,
                "user": {"id": f"demo-{username}", "username": username, "role": demo_user["role"]}
            })
        return jsonify({"error": "Invalid credentials. Create an investigator account or check your credentials."}), 401

    user = db.users.find_one({"username": username})
    if not user:
        # Keep the credentials displayed by the frontend usable when a fresh
        # MongoDB database has not yet been seeded with an investigator user.
        demo_user = _DEMO_USERS.get(username)
        if demo_user and verify_password(demo_user["password_hash"], password):
            token = create_access_token(identity=f"demo-{username}", expires_delta=timedelta(hours=12))
            return jsonify({
                "access_token": token,
                "user": {"id": f"demo-{username}", "username": username, "role": demo_user["role"]}
            })
        return jsonify({"error": "Invalid credentials"}), 401

    if not verify_password(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    token = create_access_token(identity=str(user["_id"]), expires_delta=timedelta(hours=12))

    return jsonify({
        "access_token": token,
        "user": serialize_user(user)
    })


@bp.route("/biometric", methods=["POST"])
@jwt_required(optional=True)
def biometric_auth():
    """
    Simulated biometric gate.
    In real system this would verify fingerprint / face.
    Here we accept a simple flag or dummy fingerprint file.
    """
    data = request.get_json() or {}
    verified = data.get("verified", False)

    if verified or data.get("biometric_token") == "demo-biometric-ok":
        return jsonify({
            "status": "success",
            "message": "Biometric authentication successful",
            "level": "high"
        })
    return jsonify({"status": "failed", "message": "Biometric verification failed"}), 401


@bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    identity = get_jwt_identity()
    return jsonify({"user_id": identity})
