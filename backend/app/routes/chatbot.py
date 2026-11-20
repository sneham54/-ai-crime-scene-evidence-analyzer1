from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from app.services.chatbot_service import ChatbotService, OllamaUnavailableError

bp = Blueprint("chatbot", __name__)

@bp.route("/health", methods=["GET"])
def health():
    result = ChatbotService().health()
    return jsonify(result), (200 if result["status"] == "connected" else 503)

@bp.route("/query", methods=["POST"])
@jwt_required(optional=True)
def query():
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    context = data.get("context", {})  # detections, risk, case

    if not query_text:
        return jsonify({"error": "Query text required"}), 400

    service = ChatbotService()
    try:
        answer = service.answer(query_text, context)
    except OllamaUnavailableError as error:
        return jsonify({"error": str(error)}), 503

    return jsonify({
        "status": "success",
        "query": query_text,
        "answer": answer
    })
