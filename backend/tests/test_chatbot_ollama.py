import json
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from flask import Flask
from flask_jwt_extended import JWTManager

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.routes.chatbot import bp  # noqa: E402
from app.services.chatbot_service import ChatbotService  # noqa: E402


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def _app(base_url):
    app = Flask(__name__)
    app.config.update(TESTING=True, JWT_SECRET_KEY="chatbot-test", OLLAMA_BASE_URL=base_url,
                      OLLAMA_HOST=base_url, OLLAMA_MODEL="llama3.2", OLLAMA_HEALTH_TIMEOUT_SEC=2,
                      OLLAMA_CHAT_TIMEOUT_SEC=5)
    JWTManager(app)
    app.register_blueprint(bp, url_prefix="/api/chatbot")
    return app


def test_ollama_health_checks_native_loopback_and_configured_model():
    with _app("http://127.0.0.1:11434").app_context(), patch(
        "app.services.chatbot_service.urlopen", return_value=Response({"models": [{"name": "llama3.2:latest"}]})
    ) as mocked:
        result = ChatbotService().health()
    assert mocked.call_args.args[0].full_url == "http://127.0.0.1:11434/api/tags"
    assert mocked.call_args.kwargs["timeout"] == 2
    assert result["status"] == "connected"
    assert result["model_available"] is True


def test_ollama_health_uses_docker_windows_host_and_reports_missing_model():
    app = _app("http://host.docker.internal:11434")
    with app.app_context(), patch(
        "app.services.chatbot_service.urlopen", return_value=Response({"models": [{"name": "mistral:latest"}]})
    ) as mocked:
        result = ChatbotService().health()
    assert mocked.call_args.args[0].full_url == "http://host.docker.internal:11434/api/tags"
    assert result["status"] == "unavailable"
    assert result["model_available"] is False
    assert "ollama pull llama3.2" in result["error"]


def test_ollama_unavailable_health_and_query_return_actionable_errors():
    app = _app("http://host.docker.internal:11434")
    with app.test_client() as client, patch(
        "app.services.chatbot_service.urlopen", side_effect=URLError("connection refused")
    ):
        health = client.get("/api/chatbot/health")
        answer = client.post("/api/chatbot/query", json={"query": "Summarize this case", "context": {}})
    assert health.status_code == 503
    assert "host.docker.internal" in health.json["error"]
    assert answer.status_code == 503
    assert "host.docker.internal" in answer.json["error"]


def test_chat_endpoint_returns_existing_assistant_response():
    app = _app("http://127.0.0.1:11434")
    with app.test_client() as client, patch(
        "app.services.chatbot_service.urlopen",
        return_value=Response({"message": {"content": "Review the source video and related evidence."}}),
    ) as mocked:
        response = client.post("/api/chatbot/query", json={"query": "What next?", "context": {"case_id": "case-1"}})
    request = mocked.call_args.args[0]
    body = json.loads(request.data.decode())
    assert response.status_code == 200
    assert response.json["answer"] == "Review the source video and related evidence."
    assert body["model"] == "llama3.2"
    assert body["messages"][1]["content"].find('"case_id": "case-1"') >= 0
