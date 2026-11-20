import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import current_app


class OllamaUnavailableError(RuntimeError):
    pass


class ChatbotService:
    @staticmethod
    def _base_url():
        return (current_app.config.get("OLLAMA_BASE_URL") or current_app.config.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def health(self) -> dict:
        host = self._base_url()
        model = current_app.config.get("OLLAMA_MODEL", "llama3.2")
        try:
            request = Request(f"{host}/api/tags", headers={"Accept": "application/json"})
            with urlopen(request, timeout=current_app.config.get("OLLAMA_HEALTH_TIMEOUT_SEC", current_app.config.get("OLLAMA_HEALTH_TIMEOUT", 4))) as response:
                result = json.loads(response.read().decode("utf-8"))
            available = [item.get("name", item.get("model", "")) for item in result.get("models", [])]
            model_available = any(name == model or name.split(":", 1)[0] == model for name in available)
            return {"status": "connected" if model_available else "unavailable", "base_url": host,
                    "model": model, "model_available": model_available,
                    "error": None if model_available else f"Ollama is reachable, but model '{model}' is not installed. Run 'ollama pull {model}' on the Ollama host, then retry."}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            return {"status": "unavailable", "base_url": host, "model": model,
                    "model_available": False,
                    "error": f"Cannot reach Ollama at {host}. Start Ollama and verify Docker can reach the host address. {error}"}

    def answer(self, query: str, context: dict) -> str:
        host = self._base_url()
        try:
            payload = json.dumps({
                "model": current_app.config["OLLAMA_MODEL"],
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI forensic assistant supporting an investigator. Use only the supplied context. "
                            "Answer factual questions directly and concisely. "
                            "For project-wide counts, use dashboard_summary totals only when its status is success. "
                            "recent_evidence is only a sample; never count it as the full evidence collection. "
                            "If dashboard_summary is missing or in demo/unavailable status, say live records are unavailable "
                            "instead of reporting zero. Use current_analysis, recent evidence, and custody logs when relevant. "
                            "Do not invent evidence or present an AI detection as confirmed fact. State uncertainty clearly "
                            "and offer practical investigative support, not a legal or final forensic conclusion."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Case context (JSON): {json.dumps(context, default=str)}\n\n"
                            f"Question: {query}"
                        ),
                    },
                ],
                "options": {
                    "temperature": 0.2,
                    "num_predict": 256,
                },
            }).encode("utf-8")
            request = Request(
                f"{host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=current_app.config.get("OLLAMA_CHAT_TIMEOUT_SEC", current_app.config.get("OLLAMA_CHAT_TIMEOUT", 120))) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result.get("message", {}).get("content", "").strip()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {error.code}: {detail or 'check the configured model.'}"
            ) from error
        except TimeoutError as error:
            raise OllamaUnavailableError(
                f"Ollama did not respond within 120 seconds at {host}. Please retry."
            ) from error
        except URLError as error:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {host}. Start Ollama and try again."
            ) from error

        if not answer:
            raise OllamaUnavailableError("Ollama returned an empty response. Please retry.")
        return answer
