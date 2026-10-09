import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent if BACKEND_DIR.name == "backend" else BACKEND_DIR
load_dotenv(BACKEND_DIR / ".env")


def _jwt_secret_key():
    """Keep demo JWTs valid across backend restarts without hardcoding a secret."""
    configured = os.getenv("JWT_SECRET_KEY")
    if configured:
        return configured

    secret_path = BACKEND_DIR / ".jwt_secret"
    try:
        return secret_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        generated = secrets.token_hex(32)
        try:
            with secret_path.open("x", encoding="utf-8") as secret_file:
                secret_file.write(generated)
            return generated
        except FileExistsError:
            # Another server process created it at the same time.
            return secret_path.read_text(encoding="utf-8").strip()


def _resolve_existing_path(env_name, default_value, default_fallbacks=()):
    """Resolve model files across the active project and any known working project roots."""
    candidates = []
    if env_name and os.getenv(env_name):
        candidates.append(Path(os.getenv(env_name)))
    if default_value:
        candidates.append(Path(default_value))
    for fallback in default_fallbacks:
        candidates.append(Path(fallback))

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = candidate if candidate.is_absolute() else (BACKEND_DIR / candidate)
        norm = str(path.resolve())
        if norm in seen:
            continue
        seen.add(norm)
        if path.exists():
            return str(path)

    fallback = candidates[0] if candidates else Path(default_value)
    if isinstance(fallback, Path) and not fallback.is_absolute():
        fallback = BACKEND_DIR / fallback
    return str(fallback)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()
    JWT_SECRET_KEY = _jwt_secret_key()
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME = os.getenv("DB_NAME", "crime_evidence_db")
    UPLOAD_FOLDER = _resolve_existing_path("UPLOAD_FOLDER", BACKEND_DIR / "uploads")
    YOLO_MODEL_PATH = _resolve_existing_path(
        "YOLO_MODEL_PATH",
        BACKEND_DIR / "models" / "best.pt",
        (
            Path(r"C:\Users\admin\AI-Crime-Scene-Analyzer\backend\models\best.pt"),
            Path(r"C:\Users\admin\OneDrive\Documents\ai-crime-scene-evidence-analyzer\backend\models\best.pt"),
        ),
    )
    VIDEO_MODEL_PATH = _resolve_existing_path(
        "VIDEO_MODEL_PATH",
        Path(r"C:\Users\admin\AI-Crime-Scene-Analyzer\backend\models\video_classifier.pth"),
        (
            BACKEND_DIR / "models" / "video_classifier.pth",
            Path(r"C:\Users\admin\OneDrive\Documents\ai-crime-scene-evidence-analyzer\backend\models\video_classifier.pth"),
        ),
    )
    CLASS_NAMES_PATH = _resolve_existing_path(
        "CLASS_NAMES_PATH",
        Path(r"C:\Users\admin\AI-Crime-Scene-Analyzer\backend\models\class_names.json"),
        (
            BACKEND_DIR / "models" / "class_names.json",
            Path(r"C:\Users\admin\OneDrive\Documents\ai-crime-scene-evidence-analyzer\backend\models\class_names.json"),
        ),
    )
    VIDEO_MIN_CONFIDENCE = float(os.getenv("VIDEO_MIN_CONFIDENCE", "0.0"))
    VIDEO_MAX_SAMPLED_FRAMES = int(os.getenv("VIDEO_MAX_SAMPLED_FRAMES", "12"))
    # Higher resolution preserves detail for small forensic items such as guns.
    # Both values can be tuned through the backend .env file when needed.
    # Lower default resolution for CPU-heavy video analysis to avoid long request timeouts.
    # Increase this only if you need more detail on short, high-quality clips.
    YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "832"))
    YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
