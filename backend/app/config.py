import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent if BACKEND_DIR.name == "backend" else BACKEND_DIR
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _bool(val: str, default: bool) -> bool:
    """Parse a boolean environment value.

    Recognizes ``1``, ``true`` and ``yes`` (case and surrounding whitespace
    ignored) as true. A missing value returns ``default``; other provided
    values are treated as false.
    """
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes"}


def _float(val: str, default: float) -> float:
    """Parse a numeric environment value and clamp it to ``[0.0, 1.0]``.

    Malformed or missing values use ``default``. This is used for per-class
    confidence cutoffs so an environment typo cannot create an invalid
    probability threshold.
    """
    try:
        value = float(val)
        return max(0.0, min(1.0, value))
    except (TypeError, ValueError):
        return default


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


def _resolve_video_model_path():
    """Resolve VIDEO_MODEL_PATH from either backend-relative or repo-relative form.

    Supported examples are ``models/video_classifier_v2.pth`` and
    ``backend/models/video_classifier_v2.pth``. An explicit missing path is
    preserved so the model loader reports the configuration error instead of
    silently reverting to the production checkpoint.
    """
    configured = os.getenv("VIDEO_MODEL_PATH")
    if not configured:
        return str((BACKEND_DIR / "models" / "video_classifier.pth").resolve())
    candidate = Path(configured).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    if candidate.parts and candidate.parts[0].casefold() == "backend":
        return str((PROJECT_ROOT / candidate).resolve())
    return str((BACKEND_DIR / candidate).resolve())


def _resolve_config_path(env_name, default_value):
    """Resolve configurable data paths relative to the repository root."""
    value = os.getenv(env_name)
    candidate = Path(value).expanduser() if value else Path(default_value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate.resolve())


class Config:
    # Environment defaults (also listed in the repository-root .env.example):
    # SECRET_KEY=generated, JWT_SECRET_KEY=persisted local secret,
    # MONGO_URI=mongodb://localhost:27017, DB_NAME=crime_evidence_db,
    # UPLOAD_FOLDER=backend/uploads, YOLO_MODEL_PATH=backend/models/best.pt,
    # CNN_MODEL_PATH=backend/models/best_cnn.pth, CNN_IMAGE_SIZE=224,
    # CNN_MEAN=0.485,0.456,0.406, CNN_STD=0.229,0.224,0.225,
    # CNN_CONFIDENCE=0.0,
    # CNN_PREPROCESSING_SOURCE="config_default; verify against training transforms",
    # VIDEO_MODEL_PATH=backend/models/video_classifier.pth,
    # CLASS_NAMES_PATH=backend/models/class_names.json,
    # ANOMALY_MODEL_PATH=backend/models/bigru_mil_best.pt,
    # YOLO_IMAGE_SIZE=832, YOLO_CONFIDENCE=0.25,
    # OLLAMA_HOST=http://127.0.0.1:11434, OLLAMA_MODEL=llama3.2.
    # Video-specific defaults continue below, including all 9 class thresholds.
    SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(32).hex()
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    OLLAMA_HOST = OLLAMA_BASE_URL  # legacy configuration alias
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_HEALTH_TIMEOUT = max(1, float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "4")))
    OLLAMA_CHAT_TIMEOUT = max(1, float(os.getenv("OLLAMA_CHAT_TIMEOUT", "120")))
    OLLAMA_HEALTH_TIMEOUT_SEC = OLLAMA_HEALTH_TIMEOUT
    OLLAMA_CHAT_TIMEOUT_SEC = OLLAMA_CHAT_TIMEOUT
    JWT_SECRET_KEY = _jwt_secret_key()
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME = os.getenv("MONGO_DB_NAME", os.getenv("DB_NAME", "crime_evidence_db"))
    MONGO_SERVER_SELECTION_TIMEOUT_MS = max(250, int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "1000")))
    MONGO_SOCKET_TIMEOUT_MS = max(1000, int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "10000")))
    MONGO_RECONNECT_INTERVAL_SECONDS = max(1.0, float(os.getenv("MONGO_RECONNECT_INTERVAL_SECONDS", "5")))
    UPLOAD_FOLDER = _resolve_existing_path("UPLOAD_FOLDER", BACKEND_DIR / "uploads")
    # Read-only catalogs for previously generated category evidence. These paths
    # can point at a synced Drive folder through .env without copying artifacts.
    _default_source_root = Path.home() / "OneDrive" / "Documents" / "crime_video_dataset"
    EVIDENCE_SOURCE_DIR = _resolve_config_path("EVIDENCE_SOURCE_DIR", _default_source_root)
    ASSAULT_RESULTS_DIR = _resolve_config_path(
        "ASSAULT_RESULTS_DIR", BACKEND_DIR / "tracked_results_400" / "Assault"
    )
    ASSAULT_TRACKING_STATUS_PATH = _resolve_config_path(
        "ASSAULT_TRACKING_STATUS_PATH", Path(ASSAULT_RESULTS_DIR) / "tracking_status.csv"
    )
    ROBBERY_RESULTS_DIR = _resolve_config_path(
        "ROBBERY_RESULTS_DIR", BACKEND_DIR / "tracked_results_400" / "Robbery"
    )
    ROBBERY_TRACKING_STATUS_PATH = _resolve_config_path(
        "ROBBERY_TRACKING_STATUS_PATH", Path(ROBBERY_RESULTS_DIR) / "tracking_status.csv"
    )
    IMPORTED_TRACKING_RESULTS_DIR = _resolve_config_path(
        "IMPORTED_TRACKING_RESULTS_DIR", BACKEND_DIR / "tracked_results_drive_import" / "crime_tracking_results" / "bytetrack"
    )
    SHOOTING_RESULTS_DIR = _resolve_config_path(
        "SHOOTING_RESULTS_DIR",
        BACKEND_DIR / "tracked_results_shooting_import" / "main_archive" / "Shooting" / "detections",
    )
    NORMAL_RESULTS_DIR = _resolve_config_path(
        "NORMAL_RESULTS_DIR", BACKEND_DIR / "tracked_results_400" / "Normal"
    )
    SHOOTING_TRACKING_STATUS_PATH = _resolve_config_path(
        "SHOOTING_TRACKING_STATUS_PATH",
        BACKEND_DIR / "tracked_results_shooting_import" / "main_archive" / "Shooting" / "shooting_tracking_status.csv",
    )
    ASSAULT_LEGACY_JSON_PATH = _resolve_config_path(
        "ASSAULT_LEGACY_JSON_PATH", BACKEND_DIR / "assault_77_video_detections.json"
    )
    YOLO_MODEL_PATH = _resolve_existing_path(
        "YOLO_MODEL_PATH",
        BACKEND_DIR / "models" / "best.pt",
    )
    CNN_MODEL_PATH = _resolve_existing_path("CNN_MODEL_PATH", BACKEND_DIR / "models" / "best_cnn.pth")
    CNN_IMAGE_SIZE = int(os.getenv("CNN_IMAGE_SIZE", "224"))
    CNN_MEAN = tuple(float(v) for v in os.getenv("CNN_MEAN", "0.485,0.456,0.406").split(","))
    CNN_STD = tuple(float(v) for v in os.getenv("CNN_STD", "0.229,0.224,0.225").split(","))
    CNN_CONFIDENCE = float(os.getenv("CNN_CONFIDENCE", "0.0"))
    CNN_PREPROCESSING_SOURCE = os.getenv("CNN_PREPROCESSING_SOURCE", "config_default; verify against training transforms")
    VIDEO_MODEL_PATH = _resolve_video_model_path()
    CLASS_NAMES_PATH = _resolve_existing_path("CLASS_NAMES_PATH", BACKEND_DIR / "models" / "class_names.json")
    VIDEO_CLASSIFIER_BACKEND = os.getenv("VIDEO_CLASSIFIER_BACKEND", "legacy").strip().lower()
    VIDEO_DENSENET_MODEL_PATH = _resolve_existing_path(
        "VIDEO_DENSENET_MODEL_PATH", BACKEND_DIR / "models" / "densenet_attention_convlstm" / "best_checkpoint.pth"
    )
    ANOMALY_MODEL_PATH = _resolve_existing_path(
        "ANOMALY_MODEL_PATH", BACKEND_DIR / "models" / "bigru_mil_best.pt"
    )
    VIDEO_ANOMALY_MODEL_PATH = _resolve_existing_path(
        "VIDEO_ANOMALY_MODEL_PATH", BACKEND_DIR / "models" / "video_anomaly_autoencoder.pth"
    )
    VIDEO_ANOMALY_CONFIG_PATH = _resolve_existing_path(
        "VIDEO_ANOMALY_CONFIG_PATH", BACKEND_DIR / "models" / "video_anomaly_config.json"
    )
    VIDEO_ANOMALY_MAX_GAP_SEC = max(0.0, float(os.getenv("VIDEO_ANOMALY_MAX_GAP_SEC", "3")))
    VIDEO_MIN_CONFIDENCE = float(os.getenv("VIDEO_MIN_CONFIDENCE", "0.0"))
    # Video YOLO policy: sample at a target rate, with hard work limits.
    VIDEO_SAMPLE_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "1"))
    VIDEO_SAMPLE_INTERVAL = int(os.getenv("VIDEO_SAMPLE_INTERVAL", "0"))  # frame stride override; 0 uses target FPS
    VIDEO_MAX_PROCESSED_FRAMES = int(os.getenv("VIDEO_MAX_PROCESSED_FRAMES", "300"))
    VIDEO_MAX_DURATION_SEC = float(os.getenv("VIDEO_MAX_DURATION_SEC", "600"))
    VIDEO_TRACKING_ENABLED = os.getenv("VIDEO_TRACKING_ENABLED", "true").lower() == "true"
    VIDEO_TRACKER = os.getenv("VIDEO_TRACKER", "bytetrack.yaml")
    VIDEO_CONFIDENCE_THRESHOLD = float(os.getenv("VIDEO_CONFIDENCE_THRESHOLD", os.getenv("YOLO_CONFIDENCE", "0.25")))
    VIDEO_SCENE_CHANGE_ENABLED = os.getenv("VIDEO_SCENE_CHANGE_ENABLED", "true").lower() == "true"
    VIDEO_SCENE_CHANGE_SSIM_THRESHOLD = float(os.getenv("VIDEO_SCENE_CHANGE_SSIM_THRESHOLD", "0.55"))
    VIDEO_SCENE_CHANGE_DIFF_RATIO = float(os.getenv("VIDEO_SCENE_CHANGE_DIFF_RATIO", "0.30"))
    # Experimental R3D-18 uses 32 segments × 16 sampled frames = at most 512 reads.
    R3D18_MAX_DURATION_SEC = float(os.getenv("R3D18_MAX_DURATION_SEC", str(VIDEO_MAX_DURATION_SEC)))
    VIDEO_EVENT_MAX_GAP_SEC = float(os.getenv("VIDEO_EVENT_MAX_GAP_SEC", "3"))
    VIDEO_ACTIVITY_MERGE_GAP_SEC = max(0.0, float(os.getenv("VIDEO_ACTIVITY_MERGE_GAP_SEC", "1.0")))
    VIDEO_EVENT_MIN_OBSERVATIONS = int(os.getenv("VIDEO_EVENT_MIN_OBSERVATIONS", "1"))
    VIDEO_EVENT_MIN_IOU = float(os.getenv("VIDEO_EVENT_MIN_IOU", "0.1"))
    TEMPORAL_EVENT_WEIGHTS = os.getenv("TEMPORAL_EVENT_WEIGHTS", "")
    TEMPORAL_EVENT_BIN_SEC = max(0.1, float(os.getenv("TEMPORAL_EVENT_BIN_SEC", "1.0")))
    TEMPORAL_EVENT_MIN_SCORE = min(1.0, max(0.0, float(os.getenv("TEMPORAL_EVENT_MIN_SCORE", "0.55"))))
    TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY = min(1.0, max(0.0, float(os.getenv("TEMPORAL_EVENT_MIN_CLASSIFIER_PROBABILITY", "0.5"))))
    TEMPORAL_EVENT_MIN_SUPPORTED_BINS = max(1, int(os.getenv("TEMPORAL_EVENT_MIN_SUPPORTED_BINS", "1")))
    TEMPORAL_EVENT_MAX_TRACK_GAP_SEC = max(0.0, float(os.getenv("TEMPORAL_EVENT_MAX_TRACK_GAP_SEC", "2.0")))
    TEMPORAL_EVENT_BOUNDARY_TOLERANCE_SEC = max(0.0, float(os.getenv("TEMPORAL_EVENT_BOUNDARY_TOLERANCE_SEC", "0.0")))
    TEMPORAL_EVENT_YOLO_PERSISTENCE_OBSERVATIONS = max(1, int(os.getenv("TEMPORAL_EVENT_YOLO_PERSISTENCE_OBSERVATIONS", "2")))
    # Kept as a request compatibility ceiling; actual work is governed above.
    VIDEO_MAX_SAMPLED_FRAMES = int(os.getenv("VIDEO_MAX_SAMPLED_FRAMES", "300"))
    # Video environment variables and defaults:
    # VIDEO_SAMPLE_FPS=1, VIDEO_SAMPLE_INTERVAL=0,
    # VIDEO_MAX_PROCESSED_FRAMES=300, VIDEO_MAX_DURATION_SEC=600,
    # VIDEO_TRACKING_ENABLED=true, VIDEO_TRACKER=bytetrack.yaml,
    # VIDEO_CONFIDENCE_THRESHOLD=YOLO_CONFIDENCE or 0.25,
    # VIDEO_MIN_CONFIDENCE=0.0, R3D18_MAX_DURATION_SEC=VIDEO_MAX_DURATION_SEC,
    # VIDEO_SCENE_CHANGE_ENABLED=true, VIDEO_SCENE_CHANGE_SSIM_THRESHOLD=0.55,
    # VIDEO_SCENE_CHANGE_DIFF_RATIO=0.30, VIDEO_EVENT_MAX_GAP_SEC=3,
    # VIDEO_ACTIVITY_MERGE_GAP_SEC=1.0,
    # VIDEO_EVENT_MIN_OBSERVATIONS=1, VIDEO_EVENT_MIN_IOU=0.1,
    # VIDEO_MAX_SAMPLED_FRAMES=300, VIDEO_PERSON_ROI_ENABLED=false,
    # VIDEO_PERSON_ROI_MAX=2, VIDEO_ROI_CONFIDENCE=0.25,
    # VIDEO_ROI_IMAGE_SIZE=640, VIDEO_MIN_LAPLACIAN_VARIANCE=8.0,
    # VIDEO_MOTION_DIFF_THRESHOLD=18.0, VIDEO_MOTION_RATIO_THRESHOLD=0.025,
    # VIDEO_MOTION_MAX_FPS=5.0, VIDEO_CLAHE_ENABLED=true,
    # Per-class defaults: GUN=0.30; HAMMER, HELMET, KNIFE, MASK, PERSON,
    # PISTOL, RIFLE, WEAPON=0.25. All per-class thresholds clamp to [0, 1].
    # Environment configuration for optional person ROI and per-class confidence.
    # ROI inference is disabled by default because crop inference can add false positives.
    VIDEO_PERSON_ROI_ENABLED = _bool(os.getenv("VIDEO_PERSON_ROI_ENABLED"), False)
    VIDEO_PERSON_ROI_MAX = int(os.getenv("VIDEO_PERSON_ROI_MAX", "2"))
    VIDEO_ROI_CONFIDENCE = float(os.getenv("VIDEO_ROI_CONFIDENCE", "0.25"))
    VIDEO_ROI_IMAGE_SIZE = int(os.getenv("VIDEO_ROI_IMAGE_SIZE", "640"))
    VIDEO_MIN_LAPLACIAN_VARIANCE = float(os.getenv("VIDEO_MIN_LAPLACIAN_VARIANCE", "8.0"))
    VIDEO_MOTION_DIFF_THRESHOLD = float(os.getenv("VIDEO_MOTION_DIFF_THRESHOLD", "18.0"))
    VIDEO_MOTION_RATIO_THRESHOLD = float(os.getenv("VIDEO_MOTION_RATIO_THRESHOLD", "0.025"))
    VIDEO_MOTION_MAX_FPS = float(os.getenv("VIDEO_MOTION_MAX_FPS", "5.0"))
    VIDEO_CLAHE_ENABLED = os.getenv("VIDEO_CLAHE_ENABLED", "true").lower() == "true"
    # Class ID thresholds retain the checkpoint mapping: gun, hammer, helmet,
    # knife, mask, person, pistol, rifle, weapon.
    VIDEO_CLASS_CONFIDENCE_GUN = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_GUN"), 0.30)
    VIDEO_CLASS_CONFIDENCE_HAMMER = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_HAMMER"), 0.25)
    VIDEO_CLASS_CONFIDENCE_HELMET = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_HELMET"), 0.25)
    VIDEO_CLASS_CONFIDENCE_KNIFE = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_KNIFE"), 0.25)
    VIDEO_CLASS_CONFIDENCE_MASK = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_MASK"), 0.25)
    VIDEO_CLASS_CONFIDENCE_PERSON = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_PERSON"), 0.25)
    VIDEO_CLASS_CONFIDENCE_PISTOL = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_PISTOL"), 0.25)
    VIDEO_CLASS_CONFIDENCE_RIFLE = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_RIFLE"), 0.25)
    VIDEO_CLASS_CONFIDENCE_WEAPON = _float(os.getenv("VIDEO_CLASS_CONFIDENCE_WEAPON"), 0.25)
    # All class confidence values are clamped to [0.0, 1.0].

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
