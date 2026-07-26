"""Конфигурация из переменных окружения."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# Runtime override (worker UI): None → значение из WHISPER_BACKEND.
_runtime_backend: str | None = None
_runtime_local_max: int | None = None
_runtime_openai_max: int | None = None


def set_runtime_backend(backend: str | None) -> None:
    """Установить эффективный бэкенд без перезапуска процесса (None = снова из env)."""
    global _runtime_backend
    if backend is None:
        _runtime_backend = None
        return
    b = backend.strip().lower()
    if b not in ("local", "openai"):
        raise ValueError("backend must be 'local' or 'openai'")
    _runtime_backend = b


def get_runtime_backend() -> str | None:
    return _runtime_backend


def set_runtime_local_max(n: int | None) -> None:
    global _runtime_local_max
    if n is None:
        _runtime_local_max = None
        return
    _runtime_local_max = max(1, int(n))


def set_runtime_openai_max(n: int | None) -> None:
    global _runtime_openai_max
    if n is None:
        _runtime_openai_max = None
        return
    _runtime_openai_max = max(1, int(n))


@lru_cache
def get_settings() -> "Settings":
    return Settings()


class Settings:
    def __init__(self) -> None:
        # local = faster-whisper на GPU; openai = облачный OpenAI Audio Transcriptions API.
        _backend = _str("WHISPER_BACKEND", "local").strip().lower() or "local"
        if _backend not in ("local", "openai"):
            _backend = "local"
        self._env_backend: str = _backend
        # Параллелизм раздельно: локальный GPU обычно 1; OpenAI — по RPM tier.
        self._env_local_max_concurrent_jobs: int = max(1, _int("WHISPER_MAX_CONCURRENT_JOBS", 1))
        self._env_openai_max_concurrent_jobs: int = max(1, _int("OPENAI_MAX_CONCURRENT_JOBS", 8))
        # Оценка VRAM на слот (MB); 0 = авто по имени модели.
        self.vram_per_slot_mb: int = max(0, _int("WHISPER_VRAM_PER_SLOT_MB", 0))
        self.vram_reserve_mb: float = max(0.0, _float("WHISPER_VRAM_RESERVE_MB", 800.0))
        _oa_key = _str("OPENAI_API_KEY", "").strip()
        self.openai_api_key: str | None = _oa_key or None
        _oa_base = _str("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        self.openai_base_url: str = _oa_base if _oa_base else "https://api.openai.com/v1"
        _oa_model = _str("OPENAI_TRANSCRIBE_MODEL", "whisper-1").strip()
        self.openai_transcribe_model: str = _oa_model if _oa_model else "whisper-1"
        self.openai_timeout_sec: float = max(30.0, _float("OPENAI_TIMEOUT_SEC", 600.0))
        self.openai_max_upload_mb: float = max(1.0, _float("OPENAI_MAX_UPLOAD_MB", 24.0))
        self.openai_max_retries: int = max(0, _int("OPENAI_MAX_RETRIES", 3))
        # api: только HTTP + Redis (без загрузки Whisper в память API-контейнера); worker: обработчик очереди.
        self.process_role: str = _str("WHISPER_PROCESS_ROLE", "api").strip().lower() or "api"
        self.redis_url: str | None = (
            None if _str("REDIS_URL", "").strip() == "" else _str("REDIS_URL", "").strip()
        )
        self.job_heartbeat_sec: float = max(3.0, _float("WHISPER_JOB_HEARTBEAT_SEC", 15.0))
        self.job_stale_sec: float = max(30.0, _float("WHISPER_JOB_STALE_SEC", 300.0))
        # При старте whisper-worker: удалить whisper:queue и заново LPUSH все queued/waiting_gpu из Redis.
        self.worker_queue_flush_on_start: bool = _bool("WHISPER_WORKER_QUEUE_FLUSH_ON_START", True)
        # Ключ whisper:worker:alive в Redis: пока worker жив, обновляет TTL.
        self.worker_alive_ttl_sec: int = max(30, _int("WHISPER_WORKER_ALIVE_TTL_SEC", 90))
        self.worker_alive_refresh_sec: float = max(5.0, _float("WHISPER_WORKER_ALIVE_REFRESH_SEC", 25.0))
        self.listen_host: str = _str("WHISPER_LISTEN_HOST", "0.0.0.0")
        self.listen_port: int = _int("WHISPER_LISTEN_PORT", 19900)
        self.model_path: str = _str("WHISPER_MODEL_PATH", "/models/turbo")
        self.compute_type: str = _str("WHISPER_COMPUTE_TYPE", "float16")
        self.device: str = _str("WHISPER_DEVICE", "cuda")
        self.language: str | None = (
            None if _str("WHISPER_LANGUAGE", "").strip() == "" else _str("WHISPER_LANGUAGE", "").strip()
        )
        self.beam_size: int = max(1, _int("WHISPER_BEAM_SIZE", 5))
        self.vad_filter: bool = _str("WHISPER_VAD_FILTER", "1").strip() in ("1", "true", "yes", "on")
        # >0: faster-whisper с word_timestamps и разбиение одного сегмента по паузам между словами.
        self.intra_segment_split_gap_sec: float = max(0.0, _float("WHISPER_INTRA_SEGMENT_SPLIT_GAP_SEC", 0.45))
        self.download_timeout_sec: float = max(10.0, _float("WHISPER_DOWNLOAD_TIMEOUT_SEC", 300.0))
        self.max_upload_bytes: int = max(1_000_000, _int("WHISPER_MAX_DOWNLOAD_BYTES", 200_000_000))
        # Синхронизация RX/TX по общей mono (url_mix).
        self.channel_sync_downsample_step: int = max(16, _int("WHISPER_CHANNEL_SYNC_DOWNSAMPLE_STEP", 80))
        self.channel_sync_correlation_max_sec: float = max(5.0, _float("WHISPER_CHANNEL_SYNC_CORR_MAX_SEC", 120.0))
        self.channel_sync_max_lag_sec: float = max(0.25, _float("WHISPER_CHANNEL_SYNC_MAX_LAG_SEC", 5.0))
        self.channel_sync_min_correlation: float = max(0.0, _float("WHISPER_CHANNEL_SYNC_MIN_CORR", 0.04))
        self.sync_max_offset_sec: float = max(0.05, _float("WHISPER_SYNC_MAX_OFFSET_SEC", 2.0))
        # Внешний API: legacy single token (синхронизируется в Redis как ключ env-default).
        self.api_token: str | None = (
            None if _str("WHISPER_API_TOKEN", "").strip() == "" else _str("WHISPER_API_TOKEN", "").strip()
        )
        _tok_name = _str("WHISPER_API_TOKEN_NAME", "default").strip()
        self.api_token_name: str = _tok_name if _tok_name else "default"
        # UI dashboard: логин/пароль (отдельно от API Bearer).
        _ui_user = _str("WHISPER_UI_USER", "").strip()
        _ui_pass = _str("WHISPER_UI_PASSWORD", "").strip()
        self.ui_user: str | None = _ui_user or None
        self.ui_password: str | None = _ui_pass or None
        self.ui_session_ttl_sec: int = max(300, _int("WHISPER_UI_SESSION_TTL_SEC", 86400))
        self.diarization_enabled: bool = _bool("WHISPER_DIARIZATION", False)
        self.diarize_default: bool = _bool("WHISPER_DIARIZE_DEFAULT", True)
        hf = _str("HF_TOKEN", "").strip() or _str("HUGGINGFACE_HUB_TOKEN", "").strip()
        self.hf_token: str | None = hf or None
        _pipe = _str("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1").strip()
        self.pyannote_pipeline: str = _pipe if _pipe else "pyannote/speaker-diarization-3.1"
        _pdev = _str("PYANNOTE_DEVICE", "cpu").strip()
        self.pyannote_device: str = _pdev if _pdev else "cpu"

        _logs = _str("WHISPER_LOGS_DIR", "/logs").strip()
        self.logs_dir: Path = Path(_logs if _logs else "/logs")

        self.spelling_fixes_enabled: bool = _bool("WHISPER_SPELLING_FIXES", True)
        _spell_path = _str("WHISPER_SPELLING_DICT_PATH", "").strip()
        self.spelling_dict_path: Path = (
            Path(_spell_path) if _spell_path else Path(__file__).resolve().parent / "spelling_dictionary.json"
        )

        self.speaker_roles_catalog_enabled: bool = _bool("WHISPER_SPEAKER_ROLES_CATALOG", True)
        _src_roles = _str("WHISPER_SPEAKER_ROLES_CATALOG_PATH", "").strip()
        self.speaker_roles_catalog_path: Path = (
            Path(_src_roles)
            if _src_roles
            else Path(__file__).resolve().parent / "speaker_roles_catalog.json"
        )

    @property
    def whisper_backend(self) -> str:
        return _runtime_backend if _runtime_backend is not None else self._env_backend

    @property
    def local_max_concurrent_jobs(self) -> int:
        return _runtime_local_max if _runtime_local_max is not None else self._env_local_max_concurrent_jobs

    @property
    def openai_max_concurrent_jobs(self) -> int:
        return _runtime_openai_max if _runtime_openai_max is not None else self._env_openai_max_concurrent_jobs

    @property
    def max_concurrent_jobs(self) -> int:
        # Активный лимит для текущего эффективного бэкенда (worker/health).
        if self.whisper_backend == "openai":
            return self.openai_max_concurrent_jobs
        return self.local_max_concurrent_jobs

    @property
    def ui_auth_configured(self) -> bool:
        return bool(self.ui_user and self.ui_password)