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


@lru_cache
def get_settings() -> "Settings":
    return Settings()


class Settings:
    def __init__(self) -> None:
        self.max_concurrent_jobs: int = max(1, _int("WHISPER_MAX_CONCURRENT_JOBS", 1))
        # api: только HTTP + Redis (без загрузки Whisper в память API-контейнера); worker: обработчик очереди.
        self.process_role: str = _str("WHISPER_PROCESS_ROLE", "api").strip().lower() or "api"
        self.redis_url: str | None = (
            None if _str("REDIS_URL", "").strip() == "" else _str("REDIS_URL", "").strip()
        )
        self.job_heartbeat_sec: float = max(3.0, _float("WHISPER_JOB_HEARTBEAT_SEC", 15.0))
        self.job_stale_sec: float = max(30.0, _float("WHISPER_JOB_STALE_SEC", 300.0))
        # При старте whisper-worker: удалить whisper:queue и заново LPUSH все queued/waiting_gpu из Redis (сброс «зависшей» очереди).
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
        # >0: faster-whisper с word_timestamps и разбиение одного сегмента по паузам между словами
        # (отдельные реплики в diarized_segments — корректная подсветка по времени в UI).
        self.intra_segment_split_gap_sec: float = max(0.0, _float("WHISPER_INTRA_SEGMENT_SPLIT_GAP_SEC", 0.45))
        self.download_timeout_sec: float = max(10.0, _float("WHISPER_DOWNLOAD_TIMEOUT_SEC", 300.0))
        self.max_upload_bytes: int = max(1_000_000, _int("WHISPER_MAX_DOWNLOAD_BYTES", 200_000_000))
        # Синхронизация RX/TX по общей mono (url_mix): даунсэмпл огибающей, корреляция на первых N сек.
        self.channel_sync_downsample_step: int = max(16, _int("WHISPER_CHANNEL_SYNC_DOWNSAMPLE_STEP", 80))
        self.channel_sync_correlation_max_sec: float = max(5.0, _float("WHISPER_CHANNEL_SYNC_CORR_MAX_SEC", 120.0))
        self.channel_sync_max_lag_sec: float = max(0.25, _float("WHISPER_CHANNEL_SYNC_MAX_LAG_SEC", 5.0))
        self.channel_sync_min_correlation: float = max(0.0, _float("WHISPER_CHANNEL_SYNC_MIN_CORR", 0.04))
        # Макс. |offset| (сек) по каждому каналу после auto-корреляции; выше — канал «не доверен», см. sync.max_offset_sec в запросе
        self.sync_max_offset_sec: float = max(0.05, _float("WHISPER_SYNC_MAX_OFFSET_SEC", 2.0))
        self.api_token: str | None = (
            None if _str("WHISPER_API_TOKEN", "").strip() == "" else _str("WHISPER_API_TOKEN", "").strip()
        )
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
