"""Модели задач транскрипции (Redis)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "queued",
    "waiting_gpu",
    "downloading",
    "transcribing_rx",
    "transcribing_tx",
    "transcribing_mix",
    "syncing_channels",
    "merging_segments",
    "completed",
    "failed",
    "cancelled",
    "stale_failed",
]

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "stale_failed"})
RESTART_ALLOWED_STATUSES: frozenset[str] = frozenset({"failed", "cancelled", "stale_failed"})

# Задача считается «активной» для дедупа: не создаём новую, возвращаем существующий job_id.
DEDUP_ACTIVE_STATUSES: frozenset[str] = frozenset(
    {
        "queued",
        "waiting_gpu",
        "downloading",
        "transcribing_rx",
        "transcribing_tx",
        "transcribing_mix",
        "syncing_channels",
        "merging_segments",
        "completed",
    }
)


class TranscribeJobRecord(BaseModel):
    job_id: str
    dedup_key: str
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    updated_at: str
    completed_at: str | None = None
    progress: int = Field(0, ge=0, le=100)
    current_step: str = ""
    heartbeat_at: str | None = None
    attempts: int = 0


class TranscribeJobEnqueueResponse(BaseModel):
    job_id: str
    dedup_key: str
    status: JobStatus
    existing: bool


class TranscribeJobStatusResponse(BaseModel):
    job_id: str
    dedup_key: str
    status: JobStatus
    progress: int
    current_step: str
    created_at: str
    started_at: str | None
    updated_at: str
    heartbeat_at: str | None
    completed_at: str | None
    error: str | None
    result: dict[str, Any] | None
    worker_available: bool = Field(
        ...,
        description="True, если whisper-worker недавно обновлял ключ `whisper:worker:alive` в Redis.",
    )
    retry_recommended: bool = Field(
        ...,
        description="True — имеет смысл снова вызвать POST /transcribe (или дождаться worker).",
    )
    client_hint: str | None = Field(
        None,
        description="Краткая подсказка при остановке worker, сбросе задачи или stale.",
    )
    in_redis_queue: bool | None = Field(
        None,
        description="Для status=queued: есть ли job_id в списке whisper:queue (диагностика рассинхрона).",
    )
