"""Runtime-конфиг и stats воркера в Redis (переключение backend без правки .env)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from app.job_models import TERMINAL_STATUSES
from app.job_store import QUEUE_KEY, get_job, list_active_job_ids

log = logging.getLogger(__name__)

DESIRED_BACKEND_KEY = "whisper:runtime:desired_backend"
DESIRED_LOCAL_MAX_KEY = "whisper:runtime:local_max_concurrent"
DESIRED_OPENAI_MAX_KEY = "whisper:runtime:openai_max_concurrent"
WORKER_STATS_KEY = "whisper:worker:stats"
WORKER_STATS_TTL_SEC = 90
OPENAI_MAX_CONCURRENT_HARD_CAP = 200

ACTIVE_RUN_STATUSES = frozenset(
    {
        "downloading",
        "transcribing_rx",
        "transcribing_tx",
        "transcribing_mix",
        "syncing_channels",
        "merging_segments",
    }
)
WAITING_STATUSES = frozenset({"waiting_gpu"})
QUEUED_STATUSES = frozenset({"queued"})


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_backend(raw: str | None) -> str | None:
    if raw is None:
        return None
    v = str(raw).strip().lower()
    if v in ("local", "openai"):
        return v
    return None


async def get_desired_backend(redis_c: redis.Redis) -> str | None:
    raw = await redis_c.get(DESIRED_BACKEND_KEY)
    return normalize_backend(raw if isinstance(raw, str) else None)


async def set_desired_backend(redis_c: redis.Redis, backend: str) -> str:
    b = normalize_backend(backend)
    if b is None:
        raise ValueError("backend must be 'local' or 'openai'")
    await redis_c.set(DESIRED_BACKEND_KEY, b)
    return b


async def ensure_desired_backend(redis_c: redis.Redis, fallback: str) -> str:
    """Если ключа нет — засеять из env (fallback)."""
    cur = await get_desired_backend(redis_c)
    if cur is not None:
        return cur
    b = normalize_backend(fallback) or "local"
    await redis_c.set(DESIRED_BACKEND_KEY, b)
    return b


async def get_desired_local_max(redis_c: redis.Redis) -> int | None:
    raw = await redis_c.get(DESIRED_LOCAL_MAX_KEY)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return None


async def get_desired_openai_max(redis_c: redis.Redis) -> int | None:
    raw = await redis_c.get(DESIRED_OPENAI_MAX_KEY)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(1, min(OPENAI_MAX_CONCURRENT_HARD_CAP, int(raw)))
    except (TypeError, ValueError):
        return None


async def set_desired_local_max(redis_c: redis.Redis, n: int) -> int:
    v = max(1, int(n))
    await redis_c.set(DESIRED_LOCAL_MAX_KEY, str(v))
    return v


async def set_desired_openai_max(redis_c: redis.Redis, n: int) -> int:
    v = max(1, min(OPENAI_MAX_CONCURRENT_HARD_CAP, int(n)))
    await redis_c.set(DESIRED_OPENAI_MAX_KEY, str(v))
    return v


async def ensure_desired_concurrency(
    redis_c: redis.Redis,
    *,
    local_fallback: int,
    openai_fallback: int,
) -> tuple[int, int]:
    loc = await get_desired_local_max(redis_c)
    if loc is None:
        loc = await set_desired_local_max(redis_c, max(1, local_fallback))
    oa = await get_desired_openai_max(redis_c)
    if oa is None:
        oa = await set_desired_openai_max(redis_c, max(1, openai_fallback))
    return loc, oa


async def publish_worker_stats(redis_c: redis.Redis, stats: dict[str, Any]) -> None:
    payload = dict(stats)
    payload["updated_at"] = _utc_iso()
    await redis_c.set(WORKER_STATS_KEY, json.dumps(payload, ensure_ascii=False), ex=WORKER_STATS_TTL_SEC)


async def get_worker_stats(redis_c: redis.Redis) -> dict[str, Any] | None:
    raw = await redis_c.get(WORKER_STATS_KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def count_jobs_by_bucket(redis_c: redis.Redis) -> dict[str, Any]:
    """Счётчики задач по статусам — только по индексу активных (без полного SCAN)."""
    active: list[dict[str, Any]] = []
    waiting = 0
    queued = 0
    for jid in await list_active_job_ids(redis_c):
        rec = await get_job(redis_c, jid)
        if rec is None or rec.status in TERMINAL_STATUSES:
            continue
        if rec.status in ACTIVE_RUN_STATUSES:
            active.append(
                {
                    "job_id": rec.job_id,
                    "dedup_key": rec.dedup_key,
                    "status": rec.status,
                    "progress": rec.progress,
                    "current_step": rec.current_step,
                    "api_key_id": getattr(rec, "api_key_id", None),
                    "api_key_name": getattr(rec, "api_key_name", None),
                }
            )
        elif rec.status in WAITING_STATUSES:
            waiting += 1
        elif rec.status in QUEUED_STATUSES:
            queued += 1
    try:
        queue_len = int(await redis_c.llen(QUEUE_KEY))
    except Exception:
        queue_len = queued
    return {
        "active_run": len(active),
        "waiting_slot": waiting,
        "queued": queued,
        "queue_len": queue_len,
        "active_jobs": active[:40],
    }
