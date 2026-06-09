"""Пометка зависших задач по heartbeat_at."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import redis.asyncio as redis

from app.job_models import TERMINAL_STATUSES, TranscribeJobRecord
from app.job_store import _utc_iso, get_job, save_job, scan_job_ids
from app.settings import get_settings

log = logging.getLogger(__name__)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


async def maybe_mark_job_stale(redis: redis.Redis, job_id: str) -> TranscribeJobRecord | None:
    """Если heartbeat просрочен — помечает stale_failed. Возвращает актуальную запись."""
    settings = get_settings()
    rec = await get_job(redis, job_id)
    if rec is None:
        return None
    if rec.status in TERMINAL_STATUSES:
        return rec
    hb = _parse_iso(rec.heartbeat_at) or _parse_iso(rec.updated_at)
    if hb is None:
        return rec
    age = (datetime.now(timezone.utc) - hb).total_seconds()
    if age <= settings.job_stale_sec:
        return rec
    err = f"job_stale: heartbeat older than {settings.job_stale_sec}s"
    now = _utc_iso()
    patched = rec.model_copy(
        update={
            "status": "stale_failed",
            "error": err,
            "completed_at": now,
            "updated_at": now,
            "current_step": rec.current_step or "stale watchdog",
        }
    )
    await save_job(redis, patched)
    log.warning(
        "transcribe_job_failed job_id=%s dedup_key=%s status=stale_failed current_step=%s error=%s",
        job_id,
        rec.dedup_key,
        rec.current_step,
        err,
    )
    return patched


async def scan_stale_jobs(redis: redis.Redis) -> int:
    """Проходит по job-ключам и помечает зависшие. Возвращает число помеченных."""
    n = 0
    for jid in await scan_job_ids(redis):
        before = await get_job(redis, jid)
        if before is None or before.status in TERMINAL_STATUSES:
            continue
        after = await maybe_mark_job_stale(redis, jid)
        if after is not None and after.status == "stale_failed" and (before is None or before.status != "stale_failed"):
            n += 1
    return n
