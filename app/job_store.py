"""Redis: ключи задач, очередь, атомарное создание / обновление."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from app.dedup import compute_dedup_key_from_payload_dict
from app.job_models import (
    DEDUP_ACTIVE_STATUSES,
    RESTART_ALLOWED_STATUSES,
    TERMINAL_STATUSES,
    TranscribeJobRecord,
)
from app.settings import get_settings

log = logging.getLogger(__name__)

KEY_PREFIX_JOB = "whisper:job:"
KEY_PREFIX_DEDUP = "whisper:dedup:"
QUEUE_KEY = "whisper:queue"
LOCK_PREFIX = "whisper:dedup_lock:"
WORKER_ALIVE_KEY = "whisper:worker:alive"
# Префикс для программной проверки; полный текст — в recover_jobs_on_worker_startup.
WORKER_ORPHAN_ERROR = (
    "worker_orphaned: whisper-worker was restarted during processing; "
    "repeat POST /transcribe with the same JSON to enqueue a new job."
)
_REQUEUE_AFTER_RESTART_STATUSES = frozenset({"waiting_gpu"})
_PENDING_QUEUE_STATUSES = frozenset({"queued", "waiting_gpu"})
_PROCESSING_ORPHAN_STATUSES = frozenset(
    {
        "downloading",
        "transcribing_rx",
        "transcribing_tx",
        "transcribing_mix",
        "syncing_channels",
        "merging_segments",
    }
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_key(job_id: str) -> str:
    return f"{KEY_PREFIX_JOB}{job_id}"


def dedup_key_index(dedup_key: str) -> str:
    return f"{KEY_PREFIX_DEDUP}{dedup_key}"


def dedup_lock_key(dedup_key: str) -> str:
    return f"{LOCK_PREFIX}{dedup_key}"


async def connect_redis(url: str) -> redis.Redis:
    r = redis.from_url(url, decode_responses=True)
    await r.ping()
    return r


def _parse_job(raw: str | None) -> TranscribeJobRecord | None:
    if not raw:
        return None
    try:
        return TranscribeJobRecord.model_validate_json(raw)
    except Exception:
        log.warning("job_parse_failed raw_len=%s", len(raw) if raw else 0)
        return None


async def get_job(redis: redis.Redis, job_id: str) -> TranscribeJobRecord | None:
    raw = await redis.get(job_key(job_id))
    return _parse_job(raw)


async def save_job(redis: redis.Redis, rec: TranscribeJobRecord) -> None:
    await redis.set(job_key(rec.job_id), rec.model_dump_json())


async def get_job_id_for_dedup(redis: redis.Redis, dedup_key: str) -> str | None:
    jid = await redis.get(dedup_key_index(dedup_key))
    return jid if jid else None


# LPOS → при отсутствии в Lua не number — LPUSH (атомарно, без отдельного guard-ключа).
_ENQUEUE_IF_ABSENT_LUA = """
local p = redis.call('LPOS', KEYS[1], ARGV[1])
if type(p) == 'number' then
  return 0
end
redis.call('LPUSH', KEYS[1], ARGV[1])
return 1
"""


def _lpos_is_found(pos: Any) -> bool:
    """Redis LPOS: позиция int >= 0; иное (nil/false) — не найдено."""
    return isinstance(pos, int)


async def enqueue_job_id(redis: redis.Redis, job_id: str) -> None:
    await redis.lpush(QUEUE_KEY, job_id)


async def is_job_id_in_queue(redis: redis.Redis, job_id: str) -> bool:
    """Проверяет, есть ли job_id в списке whisper:queue (LPOS или LRANGE)."""
    try:
        pos = await redis.lpos(QUEUE_KEY, job_id)
    except Exception as e:
        log.warning("queue_lpos_error job_id=%s err=%s", job_id, e)
        pos = None
    if _lpos_is_found(pos):
        return True
    if pos is not None and not _lpos_is_found(pos):
        log.warning("queue_lpos_unexpected job_id=%s value=%r", job_id, pos)
    try:
        items = await redis.lrange(QUEUE_KEY, 0, -1)
    except Exception as e:
        log.warning("queue_lrange_failed job_id=%s err=%s", job_id, e)
        return False
    return job_id in items


async def ensure_queued_job_in_redis_queue(redis: redis.Redis, job_id: str) -> bool:
    """Атомарно: LPUSH job_id, если его ещё нет в whisper:queue. Возвращает True, если добавили."""
    try:
        added = await redis.eval(_ENQUEUE_IF_ABSENT_LUA, 1, QUEUE_KEY, job_id)
        if int(added) == 1:
            log.info(
                "queue_self_heal job_id=%s status=queued current_step=atomic_lpush_missing_queue",
                job_id,
            )
            return True
        return False
    except Exception as e:
        log.warning("queue_atomic_enqueue_failed job_id=%s err=%s; fallback_lpush", job_id, e)
        if await is_job_id_in_queue(redis, job_id):
            return False
        try:
            await enqueue_job_id(redis, job_id)
            log.info("queue_self_heal job_id=%s status=queued current_step=fallback_lpush", job_id)
            return True
        except Exception:
            log.exception("queue_lpush_fallback_failed job_id=%s", job_id)
            return False


async def mark_job_claimed_after_brpop(redis: redis.Redis, job_id: str) -> None:
    """В Redis: job ещё может быть `queued`, хотя id уже снят с whisper:queue — фиксируем waiting_gpu для GET/self-heal.

    Вызывается из корутины обработки **после** валидации payload (не в цикле BRPOP), чтобы не писать waiting_gpu, если задача так и не стартовала.
    """
    rec = await get_job(redis, job_id)
    if rec is None:
        log.warning("brpop_unknown_job_id job_id=%s", job_id)
        return
    if rec.status in TERMINAL_STATUSES:
        log.info("brpop_terminal_job job_id=%s status=%s", job_id, rec.status)
        return
    if rec.status not in ("queued", "waiting_gpu"):
        return
    now = _utc_iso()
    patched = rec.model_copy(
        update={
            "status": "waiting_gpu",
            "current_step": "worker claimed, waiting for run slot",
            "updated_at": now,
            "heartbeat_at": now,
        }
    )
    await save_job(redis, patched)


async def brpop_job_id(redis: redis.Redis, timeout_sec: int = 5) -> str | None:
    out = await redis.brpop(QUEUE_KEY, timeout=timeout_sec)
    if not out:
        return None
    _k, job_id = out
    return str(job_id)


async def acquire_dedup_lock(redis: redis.Redis, dedup_key: str, ttl_sec: int = 30) -> str | None:
    """Возвращает lock_token при успехе SET NX."""
    token = str(uuid.uuid4())
    ok = await redis.set(dedup_lock_key(dedup_key), token, nx=True, ex=ttl_sec)
    return token if ok else None


async def release_dedup_lock(redis: redis.Redis, dedup_key: str, token: str | None) -> None:
    if not token:
        return
    key = dedup_lock_key(dedup_key)
    lua = """
if redis.call("get", KEYS[1]) == ARGV[1] then
  return redis.call("del", KEYS[1])
else
  return 0
end
"""
    await redis.eval(lua, 1, key, token)


async def claim_or_get_existing_job(
    redis: redis.Redis,
    *,
    payload: dict[str, Any],
) -> tuple[TranscribeJobRecord, bool]:
    """Создаёт новую задачу или возвращает существующую по dedup_key. existing=True если не создавали новую."""
    dedup_key = compute_dedup_key_from_payload_dict(payload)
    lock_token = await acquire_dedup_lock(redis, dedup_key)
    if lock_token is None:
        # Ждём коротко и пробуем прочитать индекс без блокировки (редкий гон).
        import asyncio

        await asyncio.sleep(0.05)
        cur_id = await get_job_id_for_dedup(redis, dedup_key)
        if cur_id:
            job = await get_job(redis, cur_id)
            if job is not None and job.status in DEDUP_ACTIVE_STATUSES:
                log.info(
                    "transcribe_job_existing job_id=%s dedup_key=%s status=%s current_step=%s",
                    job.job_id,
                    dedup_key,
                    job.status,
                    job.current_step,
                )
                log.info(
                    "transcribe_job_duplicate_skipped job_id=%s dedup_key=%s status=%s current_step=%s",
                    job.job_id,
                    dedup_key,
                    job.status,
                    job.current_step,
                )
                return job, True
        # Повторная попытка лока
        lock_token = await acquire_dedup_lock(redis, dedup_key)
        if lock_token is None:
            raise RuntimeError("dedup_lock_busy")

    try:
        cur_id = await get_job_id_for_dedup(redis, dedup_key)
        if cur_id:
            job = await get_job(redis, cur_id)
            if job is not None:
                if job.status in DEDUP_ACTIVE_STATUSES:
                    log.info(
                        "transcribe_job_existing job_id=%s dedup_key=%s status=%s current_step=%s",
                        job.job_id,
                        dedup_key,
                        job.status,
                        job.current_step,
                    )
                    log.info(
                        "transcribe_job_duplicate_skipped job_id=%s dedup_key=%s status=%s current_step=%s",
                        job.job_id,
                        dedup_key,
                        job.status,
                        job.current_step,
                    )
                    return job, True
                if job.status not in RESTART_ALLOWED_STATUSES:
                    # неожиданное состояние — отдаём как есть
                    return job, True

        now = _utc_iso()
        job_id = str(uuid.uuid4())
        rec = TranscribeJobRecord(
            job_id=job_id,
            dedup_key=dedup_key,
            status="queued",
            payload=payload,
            result=None,
            error=None,
            created_at=now,
            started_at=None,
            updated_at=now,
            completed_at=None,
            progress=0,
            current_step="queued",
            heartbeat_at=now,
            attempts=0,
        )
        pipe = redis.pipeline(transaction=True)
        pipe.set(job_key(job_id), rec.model_dump_json())
        pipe.set(dedup_key_index(dedup_key), job_id)
        await pipe.execute()
        await enqueue_job_id(redis, job_id)
        log.info(
            "transcribe_job_created job_id=%s dedup_key=%s status=%s current_step=%s",
            job_id,
            dedup_key,
            rec.status,
            rec.current_step,
        )
        return rec, False
    finally:
        await release_dedup_lock(redis, dedup_key, lock_token)


async def update_job_fields(
    redis: redis.Redis,
    job_id: str,
    *,
    patch: dict[str, Any],
) -> TranscribeJobRecord | None:
    cur = await get_job(redis, job_id)
    if cur is None:
        return None
    data = cur.model_dump()
    for k, v in patch.items():
        if k in data:
            data[k] = v
    data["updated_at"] = _utc_iso()
    out = TranscribeJobRecord.model_validate(data)
    await save_job(redis, out)
    return out


async def scan_job_ids(redis: redis.Redis) -> list[str]:
    ids: list[str] = []
    async for k in redis.scan_iter(f"{KEY_PREFIX_JOB}*", count=64):
        if isinstance(k, str) and k.startswith(KEY_PREFIX_JOB):
            ids.append(k.removeprefix(KEY_PREFIX_JOB))
    return ids


async def touch_worker_alive(redis: redis.Redis, *, ttl_sec: int) -> None:
    await redis.set(WORKER_ALIVE_KEY, _utc_iso(), ex=max(30, int(ttl_sec)))


async def clear_worker_alive(redis: redis.Redis) -> None:
    await redis.delete(WORKER_ALIVE_KEY)


async def is_worker_alive(redis: redis.Redis) -> bool:
    return bool(await redis.exists(WORKER_ALIVE_KEY))


async def flush_queue_and_reenqueue_pending_jobs(redis: redis.Redis) -> int:
    """Удаляет whisper:queue и снова ставит в неё все задачи со статусом queued или waiting_gpu.

    Убирает «хвосты» после падений: job в Redis waiting_gpu, а воркер уже другой процесс;
    плюс задачи queued, которые потеряли LPUSH в список очереди.
    """
    await redis.delete(QUEUE_KEY)
    n = 0
    now = _utc_iso()
    step = "requeued after worker restart (queue reset)"
    for jid in await scan_job_ids(redis):
        rec = await get_job(redis, jid)
        if rec is None or rec.status in TERMINAL_STATUSES:
            continue
        if rec.status not in _PENDING_QUEUE_STATUSES:
            continue
        patched = rec.model_copy(
            update={
                "status": "queued",
                "current_step": step,
                "updated_at": now,
                "heartbeat_at": now,
            }
        )
        await save_job(redis, patched)
        await enqueue_job_id(redis, jid)
        n += 1
        log.info(
            "transcribe_job_requeued job_id=%s dedup_key=%s status=queued current_step=%s",
            jid,
            rec.dedup_key,
            step,
        )
    log.info("queue_reset_on_worker_start jobs_reenqueued=%s", n)
    return n


async def recover_jobs_on_worker_startup(redis: redis.Redis) -> dict[str, int]:
    """После рестарта worker: опционально сброс очереди; waiting_gpu → очередь; незавершённые processing → failed."""
    settings = get_settings()
    flushed = 0
    if settings.worker_queue_flush_on_start:
        flushed = await flush_queue_and_reenqueue_pending_jobs(redis)
    requeued = 0
    orphaned = 0
    for jid in await scan_job_ids(redis):
        rec = await get_job(redis, jid)
        if rec is None or rec.status in TERMINAL_STATUSES:
            continue
        if rec.status in _REQUEUE_AFTER_RESTART_STATUSES:
            now = _utc_iso()
            patched = rec.model_copy(
                update={
                    "status": "queued",
                    "current_step": "requeued after worker restart",
                    "updated_at": now,
                    "heartbeat_at": now,
                }
            )
            await save_job(redis, patched)
            await enqueue_job_id(redis, jid)
            requeued += 1
            log.info(
                "transcribe_job_requeued job_id=%s dedup_key=%s status=queued current_step=requeued after worker restart",
                jid,
                rec.dedup_key,
            )
            continue
        if rec.status in _PROCESSING_ORPHAN_STATUSES or rec.started_at is not None:
            now = _utc_iso()
            patched = rec.model_copy(
                update={
                    "status": "failed",
                    "error": WORKER_ORPHAN_ERROR,
                    "completed_at": now,
                    "updated_at": now,
                    "current_step": "worker restart",
                    "result": None,
                }
            )
            await save_job(redis, patched)
            orphaned += 1
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=worker restart error=worker_orphaned",
                jid,
                rec.dedup_key,
            )
    return {"requeued": requeued, "orphaned": orphaned, "flushed": flushed}
