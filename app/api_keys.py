"""Именованные API-ключи для внешнего доступа (Bearer) + статистика нагрузки."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

from app.job_models import TERMINAL_STATUSES
from app.job_store import get_job, list_active_job_ids
from app.runtime_state import ACTIVE_RUN_STATUSES, QUEUED_STATUSES, WAITING_STATUSES

log = logging.getLogger(__name__)

KEYS_HASH = "whisper:api_keys"
TOKEN_INDEX_PREFIX = "whisper:api_key_token:"
STATS_PREFIX = "whisper:api_key_stats:"
ENV_KEY_ID = "env-default"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_index_key(token: str) -> str:
    return f"{TOKEN_INDEX_PREFIX}{hash_token(token)}"


def _stats_key(key_id: str) -> str:
    return f"{STATS_PREFIX}{key_id}"


def _parse_key(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _public_key(meta: dict[str, Any], *, include_token: bool = False) -> dict[str, Any]:
    out = {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "source": meta.get("source", "created"),
        "created_at": meta.get("created_at"),
        "revoked": bool(meta.get("revoked")),
        "token_preview": meta.get("token_preview"),
    }
    if include_token and meta.get("token"):
        out["token"] = meta["token"]
    return out


async def ensure_env_api_token(redis_c: redis.Redis, token: str | None, *, name: str = "default") -> None:
    """Синхронизировать WHISPER_API_TOKEN как системный ключ (не удаляется из UI)."""
    if not token:
        # Убрать индекс env-ключа, если токен очистили в env.
        existing = await redis_c.hget(KEYS_HASH, ENV_KEY_ID)
        meta = _parse_key(existing)
        if meta and meta.get("token"):
            await redis_c.delete(_token_index_key(str(meta["token"])))
        await redis_c.hdel(KEYS_HASH, ENV_KEY_ID)
        await redis_c.delete(_stats_key(ENV_KEY_ID))
        return

    existing = await redis_c.hget(KEYS_HASH, ENV_KEY_ID)
    old = _parse_key(existing)
    if old and old.get("token") and old["token"] != token:
        await redis_c.delete(_token_index_key(str(old["token"])))

    meta = {
        "id": ENV_KEY_ID,
        "name": name.strip() or "default",
        "token": token,
        "token_preview": _preview(token),
        "source": "env",
        "created_at": (old or {}).get("created_at") or _utc_iso(),
        "revoked": False,
    }
    await redis_c.hset(KEYS_HASH, ENV_KEY_ID, json.dumps(meta, ensure_ascii=False))
    await redis_c.set(_token_index_key(token), ENV_KEY_ID)
    if not await redis_c.exists(_stats_key(ENV_KEY_ID)):
        await _init_stats(redis_c, ENV_KEY_ID)


def _preview(token: str) -> str:
    if len(token) <= 10:
        return "***"
    return f"{token[:4]}…{token[-4:]}"


async def _init_stats(redis_c: redis.Redis, key_id: str) -> None:
    await redis_c.hset(
        _stats_key(key_id),
        mapping={
            "jobs_total": "0",
            "jobs_completed": "0",
            "jobs_failed": "0",
            "last_used_at": "",
        },
    )


async def create_api_key(redis_c: redis.Redis, name: str) -> dict[str, Any]:
    clean = (name or "").strip()
    if not clean:
        raise ValueError("name_required")
    if len(clean) > 64:
        raise ValueError("name_too_long")
    key_id = str(uuid.uuid4())
    token = secrets.token_urlsafe(32)
    meta = {
        "id": key_id,
        "name": clean,
        "token": token,
        "token_preview": _preview(token),
        "source": "created",
        "created_at": _utc_iso(),
        "revoked": False,
    }
    await redis_c.hset(KEYS_HASH, key_id, json.dumps(meta, ensure_ascii=False))
    await redis_c.set(_token_index_key(token), key_id)
    await _init_stats(redis_c, key_id)
    return _public_key(meta, include_token=True)


async def revoke_api_key(redis_c: redis.Redis, key_id: str) -> bool:
    if key_id == ENV_KEY_ID:
        raise ValueError("cannot_revoke_env_token")
    raw = await redis_c.hget(KEYS_HASH, key_id)
    meta = _parse_key(raw)
    if meta is None:
        return False
    if meta.get("token"):
        await redis_c.delete(_token_index_key(str(meta["token"])))
    meta["revoked"] = True
    meta["token"] = ""
    await redis_c.hset(KEYS_HASH, key_id, json.dumps(meta, ensure_ascii=False))
    return True


async def delete_api_key(redis_c: redis.Redis, key_id: str) -> bool:
    if key_id == ENV_KEY_ID:
        raise ValueError("cannot_delete_env_token")
    raw = await redis_c.hget(KEYS_HASH, key_id)
    meta = _parse_key(raw)
    if meta is None:
        return False
    if meta.get("token"):
        await redis_c.delete(_token_index_key(str(meta["token"])))
    await redis_c.hdel(KEYS_HASH, key_id)
    await redis_c.delete(_stats_key(key_id))
    return True


async def resolve_api_token(redis_c: redis.Redis, token: str | None) -> dict[str, Any] | None:
    """Вернуть meta ключа по Bearer-токену или None."""
    if not token:
        return None
    key_id = await redis_c.get(_token_index_key(token))
    if not key_id:
        # Fallback: прямой env-токен до синка Redis.
        return None
    raw = await redis_c.hget(KEYS_HASH, key_id)
    meta = _parse_key(raw)
    if meta is None or meta.get("revoked"):
        return None
    if meta.get("token") != token:
        return None
    return meta


async def list_api_keys(redis_c: redis.Redis) -> list[dict[str, Any]]:
    raw_map = await redis_c.hgetall(KEYS_HASH)
    items: list[dict[str, Any]] = []
    for _kid, raw in (raw_map or {}).items():
        meta = _parse_key(raw)
        if meta is None or meta.get("revoked"):
            continue
        pub = _public_key(meta)
        stats = await get_key_stats(redis_c, str(meta.get("id")))
        pub["stats"] = stats
        items.append(pub)
    items.sort(key=lambda x: (0 if x.get("source") == "env" else 1, str(x.get("name") or "")))
    return items


async def get_key_stats(redis_c: redis.Redis, key_id: str) -> dict[str, Any]:
    raw = await redis_c.hgetall(_stats_key(key_id))
    def _i(name: str) -> int:
        try:
            return int((raw or {}).get(name) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "jobs_total": _i("jobs_total"),
        "jobs_completed": _i("jobs_completed"),
        "jobs_failed": _i("jobs_failed"),
        "last_used_at": (raw or {}).get("last_used_at") or None,
        "active": 0,
        "queued": 0,
        "waiting_slot": 0,
    }


async def bump_key_enqueue(redis_c: redis.Redis, key_id: str | None) -> None:
    if not key_id:
        return
    pipe = redis_c.pipeline()
    pipe.hincrby(_stats_key(key_id), "jobs_total", 1)
    pipe.hset(_stats_key(key_id), "last_used_at", _utc_iso())
    await pipe.execute()


async def bump_key_terminal(redis_c: redis.Redis, key_id: str | None, *, status: str) -> None:
    if not key_id:
        return
    field = "jobs_completed" if status == "completed" else "jobs_failed"
    if status not in ("completed", "failed", "stale_failed", "cancelled"):
        return
    if status != "completed":
        field = "jobs_failed"
    await redis_c.hincrby(_stats_key(key_id), field, 1)


async def load_by_key(redis_c: redis.Redis) -> list[dict[str, Any]]:
    """Агрегация нагрузки по ключам: counters + live active/queued из job records."""
    keys = await list_api_keys(redis_c)
    live: dict[str, dict[str, int]] = {
        str(k["id"]): {"active": 0, "queued": 0, "waiting_slot": 0} for k in keys
    }
    for jid in await list_active_job_ids(redis_c):
        rec = await get_job(redis_c, jid)
        if rec is None or rec.status in TERMINAL_STATUSES:
            continue
        kid = getattr(rec, "api_key_id", None)
        if not kid or kid not in live:
            continue
        if rec.status in ACTIVE_RUN_STATUSES:
            live[kid]["active"] += 1
        elif rec.status in WAITING_STATUSES:
            live[kid]["waiting_slot"] += 1
        elif rec.status in QUEUED_STATUSES:
            live[kid]["queued"] += 1

    out: list[dict[str, Any]] = []
    for k in keys:
        kid = str(k["id"])
        stats = dict(k.get("stats") or {})
        stats.update(live.get(kid) or {})
        row = {**k, "stats": stats}
        out.append(row)
    return out
