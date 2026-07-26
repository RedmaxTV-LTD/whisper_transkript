"""Сессии UI (логин/пароль из .env) и проверка cookie."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

SESSION_PREFIX = "whisper:ui:session:"
SESSION_COOKIE = "whisper_ui_session"
DEFAULT_TTL_SEC = 60 * 60 * 24  # 24h


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def create_session(
    redis_c: redis.Redis,
    username: str,
    *,
    ttl_sec: int = DEFAULT_TTL_SEC,
) -> str:
    token = secrets.token_urlsafe(32)
    payload = {
        "username": username,
        "created_at": _utc_iso(),
    }
    import json

    await redis_c.set(f"{SESSION_PREFIX}{token}", json.dumps(payload, ensure_ascii=False), ex=max(300, int(ttl_sec)))
    return token


async def get_session(redis_c: redis.Redis, token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    import json

    raw = await redis_c.get(f"{SESSION_PREFIX}{token}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def destroy_session(redis_c: redis.Redis, token: str | None) -> None:
    if not token:
        return
    await redis_c.delete(f"{SESSION_PREFIX}{token}")


async def touch_session(redis_c: redis.Redis, token: str | None, *, ttl_sec: int = DEFAULT_TTL_SEC) -> None:
    if not token:
        return
    key = f"{SESSION_PREFIX}{token}"
    if await redis_c.exists(key):
        await redis_c.expire(key, max(300, int(ttl_sec)))
