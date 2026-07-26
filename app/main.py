"""HTTP API: постановка задач транскрипции в Redis и выдача статуса."""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import redis.asyncio as redis
from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.api_keys import (
    bump_key_enqueue,
    create_api_key,
    delete_api_key,
    ensure_env_api_token,
    list_api_keys,
    load_by_key,
    resolve_api_token,
    revoke_api_key,
)
from app.container_metrics import collect_container_metrics, query_nvidia_smi_realtime
from app.dedup import compute_dedup_key
from app.job_models import TERMINAL_STATUSES, TranscribeJobEnqueueResponse, TranscribeJobRecord, TranscribeJobStatusResponse
from app.job_store import (
    claim_or_get_existing_job,
    clear_openai_quota_exceeded,
    connect_redis,
    ensure_queued_job_in_redis_queue,
    get_job,
    get_job_id_for_dedup,
    get_openai_quota_exceeded,
    is_job_id_in_queue,
    is_worker_alive,
    rebuild_active_jobs_index,
)
from app.job_watchdog import maybe_mark_job_stale
from app.runtime_state import (
    OPENAI_MAX_CONCURRENT_HARD_CAP,
    count_jobs_by_bucket,
    ensure_desired_backend,
    ensure_desired_concurrency,
    get_desired_backend,
    get_desired_local_max,
    get_desired_openai_max,
    get_worker_stats,
    set_desired_backend,
    set_desired_local_max,
    set_desired_openai_max,
)
from app.settings import get_settings
from app.transcribe_body import TranscribeBody
from app.ui_auth import SESSION_COOKIE, create_session, destroy_session, get_session, touch_session
from app.vram_limits import estimate_max_local_slots, estimate_vram_per_slot_mb

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DASHBOARD_HTML = _STATIC_DIR / "dashboard.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.redis = None
    if settings.redis_url:
        try:
            app.state.redis = await connect_redis(settings.redis_url)
            await ensure_desired_backend(app.state.redis, settings._env_backend)
            await ensure_desired_concurrency(
                app.state.redis,
                local_fallback=settings._env_local_max_concurrent_jobs,
                openai_fallback=settings._env_openai_max_concurrent_jobs,
            )
            await ensure_env_api_token(
                app.state.redis,
                settings.api_token,
                name=settings.api_token_name,
            )
            try:
                await rebuild_active_jobs_index(app.state.redis)
            except Exception:
                log.exception("active_jobs_index_rebuild_failed")
        except Exception:
            log.exception("redis_connect_failed url=%s", settings.redis_url)
            app.state.redis = None
    collect_container_metrics(include_gpu=False)
    yield
    r: redis.Redis | None = getattr(app.state, "redis", None)
    if r is not None:
        try:
            await r.aclose()
        except Exception:
            log.exception("redis_close_failed")


_APP_DESCRIPTION = """## Назначение
Сервис ставит задачи распознавания в **очередь Redis** и обрабатывает их процессом **whisper-worker** (GPU).
Клиент получает `job_id` и опрашивает `GET /jobs/{job_id}` до `status=completed`.

### Документация API (OpenAPI)
| UI | Путь |
|----|------|
| **Swagger** | [`/docs`](/docs) |
| **ReDoc** | [`/redoc`](/redoc) |
| **Схема JSON** | [`/openapi.json`](/openapi.json) |

### Авторизация
- **Внешний API** (`/transcribe`, `/jobs/*`): `Authorization: Bearer <токен>` — `WHISPER_API_TOKEN` или ключ, созданный в UI.
- **Dashboard** (`/ui`, `/admin/*`, `/metrics/live`): логин/пароль `WHISPER_UI_USER` / `WHISPER_UI_PASSWORD` (cookie-сессия).

### Режимы запроса
1. **Один файл** — поле `url` (моно или стерео; стерео сводится в mono для распознавания). Опционально `diarize`.
2. **Два mono-файла** — `url_rx`, `url_tx`, **`call_direction`**. Опционально **`url_mix`** и объект **`sync`**: `mode` — `auto` (корреляция с mix), `manual` (смещения `offset_rx_sec` / `offset_tx_sec`), `off`; **`max_offset_sec`** — порог доверия к |offset|; **`fallback`** — `none` | `use_mix` | `use_rx_tx` при сбое auto/manual. Если передан только `url_mix` без `sync`, подразумевается `sync.mode=auto`.
"""

app = FastAPI(
    title="Whisper STT sidecar",
    description=_APP_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "transcription", "description": "Постановка задачи распознавания (асинхронно)."},
        {"name": "jobs", "description": "Статус и результат фоновой транскрипции."},
        {"name": "service", "description": "Проверка готовности и ссылки на документацию."},
        {"name": "ops", "description": "Dashboard, метрики и переключение backend."},
    ],
)
_bearer = HTTPBearer(auto_error=False)

_TRANSCRIBE_BODY_EXAMPLES: dict[str, dict[str, Any]] = {
    "mono_diarize": {
        "summary": "Одна запись + диаризация",
        "description": "Моно/стерео по одному URL",
        "value": {
            "url": "https://example.com/records/call.wav",
            "diarize": True,
        },
    },
    "dual_incoming": {
        "summary": "Два канала (входящий звонок)",
        "description": (
            "Два отдельных mono по HTTPS: `url_rx` и `url_tx`, без поля `url`. "
            "Входящий (`call_direction`: incoming): RX — клиент, TX — оператор "
            "(как файлы `external-*-rx.wav` / `external-*-tx.wav`). Диаризация не используется."
        ),
        "value": {
            "url_rx": "https://example.com/records/external-7164-0558835009-20260502-194233-1777740153.38508-rx.wav",
            "url_tx": "https://example.com/records/external-7164-0558835009-20260502-194233-1777740153.38508-tx.wav",
            "call_direction": "incoming",
            "diarize": False,
        },
    },
    "dual_outgoing": {
        "summary": "Два канала (исходящий звонок)",
        "description": (
            "Два отдельных mono по HTTPS: `url_rx` и `url_tx`, без поля `url`. "
            "Исходящий (`call_direction`: outgoing): RX — оператор, TX — клиент "
            "(как файлы `out-*-rx.wav` / `out-*-tx.wav`). Диаризация не используется."
        ),
        "value": {
            "url_rx": "https://example.com/records/out-0558835009-7164-20260502-194859-1777740539.38520-rx.wav",
            "url_tx": "https://example.com/records/out-0558835009-7164-20260502-194859-1777740539.38520-tx.wav",
            "call_direction": "outgoing",
            "diarize": False,
        },
    },
    "dual_incoming_mix_sync": {
        "summary": "Два канала + общая mono для синхронизации",
        "description": (
            "`url_mix` + `sync.mode=auto`: выравнивание RX/TX по шкале mix (корреляция огибающих). "
            "`max_offset_sec` отсекает неверные сдвиги; `fallback` — поведение при полном сбое доверия к auto."
        ),
        "value": {
            "url_rx": "https://example.com/records/call-rx.wav",
            "url_tx": "https://example.com/records/call-tx.wav",
            "url_mix": "https://example.com/records/call.wav",
            "call_direction": "incoming",
            "diarize": False,
            "sync": {"mode": "auto", "max_offset_sec": 2.0, "fallback": "none"},
        },
    },
}


async def _require_redis(request: Request) -> redis.Redis:
    r: redis.Redis | None = getattr(request.app.state, "redis", None)
    if r is None:
        raise HTTPException(status_code=503, detail="redis_unavailable")
    return r


async def require_api_key(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    settings = get_settings()
    token = creds.credentials if creds and creds.scheme.lower() == "bearer" else None
    redis_c: redis.Redis | None = getattr(request.app.state, "redis", None)

    if token is None:
        # Открытый режим только если нет env-токена и нет ключей в Redis.
        if settings.api_token:
            raise HTTPException(status_code=401, detail="invalid_or_missing_token")
        if redis_c is not None:
            keys = await list_api_keys(redis_c)
            if keys:
                raise HTTPException(status_code=401, detail="invalid_or_missing_token")
        return {"id": None, "name": "anonymous", "source": "open"}

    if settings.api_token and len(token) == len(settings.api_token) and secrets.compare_digest(token, settings.api_token):
        return {"id": "env-default", "name": settings.api_token_name, "source": "env"}

    if redis_c is None:
        raise HTTPException(status_code=401, detail="invalid_or_missing_token")
    meta = await resolve_api_token(redis_c, token)
    if meta is None:
        raise HTTPException(status_code=401, detail="invalid_or_missing_token")
    return {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "source": meta.get("source"),
    }


async def require_ui_session(
    request: Request,
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.ui_auth_configured:
        raise HTTPException(
            status_code=503,
            detail="ui_auth_not_configured: set WHISPER_UI_USER and WHISPER_UI_PASSWORD",
        )
    token = request.cookies.get(SESSION_COOKIE)
    sess = await get_session(redis_c, token)
    if sess is None:
        raise HTTPException(status_code=401, detail="ui_session_required")
    await touch_session(redis_c, token, ttl_sec=settings.ui_session_ttl_sec)
    return sess


def _compute_retry_hint(
    rec: TranscribeJobRecord,
    worker_available: bool,
    *,
    in_redis_queue: bool | None,
) -> tuple[bool, str | None]:
    """retry_recommended + client_hint (RU) для интеграций и n8n."""
    err = rec.error or ""
    err_l = err.lower()
    if rec.status not in TERMINAL_STATUSES:
        if not worker_available:
            return (
                True,
                "Whisper-worker сейчас не активен (в Redis нет ключа whisper:worker:alive). "
                "Запустите контейнер whisper-worker. Пока worker выключен, задача не обрабатывается. "
                "После запуска worker задачи в состоянии waiting_gpu снова попадут в очередь; "
                "незавершённые обработки будут помечены failed — тогда повторите POST /transcribe с тем же JSON.",
            )
        if rec.status == "queued" and in_redis_queue is False:
            return (
                True,
                "Статус queued, но job_id отсутствует в списке whisper:queue (рассинхрон с очередью). "
                "API попыталось выполнить LPUSH; если поле in_redis_queue всё ещё false — повторите POST /transcribe с тем же JSON "
                "или проверьте логи whisper-worker и одинаковый REDIS_URL у API и worker.",
            )
        if rec.status == "waiting_gpu" and rec.current_step:
            cs = rec.current_step.lower()
            if "another job" in cs or "run slot busy" in cs:
                return (
                    False,
                    "Другая задача уже заняла слот выполнения воркера (скачивание или распознавание на GPU). "
                    "Дождитесь её завершения или увеличьте WHISPER_MAX_CONCURRENT_JOBS при достаточной VRAM.",
                )
        return False, None
    if rec.status in ("failed", "stale_failed"):
        if "worker_orphaned" in err_l:
            return (
                True,
                "Задача прервана из-за перезапуска whisper-worker во время распознавания. "
                "Повторите POST /transcribe с тем же телом запроса (будет создана новая задача).",
            )
        if "job_stale" in err_l:
            return (
                True,
                "Превышен таймаут heartbeat (WHISPER_JOB_STALE_SEC). Для длинных звонков увеличьте переменную "
                "или повторите POST /transcribe.",
            )
    return False, None


async def _job_to_status(
    redis_c: redis.Redis,
    rec: TranscribeJobRecord,
    *,
    in_redis_queue: bool | None = None,
) -> TranscribeJobStatusResponse:
    worker_ok = await is_worker_alive(redis_c)
    retry, hint = _compute_retry_hint(rec, worker_ok, in_redis_queue=in_redis_queue)
    return TranscribeJobStatusResponse(
        job_id=rec.job_id,
        dedup_key=rec.dedup_key,
        status=rec.status,
        progress=int(rec.progress),
        current_step=rec.current_step,
        created_at=rec.created_at,
        started_at=rec.started_at,
        updated_at=rec.updated_at,
        heartbeat_at=rec.heartbeat_at,
        completed_at=rec.completed_at,
        error=rec.error,
        result=rec.result,
        worker_available=worker_ok,
        retry_recommended=retry,
        client_hint=hint,
        in_redis_queue=in_redis_queue,
    )


class BackendSwitchBody(BaseModel):
    backend: Literal["local", "openai"] = Field(
        ...,
        description="local = faster-whisper GPU; openai = OpenAI Audio Transcriptions API",
    )


class ConcurrentBody(BaseModel):
    local_max: int | None = Field(None, ge=1, description="Параллелизм для GPU (local)")
    openai_max: int | None = Field(None, ge=1, description="Параллелизм для OpenAI API")


class LoginBody(BaseModel):
    username: str
    password: str


class ApiKeyCreateBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


def _gpu_cap_from_metrics(settings: Any, worker_stats: dict[str, Any] | None) -> tuple[int, int, float | None]:
    """(gpu_cap_slots, vram_per_slot_mb, gpu_total_mb)."""
    per = estimate_vram_per_slot_mb(
        settings.model_path,
        override_mb=settings.vram_per_slot_mb or None,
    )
    total = None
    if isinstance(worker_stats, dict):
        if worker_stats.get("gpu_vram_cap_slots") is not None:
            try:
                cap = int(worker_stats["gpu_vram_cap_slots"])
                per2 = int(worker_stats.get("vram_per_slot_mb") or per)
                ctn = (worker_stats.get("container") or {}) if isinstance(worker_stats.get("container"), dict) else {}
                total = ctn.get("gpu_mem_total_mb")
                return max(1, cap), per2, float(total) if total is not None else None
            except (TypeError, ValueError):
                pass
        ctn = worker_stats.get("container") if isinstance(worker_stats.get("container"), dict) else {}
        total = ctn.get("gpu_mem_total_mb") if ctn else None
    if total is None:
        m = collect_container_metrics(True, False)
        total = m.gpu_mem_total_mb
    cap = estimate_max_local_slots(
        total,
        model_path=settings.model_path,
        vram_per_slot_mb=settings.vram_per_slot_mb or None,
        reserve_mb=settings.vram_reserve_mb,
    )
    return cap, per, float(total) if total is not None else None


@app.get(
    "/",
    tags=["service"],
    summary="Корень API",
    description="Короткая сводка и ссылки на интерактивную документацию OpenAPI.",
)
async def api_root() -> dict[str, str]:
    return {
        "service": "Whisper STT sidecar",
        "ui": "/ui",
        "docs_swagger": "/docs",
        "docs_redoc": "/redoc",
        "openapi_schema": "/openapi.json",
        "health": "/health",
        "metrics_live": "/metrics/live",
        "admin_backend": "/admin/backend",
        "admin_concurrent": "/admin/concurrent",
        "admin_api_keys": "/admin/api-keys",
        "auth_login": "/auth/login",
        "transcribe": "/transcribe",
        "job": "/jobs/{job_id}",
        "job_by_dedup": "/jobs/by-dedup/{dedup_key}",
    }


@app.get(
    "/ui",
    tags=["ops"],
    summary="Ops dashboard",
    include_in_schema=False,
)
async def ops_ui() -> FileResponse:
    if not _DASHBOARD_HTML.is_file():
        raise HTTPException(status_code=404, detail="dashboard_not_found")
    return FileResponse(_DASHBOARD_HTML, media_type="text/html; charset=utf-8")


@app.get("/auth/status", tags=["ops"], summary="Статус UI-сессии")
async def auth_status(request: Request, redis_c: redis.Redis = Depends(_require_redis)) -> dict[str, Any]:
    settings = get_settings()
    token = request.cookies.get(SESSION_COOKIE)
    sess = await get_session(redis_c, token) if token else None
    return {
        "ui_auth_configured": settings.ui_auth_configured,
        "authenticated": sess is not None,
        "username": (sess or {}).get("username"),
    }


@app.post("/auth/login", tags=["ops"], summary="Вход в dashboard")
async def auth_login(
    body: LoginBody,
    response: Response,
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.ui_auth_configured:
        raise HTTPException(status_code=503, detail="ui_auth_not_configured")
    user_ok = body.username.strip() == (settings.ui_user or "")
    pass_ok = body.password == (settings.ui_password or "")
    # compare_digest только при равной длине; иначе всегда False без исключения.
    if settings.ui_user and len(body.username.strip()) == len(settings.ui_user):
        user_ok = secrets.compare_digest(body.username.strip(), settings.ui_user)
    if settings.ui_password and len(body.password) == len(settings.ui_password):
        pass_ok = secrets.compare_digest(body.password, settings.ui_password)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = await create_session(redis_c, settings.ui_user or body.username, ttl_sec=settings.ui_session_ttl_sec)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.ui_session_ttl_sec,
        path="/",
    )
    return {"ok": True, "username": settings.ui_user}


@app.post("/auth/logout", tags=["ops"], summary="Выход из dashboard")
async def auth_logout(
    request: Request,
    response: Response,
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, bool]:
    token = request.cookies.get(SESSION_COOKIE)
    await destroy_session(redis_c, token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get(
    "/metrics/live",
    tags=["ops"],
    summary="Живые метрики worker и API",
)
async def metrics_live(
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    settings = get_settings()
    worker_ok = await is_worker_alive(redis_c)
    desired = await get_desired_backend(redis_c)
    worker_stats = await get_worker_stats(redis_c)
    jobs = await count_jobs_by_bucket(redis_c)
    effective = None
    if isinstance(worker_stats, dict):
        effective = worker_stats.get("backend")
    if not effective:
        effective = desired or settings._env_backend
    pending = bool(desired and effective and desired != effective)
    if isinstance(worker_stats, dict) and worker_stats.get("backend_pending"):
        pending = True
    api_m = collect_container_metrics(include_gpu=False)
    smi = query_nvidia_smi_realtime(use_cache=True)
    desired_loc = await get_desired_local_max(redis_c)
    desired_oa = await get_desired_openai_max(redis_c)
    gpu_cap, per_slot, gpu_total = _gpu_cap_from_metrics(settings, worker_stats)
    if smi and smi.memory_total_mb:
        gpu_total = smi.memory_total_mb
    max_conc = (
        (desired_oa or settings.openai_max_concurrent_jobs)
        if effective == "openai"
        else (desired_loc or settings.local_max_concurrent_jobs)
    )
    if isinstance(worker_stats, dict) and worker_stats.get("max_concurrent_jobs") is not None:
        try:
            max_conc = int(worker_stats["max_concurrent_jobs"])
        except (TypeError, ValueError):
            pass
    keys_load = await load_by_key(redis_c)
    return {
        "worker_available": worker_ok,
        "desired_backend": desired or settings._env_backend,
        "effective_backend": effective,
        "backend_pending": pending,
        "env_backend": settings._env_backend,
        "max_concurrent_jobs": max_conc,
        "local_max_concurrent_jobs": desired_loc or settings.local_max_concurrent_jobs,
        "openai_max_concurrent_jobs": desired_oa or settings.openai_max_concurrent_jobs,
        "desired_local_max": desired_loc,
        "desired_openai_max": desired_oa,
        "gpu_vram_cap_slots": gpu_cap,
        "vram_per_slot_mb": per_slot,
        "gpu_mem_total_mb": gpu_total,
        "openai_configured": bool(settings.openai_api_key),
        "gpu": smi.as_dict() if smi else None,
        "worker": worker_stats,
        "jobs": jobs,
        "api_keys": keys_load,
        "api_container": api_m.as_dict(),
    }


@app.post(
    "/admin/backend",
    tags=["ops"],
    summary="Переключить бэкенд local ↔ openai",
)
async def admin_set_backend(
    body: BackendSwitchBody,
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    settings = get_settings()
    if body.backend == "openai" and not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="openai_api_key_not_configured")
    desired = await set_desired_backend(redis_c, body.backend)
    if desired == "local":
        # Квота OpenAI не блокирует local; снимаем stale-флаг после переключения.
        try:
            await clear_openai_quota_exceeded(redis_c)
        except Exception:
            pass
    worker_stats = await get_worker_stats(redis_c)
    effective = None
    if isinstance(worker_stats, dict):
        effective = worker_stats.get("backend")
    pending = effective is None or effective != desired
    return {
        "desired_backend": desired,
        "effective_backend": effective,
        "pending": pending,
        "worker_available": await is_worker_alive(redis_c),
        "message": "backend_switch_requested" if pending else "backend_already_active",
    }


@app.post(
    "/admin/concurrent",
    tags=["ops"],
    summary="Задать число одновременных потоков",
    description=(
        "Для local (GPU) значение ограничено оценкой VRAM (`gpu_vram_cap_slots`). "
        "Для openai — до 200. Worker применит при свободных слотах."
    ),
)
async def admin_set_concurrent(
    body: ConcurrentBody,
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    if body.local_max is None and body.openai_max is None:
        raise HTTPException(status_code=400, detail="local_max_or_openai_max_required")
    ws = await get_worker_stats(redis_c)
    settings = get_settings()
    gpu_cap, per_slot, gpu_total = _gpu_cap_from_metrics(settings, ws)
    out: dict[str, Any] = {
        "gpu_vram_cap_slots": gpu_cap,
        "vram_per_slot_mb": per_slot,
        "gpu_mem_total_mb": gpu_total,
    }
    if body.local_max is not None:
        if body.local_max > gpu_cap:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "local_max_exceeds_vram",
                    "requested": body.local_max,
                    "gpu_vram_cap_slots": gpu_cap,
                    "vram_per_slot_mb": per_slot,
                    "gpu_mem_total_mb": gpu_total,
                    "hint": "Уменьшите число потоков: на GPU не хватает VRAM для стольких параллельных Whisper.",
                },
            )
        out["desired_local_max"] = await set_desired_local_max(redis_c, body.local_max)
    else:
        out["desired_local_max"] = await get_desired_local_max(redis_c)
    if body.openai_max is not None:
        if body.openai_max > OPENAI_MAX_CONCURRENT_HARD_CAP:
            raise HTTPException(status_code=400, detail="openai_max_too_high")
        out["desired_openai_max"] = await set_desired_openai_max(redis_c, body.openai_max)
    else:
        out["desired_openai_max"] = await get_desired_openai_max(redis_c)
    out["pending"] = True
    out["message"] = "concurrent_update_requested"
    return out


@app.get("/admin/api-keys", tags=["ops"], summary="Список API-ключей и нагрузка")
async def admin_list_api_keys(
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    return {"keys": await load_by_key(redis_c)}


@app.post("/admin/api-keys", tags=["ops"], summary="Создать именованный API-ключ")
async def admin_create_api_key(
    body: ApiKeyCreateBody,
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    try:
        created = await create_api_key(redis_c, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return created


@app.delete("/admin/api-keys/{key_id}", tags=["ops"], summary="Удалить API-ключ")
async def admin_delete_api_key(
    key_id: str,
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    try:
        ok = await delete_api_key(redis_c, key_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="key_not_found")
    return {"ok": True, "id": key_id}


@app.post("/admin/api-keys/{key_id}/revoke", tags=["ops"], summary="Отозвать API-ключ")
async def admin_revoke_api_key(
    key_id: str,
    _: dict[str, Any] = Depends(require_ui_session),
    redis_c: redis.Redis = Depends(_require_redis),
) -> dict[str, Any]:
    try:
        ok = await revoke_api_key(redis_c, key_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="key_not_found")
    return {"ok": True, "id": key_id}


@app.get(
    "/health",
    tags=["service"],
    summary="Проверка готовности API и Redis",
    description=(
        "API-контейнер не загружает модель Whisper; распознавание выполняет `whisper-worker`. "
        "При исчерпании квоты OpenAI и активном бэкенде `openai`: HTTP 503, `status=error`, "
        "`openai_quota_exceeded=true`. На `local` флаг квоты не влияет на статус."
    ),
    response_model=None,
    responses={
        200: {"description": "Сервис готов (или degraded: Redis/worker)."},
        503: {
            "description": "Исчерпана квота OpenAI при effective_backend=openai — "
            "клиенту следует остановить отправку задач или переключиться на local."
        },
    },
)
async def health(request: Request) -> JSONResponse:
    settings = get_settings()
    out: dict[str, Any] = {
        "status": "ok",
        "process_role": settings.process_role,
        "whisper_backend": settings._env_backend,
        "ui": "/ui",
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "local_max_concurrent_jobs": settings.local_max_concurrent_jobs,
        "openai_max_concurrent_jobs": settings.openai_max_concurrent_jobs,
        "transcription": "worker",
        "openai_quota_exceeded": False,
    }
    r: redis.Redis | None = getattr(request.app.state, "redis", None)
    if not settings.redis_url:
        out["redis"] = "unset"
        out["status"] = "degraded"
        out["worker_available"] = False
        return JSONResponse(content=out)
    if r is None:
        out["redis"] = "not_connected"
        out["status"] = "degraded"
        out["worker_available"] = False
        return JSONResponse(content=out)
    try:
        await r.ping()
        out["redis"] = "ok"
        out["worker_available"] = bool(await is_worker_alive(r))
        desired = await get_desired_backend(r)
        ws = await get_worker_stats(r)
        effective = (ws or {}).get("backend") if isinstance(ws, dict) else None
        out["desired_backend"] = desired or settings._env_backend
        out["effective_backend"] = effective or out["desired_backend"]
        out["whisper_backend"] = out["effective_backend"]
        if effective == "openai":
            out["max_concurrent_jobs"] = settings.openai_max_concurrent_jobs
        elif effective == "local":
            out["max_concurrent_jobs"] = settings.local_max_concurrent_jobs
        if isinstance(ws, dict) and ws.get("active_slots") is not None:
            out["active_slots"] = ws.get("active_slots")
    except Exception as e:
        out["redis"] = f"error:{type(e).__name__}"
        out["status"] = "degraded"
        out["worker_available"] = False
        return JSONResponse(content=out)
    if not out.get("worker_available", False):
        out["status"] = "degraded"

    # Квота OpenAI критична только пока реально работает openai-бэкенд.
    # Env WHISPER_BACKEND=openai при runtime local (UI-переключение) не должен давать 503.
    if out.get("effective_backend") == "openai":
        quota = await get_openai_quota_exceeded(r)
        if quota is not None:
            out["status"] = "error"
            out["openai_quota_exceeded"] = True
            out["error"] = "openai_quota_exceeded"
            if isinstance(quota.get("detail"), str) and quota["detail"]:
                out["error_detail"] = quota["detail"][:300]
            if isinstance(quota.get("at"), str):
                out["openai_quota_exceeded_at"] = quota["at"]
            return JSONResponse(status_code=503, content=out)

    return JSONResponse(content=out)


@app.post(
    "/transcribe",
    tags=["transcription"],
    summary="Поставить задачу распознавания",
    description=(
        "Создаёт (или возвращает существующую) задачу в Redis. **Не ждёт** завершения распознавания. "
        "Статус и результат — **GET /jobs/{job_id}** или, для интеграций по «файлу записи», **GET /jobs/by-dedup/{dedup_key}** "
        "(тот же JSON, что у `POST`, при `status=completed`)."
    ),
    responses={
        200: {"description": "Существующая задача (в т.ч. уже завершённая) по dedup_key."},
        202: {"description": "Создана новая задача."},
        401: {"description": "Неверный или отсутствующий Bearer-токен (если задан WHISPER_API_TOKEN)."},
        400: {"description": "Некорректное тело запроса."},
        503: {"description": "Redis недоступен или не настроен."},
    },
)
async def transcribe(
    body: Annotated[TranscribeBody, Body(examples=_TRANSCRIBE_BODY_EXAMPLES)],
    api_key: dict[str, Any] = Depends(require_api_key),
    redis_c: redis.Redis = Depends(_require_redis),
) -> JSONResponse:
    dk = compute_dedup_key(body)
    log.info(
        "transcribe_job_received job_id=- dedup_key=%s status=- current_step=validate api_key=%s payload=%s",
        dk,
        api_key.get("name") or api_key.get("id") or "-",
        json.dumps(body.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
    )
    try:
        rec, existing = await claim_or_get_existing_job(
            redis_c,
            payload=body.model_dump(mode="json"),
            api_key_id=api_key.get("id"),
            api_key_name=api_key.get("name"),
        )
    except RuntimeError as e:
        if str(e) == "dedup_lock_busy":
            raise HTTPException(status_code=503, detail="dedup_lock_busy") from e
        raise HTTPException(status_code=500, detail="job_claim_failed") from e

    if not existing:
        try:
            await bump_key_enqueue(redis_c, api_key.get("id"))
        except Exception:
            log.exception("api_key_enqueue_bump_failed")

    payload = TranscribeJobEnqueueResponse(
        job_id=rec.job_id,
        dedup_key=rec.dedup_key,
        status=rec.status,
        existing=existing,
    )
    code = 200 if existing else 202
    return JSONResponse(status_code=code, content=json.loads(payload.model_dump_json()))


async def _get_job_status_after_stale_and_queue_heal(
    redis_c: redis.Redis,
    job_id: str,
) -> TranscribeJobStatusResponse:
    await maybe_mark_job_stale(redis_c, job_id)
    rec = await get_job(redis_c, job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    in_q: bool | None = None
    if rec.status == "queued":
        in_q = await is_job_id_in_queue(redis_c, rec.job_id)
        if not in_q:
            await ensure_queued_job_in_redis_queue(redis_c, rec.job_id)
        rec = await get_job(redis_c, rec.job_id) or rec
        if rec.status == "queued":
            in_q = await is_job_id_in_queue(redis_c, rec.job_id)
        else:
            in_q = None
    return await _job_to_status(redis_c, rec, in_redis_queue=in_q)


@app.get(
    "/jobs/{job_id}",
    response_model=TranscribeJobStatusResponse,
    tags=["jobs"],
    summary="Статус задачи по job_id",
    description="Прямой опрос по UUID из ответа **POST /transcribe**. Если у вас своя очередь по «файлу», смотрите **GET /jobs/by-dedup/{dedup_key}**.",
)
async def get_job_status(
    job_id: str,
    _: dict[str, Any] = Depends(require_api_key),
    redis_c: redis.Redis = Depends(_require_redis),
) -> TranscribeJobStatusResponse:
    return await _get_job_status_after_stale_and_queue_heal(redis_c, job_id)


@app.get(
    "/jobs/by-dedup/{dedup_key}",
    response_model=TranscribeJobStatusResponse,
    tags=["jobs"],
    summary="Статус задачи по dedup_key",
    description=(
        "Резолвит `whisper:dedup:{dedup_key}` → актуальный `job_id` и возвращает тот же ответ, что **GET /jobs/{job_id}**. "
        "Для внешней очереди, завязанной на конкретный звонок/файл, **предпочтительнее** опрос по `dedup_key`: после рестарта "
        "воркера или повторного POST может смениться `job_id`, а `dedup_key` (из тех же URL) остаётся тем же. "
        "В пути передавайте ключ в **URL-encoded** виде (слэши и т.п.)."
    ),
)
async def get_job_status_by_dedup(
    dedup_key: str,
    _: dict[str, Any] = Depends(require_api_key),
    redis_c: redis.Redis = Depends(_require_redis),
) -> TranscribeJobStatusResponse:
    jid = await get_job_id_for_dedup(redis_c, dedup_key)
    if not jid:
        raise HTTPException(status_code=404, detail="dedup_not_found")
    return await _get_job_status_after_stale_and_queue_heal(redis_c, jid)