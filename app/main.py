"""HTTP API: постановка задач транскрипции в Redis и выдача статуса."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import redis.asyncio as redis
from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.dedup import compute_dedup_key
from app.job_models import TERMINAL_STATUSES, TranscribeJobEnqueueResponse, TranscribeJobRecord, TranscribeJobStatusResponse
from app.job_store import (
    claim_or_get_existing_job,
    connect_redis,
    ensure_queued_job_in_redis_queue,
    get_job,
    get_job_id_for_dedup,
    is_job_id_in_queue,
    is_worker_alive,
)
from app.job_watchdog import maybe_mark_job_stale
from app.settings import get_settings
from app.transcribe_body import TranscribeBody

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.redis = None
    if settings.redis_url:
        try:
            app.state.redis = await connect_redis(settings.redis_url)
        except Exception:
            log.exception("redis_connect_failed url=%s", settings.redis_url)
            app.state.redis = None
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
Если задан `WHISPER_API_TOKEN`, передавайте заголовок `Authorization: Bearer <токен>`.

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


def _require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    settings = get_settings()
    if settings.api_token is None:
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != settings.api_token:
        raise HTTPException(status_code=401, detail="invalid_or_missing_token")


async def _require_redis(request: Request) -> redis.Redis:
    r: redis.Redis | None = getattr(request.app.state, "redis", None)
    if r is None:
        raise HTTPException(status_code=503, detail="redis_unavailable")
    return r


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


@app.get(
    "/",
    tags=["service"],
    summary="Корень API",
    description="Короткая сводка и ссылки на интерактивную документацию OpenAPI.",
)
async def api_root() -> dict[str, str]:
    return {
        "service": "Whisper STT sidecar",
        "docs_swagger": "/docs",
        "docs_redoc": "/redoc",
        "openapi_schema": "/openapi.json",
        "health": "/health",
        "transcribe": "/transcribe",
        "job": "/jobs/{job_id}",
        "job_by_dedup": "/jobs/by-dedup/{dedup_key}",
    }


@app.get(
    "/health",
    tags=["service"],
    summary="Проверка готовности API и Redis",
    description="API-контейнер не загружает модель Whisper; распознавание выполняет `whisper-worker`.",
)
async def health(request: Request) -> dict[str, Any]:
    settings = get_settings()
    out: dict[str, Any] = {
        "status": "ok",
        "process_role": settings.process_role,
        "whisper_backend": settings.whisper_backend,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "local_max_concurrent_jobs": settings.local_max_concurrent_jobs,
        "openai_max_concurrent_jobs": settings.openai_max_concurrent_jobs,
        "transcription": "worker",
    }
    r: redis.Redis | None = getattr(request.app.state, "redis", None)
    if not settings.redis_url:
        out["redis"] = "unset"
        out["status"] = "degraded"
        out["worker_available"] = False
        return out
    if r is None:
        out["redis"] = "not_connected"
        out["status"] = "degraded"
        out["worker_available"] = False
        return out
    try:
        await r.ping()
        out["redis"] = "ok"
        out["worker_available"] = bool(await is_worker_alive(r))
    except Exception as e:
        out["redis"] = f"error:{type(e).__name__}"
        out["status"] = "degraded"
        out["worker_available"] = False
        return out
    if not out.get("worker_available", False):
        out["status"] = "degraded"
    return out


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
    _: None = Depends(_require_auth),
    redis_c: redis.Redis = Depends(_require_redis),
) -> JSONResponse:
    dk = compute_dedup_key(body)
    log.info(
        "transcribe_job_received job_id=- dedup_key=%s status=- current_step=validate payload=%s",
        dk,
        json.dumps(body.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")),
    )
    try:
        rec, existing = await claim_or_get_existing_job(redis_c, payload=body.model_dump(mode="json"))
    except RuntimeError as e:
        if str(e) == "dedup_lock_busy":
            raise HTTPException(status_code=503, detail="dedup_lock_busy") from e
        raise HTTPException(status_code=500, detail="job_claim_failed") from e

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
    _: None = Depends(_require_auth),
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
    _: None = Depends(_require_auth),
    redis_c: redis.Redis = Depends(_require_redis),
) -> TranscribeJobStatusResponse:
    jid = await get_job_id_for_dedup(redis_c, dedup_key)
    if not jid:
        raise HTTPException(status_code=404, detail="dedup_not_found")
    return await _get_job_status_after_stale_and_queue_heal(redis_c, jid)