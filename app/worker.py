"""Фоновый worker: BRPOP очереди Redis, транскрипция, heartbeat, watchdog."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
import redis as redis_sync
import redis.asyncio as redis
from pydantic import ValidationError

from app.api_keys import bump_key_terminal
from app.container_metrics import (
    cache_worker_gpu_total,
    collect_container_metrics,
    collect_worker_metrics,
    get_cached_gpu_total_mb,
)
from app.diarize_engine import ensure_diarization_pipeline_loaded
from app.job_models import TERMINAL_STATUSES, TranscribeJobRecord
from app.job_store import (
    QUEUE_KEY,
    WORKER_ALIVE_KEY,
    _utc_iso,
    brpop_job_id,
    clear_openai_quota_exceeded,
    clear_worker_alive,
    connect_redis,
    get_job,
    mark_job_claimed_after_brpop,
    rebuild_active_jobs_index,
    recover_jobs_on_worker_startup,
    save_job,
    set_openai_quota_exceeded,
    touch_worker_alive,
)
from app.job_watchdog import scan_stale_jobs
from app.persistent_logs import install_app_console_logging, install_persistent_logging
from app.runtime_state import (
    DESIRED_BACKEND_KEY,
    DESIRED_LOCAL_MAX_KEY,
    DESIRED_OPENAI_MAX_KEY,
    WORKER_STATS_KEY,
    WORKER_STATS_TTL_SEC,
    ensure_desired_backend,
    ensure_desired_concurrency,
    get_desired_backend,
    get_desired_local_max,
    get_desired_openai_max,
    set_desired_local_max,
)
from app.settings import get_settings, set_runtime_backend, set_runtime_local_max, set_runtime_openai_max
from app.transcribe_body import TranscribeBody
from app.transcribe_engine import ensure_model_loaded
from app.transcribe_pipeline import run_transcription_pipeline
from app.transcribe_schemas import build_transcribe_response
from app.vram_limits import estimate_max_local_slots, estimate_vram_per_slot_mb

log = logging.getLogger(__name__)


class WorkerMetricsPublisher(threading.Thread):
    """Публикует alive+stats через sync Redis вне asyncio.

    Во время инференса event loop часто блокируется (GIL / sync merge), и UI
    залипает на cpu_percent=0.0 / протухшем whisper:worker:stats.
    """

    daemon = True

    def __init__(
        self,
        redis_url: str,
        *,
        snapshot: Callable[[], dict[str, Any]],
        alive_ttl_sec: int,
        interval_sec: float = 2.0,
    ) -> None:
        super().__init__(name="worker-metrics")
        self._redis_url = redis_url
        self._snapshot = snapshot
        self._alive_ttl = max(30, int(alive_ttl_sec))
        self._interval = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._last_cpu: float | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        client = redis_sync.from_url(self._redis_url, decode_responses=True)
        try:
            while not self._stop.is_set():
                try:
                    self._tick(client)
                except Exception:
                    log.exception("worker_metrics_thread_tick_failed")
                self._stop.wait(self._interval)
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _tick(self, r: redis_sync.Redis) -> None:
        now = datetime.now(timezone.utc).isoformat()
        r.set(WORKER_ALIVE_KEY, now, ex=self._alive_ttl)

        snap = self._snapshot()
        s = get_settings()
        try:
            qlen = int(r.llen(QUEUE_KEY))
        except Exception:
            qlen = 0

        desired_raw = r.get(DESIRED_BACKEND_KEY)
        desired_b = str(desired_raw).strip().lower() if desired_raw else None
        if desired_b not in ("local", "openai"):
            desired_b = None

        def _as_int(raw: Any) -> int | None:
            if raw is None or str(raw).strip() == "":
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        desired_loc = _as_int(r.get(DESIRED_LOCAL_MAX_KEY))
        desired_oa = _as_int(r.get(DESIRED_OPENAI_MAX_KEY))

        m = collect_worker_metrics(include_gpu=False)
        container = m.as_dict()
        cpu = container.get("cpu_percent")
        if isinstance(cpu, (int, float)):
            self._last_cpu = float(cpu)
        elif self._last_cpu is not None:
            container["cpu_percent"] = self._last_cpu

        gpu_total = container.get("gpu_mem_total_mb") or get_cached_gpu_total_mb()
        per_slot = estimate_vram_per_slot_mb(
            s.model_path,
            override_mb=s.vram_per_slot_mb or None,
        )
        gpu_cap = estimate_max_local_slots(
            gpu_total,
            model_path=s.model_path,
            vram_per_slot_mb=s.vram_per_slot_mb or None,
            reserve_mb=s.vram_reserve_mb,
        )
        payload = {
            "role": "worker",
            "backend": s.whisper_backend,
            "desired_backend": desired_b or s.whisper_backend,
            "backend_pending": bool(desired_b and desired_b != s.whisper_backend),
            "switching": bool(snap.get("switching")),
            "max_concurrent_jobs": s.max_concurrent_jobs,
            "local_max_concurrent_jobs": s.local_max_concurrent_jobs,
            "openai_max_concurrent_jobs": s.openai_max_concurrent_jobs,
            "desired_local_max": desired_loc,
            "desired_openai_max": desired_oa,
            "gpu_vram_cap_slots": gpu_cap,
            "vram_per_slot_mb": per_slot,
            "active_slots": int(snap.get("active_slots") or 0),
            "inflight_tasks": int(snap.get("inflight_tasks") or 0),
            "queue_len": qlen,
            "container": container,
            "updated_at": now,
        }
        r.set(WORKER_STATS_KEY, json.dumps(payload, ensure_ascii=False), ex=WORKER_STATS_TTL_SEC)


async def _patch_job(
    redis_c: redis.Redis,
    job_id: str,
    *,
    status: str | None = None,
    current_step: str | None = None,
    progress: int | None = None,
    result: dict | None = None,
    error: str | None = None,
    started: bool = False,
    completed: bool = False,
    attempts_delta: int = 0,
) -> None:
    rec = await get_job(redis_c, job_id)
    if rec is None:
        return
    now = _utc_iso()
    data = rec.model_dump()
    if status is not None:
        data["status"] = status
    if current_step is not None:
        data["current_step"] = current_step
    if progress is not None:
        data["progress"] = progress
    if result is not None:
        data["result"] = result
    if error is not None:
        data["error"] = error
    if started:
        data["started_at"] = data.get("started_at") or now
        data["attempts"] = int(data.get("attempts") or 0) + max(0, attempts_delta)
    if completed:
        data["completed_at"] = now
    data["updated_at"] = now
    data["heartbeat_at"] = now
    await save_job(redis_c, TranscribeJobRecord.model_validate(data))
    if completed and status in ("completed", "failed", "stale_failed", "cancelled"):
        try:
            await bump_key_terminal(redis_c, data.get("api_key_id"), status=str(status))
        except Exception:
            log.exception("api_key_stats_bump_failed job_id=%s", job_id)


async def _run_watchdog(redis_c: redis.Redis, stop_event: asyncio.Event | None = None) -> None:
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            n = await scan_stale_jobs(redis_c)
            if n:
                log.info("watchdog_marked_stale count=%s", n)
        except Exception:
            log.exception("watchdog_scan_failed")
        if stop_event is not None:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=60.0)
                return
            except TimeoutError:
                continue
        await asyncio.sleep(60.0)


async def _acquire_job_sem_with_heartbeat(
    redis_c: redis.Redis,
    job_id: str,
    rec: TranscribeJobRecord,
    job_sem: asyncio.Semaphore,
) -> None:
    """Ожидание слота GPU: при WHISPER_MAX_CONCURRENT_JOBS=1 следующая задача висела на sem без heartbeat — stale watchdog."""
    settings = get_settings()
    interval = min(45.0, max(5.0, float(settings.job_heartbeat_sec)))
    if settings.whisper_backend == "openai":
        busy_step = "run slot busy (another job: download or OpenAI)"
    else:
        busy_step = "run slot busy (another job: download or GPU)"
    while True:
        try:
            await asyncio.wait_for(job_sem.acquire(), timeout=interval)
            return
        except TimeoutError:
            await _patch_job(
                redis_c,
                job_id,
                status="waiting_gpu",
                current_step=busy_step,
                progress=rec.progress,
            )
            log.info(
                "gpu_slot_wait job_id=%s dedup_key=%s max_concurrent_jobs=%s",
                job_id,
                rec.dedup_key,
                settings.max_concurrent_jobs,
            )


async def _process_job(
    redis_c: redis.Redis,
    job_id: str,
    executor: ThreadPoolExecutor,
    job_sem: asyncio.Semaphore,
) -> None:
    try:
        await _process_job_inner(redis_c, job_id, executor, job_sem)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("transcribe_job_unhandled job_id=%s", job_id)
        try:
            cur = await get_job(redis_c, job_id)
            if cur is not None and cur.status == "completed":
                return
            pr = 0 if cur is None else min(99, max(0, int(cur.progress)))
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error="worker_internal:unhandled_exception",
                current_step="failed",
                progress=pr,
                completed=True,
            )
        except Exception:
            log.exception("transcribe_job_unhandled_patch_failed job_id=%s", job_id)


async def _process_job_inner(
    redis_c: redis.Redis,
    job_id: str,
    executor: ThreadPoolExecutor,
    job_sem: asyncio.Semaphore,
) -> None:
    log.info("transcribe_job_task_begin job_id=%s", job_id)
    rec = await get_job(redis_c, job_id)
    if rec is None:
        log.warning("transcribe_job_failed job_id=%s dedup_key=%s status=missing current_step=pop", job_id, "-")
        return
    if rec.status in TERMINAL_STATUSES:
        log.info(
            "transcribe_job_duplicate_skipped job_id=%s dedup_key=%s status=%s current_step=%s",
            job_id,
            rec.dedup_key,
            rec.status,
            rec.current_step,
        )
        return

    log.info(
        "transcribe_job_worker_dequeued job_id=%s dedup_key=%s status=%s",
        job_id,
        rec.dedup_key,
        rec.status,
    )

    try:
        body = TranscribeBody.model_validate(rec.payload)
    except ValidationError as e:
        await _patch_job(
            redis_c,
            job_id,
            status="failed",
            error=f"invalid_payload: {e}",
            current_step="validate",
            completed=True,
        )
        log.warning(
            "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=validate error=%s",
            job_id,
            rec.dedup_key,
            e,
        )
        return

    await mark_job_claimed_after_brpop(redis_c, job_id)
    rec = await get_job(redis_c, job_id) or rec
    log.info(
        "transcribe_job_started job_id=%s dedup_key=%s status=waiting_gpu current_step=%s",
        job_id,
        rec.dedup_key,
        rec.current_step or "worker claimed, waiting for run slot",
    )

    await _acquire_job_sem_with_heartbeat(redis_c, job_id, rec, job_sem)
    try:
        settings = get_settings()
        stop = asyncio.Event()
        state: dict[str, str | int] = {"status": "downloading", "step": "starting transcription", "progress": 2}

        async def heartbeat_loop() -> None:
            while not stop.is_set():
                try:
                    st = str(state.get("status") or "processing")
                    step = str(state.get("step") or "")
                    pr = int(state.get("progress") or 0)
                    await _patch_job(
                        redis_c,
                        job_id,
                        status=st,
                        current_step=step,
                        progress=pr,
                    )
                    log.info(
                        "transcribe_job_heartbeat job_id=%s dedup_key=%s status=%s current_step=%s",
                        job_id,
                        rec.dedup_key,
                        st,
                        step,
                    )
                except Exception:
                    log.exception("heartbeat_update_failed job_id=%s", job_id)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=settings.job_heartbeat_sec)
                except TimeoutError:
                    continue

        async def step_cb(status: str, step: str, progress: int) -> None:
            state["status"] = status
            state["step"] = step
            state["progress"] = progress
            await _patch_job(
                redis_c,
                job_id,
                status=status,
                current_step=step,
                progress=progress,
            )
            log.info(
                "transcribe_job_step job_id=%s dedup_key=%s status=%s current_step=%s",
                job_id,
                rec.dedup_key,
                status,
                step,
            )

        hb_task = asyncio.create_task(heartbeat_loop(), name=f"hb-{job_id}")
        await _patch_job(
            redis_c,
            job_id,
            status="downloading",
            current_step="starting transcription",
            started=True,
            attempts_delta=1,
        )
        log.info(
            "transcribe_job_started job_id=%s dedup_key=%s status=downloading current_step=starting transcription",
            job_id,
            rec.dedup_key,
        )
        try:
            t_result, src_ch, lay = await run_transcription_pipeline(
                body,
                executor=executor,
                step_callback=step_cb,
            )
            resp = build_transcribe_response(source_channels=src_ch, layout=lay, result=t_result)
            result_dict = json.loads(resp.model_dump_json())
            await _patch_job(
                redis_c,
                job_id,
                status="completed",
                current_step="completed",
                progress=100,
                result=result_dict,
                completed=True,
            )
            if get_settings().whisper_backend == "openai":
                try:
                    await clear_openai_quota_exceeded(redis_c)
                except Exception:
                    log.exception("openai_quota_clear_failed job_id=%s", job_id)
            log.info(
                "transcribe_job_completed job_id=%s dedup_key=%s status=completed current_step=completed",
                job_id,
                rec.dedup_key,
            )
        except ValueError as e:
            if str(e) == "download_exceeds_max_bytes":
                detail = "download_too_large"
            else:
                detail = f"bad_request: {e}"
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error=detail,
                current_step="failed",
                completed=True,
            )
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=request error=%s",
                job_id,
                rec.dedup_key,
                detail,
            )
        except httpx.HTTPStatusError as e:
            detail = f"download_upstream_http_error:{e.response.status_code}"
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error=detail,
                current_step="download",
                completed=True,
            )
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=download error=%s",
                job_id,
                rec.dedup_key,
                detail,
            )
        except httpx.TimeoutException as e:
            detail = f"download_timeout:{e}"
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error="download_timeout",
                current_step="download",
                completed=True,
            )
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=download error=%s",
                job_id,
                rec.dedup_key,
                detail,
            )
        except httpx.RequestError as e:
            detail = f"download_connect_failed:{type(e).__name__}"
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error="download_connect_failed",
                current_step="download",
                completed=True,
            )
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=download error=%s",
                job_id,
                rec.dedup_key,
                detail,
            )
        except Exception as e:
            log.exception("transcribe_job_failed job_id=%s", job_id)
            err_name = type(e).__name__
            err_msg = str(e)
            # OpenAITranscribeError и RuntimeError с префиксом openai_*
            if "openai_" in err_msg or err_name == "OpenAITranscribeError":
                detail = err_msg if err_msg.startswith("openai_") else f"openai_failed:{err_msg}"
            else:
                detail = f"transcribe_failed:{err_name}:{e}"
            if detail.startswith("openai_quota_exceeded"):
                try:
                    await set_openai_quota_exceeded(redis_c, detail)
                    log.warning("openai_quota_exceeded_flag_set job_id=%s", job_id)
                except Exception:
                    log.exception("openai_quota_flag_set_failed job_id=%s", job_id)
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error=detail,
                current_step="failed",
                completed=True,
            )
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=failed error=%s",
                job_id,
                rec.dedup_key,
                err_name,
            )
        finally:
            stop.set()
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Иначе исключение из heartbeat (например, Redis) затирает успешный completed как worker_internal.
                log.exception("heartbeat_task_join_failed job_id=%s", job_id)
    finally:
        job_sem.release()


async def _prepare_backend(backend: str) -> None:
    """Загрузить модель / залогировать openai после смены бэкенда."""
    settings = get_settings()
    if backend == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when backend=openai")
        log.info(
            "worker_openai_backend model=%s base_url=%s max_concurrent=%s",
            settings.openai_transcribe_model,
            settings.openai_base_url,
            settings.max_concurrent_jobs,
        )
        return
    await ensure_model_loaded()
    if settings.diarization_enabled:
        await ensure_diarization_pipeline_loaded()


async def worker_main_async() -> None:
    settings = get_settings()
    if settings.redis_url is None:
        raise RuntimeError("REDIS_URL is required for worker")
    redis_c = await connect_redis(settings.redis_url)

    desired = await ensure_desired_backend(redis_c, settings._env_backend)
    set_runtime_backend(desired)
    loc_max, oa_max = await ensure_desired_concurrency(
        redis_c,
        local_fallback=settings._env_local_max_concurrent_jobs,
        openai_fallback=settings._env_openai_max_concurrent_jobs,
    )
    # VRAM cap for local (без nvidia-smi — torch ещё может быть не готов).
    m0 = collect_container_metrics(False, False)
    per_slot = estimate_vram_per_slot_mb(
        settings.model_path,
        override_mb=settings.vram_per_slot_mb or None,
    )
    gpu_cap = estimate_max_local_slots(
        m0.gpu_mem_total_mb,
        model_path=settings.model_path,
        vram_per_slot_mb=settings.vram_per_slot_mb or None,
        reserve_mb=settings.vram_reserve_mb,
    )
    if loc_max > gpu_cap:
        log.warning(
            "local_max_capped_by_vram requested=%s cap=%s vram_total_mb=%s per_slot_mb=%s",
            loc_max,
            gpu_cap,
            m0.gpu_mem_total_mb,
            per_slot,
        )
        loc_max = await set_desired_local_max(redis_c, gpu_cap)
    set_runtime_local_max(loc_max)
    set_runtime_openai_max(oa_max)
    if settings.whisper_backend == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required when WHISPER_BACKEND=openai")
    await _prepare_backend(settings.whisper_backend)
    # Total VRAM один раз после загрузки модели; дальше в hot-path CUDA не трогаем.
    try:
        cache_worker_gpu_total(force=True)
    except Exception:
        log.exception("worker_gpu_total_cache_failed")

    stats = await recover_jobs_on_worker_startup(redis_c)
    try:
        await rebuild_active_jobs_index(redis_c)
    except Exception:
        log.exception("active_jobs_index_rebuild_failed")
    if stats.get("requeued") or stats.get("orphaned") or stats.get("flushed"):
        log.info(
            "worker_startup_recovery job_id=- dedup_key=- status=- current_step=recovery requeued=%s orphaned=%s flushed=%s",
            stats["requeued"],
            stats["orphaned"],
            stats.get("flushed", 0),
        )
    await touch_worker_alive(redis_c, ttl_sec=settings.worker_alive_ttl_sec)
    log.info(
        "worker_alive_set key=%s ttl=%s",
        "whisper:worker:alive",
        settings.worker_alive_ttl_sec,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_shutdown(signame: str) -> None:
        if stop_event.is_set():
            return
        log.info("worker_shutdown_signal signal=%s", signame)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except NotImplementedError:
            pass

    # Отдельное Redis-соединение для heartbeat: BRPOP на общем клиенте блокирует pool и alive «умирает».
    redis_alive = await connect_redis(settings.redis_url)
    redis_holder: list[redis.Redis] = [redis_c]
    switching = False
    active_slots = 0
    # При openai dual RX/TX идут параллельно внутри job → больше потоков executor.
    exec_workers = max(1, settings.max_concurrent_jobs * (2 if settings.whisper_backend == "openai" else 1))
    executor = ThreadPoolExecutor(max_workers=exec_workers)
    job_sem = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))
    tasks: set[asyncio.Task[None]] = set()
    sync_busy = False  # True пока max_concurrent=1 обрабатывает job в main loop

    def _metrics_snapshot() -> dict[str, Any]:
        return {
            "switching": switching,
            "active_slots": active_slots,
            "inflight_tasks": len(tasks) + (1 if sync_busy else 0),
        }

    metrics_pub = WorkerMetricsPublisher(
        settings.redis_url,  # type: ignore[arg-type]
        snapshot=_metrics_snapshot,
        alive_ttl_sec=settings.worker_alive_ttl_sec,
        interval_sec=2.0,
    )
    metrics_pub.start()
    log.info("worker_metrics_publisher_started interval_sec=2")

    async def _reconnect_redis() -> bool:
        old = redis_holder[0]
        try:
            await old.aclose()
        except Exception:
            pass
        try:
            redis_holder[0] = await connect_redis(settings.redis_url)  # type: ignore[arg-type]
            log.info("redis_reconnected")
            return True
        except Exception:
            log.exception("redis_reconnect_failed")
            return False

    def _done(t: asyncio.Task[None]) -> None:
        tasks.discard(t)
        if (exc := t.exception()) is not None:
            log.error("job_task_failed: %s", exc)

    async def _rebuild_executor_and_sem() -> None:
        nonlocal executor, job_sem, active_slots
        s = get_settings()
        executor.shutdown(wait=False, cancel_futures=True)
        exec_n = max(1, s.max_concurrent_jobs * (2 if s.whisper_backend == "openai" else 1))
        executor = ThreadPoolExecutor(max_workers=exec_n)
        job_sem = asyncio.Semaphore(max(1, s.max_concurrent_jobs))
        active_slots = 0

    async def _try_apply_runtime_config() -> None:
        nonlocal switching
        if switching or sync_busy or tasks or active_slots > 0:
            return
        s = get_settings()
        gpu_cap = estimate_max_local_slots(
            get_cached_gpu_total_mb(),
            model_path=s.model_path,
            vram_per_slot_mb=s.vram_per_slot_mb or None,
            reserve_mb=s.vram_reserve_mb,
        )
        desired_b = await get_desired_backend(redis_holder[0])
        desired_loc = await get_desired_local_max(redis_holder[0])
        desired_oa = await get_desired_openai_max(redis_holder[0])

        need_backend = bool(desired_b and desired_b != s.whisper_backend)
        if desired_loc is not None and desired_loc > gpu_cap:
            desired_loc = await set_desired_local_max(redis_holder[0], gpu_cap)
        need_local = desired_loc is not None and desired_loc != s.local_max_concurrent_jobs
        need_openai = desired_oa is not None and desired_oa != s.openai_max_concurrent_jobs
        if not (need_backend or need_local or need_openai):
            return

        if need_backend and desired_b == "openai" and not s.openai_api_key:
            log.warning("backend_switch_blocked reason=missing_openai_api_key desired=%s", desired_b)
            need_backend = False
            if not (need_local or need_openai):
                return

        switching = True
        old_backend = s.whisper_backend
        try:
            if need_backend and desired_b:
                log.info("backend_switch_begin from=%s to=%s", old_backend, desired_b)
                set_runtime_backend(desired_b)
                await _prepare_backend(desired_b)
            if desired_loc is not None:
                set_runtime_local_max(min(desired_loc, gpu_cap))
            if desired_oa is not None:
                set_runtime_openai_max(desired_oa)
            await _rebuild_executor_and_sem()
            s2 = get_settings()
            log.info(
                "runtime_config_applied backend=%s local_max=%s openai_max=%s gpu_cap=%s",
                s2.whisper_backend,
                s2.local_max_concurrent_jobs,
                s2.openai_max_concurrent_jobs,
                gpu_cap,
            )
        except Exception:
            log.exception("runtime_config_apply_failed")
            if need_backend:
                set_runtime_backend(old_backend)
                try:
                    await _prepare_backend(old_backend)
                except Exception:
                    log.exception("backend_revert_failed")
        finally:
            switching = False

    watchdog_task = asyncio.create_task(_run_watchdog(redis_alive, stop_event), name="stale-watchdog")
    log.info(
        "worker_started backend=%s max_concurrent_jobs=%s local_max=%s openai_max=%s metrics=thread",
        settings.whisper_backend,
        settings.max_concurrent_jobs,
        settings.local_max_concurrent_jobs,
        settings.openai_max_concurrent_jobs,
    )

    try:
        while not stop_event.is_set():
            await _try_apply_runtime_config()
            if switching:
                await asyncio.sleep(0.2)
                continue
            try:
                job_id = await brpop_job_id(redis_holder[0], timeout_sec=1)
            except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
                if stop_event.is_set():
                    log.info("brpop_aborted_on_shutdown err=%s", type(e).__name__)
                    break
                log.warning("redis_brpop_error err=%s; reconnecting", type(e).__name__)
                if not await _reconnect_redis():
                    await asyncio.sleep(2.0)
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                if stop_event.is_set():
                    break
                log.exception("brpop_unexpected_error")
                await asyncio.sleep(1.0)
                continue
            if stop_event.is_set():
                break
            if not job_id:
                continue

            async def _run_tracked(jid: str) -> None:
                nonlocal active_slots
                active_slots += 1
                try:
                    await _process_job(redis_holder[0], jid, executor, job_sem)
                finally:
                    active_slots = max(0, active_slots - 1)

            if get_settings().max_concurrent_jobs <= 1:
                sync_busy = True
                try:
                    await _run_tracked(job_id)
                finally:
                    sync_busy = False
            else:
                t = asyncio.create_task(_run_tracked(job_id), name=f"job-{job_id}")
                tasks.add(t)
                t.add_done_callback(_done)
                await asyncio.sleep(0)
    finally:
        log.info("worker_stopping inflight_tasks=%s", len(tasks))
        stop_event.set()
        metrics_pub.stop()
        metrics_pub.join(timeout=5.0)
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("worker_bg_task_join_failed")
        try:
            await clear_worker_alive(redis_alive)
        except Exception:
            log.exception("worker_alive_clear_failed")
        for t in list(tasks):
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        executor.shutdown(wait=False, cancel_futures=True)
        for client in (redis_holder[0], redis_alive):
            try:
                await client.aclose()
            except Exception:
                log.exception("redis_close_failed")
        log.info("worker_stopped")


def main() -> None:
    s = get_settings()
    install_persistent_logging(s.logs_dir)
    install_app_console_logging()
    try:
        asyncio.run(worker_main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
