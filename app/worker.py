"""Фоновый worker: BRPOP очереди Redis, транскрипция, heartbeat, watchdog."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import redis.asyncio as redis
from pydantic import ValidationError

from app.diarize_engine import ensure_diarization_pipeline_loaded
from app.job_models import TERMINAL_STATUSES, TranscribeJobRecord
from app.job_store import (
    _utc_iso,
    brpop_job_id,
    clear_worker_alive,
    connect_redis,
    get_job,
    mark_job_claimed_after_brpop,
    recover_jobs_on_worker_startup,
    save_job,
    touch_worker_alive,
)
from app.job_watchdog import scan_stale_jobs
from app.persistent_logs import install_app_console_logging, install_persistent_logging
from app.settings import get_settings
from app.transcribe_body import TranscribeBody
from app.transcribe_engine import ensure_model_loaded
from app.transcribe_pipeline import run_transcription_pipeline
from app.transcribe_schemas import build_transcribe_response

log = logging.getLogger(__name__)


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


async def _run_watchdog(redis_c: redis.Redis) -> None:
    while True:
        try:
            n = await scan_stale_jobs(redis_c)
            if n:
                log.info("watchdog_marked_stale count=%s", n)
        except Exception:
            log.exception("watchdog_scan_failed")
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
            await _patch_job(
                redis_c,
                job_id,
                status="failed",
                error=f"transcribe_failed:{type(e).__name__}:{e}",
                current_step="failed",
                completed=True,
            )
            log.warning(
                "transcribe_job_failed job_id=%s dedup_key=%s status=failed current_step=failed error=%s",
                job_id,
                rec.dedup_key,
                type(e).__name__,
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


async def worker_main_async() -> None:
    settings = get_settings()
    if settings.redis_url is None:
        raise RuntimeError("REDIS_URL is required for worker")
    redis_c = await connect_redis(settings.redis_url)
    await ensure_model_loaded()
    # Нельзя BRPOP/транскрибировать параллельно с первичной загрузкой pyannote на CUDA — зависание GPU/процесса.
    if settings.diarization_enabled:
        await ensure_diarization_pipeline_loaded()
    stats = await recover_jobs_on_worker_startup(redis_c)
    if stats.get("requeued") or stats.get("orphaned") or stats.get("flushed"):
        log.info(
            "worker_startup_recovery job_id=- dedup_key=- status=- current_step=recovery requeued=%s orphaned=%s flushed=%s",
            stats["requeued"],
            stats["orphaned"],
            stats.get("flushed", 0),
        )
    await touch_worker_alive(redis_c, ttl_sec=settings.worker_alive_ttl_sec)

    async def _alive_loop() -> None:
        while True:
            try:
                await touch_worker_alive(redis_c, ttl_sec=settings.worker_alive_ttl_sec)
            except Exception:
                log.exception("worker_alive_touch_failed")
            await asyncio.sleep(settings.worker_alive_refresh_sec)

    asyncio.create_task(_alive_loop(), name="worker-alive")
    executor = ThreadPoolExecutor(max_workers=max(1, settings.max_concurrent_jobs))
    job_sem = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))
    tasks: set[asyncio.Task[None]] = set()

    def _done(t: asyncio.Task[None]) -> None:
        tasks.discard(t)
        if (exc := t.exception()) is not None:
            log.error("job_task_failed: %s", exc)

    asyncio.create_task(_run_watchdog(redis_c), name="stale-watchdog")
    log.info("worker_started max_concurrent_jobs=%s", settings.max_concurrent_jobs)

    try:
        while True:
            job_id = await brpop_job_id(redis_c, timeout_sec=5)
            if not job_id:
                await asyncio.sleep(0)
                continue
            # WHISPER_MAX_CONCURRENT_JOBS==1: обрабатываем без лишнего Task (устойчивее к планировщику).
            if settings.max_concurrent_jobs <= 1:
                await _process_job(redis_c, job_id, executor, job_sem)
            else:
                t = asyncio.create_task(_process_job(redis_c, job_id, executor, job_sem), name=f"job-{job_id}")
                tasks.add(t)
                t.add_done_callback(_done)
                await asyncio.sleep(0)
    finally:
        try:
            await clear_worker_alive(redis_c)
        except Exception:
            log.exception("worker_alive_clear_failed")
        for t in list(tasks):
            t.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    s = get_settings()
    install_persistent_logging(s.logs_dir)
    install_app_console_logging()
    asyncio.run(worker_main_async())


if __name__ == "__main__":
    main()
