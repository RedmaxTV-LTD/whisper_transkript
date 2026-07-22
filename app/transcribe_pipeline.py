"""Асинхронный пайплайн транскрипции (используется worker'ом и совпадает с логикой legacy /transcribe)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from app.audio_prep import cleanup_temp_dir, prepare_audio_from_url, prepare_dual_rx_tx_from_urls
from app.channel_sync import estimate_rx_tx_offsets_vs_mix
from app.diarize_engine import ensure_diarization_pipeline_loaded
from app.openai_engine import (
    transcribe_mix_only_result_openai,
    transcribe_mono_openai,
    transcribe_wav_to_parts_openai,
)
from app.settings import Settings, get_settings
from app.transcribe_body import SyncOptionsBody, TranscribeBody
from app.transcribe_engine import (
    TranscribeResult,
    build_dual_track_result_from_parts,
    get_model,
    transcribe_mix_only_result,
    transcribe_dual_rx_tx,
    transcribe_wav_to_parts,
    transcribe_with_diarization,
)

log = logging.getLogger(__name__)

StepCallback = Callable[[str, str, int], Awaitable[None]]


async def _notify(cb: StepCallback | None, status: str, step: str, progress: int) -> None:
    if cb is not None:
        await cb(status, step, progress)


def _is_openai(settings: Settings) -> bool:
    return settings.whisper_backend == "openai"


async def _wav_to_parts(
    executor: ThreadPoolExecutor,
    wav_path: str,
    settings: Settings,
    *,
    model: object | None,
) -> tuple[list[tuple[float, float, str]], str | None]:
    loop = asyncio.get_running_loop()
    if _is_openai(settings):
        return await loop.run_in_executor(executor, lambda: transcribe_wav_to_parts_openai(wav_path, settings))
    assert model is not None
    return await loop.run_in_executor(executor, lambda: transcribe_wav_to_parts(model, wav_path, settings))


async def _parallel_rx_tx_parts(
    executor: ThreadPoolExecutor,
    rx_wav: str,
    tx_wav: str,
    settings: Settings,
    *,
    model: object | None,
) -> tuple[
    list[tuple[float, float, str]],
    str | None,
    list[tuple[float, float, str]],
    str | None,
]:
    """RX и TX: на openai — параллельно; на local — последовательно (GPU)."""
    if _is_openai(settings):
        (parts_rx, lang_rx), (parts_tx, lang_tx) = await asyncio.gather(
            _wav_to_parts(executor, rx_wav, settings, model=None),
            _wav_to_parts(executor, tx_wav, settings, model=None),
        )
        return parts_rx, lang_rx, parts_tx, lang_tx
    parts_rx, lang_rx = await _wav_to_parts(executor, rx_wav, settings, model=model)
    parts_tx, lang_tx = await _wav_to_parts(executor, tx_wav, settings, model=model)
    return parts_rx, lang_rx, parts_tx, lang_tx


async def _mix_only(
    executor: ThreadPoolExecutor,
    mix_wav: str,
    settings: Settings,
    call_direction: Literal["incoming", "outgoing"],
    *,
    model: object | None,
) -> TranscribeResult:
    loop = asyncio.get_running_loop()
    if _is_openai(settings):
        return await loop.run_in_executor(
            executor,
            lambda: transcribe_mix_only_result_openai(mix_wav, settings, call_direction=call_direction),
        )
    assert model is not None
    return await loop.run_in_executor(
        executor,
        lambda: transcribe_mix_only_result(model, mix_wav, settings, call_direction=call_direction),
    )


async def _dual_fallback_no_offsets(
    executor: ThreadPoolExecutor,
    rx_wav: str,
    tx_wav: str,
    settings: Settings,
    call_direction: Literal["incoming", "outgoing"],
    *,
    model: object | None,
) -> TranscribeResult:
    if _is_openai(settings):
        # Параллельные RX/TX через gather внутри pipeline-хелпера.
        parts_rx, lang_rx, parts_tx, lang_tx = await _parallel_rx_tx_parts(
            executor, rx_wav, tx_wav, settings, model=None
        )
        return build_dual_track_result_from_parts(parts_rx, parts_tx, lang_rx, lang_tx, call_direction)
    assert model is not None
    return await asyncio.get_running_loop().run_in_executor(
        executor,
        lambda: transcribe_dual_rx_tx(model, rx_wav, tx_wav, settings, call_direction),
    )


async def run_transcription_pipeline(
    body: TranscribeBody,
    *,
    executor: ThreadPoolExecutor,
    step_callback: StepCallback | None = None,
) -> tuple[TranscribeResult, int, str]:
    """Возвращает (TranscribeResult, source_channels, layout)."""
    settings = get_settings()
    wav_path: Path | None = None
    source_channels = 1
    layout = "mono"
    use_openai = _is_openai(settings)
    model = None if use_openai else get_model()

    try:
        if body.url_rx is not None:
            await _notify(step_callback, "downloading", "downloading audio files", 5)
            mix_url = str(body.url_mix) if body.url_mix is not None else None
            rx_wav, tx_wav, mix_wav = await prepare_dual_rx_tx_from_urls(
                str(body.url_rx),
                str(body.url_tx),
                mix_url,
            )
            wav_path = rx_wav
            source_channels = 2
            layout = "dual_mono"
            assert body.call_direction is not None
            sync = body.sync or SyncOptionsBody()
            max_off = float(sync.max_offset_sec) if sync.max_offset_sec is not None else settings.sync_max_offset_sec
            max_off = max(0.05, max_off)

            if sync.mode == "off":
                await _notify(step_callback, "transcribing_rx", "transcribing rx/tx channels", 25)
                parts_rx, lang_rx, parts_tx, lang_tx = await _parallel_rx_tx_parts(
                    executor, str(rx_wav), str(tx_wav), settings, model=model
                )
                await _notify(step_callback, "merging_segments", "merging rx/tx segments", 85)
                t_result = build_dual_track_result_from_parts(
                    parts_rx,
                    parts_tx,
                    lang_rx,
                    lang_tx,
                    body.call_direction,
                )
                return t_result, source_channels, layout

            if sync.mode == "manual":
                bad = abs(sync.offset_rx_sec) > max_off or abs(sync.offset_tx_sec) > max_off
                if bad:
                    log.warning(
                        "dual_sync_manual_offsets_exceed_max max=%s rx=%s tx=%s",
                        max_off,
                        sync.offset_rx_sec,
                        sync.offset_tx_sec,
                    )
                    if sync.fallback == "use_mix":
                        assert mix_wav is not None
                        await _notify(step_callback, "transcribing_mix", "transcribing url_mix (fallback)", 40)
                        return (
                            await _mix_only(
                                executor, str(mix_wav), settings, body.call_direction, model=model
                            ),
                            source_channels,
                            layout,
                        )
                    return (
                        await _dual_fallback_no_offsets(
                            executor, str(rx_wav), str(tx_wav), settings, body.call_direction, model=model
                        ),
                        source_channels,
                        layout,
                    )
                await _notify(step_callback, "transcribing_rx", "transcribing rx/tx channels (manual sync)", 25)
                parts_rx, lang_rx, parts_tx, lang_tx = await _parallel_rx_tx_parts(
                    executor, str(rx_wav), str(tx_wav), settings, model=model
                )
                await _notify(step_callback, "merging_segments", "merging rx/tx segments", 85)
                t_result = build_dual_track_result_from_parts(
                    parts_rx,
                    parts_tx,
                    lang_rx,
                    lang_tx,
                    body.call_direction,
                    time_offset_rx_sec=sync.offset_rx_sec,
                    time_offset_tx_sec=sync.offset_tx_sec,
                    mix_sync_applied=True,
                    mix_sync_score_rx=None,
                    mix_sync_score_tx=None,
                    dual_mix_sync_mode="manual",
                )
                return t_result, source_channels, layout

            assert sync.mode == "auto"
            assert mix_wav is not None
            await _notify(step_callback, "syncing_channels", "syncing rx/tx to mix", 15)

            def _estimate():
                return estimate_rx_tx_offsets_vs_mix(
                    str(mix_wav),
                    str(rx_wav),
                    str(tx_wav),
                    settings,
                    max_offset_sec=max_off,
                )

            off_rx, off_tx, sr_rx, sr_tx, tr_rx, tr_tx = await asyncio.get_running_loop().run_in_executor(
                executor,
                _estimate,
            )
            if not tr_rx and not tr_tx:
                log.warning("dual_sync_auto_both_untrusted fallback=%s", sync.fallback)
                if sync.fallback == "use_mix":
                    await _notify(step_callback, "transcribing_mix", "transcribing url_mix (auto fallback)", 40)
                    return (
                        await _mix_only(
                            executor, str(mix_wav), settings, body.call_direction, model=model
                        ),
                        source_channels,
                        layout,
                    )
                return (
                    await _dual_fallback_no_offsets(
                        executor, str(rx_wav), str(tx_wav), settings, body.call_direction, model=model
                    ),
                    source_channels,
                    layout,
                )
            await _notify(step_callback, "transcribing_rx", "transcribing rx/tx channels (synced)", 30)
            parts_rx, lang_rx, parts_tx, lang_tx = await _parallel_rx_tx_parts(
                executor, str(rx_wav), str(tx_wav), settings, model=model
            )
            await _notify(step_callback, "merging_segments", "merging rx/tx segments", 85)
            t_result = build_dual_track_result_from_parts(
                parts_rx,
                parts_tx,
                lang_rx,
                lang_tx,
                body.call_direction,
                time_offset_rx_sec=off_rx,
                time_offset_tx_sec=off_tx,
                mix_sync_applied=True,
                mix_sync_score_rx=sr_rx,
                mix_sync_score_tx=sr_tx,
                dual_mix_sync_mode="auto",
            )
            return t_result, source_channels, layout

        assert body.url is not None
        await _notify(step_callback, "downloading", "downloading audio", 8)
        wav_path, source_channels, layout = await prepare_audio_from_url(str(body.url))

        if use_openai:
            if body.diarize is True or (
                body.diarize is None and settings.diarization_enabled and settings.diarize_default
            ):
                log.info("openai_backend_diarization_skipped note=use_dual_rx_tx_for_roles")
            await _notify(step_callback, "transcribing_mix", "transcribing via openai", 40)
            t_result = await asyncio.get_running_loop().run_in_executor(
                executor,
                lambda: transcribe_mono_openai(str(wav_path), settings),
            )
            return t_result, source_channels, layout

        use_diar = False
        if settings.diarization_enabled:
            use_diar = settings.diarize_default if body.diarize is None else bool(body.diarize)
        if settings.diarization_enabled and use_diar:
            await ensure_diarization_pipeline_loaded()
            await _notify(step_callback, "merging_segments", "transcribing and diarizing", 40)
        else:
            await _notify(step_callback, "transcribing_mix", "transcribing mono/stereo", 40)

        def _run() -> TranscribeResult:
            return transcribe_with_diarization(
                model,
                str(wav_path),
                settings,
                apply_diarization=use_diar,
            )

        t_result = await asyncio.get_running_loop().run_in_executor(executor, _run)
        return t_result, source_channels, layout
    finally:
        if wav_path is not None:
            cleanup_temp_dir(wav_path)
