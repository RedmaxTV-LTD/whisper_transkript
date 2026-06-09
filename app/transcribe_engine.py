"""Загрузка faster-whisper и синхронное распознавание (в executor из FastAPI)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from app.align_speakers import align_whisper_segments, speakers_chronological_order
from app.settings import Settings, get_settings
from app.speaker_roles_catalog import get_speaker_roles_catalog_live, infer_speaker_role_map
from app.spelling_fixes import apply_spelling_fixes, get_spelling_patterns_live

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

_model: WhisperModel | None = None
_model_lock = asyncio.Lock()
_onnx_preloaded = False


@dataclass
class TranscribeResult:
    transcript: str
    detected_language: str | None
    diarization_used: bool
    segments: list[tuple[str, float, float, str]] = field(default_factory=list)
    speakers_ordered: list[str] = field(default_factory=list)
    # SPEAKER_* → client|operator по каталогу фраз (IVR — в operator); None — фиксированная схема 00/01/02.
    # Для dual_track: спикеры «RX» и «TX», карта задаётся из call_direction (входящий/исходящий).
    speaker_role_map: dict[str, str] | None = None
    dual_track: bool = False
    dual_call_direction: Literal["incoming", "outgoing"] | None = None
    dual_mix_sync_applied: bool = False
    dual_mix_sync_offset_rx_sec: float | None = None
    dual_mix_sync_offset_tx_sec: float | None = None
    dual_mix_sync_score_rx: float | None = None
    dual_mix_sync_score_tx: float | None = None
    dual_mix_sync_mode: Literal["auto", "manual"] | None = None
    mix_fallback: bool = False


def _preload_onnxruntime_silently_for_vad() -> None:
    """Один раз подгружаем onnxruntime, чтобы не засорять лог при первом /transcribe.

    Silero VAD (faster-whisper) при первом import дергает обнаружение GPU через
    /sys/class/drm/...; в nvidia-docker часто нет vendor у card0 — это лишь [W],
    inference всё равно на CPUExecutionProvider. Временно глушим stderr на fd 2.
    """
    global _onnx_preloaded
    if _onnx_preloaded:
        return
    settings = get_settings()
    if not settings.vad_filter:
        _onnx_preloaded = True
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    save_err = os.dup(2)
    try:
        os.dup2(devnull, 2)
        import onnxruntime  # noqa: F401
    except ImportError:
        pass
    finally:
        os.dup2(save_err, 2)
        os.close(save_err)
        os.close(devnull)
    _onnx_preloaded = True


async def ensure_model_loaded() -> None:
    global _model
    if _model is not None:
        return
    async with _model_lock:
        if _model is not None:
            return
        settings = get_settings()
        log.info(
            "loading_whisper_model path=%s device=%s compute=%s",
            settings.model_path,
            settings.device,
            settings.compute_type,
        )

        def _load():
            from faster_whisper import WhisperModel

            return WhisperModel(
                settings.model_path,
                device=settings.device,
                compute_type=settings.compute_type,
            )

        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(None, _load)
        _model = loaded
        log.info("whisper_model_ready")
        await loop.run_in_executor(None, _preload_onnxruntime_silently_for_vad)


def get_model() -> "WhisperModel":
    if _model is None:
        raise RuntimeError("model_not_loaded")
    return _model


def _whisper_segment_to_time_parts(
    seg: Any,
    patterns: list[tuple[re.Pattern[str], str]],
    split_gap_sec: float,
) -> list[tuple[float, float, str]]:
    """Один сегмент Whisper → одна или несколько частей; при split_gap_sec>0 режем по паузам между словами."""
    words = getattr(seg, "words", None)
    if split_gap_sec <= 0 or not words:
        t = apply_spelling_fixes(seg.text.strip(), patterns)
        if not t:
            return []
        return [(float(seg.start), float(seg.end), t)]

    word_list = list(words)
    if not word_list:
        t = apply_spelling_fixes(seg.text.strip(), patterns)
        if not t:
            return []
        return [(float(seg.start), float(seg.end), t)]

    runs: list[list[Any]] = [[word_list[0]]]
    for w in word_list[1:]:
        prev = runs[-1][-1]
        gap = float(w.start) - float(prev.end)
        if gap > split_gap_sec:
            runs.append([w])
        else:
            runs[-1].append(w)

    out: list[tuple[float, float, str]] = []
    for run in runs:
        s0 = float(run[0].start)
        s1 = float(run[-1].end)
        raw = "".join(str(getattr(x, "word", x)) for x in run).strip()
        t = apply_spelling_fixes(raw, patterns)
        if t:
            out.append((s0, s1, t))
    return out


def collect_transcript(model: "WhisperModel", wav_path: str, settings: Settings) -> tuple[str, str | None]:
    patterns = get_spelling_patterns_live(settings.spelling_dict_path, settings.spelling_fixes_enabled)
    use_word_ts = settings.intra_segment_split_gap_sec > 0
    segments, info = model.transcribe(
        wav_path,
        language=settings.language,
        beam_size=settings.beam_size,
        vad_filter=settings.vad_filter,
        word_timestamps=use_word_ts,
    )
    parts: list[str] = []
    for seg in segments:
        for _s0, _s1, t in _whisper_segment_to_time_parts(
            seg, patterns, settings.intra_segment_split_gap_sec
        ):
            if t:
                parts.append(t)
    text = " ".join(parts).strip()
    lang = getattr(info, "language", None)
    return text, lang


def transcribe_wav_to_parts(
    model: "WhisperModel",
    wav_path: str,
    settings: Settings,
) -> tuple[list[tuple[float, float, str]], str | None]:
    """Whisper по одному WAV: сегменты (start, end, text) и язык."""
    patterns = get_spelling_patterns_live(settings.spelling_dict_path, settings.spelling_fixes_enabled)
    use_word_ts = settings.intra_segment_split_gap_sec > 0
    segments, info = model.transcribe(
        wav_path,
        language=settings.language,
        beam_size=settings.beam_size,
        vad_filter=settings.vad_filter,
        word_timestamps=use_word_ts,
    )
    whisper_parts: list[tuple[float, float, str]] = []
    for seg in segments:
        whisper_parts.extend(
            _whisper_segment_to_time_parts(seg, patterns, settings.intra_segment_split_gap_sec)
        )
    lang = getattr(info, "language", None)
    return whisper_parts, lang


def transcribe_mix_only_result(
    model: "WhisperModel",
    mix_wav_path: str,
    settings: Settings,
    *,
    call_direction: Literal["incoming", "outgoing"],
) -> TranscribeResult:
    """Один проход Whisper по url_mix (fallback при сбое синхронизации RX/TX)."""
    parts, lang = transcribe_wav_to_parts(model, mix_wav_path, settings)
    full_text = " ".join(p[2] for p in parts).strip()
    return TranscribeResult(
        transcript=full_text,
        detected_language=lang,
        diarization_used=False,
        mix_fallback=True,
        dual_call_direction=call_direction,
        segments=[],
        speakers_ordered=[],
    )


def transcribe_dual_rx_tx(
    model: "WhisperModel",
    rx_wav_path: str,
    tx_wav_path: str,
    settings: Settings,
    call_direction: Literal["incoming", "outgoing"],
    *,
    time_offset_rx_sec: float = 0.0,
    time_offset_tx_sec: float = 0.0,
    mix_sync_applied: bool = False,
    mix_sync_score_rx: float | None = None,
    mix_sync_score_tx: float | None = None,
    dual_mix_sync_mode: Literal["auto", "manual"] | None = None,
) -> TranscribeResult:
    """Два моно-файла rx/tx: роли по направлению звонка (входящий: rx=клиент, tx=оператор; исходящий — наоборот)."""
    parts_rx, lang_rx = transcribe_wav_to_parts(model, rx_wav_path, settings)
    parts_tx, lang_tx = transcribe_wav_to_parts(model, tx_wav_path, settings)
    return build_dual_track_result_from_parts(
        parts_rx,
        parts_tx,
        lang_rx,
        lang_tx,
        call_direction,
        time_offset_rx_sec=time_offset_rx_sec,
        time_offset_tx_sec=time_offset_tx_sec,
        mix_sync_applied=mix_sync_applied,
        mix_sync_score_rx=mix_sync_score_rx,
        mix_sync_score_tx=mix_sync_score_tx,
        dual_mix_sync_mode=dual_mix_sync_mode,
    )


def build_dual_track_result_from_parts(
    parts_rx: list[tuple[float, float, str]],
    parts_tx: list[tuple[float, float, str]],
    lang_rx: str | None,
    lang_tx: str | None,
    call_direction: Literal["incoming", "outgoing"],
    *,
    time_offset_rx_sec: float = 0.0,
    time_offset_tx_sec: float = 0.0,
    mix_sync_applied: bool = False,
    mix_sync_score_rx: float | None = None,
    mix_sync_score_tx: float | None = None,
    dual_mix_sync_mode: Literal["auto", "manual"] | None = None,
) -> TranscribeResult:
    """Собирает dual_track TranscribeResult из уже распознанных частей RX/TX (для пошагового heartbeat в worker)."""
    role_rx: str = "client" if call_direction == "incoming" else "operator"
    role_tx: str = "operator" if call_direction == "incoming" else "client"
    merged: list[tuple[str, float, float, str]] = []
    for s0, s1, t in parts_rx:
        merged.append(("RX", s0 + time_offset_rx_sec, s1 + time_offset_rx_sec, t))
    for s0, s1, t in parts_tx:
        merged.append(("TX", s0 + time_offset_tx_sec, s1 + time_offset_tx_sec, t))
    merged.sort(key=lambda x: x[1])
    full_text = " ".join(p[3] for p in merged).strip()
    lang = lang_rx or lang_tx
    return TranscribeResult(
        transcript=full_text,
        detected_language=lang,
        diarization_used=False,
        dual_track=True,
        dual_call_direction=call_direction,
        segments=merged,
        speakers_ordered=["RX", "TX"],
        speaker_role_map={"RX": role_rx, "TX": role_tx},
        dual_mix_sync_applied=mix_sync_applied,
        dual_mix_sync_offset_rx_sec=time_offset_rx_sec if mix_sync_applied else None,
        dual_mix_sync_offset_tx_sec=time_offset_tx_sec if mix_sync_applied else None,
        dual_mix_sync_score_rx=mix_sync_score_rx if mix_sync_applied else None,
        dual_mix_sync_score_tx=mix_sync_score_tx if mix_sync_applied else None,
        dual_mix_sync_mode=dual_mix_sync_mode if mix_sync_applied else None,
    )


def transcribe_with_diarization(
    model: "WhisperModel",
    wav_path: str,
    settings: Settings,
    *,
    apply_diarization: bool,
) -> TranscribeResult:
    """Распознавание с опциональной диаризацией pyannote (если включена и пайплайн загружен)."""
    whisper_parts, lang = transcribe_wav_to_parts(model, wav_path, settings)
    full_text = " ".join(p[2] for p in whisper_parts).strip()

    if not settings.diarization_enabled or not apply_diarization:
        return TranscribeResult(
            transcript=full_text,
            detected_language=lang,
            diarization_used=False,
            segments=[],
            speakers_ordered=[],
        )

    from app import diarize_engine

    if not diarize_engine.is_diarization_pipeline_ready():
        log.warning(
            "diarization_skipped: pipeline not available (%s)",
            diarize_engine.diarization_pipeline_error(),
        )
        return TranscribeResult(
            transcript=full_text,
            detected_language=lang,
            diarization_used=False,
            segments=[],
            speakers_ordered=[],
        )

    try:
        turns = diarize_engine.run_diarization(wav_path)
    except Exception:
        log.exception("diarization_run_failed")
        return TranscribeResult(
            transcript=full_text,
            detected_language=lang,
            diarization_used=False,
            segments=[],
            speakers_ordered=[],
        )

    aligned = align_whisper_segments(whisper_parts, turns)
    order = speakers_chronological_order(turns)
    role_map: dict[str, str] | None = None
    if settings.speaker_roles_catalog_enabled:
        cat = get_speaker_roles_catalog_live(
            settings.speaker_roles_catalog_path,
            settings.speaker_roles_catalog_enabled,
        )
        if cat is not None:
            role_map = infer_speaker_role_map(aligned, cat)
    return TranscribeResult(
        transcript=full_text,
        detected_language=lang,
        diarization_used=True,
        segments=aligned,
        speakers_ordered=order,
        speaker_role_map=role_map,
    )
