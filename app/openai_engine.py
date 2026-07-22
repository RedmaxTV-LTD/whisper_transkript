"""Облачный OpenAI Audio Transcriptions API (httpx)."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import httpx

from app.settings import Settings
from app.spelling_fixes import apply_spelling_fixes, get_spelling_patterns_live
from app.transcribe_engine import TranscribeResult, build_dual_track_result_from_parts

log = logging.getLogger(__name__)


class OpenAITranscribeError(RuntimeError):
    """Ошибка вызова OpenAI transcriptions."""


def _ensure_openai_key(settings: Settings) -> str:
    if not settings.openai_api_key:
        raise OpenAITranscribeError("openai_missing_api_key")
    return settings.openai_api_key


def _file_for_upload(wav_path: str, settings: Settings) -> tuple[Path, bool]:
    """Возвращает (path, is_temp). При превышении лимита сжимает в mp3."""
    src = Path(wav_path)
    max_bytes = int(settings.openai_max_upload_mb * 1024 * 1024)
    size = src.stat().st_size
    if size <= max_bytes:
        return src, False

    out = src.with_suffix(".openai.mp3")
    # Целевой битрейт ~64k обычно укладывает длинные mono 16 kHz в лимит; при необходимости понижаем.
    for bitrate in ("64k", "48k", "32k"):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(out),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if out.stat().st_size <= max_bytes:
            log.info(
                "openai_audio_compressed path=%s orig_bytes=%s out_bytes=%s bitrate=%s",
                src.name,
                size,
                out.stat().st_size,
                bitrate,
            )
            return out, True
    raise OpenAITranscribeError(
        f"openai_upload_too_large:size={size}:max={max_bytes}:compressed={out.stat().st_size}"
    )


def _parts_from_verbose_json(
    data: dict[str, Any],
    settings: Settings,
) -> tuple[list[tuple[float, float, str]], str | None]:
    patterns = get_spelling_patterns_live(settings.spelling_dict_path, settings.spelling_fixes_enabled)
    lang = data.get("language")
    if isinstance(lang, str):
        lang_out: str | None = lang
    else:
        lang_out = None

    segments = data.get("segments") or []
    parts: list[tuple[float, float, str]] = []
    split_gap = settings.intra_segment_split_gap_sec

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        words = seg.get("words")
        if split_gap > 0 and isinstance(words, list) and words:
            runs: list[list[dict[str, Any]]] = [[words[0]]]
            for w in words[1:]:
                if not isinstance(w, dict):
                    continue
                prev = runs[-1][-1]
                gap = float(w.get("start", 0)) - float(prev.get("end", 0))
                if gap > split_gap:
                    runs.append([w])
                else:
                    runs[-1].append(w)
            for run in runs:
                s0 = float(run[0].get("start", 0))
                s1 = float(run[-1].get("end", 0))
                raw = "".join(str(x.get("word", "")) for x in run).strip()
                t = apply_spelling_fixes(raw, patterns)
                if t:
                    parts.append((s0, s1, t))
            continue

        text = apply_spelling_fixes(str(seg.get("text") or "").strip(), patterns)
        if not text:
            continue
        parts.append((float(seg.get("start") or 0), float(seg.get("end") or 0), text))

    if not parts:
        full = apply_spelling_fixes(str(data.get("text") or "").strip(), patterns)
        if full:
            parts.append((0.0, 0.0, full))
    return parts, lang_out


def transcribe_wav_to_parts_openai(
    wav_path: str,
    settings: Settings,
) -> tuple[list[tuple[float, float, str]], str | None]:
    """Один файл → сегменты (start, end, text) и язык через OpenAI API."""
    api_key = _ensure_openai_key(settings)
    upload_path, is_temp = _file_for_upload(wav_path, settings)
    url = f"{settings.openai_base_url}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    use_word = settings.intra_segment_split_gap_sec > 0
    form: dict[str, Any] = {
        "model": settings.openai_transcribe_model,
        "response_format": "verbose_json",
    }
    if settings.language:
        form["language"] = settings.language
    # timestamp_granularities — повторяющиеся поля формы
    granularities = ["segment"]
    if use_word:
        granularities.append("word")

    timeout = httpx.Timeout(settings.openai_timeout_sec)
    last_err: Exception | None = None
    try:
        for attempt in range(settings.openai_max_retries + 1):
            try:
                with upload_path.open("rb") as f:
                    files = {"file": (upload_path.name, f, "application/octet-stream")}
                    data_fields: list[tuple[str, str]] = [(k, str(v)) for k, v in form.items()]
                    for g in granularities:
                        data_fields.append(("timestamp_granularities[]", g))
                    with httpx.Client(timeout=timeout) as client:
                        resp = client.post(url, headers=headers, data=data_fields, files=files)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < settings.openai_max_retries:
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0 ** attempt)
                    log.warning(
                        "openai_transcribe_retry status=%s attempt=%s delay=%s",
                        resp.status_code,
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if resp.status_code >= 400:
                    body_snip = (resp.text or "")[:400]
                    raise OpenAITranscribeError(f"openai_http_{resp.status_code}:{body_snip}")
                payload = resp.json()
                if not isinstance(payload, dict):
                    raise OpenAITranscribeError("openai_invalid_json_response")
                return _parts_from_verbose_json(payload, settings)
            except OpenAITranscribeError:
                raise
            except Exception as e:
                last_err = e
                if attempt < settings.openai_max_retries:
                    delay = min(60.0, 2.0 ** attempt)
                    log.warning(
                        "openai_transcribe_retry err=%s attempt=%s delay=%s",
                        type(e).__name__,
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise OpenAITranscribeError(f"openai_request_failed:{type(e).__name__}:{e}") from e
        raise OpenAITranscribeError(f"openai_request_failed:{last_err}")
    finally:
        if is_temp:
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                log.warning("openai_temp_cleanup_failed path=%s", upload_path)


def transcribe_mix_only_result_openai(
    mix_wav_path: str,
    settings: Settings,
    *,
    call_direction: Literal["incoming", "outgoing"],
) -> TranscribeResult:
    parts, lang = transcribe_wav_to_parts_openai(mix_wav_path, settings)
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


def transcribe_mono_openai(wav_path: str, settings: Settings) -> TranscribeResult:
    """Mono/stereo без диаризации (pyannote на openai-бэкенде не используется)."""
    parts, lang = transcribe_wav_to_parts_openai(wav_path, settings)
    full_text = " ".join(p[2] for p in parts).strip()
    return TranscribeResult(
        transcript=full_text,
        detected_language=lang,
        diarization_used=False,
        segments=[],
        speakers_ordered=[],
    )


def transcribe_dual_rx_tx_openai(
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
    """Последовательный fallback (для sync fallback без gather); предпочтительнее параллельный путь в pipeline."""
    parts_rx, lang_rx = transcribe_wav_to_parts_openai(rx_wav_path, settings)
    parts_tx, lang_tx = transcribe_wav_to_parts_openai(tx_wav_path, settings)
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
