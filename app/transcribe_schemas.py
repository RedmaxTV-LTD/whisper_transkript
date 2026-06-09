"""Pydantic-модели ответа транскрипции и сборка HTTP-ответа из TranscribeResult."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.settings import get_settings
from app.speaker_roles import (
    aggregate_by_role,
    effective_role_map,
    effective_segment_role,
    role_label_ru,
    text_matches_system_ivr,
)
from app.speaker_roles_catalog import get_speaker_roles_catalog_live
from app.transcribe_engine import TranscribeResult


class DiarizedSegment(BaseModel):
    speaker: str = Field(
        ...,
        description="Спикер pyannote (SPEAKER_00) или канал RX/TX при режиме dual_mono",
    )
    role: Literal["client", "operator", "other"] = Field(
        ...,
        description="Роль: клиент / оператор (включая IVR и линию); отдельной роли «система» нет",
    )
    role_label_ru: str = Field(..., description="Подпись роли на русском (Клиент / Оператор / Прочее)")
    ivr_hint_match: bool = Field(
        False,
        description="Текст содержит типичные маркеры IVR/системы (время ожидания, Рэдмакс и т.п.)",
    )
    start_sec: float
    end_sec: float
    text: str
    pause_before_sec: float = Field(
        0.0,
        description="Пауза на общей шкале перед сегментом: start_sec − end предыдущего (≥0); для UI не склеивать реплики в один блок.",
    )


class TranscribeResponse(BaseModel):
    source_audio_channels: int = Field(..., description="Число каналов во входном файле (до сведения в mono)")
    source_layout: Literal["mono", "stereo", "dual_mono"] = Field(
        ...,
        description="mono = 1 канал; stereo = один файл 2+ канала (сведён в mono); dual_mono = два отдельных mono URL (rx/tx)",
    )
    detected_language: str | None = None
    transcript: str = Field(..., description="Полный текст по моно (или по сведённому stereo)")
    operator_text: str | None = Field(None, description="Текст оператора (роль operator) при диаризации")
    client_text: str | None = Field(None, description="Текст клиента (роль client) и прочих спикеров при диаризации")
    system_text: str | None = Field(
        None,
        description="Зарезервировано; при диаризации текст IVR входит в operator_text",
    )
    role_separation: Literal[
        "unavailable_mono",
        "unavailable_stereo_mixed",
        "dual_track",
        "diarized",
        "mix_fallback",
    ] = Field(
        ...,
        description="Режим разделения Оператор/Клиент / диаризация / fallback только по url_mix",
    )
    formatted_text: str = Field(..., description="Текст для отображения; для моно — блок «Транскрипт»")
    note_ru: str = Field(..., description="Пояснение по ролям и будущим rx/tx")
    diarization_used: bool = Field(False, description="Была ли применена pyannote.audio для этого ответа")
    diarized_segments: list[DiarizedSegment] | None = Field(
        None,
        description="Сегменты с привязкой спикера (если diarization_used)",
    )
    mix_sync_mode_used: Literal["auto", "manual"] | None = Field(
        None,
        description="Если dual_mono и применялись сдвиги: auto (корреляция) или manual",
    )
    mix_sync_offset_rx_sec: float | None = Field(
        None,
        description="Секунды, прибавленные к таймингам RX для шкалы mix (только при mix_sync_mode_used)",
    )
    mix_sync_offset_tx_sec: float | None = Field(
        None,
        description="Секунды, прибавленные к таймингам TX для шкалы mix (только при mix_sync_mode_used)",
    )
    mix_sync_score_rx: float | None = Field(
        None,
        description="Метрика корреляции RX↔mix (выше — увереннее; см. WHISPER_CHANNEL_SYNC_MIN_CORR)",
    )
    mix_sync_score_tx: float | None = Field(
        None,
        description="Метрика корреляции TX↔mix",
    )


def build_transcribe_response(
    *,
    source_channels: int,
    layout: str,
    result: TranscribeResult,
) -> TranscribeResponse:
    transcript = result.transcript
    detected_language = result.detected_language
    stereo_mixed = layout != "dual_mono" and (layout == "stereo" or source_channels >= 2)
    settings = get_settings()

    if result.mix_fallback:
        if result.dual_call_direction == "incoming":
            direction_ru = "входящий"
        elif result.dual_call_direction == "outgoing":
            direction_ru = "исходящий"
        else:
            direction_ru = "—"
        note = (
            f"Сработал sync.fallback=use_mix: один транскрипт по **url_mix** (звонок {direction_ru}). "
            "Разделения RX/TX в этом ответе нет."
        )
        tx = (transcript or "").strip()
        return TranscribeResponse(
            source_audio_channels=1,
            source_layout="mono",
            detected_language=detected_language,
            transcript=tx,
            operator_text=None,
            client_text=None,
            system_text=None,
            role_separation="mix_fallback",
            formatted_text="Транскрипт (url_mix, fallback):\n" + tx,
            note_ru=note,
            diarization_used=False,
            diarized_segments=None,
            mix_sync_mode_used=None,
            mix_sync_offset_rx_sec=None,
            mix_sync_offset_tx_sec=None,
            mix_sync_score_rx=None,
            mix_sync_score_tx=None,
        )

    if result.dual_track and not result.segments:
        direction_ru = (
            "входящий (rx — клиент, tx — оператор)"
            if result.dual_call_direction == "incoming"
            else "исходящий (rx — оператор, tx — клиент)"
        )
        note = (
            f"Два отдельных mono-файла (url_rx / url_tx), звонок {direction_ru}. "
            "Распознанный текст пуст по обоим каналам."
        )
        return TranscribeResponse(
            source_audio_channels=2,
            source_layout="dual_mono",
            detected_language=detected_language,
            transcript=transcript.strip(),
            operator_text=None,
            client_text=None,
            system_text=None,
            role_separation="dual_track",
            formatted_text="Транскрипт по каналам (Whisper, rx/tx):",
            note_ru=note,
            diarization_used=False,
            diarized_segments=None,
        )
    roles_cat = get_speaker_roles_catalog_live(
        settings.speaker_roles_catalog_path,
        settings.speaker_roles_catalog_enabled,
    )
    system_phrases = roles_cat.phrases_system if roles_cat is not None else None
    ivr_early_max = roles_cat.ivr_early_segment_max_start_sec if roles_cat is not None else None

    diarized_segments: list[DiarizedSegment] | None = None
    role_by: dict[str, str] = {}
    if result.segments and (result.diarization_used or result.dual_track):
        role_by = effective_role_map(result.segments, result.speaker_role_map)
        diarized_segments = []
        prev_end: float | None = None
        for sp, s0, s1, tx in result.segments:
            pause_before = 0.0 if prev_end is None else max(0.0, float(s0) - float(prev_end))
            prev_end = float(s1)
            r = effective_segment_role(
                sp,
                s0,
                tx,
                role_by,
                system_phrases=system_phrases,
                ivr_early_segment_max_start_sec=ivr_early_max,
            )
            diarized_segments.append(
                DiarizedSegment(
                    speaker=sp,
                    role=r,
                    role_label_ru=role_label_ru(r),
                    ivr_hint_match=text_matches_system_ivr(tx, system_phrases),
                    start_sec=s0,
                    end_sec=s1,
                    text=tx,
                    pause_before_sec=pause_before,
                )
            )

        cl_text, op_text, _sys_unused = aggregate_by_role(
            result.segments,
            role_by,
            system_phrases=system_phrases,
            ivr_early_segment_max_start_sec=ivr_early_max,
        )

        if result.dual_track:
            role = "dual_track"
            direction_ru = (
                "входящий (rx — клиент, tx — оператор)"
                if result.dual_call_direction == "incoming"
                else "исходящий (rx — оператор, tx — клиент)"
            )
            note = (
                f"Два отдельных mono-файла (url_rx / url_tx), звонок {direction_ru}. "
                "Каждый канал распознан отдельно Whisper; сегменты отсортированы по времени начала. "
                "Длинные паузы внутри одной фразы модели режутся на отдельные элементы (см. WHISPER_INTRA_SEGMENT_SPLIT_GAP_SEC); "
                "pause_before_sec — зазор до сегмента на общей шкале для UI. "
                "ivr_hint_match — эвристика по типичным фразам IVR (как при диаризации)."
            )
            if result.dual_mix_sync_applied:
                how = "вручную (sync.mode=manual)" if result.dual_mix_sync_mode == "manual" else "по url_mix (корреляция огибающих)"
                note += (
                    f" Выравнивание {how}: смещения RX/TX — mix_sync_offset_*; "
                    "низкие mix_sync_score_* при auto — слабая уверенность (см. WHISPER_CHANNEL_SYNC_MIN_CORR)."
                )
            lines = ["Транскрипт по каналам (Whisper, rx/tx):"]
            for sp, s0, s1, tx in result.segments:
                r = effective_segment_role(
                    sp,
                    s0,
                    tx,
                    role_by,
                    system_phrases=system_phrases,
                    ivr_early_segment_max_start_sec=ivr_early_max,
                )
                label = role_label_ru(r)
                hint = " [маркеры IVR]" if text_matches_system_ivr(tx, system_phrases) else ""
                lines.append(f"[{s0:.2f}–{s1:.2f}s] {label}{hint} ({sp}): {tx}")
            formatted = "\n".join(lines).strip()
            return TranscribeResponse(
                source_audio_channels=2,
                source_layout="dual_mono",
                detected_language=detected_language,
                transcript=transcript.strip(),
                operator_text=op_text,
                client_text=cl_text,
                system_text=None,
                role_separation=role,
                formatted_text=formatted,
                note_ru=note,
                diarization_used=False,
                diarized_segments=diarized_segments,
                mix_sync_mode_used=result.dual_mix_sync_mode if result.dual_mix_sync_applied else None,
                mix_sync_offset_rx_sec=result.dual_mix_sync_offset_rx_sec,
                mix_sync_offset_tx_sec=result.dual_mix_sync_offset_tx_sec,
                mix_sync_score_rx=result.dual_mix_sync_score_rx,
                mix_sync_score_tx=result.dual_mix_sync_score_tx,
            )

        role = "diarized"
        note = (
            "Диаризация pyannote.audio: оператор (включая IVR и приветствие линии) — по фразам в match_window_sec "
            "и по ранним маркерам из каталога WHISPER_SPEAKER_ROLES_CATALOG_PATH (system_phrase_window_sec, "
            "system_speaker_first_start_before_sec). Ранняя IVR-реплика при моно в одном кластере с клиентом "
            "до ivr_early_segment_max_start_sec с тоже относится к оператору. "
            "Если совпадений мало — схема SPEAKER_00=Клиент, SPEAKER_01/02=Оператор. "
            "pause_before_sec — пауза на шкале времени перед сегментом (для подсветки в UI). "
            "ivr_hint_match: текст похож на типичные фразы IVR из каталога (подсказка)."
        )
        if stereo_mixed:
            note = (
                "Вход был стерео и сведён в mono для распознавания; диаризация выполнялась по этому mono. " + note
            )
        lines = ["Транскрипт по спикерам (Whisper + pyannote):"]
        for sp, s0, s1, tx in result.segments:
            r = effective_segment_role(
                sp,
                s0,
                tx,
                role_by,
                system_phrases=system_phrases,
                ivr_early_segment_max_start_sec=ivr_early_max,
            )
            label = role_label_ru(r)
            hint = " [маркеры IVR]" if text_matches_system_ivr(tx, system_phrases) else ""
            lines.append(f"[{s0:.2f}–{s1:.2f}s] {label}{hint} ({sp}): {tx}")
        formatted = "\n".join(lines).strip()
        return TranscribeResponse(
            source_audio_channels=source_channels,
            source_layout="stereo" if stereo_mixed else "mono",
            detected_language=detected_language,
            transcript=transcript.strip(),
            operator_text=op_text,
            client_text=cl_text,
            system_text=None,
            role_separation=role,
            formatted_text=formatted,
            note_ru=note,
            diarization_used=True,
            diarized_segments=diarized_segments,
        )

    if stereo_mixed:
        role = "unavailable_stereo_mixed"
        note = (
            "Вход был многоканальным: для распознавания дорожки сведены в mono. "
            "Разделение «Оператор» / «Клиент» появится, когда будут отдельные rx/tx или стерео без сведения."
        )
    else:
        role = "unavailable_mono"
        note = (
            "Запись моно: без отдельных каналов или диаризации нельзя надёжно разнести реплики на «Оператор» и «Клиент». "
            "Включите WHISPER_DIARIZATION=1 и HF_TOKEN (модели pyannote на Hugging Face) или раздельные каналы (rx/tx)."
        )
    formatted = "Транскрипт:\n" + (transcript or "").strip()
    return TranscribeResponse(
        source_audio_channels=source_channels,
        source_layout="stereo" if stereo_mixed else "mono",
        detected_language=detected_language,
        transcript=transcript.strip(),
        operator_text=None,
        client_text=None,
        system_text=None,
        role_separation=role,
        formatted_text=formatted.strip(),
        note_ru=note,
        diarization_used=False,
        diarized_segments=None,
    )
