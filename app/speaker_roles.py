"""Сопоставление идентификаторов pyannote с ролями колл-центра и эвристики для реплик системы/IVR."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Role = Literal["client", "operator", "other"]

# Фиксированная схема для записей Redmax / колл-центра:
# SPEAKER_00 — клиент, SPEAKER_01/02 — оператор (IVR и линия — в одной роли «оператор»).
SPEAKER_TO_ROLE: dict[str, Role] = {
    "SPEAKER_00": "client",
    "SPEAKER_01": "operator",
    "SPEAKER_02": "operator",
}

ROLE_LABEL_RU: dict[str, str] = {
    "client": "Клиент",
    "operator": "Оператор",
    "other": "Прочее",
}


def role_for_speaker(speaker_id: str, role_by_speaker: dict[str, Role] | None = None) -> Role:
    if role_by_speaker is not None:
        return role_by_speaker.get(speaker_id, "other")
    return SPEAKER_TO_ROLE.get(speaker_id, "other")


def effective_role_map(
    segments: list[tuple[str, float, float, str]],
    inferred: dict[str, Role] | None,
) -> dict[str, Role]:
    """Словарь SPEAKER_* → роль: из инференса по каталогу или фиксированная схема pyannote."""
    speakers = sorted({sp for sp, _, _, _ in segments})
    if inferred is None:
        return {sp: SPEAKER_TO_ROLE.get(sp, "other") for sp in speakers}
    return {sp: inferred.get(sp, "other") for sp in speakers}


def role_label_ru(role: str) -> str:
    return ROLE_LABEL_RU.get(role, ROLE_LABEL_RU["other"])


def effective_segment_role(
    speaker_id: str,
    start_sec: float,
    text: str,
    role_by_speaker: dict[str, Role],
    *,
    system_phrases: tuple[str, ...] | None = None,
    ivr_early_segment_max_start_sec: float | None = None,
) -> Role:
    """Роль для одной реплики. Реплики IVR/«система» относятся к оператору (в т.ч. ранняя фраза при моно)."""
    base = role_for_speaker(speaker_id, role_by_speaker)
    if base == "operator":
        return "operator"
    if (
        ivr_early_segment_max_start_sec is not None
        and system_phrases is not None
        and start_sec < ivr_early_segment_max_start_sec
        and text_matches_system_ivr(text, system_phrases)
    ):
        return "operator"
    if base == "other":
        return "client"
    return base


def _normalize_for_match(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).casefold()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Фразы, по которым реплику удобнее отнести к системе/IVR (точное определение — по SPEAKER_02;
# совпадение используется как дополнительная проверка и подсказка в ответе API).
_SYSTEM_PHRASES = (
    "время ожидания",
    "редмакс",
    "рэдмакс",
    "radmax",
    "ожидайте",
    "оставайтесь на линии",
    "не кладите трубку",
    "все операторы заняты",
    "операторы заняты",
    "свободный оператор",
    "перевожу на оператора",
    "соединяю с оператором",
    "ваш звонок очень важен",
    "вы позвонили",
)


def text_matches_system_ivr(text: str, system_phrases: tuple[str, ...] | None = None) -> bool:
    """Текст похож на фразы роли «система» из каталога или встроенный список."""
    if not text or not text.strip():
        return False
    n = _normalize_for_match(text)
    phrases = system_phrases if system_phrases is not None else tuple(p.casefold() for p in _SYSTEM_PHRASES)
    return any(p in n for p in phrases)


def aggregate_by_role(
    segments: list[tuple[str, float, float, str]],
    role_by_speaker: dict[str, Role],
    *,
    system_phrases: tuple[str, ...] | None = None,
    ivr_early_segment_max_start_sec: float | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Собирает тексты по ролям: клиент / оператор / система; прочие спикеры — в хвост клиента.

    Если заданы system_phrases и ivr_early_segment_max_start_sec, ранняя IVR-реплика у спикера «клиент»
    учитывается как оператор (см. effective_segment_role). Остатки роли system сливаются в оператора.
    """
    from collections import defaultdict

    by_role: dict[str, list[str]] = defaultdict(list)
    other: list[str] = []

    for sp, s0, _e, txt in segments:
        t = txt.strip()
        if not t:
            continue
        role = effective_segment_role(
            sp,
            s0,
            txt,
            role_by_speaker,
            system_phrases=system_phrases,
            ivr_early_segment_max_start_sec=ivr_early_segment_max_start_sec,
        )
        if role == "other":
            other.append(t)
        else:
            by_role[role].append(t)

    if other:
        by_role["client"].extend(other)

    if by_role.get("system"):
        by_role["operator"].extend(by_role["system"])
        by_role["system"].clear()

    def join_role(r: str) -> str | None:
        s = " ".join(by_role.get(r, [])).strip()
        return s or None

    return join_role("client"), join_role("operator"), join_role("system")
