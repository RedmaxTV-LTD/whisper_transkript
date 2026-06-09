"""Сопоставление сегментов Whisper с интервалами диаризации pyannote."""

from __future__ import annotations


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(
    seg_start: float,
    seg_end: float,
    turns: list[tuple[float, float, str]],
) -> str:
    """Назначает спикера по максимальному пересечению по времени; иначе — ближайший интервал по центру сегмента."""
    if not turns:
        return "UNKNOWN"
    best_sp, best_o = "UNKNOWN", 0.0
    for t0, t1, sp in turns:
        o = _overlap(seg_start, seg_end, t0, t1)
        if o > best_o:
            best_o, best_sp = o, sp
    if best_o > 1e-6:
        return best_sp
    mid = 0.5 * (seg_start + seg_end)

    def dist_to_turn(item: tuple[float, float, str]) -> float:
        t0, t1, _ = item
        if t0 <= mid <= t1:
            return 0.0
        return min(abs(mid - t0), abs(mid - t1))

    t0, t1, sp = min(turns, key=dist_to_turn)
    return sp


def speakers_chronological_order(turns: list[tuple[float, float, str]]) -> list[str]:
    """Порядок первого появления спикера по времени (для эвристики Оператор/Клиент)."""
    first: dict[str, float] = {}
    for t0, t1, sp in sorted(turns, key=lambda x: x[0]):
        if sp not in first:
            first[sp] = t0
    return sorted(first.keys(), key=lambda s: first[s])


def align_whisper_segments(
    whisper_parts: list[tuple[float, float, str]],
    diar_turns: list[tuple[float, float, str]],
) -> list[tuple[str, float, float, str]]:
    """Список (speaker_id, start, end, text) в порядке Whisper."""
    out: list[tuple[str, float, float, str]] = []
    for ws, we, text in whisper_parts:
        sp = assign_speaker(ws, we, diar_turns)
        out.append((sp, ws, we, text))
    return out
