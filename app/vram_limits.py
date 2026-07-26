"""Оценка макс. параллелизма local GPU по VRAM."""

from __future__ import annotations

from pathlib import Path


# Грубая оценка VRAM на один слот (модель + активации) для float16 faster-whisper.
_MODEL_SLOT_MB: dict[str, int] = {
    "tiny": 1200,
    "tiny.en": 1200,
    "base": 1400,
    "base.en": 1400,
    "small": 2200,
    "small.en": 2200,
    "medium": 4200,
    "medium.en": 4200,
    "large-v2": 6500,
    "large-v3": 6500,
    "large": 6500,
    "turbo": 4800,
    "distil-large-v3": 4500,
}


def estimate_vram_per_slot_mb(model_path: str, *, override_mb: int | None = None) -> int:
    if override_mb is not None and override_mb > 0:
        return int(override_mb)
    name = Path(model_path.rstrip("/")).name.lower()
    for key, mb in _MODEL_SLOT_MB.items():
        if key in name:
            return mb
    return 4800  # conservative default (turbo-like)


def estimate_max_local_slots(
    gpu_mem_total_mb: float | None,
    *,
    model_path: str,
    vram_per_slot_mb: int | None = None,
    reserve_mb: float = 800.0,
) -> int:
    """Сколько одновременных local-задач влезает в VRAM (минимум 1)."""
    per = estimate_vram_per_slot_mb(model_path, override_mb=vram_per_slot_mb)
    if gpu_mem_total_mb is None or gpu_mem_total_mb <= 0:
        return 1
    usable = max(0.0, float(gpu_mem_total_mb) - float(reserve_mb))
    n = int(usable // per)
    return max(1, n)
