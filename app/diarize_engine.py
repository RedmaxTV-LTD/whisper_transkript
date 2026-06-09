"""Загрузка pyannote.audio и диаризация моно-WAV (16 kHz)."""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.settings import get_settings

# pyannote → lightning_fabric вызывает torch.load(..., weights_only=False) для весов с HF;
# файлы доверенные; иначе PyTorch засыпает лог длинным предупреждением на каждый старт.
warnings.filterwarnings(
    "ignore",
    message=r"You are using .*torch\.load.*weights_only=False",
)

log = logging.getLogger(__name__)

_pipeline: Any = None
_pipeline_lock = asyncio.Lock()
_pipeline_failed: str | None = None


def get_diarization_pipeline() -> Any:
    if _pipeline is None:
        raise RuntimeError("diarization_pipeline_not_loaded")
    return _pipeline


def is_diarization_pipeline_ready() -> bool:
    return _pipeline is not None


def diarization_pipeline_error() -> str | None:
    return _pipeline_failed


async def ensure_diarization_pipeline_loaded() -> None:
    global _pipeline, _pipeline_failed
    settings = get_settings()
    if not settings.diarization_enabled:
        return
    if _pipeline is not None:
        return
    if _pipeline_failed is not None:
        return
    async with _pipeline_lock:
        if _pipeline is not None or _pipeline_failed is not None:
            return
        token = settings.hf_token
        if not token:
            _pipeline_failed = "missing_hf_token"
            log.error("WHISPER_DIARIZATION=1 but HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) is empty")
            return

        log.info(
            "loading_pyannote_pipeline model=%s device=%s",
            settings.pyannote_pipeline,
            settings.pyannote_device,
        )

        def _load() -> Any:
            import os

            import torch
            from pyannote.audio import Pipeline

            # Часть вложенных hf_hub_download в pyannote подхватывает только переменные окружения.
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token

            p = Pipeline.from_pretrained(settings.pyannote_pipeline, use_auth_token=token)
            # pyannote при RepositoryNotFoundError (нет доступа к gated-репо, неверный токен и т.п.)
            # возвращает None вместо исключения — см. pyannote/audio/core/pipeline.py
            if p is None:
                raise RuntimeError(
                    "pyannote_from_pretrained_returned_none: "
                    "проверьте HF_TOKEN, примите условия на странице модели на Hugging Face "
                    f"({settings.pyannote_pipeline}) и id пайплайна в PYANNOTE_PIPELINE"
                )
            dev = torch.device(settings.pyannote_device)
            p.to(dev)
            return p

        loop = asyncio.get_running_loop()
        try:
            loaded = await loop.run_in_executor(None, _load)
        except Exception as e:
            err = f"{type(e).__name__}:{e}"
            hint = ""
            if isinstance(e, AttributeError) and "NoneType" in str(e) and "eval" in str(e):
                hint = (
                    " | gated_or_auth: HF отдаёт GatedRepoError как «нет репо» — "
                    "примите условия на https://huggingface.co/pyannote/segmentation-3.0 "
                    "и https://huggingface.co/pyannote/speaker-diarization-3.1 под тем же пользователем, "
                    "что и HF_TOKEN; при сомнениях очистите кэш: "
                    "rm -rf ~/.cache/huggingface/hub/models--pyannote--segmentation-3.0*"
                )
            _pipeline_failed = f"load_failed:{err}{hint}"
            log.exception("pyannote_pipeline_load_failed")
            return
        _pipeline = loaded
        log.info("pyannote_pipeline_ready device=%s model=%s", settings.pyannote_device, settings.pyannote_pipeline)


def run_diarization(wav_path: str) -> list[tuple[float, float, str]]:
    """Возвращает отсортированные по времени интервалы (start_sec, end_sec, speaker_label)."""
    pipeline = get_diarization_pipeline()
    diar = pipeline({"uri": "utterance", "audio": wav_path})
    rows: list[tuple[float, float, str]] = []
    for segment, _track, label in diar.itertracks(yield_label=True):
        rows.append((float(segment.start), float(segment.end), str(label)))
    rows.sort(key=lambda x: x[0])
    return rows
