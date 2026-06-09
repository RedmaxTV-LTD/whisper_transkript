"""Подготовка аудио: скачивание по URL, нормализация в WAV для Whisper."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import httpx

from app.settings import get_settings


async def download_url_to_file(url: str, dest: Path) -> None:
    settings = get_settings()
    timeout = httpx.Timeout(settings.download_timeout_sec)
    limits = httpx.Limits(max_keepalive_connections=2, max_connections=4)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise ValueError("download_exceeds_max_bytes")
                    f.write(chunk)


def _ffprobe_channels(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if not out:
        return 1
    try:
        return max(1, int(float(out)))
    except ValueError:
        return 1


def normalize_to_wav_16k_mono(src: Path, dst_wav: Path) -> tuple[int, str]:
    ch = _ffprobe_channels(src)
    layout = "stereo" if ch >= 2 else "mono"
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(dst_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return ch, layout


async def prepare_dual_rx_tx_from_urls(
    url_rx: str,
    url_tx: str,
    url_mix: str | None = None,
) -> tuple[Path, Path, Path | None]:
    """Скачивает два URL (и опционально mix), нормализует в mono 16 kHz WAV в одном temp-каталоге."""
    tmp = Path(tempfile.mkdtemp(prefix="whisper_"))
    rx_raw = tmp / "rx.bin"
    tx_raw = tmp / "tx.bin"
    rx_wav = tmp / "rx.wav"
    tx_wav = tmp / "tx.wav"
    mix_raw = tmp / "mix.bin" if url_mix else None
    mix_wav = tmp / "mix.wav" if url_mix else None
    await download_url_to_file(url_rx, rx_raw)
    await download_url_to_file(url_tx, tx_raw)
    if url_mix and mix_raw is not None and mix_wav is not None:
        await download_url_to_file(url_mix, mix_raw)
    loop = asyncio.get_running_loop()

    def _run() -> None:
        normalize_to_wav_16k_mono(rx_raw, rx_wav)
        normalize_to_wav_16k_mono(tx_raw, tx_wav)
        if url_mix and mix_raw is not None and mix_wav is not None:
            normalize_to_wav_16k_mono(mix_raw, mix_wav)

    await loop.run_in_executor(None, _run)
    try:
        rx_raw.unlink(missing_ok=True)
        tx_raw.unlink(missing_ok=True)
        if mix_raw is not None:
            mix_raw.unlink(missing_ok=True)
    except OSError:
        pass
    return rx_wav, tx_wav, mix_wav


async def prepare_audio_from_url(url: str) -> tuple[Path, int, str]:
    tmp = Path(tempfile.mkdtemp(prefix="whisper_"))
    raw = tmp / "input.bin"
    wav = tmp / "speech.wav"
    await download_url_to_file(url, raw)
    loop = asyncio.get_running_loop()

    def _run() -> tuple[int, str]:
        return normalize_to_wav_16k_mono(raw, wav)

    channels, layout = await loop.run_in_executor(None, _run)
    try:
        raw.unlink(missing_ok=True)
    except OSError:
        pass
    return wav, channels, layout


def cleanup_temp_dir(wav_path: Path) -> None:
    try:
        root = wav_path.parent
        if root.name.startswith("whisper_"):
            for p in root.glob("*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                root.rmdir()
            except OSError:
                pass
    except Exception:
        pass
