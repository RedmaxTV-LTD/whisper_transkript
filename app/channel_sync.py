"""Оценка сдвига RX/TX относительно общей mono (url_mix) по огибающей сигнала."""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from app.settings import Settings

log = logging.getLogger(__name__)


def _downsample_envelope(samples: np.ndarray, step: int) -> np.ndarray:
    """Абсолютное значение + усреднение по блокам step сэмплов (16 kHz → step)."""
    x = np.abs(samples.astype(np.float64, copy=False))
    n = (len(x) // step) * step
    if n < step * 4:
        return np.array([], dtype=np.float64)
    x = x[:n].reshape(-1, step).mean(axis=1)
    x -= x.mean()
    s = x.std()
    if s < 1e-9:
        return np.array([], dtype=np.float64)
    return x / s


def _best_lag_pearson(mix: np.ndarray, ch: np.ndarray, max_lag_ds: int) -> tuple[int, float]:
    """Ищем lag (в индексах даунсэмпла), при котором mix[lag:] накладывается на ch[:]. Возвращаем (lag, score)."""
    best_lag = 0
    best_score = -2.0
    n_m, n_c = len(mix), len(ch)
    for lag in range(-max_lag_ds, max_lag_ds + 1):
        if lag >= 0:
            L = min(n_m - lag, n_c)
            if L < 20:
                continue
            a = mix[lag : lag + L]
            b = ch[0:L]
        else:
            skip = -lag
            L = min(n_m, n_c - skip)
            if L < 20:
                continue
            a = mix[0:L]
            b = ch[skip : skip + L]
        score = float(np.dot(a, b) / len(a))
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag, best_score


def estimate_rx_tx_offsets_vs_mix(
    mix_wav_path: str,
    rx_wav_path: str,
    tx_wav_path: str,
    settings: Settings,
    *,
    max_offset_sec: float | None = None,
) -> tuple[float, float, float | None, float | None, bool, bool]:
    """
    Секунды, которые нужно **прибавить** к таймштампам Whisper на RX/TX,
    чтобы выровнять их на общую шкалу **mix** (mono сводка).

    Возвращает (offset_rx_sec, offset_tx_sec, score_rx, score_tx, trusted_rx, trusted_tx).
    Канал «не доверен» (trusted=False), если корреляция ниже порога или |offset| > max_offset_sec
    (тогда offset для канала обнуляется).
    """
    step = max(16, settings.channel_sync_downsample_step)
    max_sec = max(5.0, settings.channel_sync_correlation_max_sec)
    max_lag_sec = max(0.5, settings.channel_sync_max_lag_sec)
    sr = 16000

    try:
        mix, _sr_m = sf.read(mix_wav_path, dtype="float64", always_2d=False)
        rx, _sr_r = sf.read(rx_wav_path, dtype="float64", always_2d=False)
        tx, _sr_t = sf.read(tx_wav_path, dtype="float64", always_2d=False)
    except OSError:
        log.warning("channel_sync_read_failed")
        return 0.0, 0.0, None, None, False, False

    def _mono(a: np.ndarray) -> np.ndarray:
        if a.ndim > 1:
            return np.asarray(a[:, 0], dtype=np.float64)
        return np.asarray(a, dtype=np.float64)

    mix = _mono(mix)
    rx = _mono(rx)
    tx = _mono(tx)

    max_samples = int(max_sec * sr)
    n_cap = min(len(mix), len(rx), len(tx), max_samples)
    if n_cap < sr:  # < 1 c
        log.warning("channel_sync_too_short n=%s", n_cap)
        return 0.0, 0.0, None, None, False, False

    mix = mix[:n_cap]
    rx = rx[:n_cap]
    tx = tx[:n_cap]

    mix_ds = _downsample_envelope(mix, step)
    rx_ds = _downsample_envelope(rx, step)
    tx_ds = _downsample_envelope(tx, step)
    if mix_ds.size < 50 or rx_ds.size < 50 or tx_ds.size < 50:
        log.warning("channel_sync_empty_envelope")
        return 0.0, 0.0, None, None, False, False

    n_ds = min(len(mix_ds), len(rx_ds), len(tx_ds))
    mix_ds = mix_ds[:n_ds]
    rx_ds = rx_ds[:n_ds]
    tx_ds = tx_ds[:n_ds]

    fs_ds = sr / float(step)
    max_lag_ds = int(max_lag_sec * fs_ds)
    max_lag_ds = max(1, min(max_lag_ds, n_ds // 4))

    lag_rx, score_rx = _best_lag_pearson(mix_ds, rx_ds, max_lag_ds)
    lag_tx, score_tx = _best_lag_pearson(mix_ds, tx_ds, max_lag_ds)

    raw_off_rx = lag_rx * step / float(sr)
    raw_off_tx = lag_tx * step / float(sr)

    thr = settings.channel_sync_min_correlation
    lim = max_offset_sec if max_offset_sec is not None else settings.sync_max_offset_sec

    trusted_rx = bool(score_rx >= thr and abs(raw_off_rx) <= lim)
    trusted_tx = bool(score_tx >= thr and abs(raw_off_tx) <= lim)

    if not trusted_rx:
        log.warning(
            "channel_sync_rx_untrusted score=%s thr=%s raw_off=%s lim=%s",
            score_rx,
            thr,
            raw_off_rx,
            lim,
        )
    if not trusted_tx:
        log.warning(
            "channel_sync_tx_untrusted score=%s thr=%s raw_off=%s lim=%s",
            score_tx,
            thr,
            raw_off_tx,
            lim,
        )

    offset_rx = raw_off_rx if trusted_rx else 0.0
    offset_tx = raw_off_tx if trusted_tx else 0.0

    log.info(
        "channel_sync offsets_rx=%.4f offsets_tx=%.4f raw_rx=%.4f raw_tx=%.4f score_rx=%.4f score_tx=%.4f trusted_rx=%s trusted_tx=%s",
        offset_rx,
        offset_tx,
        raw_off_rx,
        raw_off_tx,
        score_rx,
        score_tx,
        trusted_rx,
        trusted_tx,
    )
    return offset_rx, offset_tx, float(score_rx), float(score_tx), trusted_rx, trusted_tx
