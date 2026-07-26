"""CPU / RAM / GPU метрики процесса (cgroup + /proc; без psutil).

Важно: в процессе whisper-worker с уже загруженной CUDA вызов nvidia-smi
часто зависает навсегда. Для worker используйте collect_worker_metrics().
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ContainerMetrics:
    cpu_percent: float | None
    mem_used_mb: float | None
    mem_limit_mb: float | None
    mem_percent: float | None
    gpu_util_percent: float | None
    gpu_mem_used_mb: float | None
    gpu_mem_total_mb: float | None
    gpu_name: str | None
    gpu_mem_free_mb: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NvidiaSmiSnapshot:
    """Снимок nvidia-smi (realtime), безопасен из API-процесса без CUDA-модели."""

    memory_total_mb: float | None
    memory_used_mb: float | None
    memory_free_mb: float | None
    utilization_gpu_percent: float | None
    name: str | None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.memory_total_mb and self.memory_used_mb is not None and self.memory_total_mb > 0:
            d["memory_used_percent"] = round(min(100.0, self.memory_used_mb / self.memory_total_mb * 100.0), 1)
        else:
            d["memory_used_percent"] = None
        return d


# (usage_usec_or_ticks, monotonic_sec) — для дельты CPU между опросами.
_prev_cpu: tuple[float, float] | None = None
_prev_cpu_lock = __import__("threading").Lock()
_smi_rt_cache: tuple[float, NvidiaSmiSnapshot] | None = None
_SMI_RT_CACHE_TTL_SEC = 3.0  # UI опрашивает раз в 3с — сглаживаем скачки util


def query_nvidia_smi_realtime(*, use_cache: bool = True) -> NvidiaSmiSnapshot | None:
    """nvidia-smi --query-gpu=memory.total,memory.used,memory.free (+ util, name).

    Вызывать из API (whisper-stt), не из worker с загруженной моделью.
    """
    global _smi_rt_cache
    now = time.monotonic()
    if use_cache and _smi_rt_cache is not None and now - _smi_rt_cache[0] < _SMI_RT_CACHE_TTL_SEC:
        return _smi_rt_cache[1]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception as e:
        log.warning("nvidia_smi_realtime_failed err=%s", type(e).__name__)
        return _smi_rt_cache[1] if _smi_rt_cache else None
    if out.returncode != 0 or not out.stdout.strip():
        return _smi_rt_cache[1] if _smi_rt_cache else None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 5:
        return _smi_rt_cache[1] if _smi_rt_cache else None
    try:
        snap = NvidiaSmiSnapshot(
            memory_total_mb=float(parts[0]),
            memory_used_mb=float(parts[1]),
            memory_free_mb=float(parts[2]),
            utilization_gpu_percent=float(parts[3]),
            name=parts[4] or None,
        )
    except ValueError:
        return _smi_rt_cache[1] if _smi_rt_cache else None
    _smi_rt_cache = (now, snap)
    return snap


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _parse_kb_status(status: str, key: str) -> float | None:
    for line in status.splitlines():
        if line.startswith(key):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0
                except ValueError:
                    return None
    return None


def _cgroup_memory() -> tuple[float | None, float | None]:
    v2_cur = Path("/sys/fs/cgroup/memory.current")
    v2_max = Path("/sys/fs/cgroup/memory.max")
    if v2_cur.is_file():
        raw_cur = _read_text(v2_cur)
        raw_max = _read_text(v2_max)
        used = float(raw_cur) / (1024 * 1024) if raw_cur and raw_cur.isdigit() else None
        limit: float | None = None
        if raw_max and raw_max != "max" and raw_max.isdigit():
            limit = float(raw_max) / (1024 * 1024)
        return used, limit

    v1_usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    v1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1_usage.is_file():
        raw_u = _read_text(v1_usage)
        raw_l = _read_text(v1_limit)
        used = float(raw_u) / (1024 * 1024) if raw_u and raw_u.isdigit() else None
        limit = None
        if raw_l and raw_l.isdigit():
            lim = float(raw_l) / (1024 * 1024)
            if lim < 1e9:
                limit = lim
        return used, limit
    return None, None


def _cgroup_cpu_count() -> float | None:
    """Число CPU из cgroup quota (например 2.0 при cpus: 2). None = без лимита."""
    v2 = Path("/sys/fs/cgroup/cpu.max")
    if v2.is_file():
        raw = _read_text(v2)
        if not raw or raw.startswith("max"):
            return None
        parts = raw.split()
        if len(parts) >= 2:
            try:
                quota, period = float(parts[0]), float(parts[1])
                if period > 0 and quota >= 0:
                    return quota / period
            except ValueError:
                return None
        return None

    quota_p = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_p.is_file() and period_p.is_file():
        raw_q, raw_p = _read_text(quota_p), _read_text(period_p)
        try:
            quota = float(raw_q) if raw_q else -1.0
            period = float(raw_p) if raw_p else 0.0
        except ValueError:
            return None
        if quota < 0 or period <= 0:
            return None
        return quota / period
    return None


def _cgroup_cpu_usage_usec() -> float | None:
    """Накопленное CPU-время контейнера в микросекундах (весь cgroup, не только main PID)."""
    v2 = Path("/sys/fs/cgroup/cpu.stat")
    if v2.is_file():
        raw = _read_text(v2)
        if not raw:
            return None
        for line in raw.splitlines():
            if line.startswith("usage_usec"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        return None
        return None
    # cgroup v1: наносекунды
    v1 = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if v1.is_file():
        raw = _read_text(v1)
        if raw and raw.isdigit():
            return float(raw) / 1000.0
    return None


def _process_rss_mb() -> float | None:
    status = _read_text(Path("/proc/self/status"))
    if not status:
        return None
    return _parse_kb_status(status, "VmRSS:")


def _process_cpu_usage_usec() -> float | None:
    """Fallback: utime+stime процесса → микросекунды."""
    try:
        with open("/proc/self/stat", encoding="utf-8") as f:
            fields = f.read().split()
        ticks = float(int(fields[13]) + int(fields[14]))
    except Exception:
        return None
    try:
        hz = float(os.sysconf("SC_CLK_TCK"))
    except Exception:
        hz = 100.0
    if hz <= 0:
        return None
    return ticks / hz * 1_000_000.0


def _cpu_percent() -> float | None:
    """Загрузка CPU контейнера: 100% = полная квота (cpus: 2 → оба ядра).

    Первый образец — None (не 0.0), иначе UI залипает на нуле до второй публикации.
    """
    global _prev_cpu
    usage = _cgroup_cpu_usage_usec()
    if usage is None:
        usage = _process_cpu_usage_usec()
    if usage is None:
        return None

    now = time.monotonic()
    with _prev_cpu_lock:
        if _prev_cpu is None:
            _prev_cpu = (usage, now)
            return None

        prev_usage, prev_wall = _prev_cpu
        _prev_cpu = (usage, now)
        dt = now - prev_wall
        if dt <= 0:
            return None
        # usage_usec дельта → секунды CPU / wall → доля одного ядра * 100.
        delta_sec = (usage - prev_usage) / 1_000_000.0
        if delta_sec < 0:
            return None
        pct = (delta_sec / dt) * 100.0
    ncpus = _cgroup_cpu_count()
    if ncpus and ncpus > 0:
        pct = pct / ncpus
    return max(0.0, round(min(100.0, pct), 1))


def _cpu_ram_only() -> tuple[float | None, float | None, float | None, float | None]:
    used_cg, limit_cg = _cgroup_memory()
    rss = _process_rss_mb()
    mem_used = used_cg if used_cg is not None else rss
    mem_limit = limit_cg
    mem_pct: float | None = None
    if mem_used is not None and mem_limit and mem_limit > 0:
        mem_pct = round(min(100.0, mem_used / mem_limit * 100.0), 1)
    return (
        _cpu_percent(),
        round(mem_used, 1) if mem_used is not None else None,
        round(mem_limit, 1) if mem_limit is not None else None,
        mem_pct,
    )


def _torch_gpu_metrics() -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """VRAM через уже открытый CUDA-контекст (безопасно в worker). util% здесь нет."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, None, None, None
        free_b, total_b = torch.cuda.mem_get_info(0)
        used_mb = (total_b - free_b) / (1024 * 1024)
        total_mb = total_b / (1024 * 1024)
        free_mb = free_b / (1024 * 1024)
        name = torch.cuda.get_device_name(0)
        return None, round(used_mb, 1), round(total_mb, 1), round(free_mb, 1), name
    except Exception:
        return None, None, None, None, None


def _nvidia_smi() -> tuple[float | None, float | None, float | None, str | None]:
    """Legacy helper; предпочтительно query_nvidia_smi_realtime() из API."""
    snap = query_nvidia_smi_realtime(use_cache=True)
    if snap is None:
        return None, None, None, None
    return snap.utilization_gpu_percent, snap.memory_used_mb, snap.memory_total_mb, snap.name


def collect_container_metrics(include_gpu: bool = True, force_gpu: bool = False) -> ContainerMetrics:
    """Метрики для API-процесса. GPU через nvidia-smi realtime."""
    cpu, mem_used, mem_limit, mem_pct = _cpu_ram_only()
    gpu_u = gpu_mu = gpu_mt = gpu_free = gpu_name = None
    if include_gpu:
        snap = query_nvidia_smi_realtime(use_cache=not force_gpu)
        if snap is not None:
            gpu_u = snap.utilization_gpu_percent
            gpu_mu = snap.memory_used_mb
            gpu_mt = snap.memory_total_mb
            gpu_free = snap.memory_free_mb
            gpu_name = snap.name
    return ContainerMetrics(
        cpu_percent=cpu,
        mem_used_mb=mem_used,
        mem_limit_mb=mem_limit,
        mem_percent=mem_pct,
        gpu_util_percent=gpu_u,
        gpu_mem_used_mb=gpu_mu,
        gpu_mem_total_mb=gpu_mt,
        gpu_name=gpu_name,
        gpu_mem_free_mb=gpu_free,
    )


_cached_gpu_total_mb: float | None = None
_cached_gpu_name: str | None = None


def cache_worker_gpu_total(*, force: bool = False) -> float | None:
    """Однократный снимок total VRAM (только на старте / idle). Не вызывать во время инференса."""
    global _cached_gpu_total_mb, _cached_gpu_name
    if not force and _cached_gpu_total_mb is not None:
        return _cached_gpu_total_mb
    _u, _used, total, _free, name = _torch_gpu_metrics()
    if total is not None:
        _cached_gpu_total_mb = total
        _cached_gpu_name = name
    return _cached_gpu_total_mb


def get_cached_gpu_total_mb() -> float | None:
    return _cached_gpu_total_mb


def collect_worker_metrics(*, include_gpu: bool = False) -> ContainerMetrics:
    """CPU/RAM worker. GPU по умолчанию выкл: torch.cuda.mem_get_info в event loop
    блокируется на время инференса и роняет whisper:worker:alive.
    Live VRAM/util — из API (nvidia-smi). Total VRAM — из кэша после cache_worker_gpu_total().
    """
    global _cached_gpu_total_mb, _cached_gpu_name
    cpu, mem_used, mem_limit, mem_pct = _cpu_ram_only()
    gpu_u = gpu_mu = gpu_free = None
    gpu_mt = _cached_gpu_total_mb
    gpu_name = _cached_gpu_name
    if include_gpu:
        gpu_u, gpu_mu, gpu_mt, gpu_free, gpu_name = _torch_gpu_metrics()
        if gpu_mt is not None:
            _cached_gpu_total_mb = gpu_mt
            _cached_gpu_name = gpu_name
    return ContainerMetrics(
        cpu_percent=cpu,
        mem_used_mb=mem_used,
        mem_limit_mb=mem_limit,
        mem_percent=mem_pct,
        gpu_util_percent=gpu_u,
        gpu_mem_used_mb=gpu_mu,
        gpu_mem_total_mb=gpu_mt,
        gpu_name=gpu_name,
        gpu_mem_free_mb=gpu_free,
    )
