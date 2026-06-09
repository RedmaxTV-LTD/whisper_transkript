"""Каталог шаблонов фраз для сопоставления pyannote SPEAKER_* с ролями (система / оператор / клиент)."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.speaker_roles import Role, SPEAKER_TO_ROLE, _normalize_for_match

log = logging.getLogger(__name__)

BUNDLED_CATALOG_PATH = Path(__file__).resolve().parent / "speaker_roles_catalog.json"

_lock = threading.Lock()
_cache: dict[str, Any] = {"path": None, "mtime": None, "catalog": None}


@dataclass(frozen=True)
class SpeakerRolesCatalog:
    """Параметры каталога: match_window — оператор; system_* — только начало звонка для роли IVR."""

    match_window_sec: float
    system_phrase_window_sec: float
    system_speaker_first_start_before_sec: float
    min_hits_system: int
    min_hits_operator: int
    max_system_speaker_total_sec: float
    max_system_speaker_segments: int
    ivr_early_segment_max_start_sec: float
    phrases_system: tuple[str, ...]
    phrases_operator: tuple[str, ...]
    phrases_client: tuple[str, ...]


def _resolve_catalog_path(primary: Path) -> Path:
    try:
        if primary.is_file():
            return primary
    except OSError:
        pass
    if BUNDLED_CATALOG_PATH.is_file():
        log.warning(
            "speaker_roles_catalog_primary_missing path=%s using_bundled=%s",
            primary,
            BUNDLED_CATALOG_PATH,
        )
        return BUNDLED_CATALOG_PATH
    return primary


def _load_catalog(path: Path) -> SpeakerRolesCatalog | None:
    try:
        raw = path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except FileNotFoundError:
        log.warning("speaker_roles_catalog_not_found path=%s", path)
        return None
    except (json.JSONDecodeError, OSError, ValueError) as e:
        log.warning("speaker_roles_catalog_load_failed path=%s err=%s", path, e)
        return None

    if not isinstance(data, dict):
        return None

    window = float(data.get("match_window_sec", 120))
    window = max(30.0, min(600.0, window))

    def _opt_float(key: str, default: float) -> float:
        v = data.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        return default

    # Система: только начало разговора (отдельно от окна оператора).
    sys_phrase_w = _opt_float("system_phrase_window_sec", min(45.0, window))
    sys_phrase_w = max(10.0, min(600.0, sys_phrase_w))
    sys_first_before = _opt_float("system_speaker_first_start_before_sec", min(60.0, window))
    sys_first_before = max(5.0, min(600.0, sys_first_before))

    mh = data.get("min_hits") if isinstance(data.get("min_hits"), dict) else {}
    min_sys = int(mh.get("system", 1)) if isinstance(mh.get("system"), (int, float)) else 1
    min_op = int(mh.get("operator", 1)) if isinstance(mh.get("operator"), (int, float)) else 1
    min_sys = max(1, min_sys)
    min_op = max(1, min_op)

    max_sys_total = _opt_float("max_system_speaker_total_sec", 45.0)
    max_sys_total = max(5.0, min(600.0, max_sys_total))
    mss = data.get("max_system_speaker_segments")
    if isinstance(mss, (int, float)):
        max_sys_seg = int(mss)
    else:
        max_sys_seg = 20
    max_sys_seg = max(2, min(500, max_sys_seg))

    ivr_early = _opt_float("ivr_early_segment_max_start_sec", 18.0)
    ivr_early = max(2.0, min(120.0, ivr_early))

    roles = data.get("roles")
    if not isinstance(roles, dict):
        return None

    def phrases_for(key: str) -> tuple[str, ...]:
        block = roles.get(key)
        if not isinstance(block, dict):
            return ()
        arr = block.get("phrases")
        if not isinstance(arr, list):
            return ()
        out: list[str] = []
        for x in arr:
            if isinstance(x, str) and (t := x.strip()):
                out.append(t.casefold())
        return tuple(dict.fromkeys(out))

    ps, po, pc = phrases_for("system"), phrases_for("operator"), phrases_for("client")
    if not ps and not po:
        return None

    cat = SpeakerRolesCatalog(
        match_window_sec=window,
        system_phrase_window_sec=sys_phrase_w,
        system_speaker_first_start_before_sec=sys_first_before,
        min_hits_system=min_sys,
        min_hits_operator=min_op,
        max_system_speaker_total_sec=max_sys_total,
        max_system_speaker_segments=max_sys_seg,
        ivr_early_segment_max_start_sec=ivr_early,
        phrases_system=ps,
        phrases_operator=po,
        phrases_client=pc,
    )
    log.info(
        "speaker_roles_catalog_loaded path=%s window_sec=%s system_phrase_window_sec=%s "
        "system_speaker_first_start_before_sec=%s max_system_total_sec=%s max_system_segments=%s "
        "ivr_early_segment_max_start_sec=%s phrases(system=%s operator=%s client=%s)",
        path,
        window,
        sys_phrase_w,
        sys_first_before,
        max_sys_total,
        max_sys_seg,
        ivr_early,
        len(ps),
        len(po),
        len(pc),
    )
    return cat


def get_speaker_roles_catalog_live(path: Path, enabled: bool) -> SpeakerRolesCatalog | None:
    if not enabled:
        return None
    resolved = _resolve_catalog_path(path)
    path_key = str(resolved.resolve())
    with _lock:
        try:
            mtime = resolved.stat().st_mtime
        except OSError:
            if _cache["path"] == path_key and _cache["catalog"] is not None:
                return _cache["catalog"]
            return None
        if _cache["path"] == path_key and _cache["mtime"] == mtime:
            return _cache["catalog"]
        cat = _load_catalog(resolved)
        _cache["path"] = path_key
        _cache["mtime"] = mtime
        _cache["catalog"] = cat
        return cat


def _hits(normalized_text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for p in phrases if p in normalized_text)


def _speaker_speech_stats(
    segments: list[tuple[str, float, float, str]], speaker: str
) -> tuple[float, int]:
    """Суммарная длительность сегментов и число непустых реплик (для отсечения «IVR+клиент» в одном SPEAKER_*)."""
    total_sec = 0.0
    n_nonempty = 0
    for sp, s0, s1, tx in segments:
        if sp != speaker:
            continue
        total_sec += max(0.0, float(s1) - float(s0))
        if tx.strip():
            n_nonempty += 1
    return total_sec, n_nonempty


def infer_speaker_role_map(
    segments: list[tuple[str, float, float, str]],
    catalog: SpeakerRolesCatalog,
) -> dict[str, Role]:
    """По тексту в начале записи назначает роли: система только по раннему окну и раннему первому выходу спикера."""
    if not segments:
        return {}

    first_start: dict[str, float] = {}
    for sp, s0, _s1, _tx in segments:
        if sp not in first_start or s0 < first_start[sp]:
            first_start[sp] = s0
    speakers = sorted(first_start.keys(), key=lambda s: first_start[s])

    op_window = catalog.match_window_sec
    sys_window = catalog.system_phrase_window_sec
    sys_first_deadline = catalog.system_speaker_first_start_before_sec

    buf_op: dict[str, list[str]] = {s: [] for s in speakers}
    buf_sys: dict[str, list[str]] = {s: [] for s in speakers}
    for sp, s0, _s1, tx in segments:
        t = tx.strip()
        if not t:
            continue
        if s0 < op_window:
            buf_op[sp].append(t)
        if s0 < sys_window:
            buf_sys[sp].append(t)

    scores_sys: dict[str, int] = {}
    scores_op: dict[str, int] = {}
    scores_cli: dict[str, int] = {}
    for sp in speakers:
        n_op = _normalize_for_match(" ".join(buf_op[sp]))
        n_sys = _normalize_for_match(" ".join(buf_sys[sp]))
        scores_sys[sp] = _hits(n_sys, catalog.phrases_system)
        scores_op[sp] = _hits(n_op, catalog.phrases_operator)
        scores_cli[sp] = _hits(n_op, catalog.phrases_client)

    max_sys = max(scores_sys.values(), default=0)
    max_op = max(scores_op.values(), default=0)

    if max_sys < catalog.min_hits_system and max_op < catalog.min_hits_operator:
        legacy: dict[str, Role] = {sp: SPEAKER_TO_ROLE.get(sp, "other") for sp in speakers}
        log.info("speaker_roles_inference_skipped using_legacy_map %s", legacy)
        return legacy

    assigned: set[str] = set()
    out: dict[str, Role] = {}

    # Кандидат в «голос IVR» в начале звонка — в ответе API относится к оператору (роль operator).
    sys_candidates = [
        s
        for s in speakers
        if first_start[s] < sys_first_deadline and scores_sys[s] >= catalog.min_hits_system
    ]
    if sys_candidates:
        system_sp = max(sys_candidates, key=lambda s: (scores_sys[s], -first_start[s]))
        tot_sec, n_seg = _speaker_speech_stats(segments, system_sp)
        too_much_speech = tot_sec > catalog.max_system_speaker_total_sec
        too_many_turns = n_seg > catalog.max_system_speaker_segments
        if too_much_speech or too_many_turns:
            log.info(
                "speaker_roles_system_candidate_rejected speaker=%s total_sec=%.2f n_segments=%s "
                "thresholds_total_sec=%.2f thresholds_segments=%s over_duration=%s over_segments=%s",
                system_sp,
                tot_sec,
                n_seg,
                catalog.max_system_speaker_total_sec,
                catalog.max_system_speaker_segments,
                too_much_speech,
                too_many_turns,
            )
        else:
            out[system_sp] = "operator"
            assigned.add(system_sp)

    remaining = [s for s in speakers if s not in assigned]
    if remaining and max(scores_op.get(s, 0) for s in remaining) >= catalog.min_hits_operator:
        operator_sp = max(remaining, key=lambda s: (scores_op[s], -first_start[s]))
        if scores_op[operator_sp] >= catalog.min_hits_operator:
            out[operator_sp] = "operator"
            assigned.add(operator_sp)

    for sp in speakers:
        if sp not in out:
            out[sp] = "client"

    log.info(
        "speaker_roles_inferred window_sec=%s system_phrase_window_sec=%s "
        "system_speaker_first_start_before_sec=%s scores_sys=%s scores_op=%s map=%s",
        catalog.match_window_sec,
        sys_window,
        sys_first_deadline,
        scores_sys,
        scores_op,
        out,
    )
    return out
