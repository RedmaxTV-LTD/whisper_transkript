"""Замена частых ошибок распознавания по словарю «как услышано → каноническое написание»."""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BUNDLED_SPELLING_PATH: Path = Path(__file__).resolve().parent / "spelling_dictionary.json"

_live_lock = threading.Lock()
_live_cache: dict[str, Any] = {"path": None, "mtime": None, "patterns": []}


def _resolve_spelling_file(primary: Path) -> Path:
    """Если смонтированный JSON отсутствует или это не файл — берём встроенный словарь из образа."""
    try:
        if primary.is_file():
            return primary
    except OSError:
        pass
    if BUNDLED_SPELLING_PATH.is_file():
        log.warning(
            "spelling_dictionary_primary_missing_or_not_file path=%s using_bundled=%s",
            primary,
            BUNDLED_SPELLING_PATH,
        )
        return BUNDLED_SPELLING_PATH
    return primary


def _load_pairs_from_json(path: Path) -> list[tuple[str, str]]:
    raw = path.read_text(encoding="utf-8")
    data: Any = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("spelling_dictionary: ожидается JSON-объект { \"ошибка\": \"правильно\" }")
    pairs: list[tuple[str, str]] = []
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        wrong, right = k.strip(), v.strip()
        if wrong and right:
            pairs.append((wrong, right))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def compile_spelling_patterns(pairs: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    """Границы «слова» (?<!\\w)…(?!\\w), без учёта регистра при поиске."""
    out: list[tuple[re.Pattern[str], str]] = []
    for wrong, right in pairs:
        pat = re.compile(r"(?i)(?<!\w)" + re.escape(wrong) + r"(?!\w)", re.UNICODE)
        out.append((pat, right))
    return out


def load_spelling_patterns(path: Path) -> list[tuple[re.Pattern[str], str]]:
    try:
        pairs = _load_pairs_from_json(path)
    except FileNotFoundError:
        log.warning("spelling_dictionary_not_found path=%s", path)
        return []
    except (json.JSONDecodeError, OSError, ValueError) as e:
        log.warning("spelling_dictionary_load_failed path=%s err=%s", path, e)
        return []
    if pairs:
        log.info("spelling_dictionary_loaded path=%s entries=%s", path, len(pairs))
    return compile_spelling_patterns(pairs)


def apply_spelling_fixes(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> str:
    if not text or not patterns:
        return text
    s = text
    for pat, repl in patterns:
        s = pat.sub(repl, s)
    return s


def get_spelling_patterns_live(path: Path, enabled: bool) -> list[tuple[re.Pattern[str], str]]:
    """Паттерны для текущего запроса: перечитываем JSON, если изменился mtime (без перезапуска контейнера)."""
    if not enabled:
        return []
    resolved = _resolve_spelling_file(path)
    path_key = str(resolved.resolve())
    with _live_lock:
        try:
            mtime = resolved.stat().st_mtime
        except OSError:
            if _live_cache["path"] == path_key and _live_cache["patterns"]:
                return _live_cache["patterns"]
            return []
        if _live_cache["path"] == path_key and _live_cache["mtime"] == mtime:
            return _live_cache["patterns"]
        patterns = load_spelling_patterns(resolved)
        _live_cache["path"] = path_key
        _live_cache["mtime"] = mtime
        _live_cache["patterns"] = patterns
        return patterns
