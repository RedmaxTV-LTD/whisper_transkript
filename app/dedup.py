"""Стабильный dedup_key для фоновых задач транскрипции."""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlparse

from pydantic import BaseModel


def _basename_no_ext(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    if not path or path.endswith("/"):
        return ""
    name = path.rsplit("/", 1)[-1]
    if "." in name:
        return name.rsplit(".", 1)[0]
    return name


_rx_suffix = re.compile(r"-(?i)rx$")
_tx_suffix = re.compile(r"-(?i)tx$")


def _strip_rx_tx_suffix(stem: str) -> str:
    s = _rx_suffix.sub("", stem)
    s = _tx_suffix.sub("", s)
    return s


def _longest_common_prefix(a: str, b: str) -> str:
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i].rstrip("-_.")


def compute_dedup_key_from_payload_dict(payload: dict) -> str:
    """Вычисляет dedup_key из JSON-сериализуемого dict (как в Redis payload)."""
    url_mix = payload.get("url_mix")
    url_rx = payload.get("url_rx")
    url_tx = payload.get("url_tx")
    url_single = payload.get("url")

    if url_mix:
        stem = _basename_no_ext(str(url_mix))
        if stem:
            return stem

    if url_rx and url_tx:
        rx = _basename_no_ext(str(url_rx))
        tx = _basename_no_ext(str(url_tx))
        if rx and tx:
            crx = _strip_rx_tx_suffix(rx)
            ctx = _strip_rx_tx_suffix(tx)
            if crx and crx == ctx:
                return crx
            pfx = _longest_common_prefix(rx, tx)
            if len(pfx) >= 8:
                return pfx.rstrip("-_.")

    if url_single:
        stem = _basename_no_ext(str(url_single))
        if stem:
            return stem

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_dedup_key(body: BaseModel) -> str:
    return compute_dedup_key_from_payload_dict(body.model_dump(mode="json"))
