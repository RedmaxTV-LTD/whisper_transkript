"""Файловые логи на смонтированный том: ошибки приложения и необработанные исключения (краши процесса)."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

_installed = False
_app_console_installed = False


def install_app_console_logging() -> None:
    """INFO из пакета ``app`` в stderr.

    Uvicorn пишет access/error в свои логгеры; без отдельного handler записи
    ``logging.getLogger(__name__)`` из ``app.*`` доходят до root, где только
    файловый handler уровня ERROR — строки INFO в docker logs не появляются.
    """
    global _app_console_installed
    if _app_console_installed:
        return
    lg = logging.getLogger("app")
    if lg.handlers:
        _app_console_installed = True
        return
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    lg.addHandler(h)
    _app_console_installed = True


def install_persistent_logging(logs_dir: Path) -> None:
    """Создаёт каталог, пишет ERROR в errors.log, при падении процесса — отдельный crash-*.log."""
    global _installed
    logs_dir.mkdir(parents=True, exist_ok=True)
    if _installed:
        return

    err_path = logs_dir / "errors.log"
    fh = RotatingFileHandler(
        err_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.ERROR)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(fh)

    def _write_crash(
        etype: type[BaseException] | None,
        value: BaseException | None,
        tb: TracebackType | None,
        *,
        context: str,
    ) -> None:
        if etype is None:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"crash-{stamp}.log"
        path = logs_dir / name
        try:
            body = "".join(traceback.format_exception(etype, value, tb))
            path.write_text(f"context: {context}\n\n{body}", encoding="utf-8")
            idx = logs_dir / "crashes-index.log"
            with idx.open("a", encoding="utf-8") as f:
                f.write(f"{stamp} {context} -> {name}\n")
        except OSError:
            pass

    def _sys_excepthook(etype: type[BaseException] | None, value: BaseException | None, tb) -> None:  # type: ignore[no-untyped-def]
        if etype is not None and issubclass(etype, KeyboardInterrupt):
            sys.__excepthook__(etype, value, tb)
            return
        _write_crash(etype, value, tb, context="sys.excepthook")
        sys.__excepthook__(etype, value, tb)

    sys.excepthook = _sys_excepthook

    if hasattr(threading, "excepthook"):
        _prev_thread = threading.excepthook

        def _thread_excepthook(args: threading.ExceptHookArgs) -> None:  # type: ignore[name-defined]
            ctx = f"threading.excepthook thread={args.thread.name!r}"
            _write_crash(args.exc_type, args.exc_value, args.exc_traceback, context=ctx)
            _prev_thread(args)

        threading.excepthook = _thread_excepthook

    _installed = True
