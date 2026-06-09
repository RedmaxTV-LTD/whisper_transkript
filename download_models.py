#!/usr/bin/env python3
"""
Скачивание и обновление моделей в формате **CTranslate2** для **faster-whisper**.

Скрипт с `import whisper` и файлами `*.pt` относится к пакету **openai-whisper**;
этот sidecar использует **faster-whisper**, ему нужны каталоги с `model.bin` и др.,
которые загружаются с Hugging Face через `faster_whisper.utils.download_model`.

В корне `--dir` создаются подкаталоги по имени модели, например `/models/turbo`.
Укажите `WHISPER_MODEL_PATH=/models/turbo` (или другой подкаталог).

Переменные окружения:
  WHISPER_MODELS_DIR — корень (как --dir), по умолчанию /models
  WHISPER_MODELS_FORCE — 1/true: удалить каталоги перечисленных моделей и скачать снова
  WHISPER_HF_TOKEN или HF_TOKEN — опционально для приватных репозиториев
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _hf_token() -> str | bool | None:
    tok = (os.environ.get("WHISPER_HF_TOKEN") or os.environ.get("HF_TOKEN") or "").strip()
    if tok:
        return tok
    return None


def _model_complete(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, "model.bin"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Скачать модели faster-whisper (CTranslate2) в подкаталоги.",
    )
    parser.add_argument(
        "--dir",
        default=os.environ.get("WHISPER_MODELS_DIR", "/models"),
        help="Корень: для каждой модели — подкаталог с именем модели (по умолчанию /models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Удалить каталог модели и скачать заново (аналог WHISPER_MODELS_FORCE=1)",
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Имена faster-whisper: tiny, base, small, medium, large, turbo, large-v3, … "
        "По умолчанию: tiny base small medium large turbo",
    )
    args = parser.parse_args()

    try:
        from faster_whisper.utils import download_model
    except ImportError as e:
        print("Нужен пакет faster-whisper: pip install faster-whisper", file=sys.stderr)
        raise SystemExit(1) from e

    root = os.path.abspath(args.dir)
    if not os.access(root, os.W_OK):
        print(
            "Каталог не доступен на запись: %r. "
            "В docker-compose у тома /models не должно быть :ro (нужна запись для download_models.py)."
            % root,
            file=sys.stderr,
        )
        return 1
    os.makedirs(root, exist_ok=True)
    force = args.force or _truthy("WHISPER_MODELS_FORCE")
    token = _hf_token()

    default_list = ["tiny", "base", "small", "medium", "large", "turbo"]
    models = list(args.models) if args.models else default_list

    for name in models:
        target = os.path.join(root, name)

        if force and os.path.isdir(target):
            print(f"Удаление {target} (--force / WHISPER_MODELS_FORCE)")
            shutil.rmtree(target, ignore_errors=True)

        if not force and os.path.isdir(target) and _model_complete(target):
            print(f"Пропуск (есть model.bin): {name} -> {target}")
            continue

        if os.path.isdir(target):
            print(f"Очистка неполного/устаревшего каталога: {target}")
            shutil.rmtree(target, ignore_errors=True)

        print(f"Загрузка {name} -> {target} …")
        os.makedirs(target, exist_ok=True)
        download_model(name, output_dir=target, local_files_only=False, use_auth_token=token)
        print(f"Готово: {name}")

    print("Все указанные модели обработаны.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())