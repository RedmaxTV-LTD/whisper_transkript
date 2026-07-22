# Компоненты и модули

## Структура репозитория

```
whisper/
├── app/                      # Python-пакет приложения
├── docs/architecture/        # Архитектурная документация
├── spelling_config/          # Том: словарь пост-обработки текста
├── speaker_roles_config/     # Том: каталог фраз для ролей спикеров
├── models/                   # Том: CTranslate2-модели (model.bin)
├── logs/                     # Том: errors.log, crash-*.log
├── download_models.py        # Загрузка моделей в volume
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env / .env.example
```

## Модули `app/`

### Точки входа

| Модуль | Назначение |
|--------|------------|
| `__main__.py` | Запуск API: uvicorn + persistent logging |
| `worker.py` | Фоновый worker: очередь, семафор GPU, watchdog |

### HTTP и задачи

| Модуль | Назначение |
|--------|------------|
| `main.py` | FastAPI: `/transcribe`, `/jobs/*`, `/health`; lifespan Redis |
| `transcribe_body.py` | Pydantic-модели запроса: `TranscribeBody`, `SyncOptionsBody` |
| `transcribe_schemas.py` | Pydantic-модели ответа: `TranscribeResponse`, `DiarizedSegment` |
| `job_models.py` | `TranscribeJobRecord`, статусы, terminal/active множества |
| `job_store.py` | Redis: ключи, очередь, атомарный claim, recovery при старте worker |
| `job_watchdog.py` | Пометка `stale_failed` по устаревшему heartbeat |
| `dedup.py` | Вычисление стабильного `dedup_key` из URL |

### Пайплайн распознавания

| Модуль | Назначение |
|--------|------------|
| `transcribe_pipeline.py` | Оркестрация: download → sync → transcribe → merge; ветка `local` / `openai` |
| `audio_prep.py` | Скачивание URL, ffmpeg → mono 16 kHz WAV |
| `channel_sync.py` | Auto-sync RX/TX по mix (корреляция огибающих) |
| `transcribe_engine.py` | faster-whisper (local): загрузка модели, transcribe, dual-track merge |
| `openai_engine.py` | OpenAI Audio Transcriptions API (`WHISPER_BACKEND=openai`) |
| `diarize_engine.py` | pyannote Pipeline: lazy load, diarize mono (только local) |
| `align_speakers.py` | Сопоставление сегментов Whisper и меток pyannote |

### Пост-обработка текста и роли

| Модуль | Назначение |
|--------|------------|
| `spelling_fixes.py` | Замены по `dictionary.json` (hot-reload по mtime) |
| `speaker_roles.py` | Маппинг SPEAKER_* → client/operator, IVR-эвристики |
| `speaker_roles_catalog.py` | Загрузка `catalog.json`, инференс ролей по фразам |
| `speaker_roles_catalog.json` | Fallback-каталог в образе (если том не смонтирован) |
| `spelling_dictionary.json` | Fallback-словарь в образе |

### Инфраструктура

| Модуль | Назначение |
|--------|------------|
| `settings.py` | Конфигурация из env (`Settings`, `get_settings`) |
| `persistent_logs.py` | `/logs/errors.log`, crash dumps, console logging |

## Зависимости между модулями (упрощённо)

```
main.py
  ├── job_store, job_watchdog, dedup
  └── transcribe_body

worker.py
  ├── job_store, job_watchdog
  ├── transcribe_pipeline
  │     ├── audio_prep, channel_sync
  │     ├── transcribe_engine ── align_speakers, spelling_fixes, speaker_roles_catalog
  │     ├── openai_engine (при WHISPER_BACKEND=openai)
  │     └── diarize_engine (local + diarize)
  └── transcribe_schemas (build_transcribe_response)

transcribe_engine
  ├── align_speakers
  ├── speaker_roles_catalog → speaker_roles
  └── spelling_fixes
```

## Конфигурационные тома

### `spelling_config/dictionary.json`

- Монтируется в `/config/spelling/dictionary.json`.
- Формат: JSON-массив или объект с парами «ошибка → канон».
- Перечитывается при изменении mtime **без перезапуска** контейнера.
- Отключение: `WHISPER_SPELLING_FIXES=0`.

### `speaker_roles_config/catalog.json`

- Монтируется в `/config/speaker_roles/catalog.json`.
- Содержит фразы для ролей **system / operator / client** и окна времени (`match_window_sec`).
- Используется при диаризации для сопоставления `SPEAKER_00/01/02` с ролями колл-центра.
- Отключение каталога: `WHISPER_SPEAKER_ROLES_CATALOG=0` (фиксированная схема 00=клиент, 01/02=оператор).

## Модели Whisper

### Local (`WHISPER_BACKEND=local`)

- Формат: **CTranslate2** (каталог с `model.bin`), не OpenAI `.pt`.
- Загрузка: `download_models.py` → подкаталоги `/models/{tiny,base,small,medium,large,turbo}`.
- Активная модель: `WHISPER_MODEL_PATH` (часто `/models/turbo` или `/models/large`).
- Worker загружает модель **один раз** при старте (`ensure_model_loaded`).
- При `WHISPER_DIARIZATION=1` pyannote загружается **до** BRPOP (избежание CUDA deadlock).

### OpenAI (`WHISPER_BACKEND=openai`)

- Локальные CTranslate2-модели **не** загружаются.
- Модель API: `OPENAI_TRANSCRIBE_MODEL` (по умолчанию `whisper-1`).
- Том `/models` можно не использовать (но compose обычно всё равно монтирует).

## Логирование

| Файл | Содержимое |
|------|------------|
| `/logs/errors.log` | Структурированные ошибки приложения |
| `/logs/crash-*.log` | Необработанные исключения |
| `/logs/crashes-index.log` | Индекс crash-файлов |

Настраивается через `WHISPER_LOGS_DIR` (в compose: `./logs` → `/logs`).

## См. также

- Схемы: [diagrams.md](./diagrams.md)
- Обзор: [overview.md](./overview.md)