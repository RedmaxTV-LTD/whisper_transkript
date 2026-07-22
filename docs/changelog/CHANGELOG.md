# Changelog

Все значимые изменения проекта документируются в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версионирование — [Semantic Versioning](https://semver.org/lang/ru/).

## [Unreleased]

### Added

- Гибридный бэкенд распознавания: `WHISPER_BACKEND=local|openai` (переключение через `.env`).
- Облачный OpenAI Audio Transcriptions (`app/openai_engine.py`): `verbose_json`, retry на 429/5xx, сжатие файла > лимита в mp3.
- Dual RX/TX через OpenAI — два параллельных запроса; роли по `call_direction` без изменений API.
- Раздельный параллелизм: `WHISPER_MAX_CONCURRENT_JOBS` (local) и `OPENAI_MAX_CONCURRENT_JOBS` (openai).
- `/health`: поля `whisper_backend`, `local_max_concurrent_jobs`, `openai_max_concurrent_jobs`.
- Подробная архитектурная документация и схемы Mermaid: `docs/architecture/` (включая `diagrams.md`).

### Changed

- Worker при `openai` не загружает faster-whisper / pyannote; при `local` поведение прежнее.

## [1.0.0] — 2026-06-09

Первый стабильный релиз **Whisper STT sidecar** — GPU-сервис распознавания речи для записей колл-центра с асинхронной очередью задач.

### Added

#### HTTP API (FastAPI)

- `POST /transcribe` — постановка задачи в очередь; ответ **202** (новая задача) или **200** (существующая по `dedup_key`).
- `GET /jobs/{job_id}` — статус, прогресс, heartbeat и результат при `completed`.
- `GET /jobs/by-dedup/{dedup_key}` — опрос по стабильному ключу записи (удобно для n8n и внешних очередей).
- `GET /health` — проверка Redis и флага `worker_available`.
- OpenAPI: Swagger (`/docs`), ReDoc (`/redoc`), схема JSON (`/openapi.json`).
- Опциональная авторизация через `Authorization: Bearer` (`WHISPER_API_TOKEN`).
- Поля интеграции в ответе jobs: `worker_available`, `retry_recommended`, `client_hint`, `in_redis_queue`.

#### Архитектура и очередь

- Разделение процессов: **whisper-stt** (`WHISPER_PROCESS_ROLE=api`) и **whisper-worker** (`worker`).
- Очередь задач в **Redis 7** (`whisper:queue`, `whisper:job:*`, `whisper:dedup:*`).
- Дедупликация задач по `dedup_key` (basename URL, stem пары RX/TX или SHA-256 payload).
- Семафор `WHISPER_MAX_CONCURRENT_JOBS` — ограничение параллельных задач на GPU (включая этап скачивания).
- Heartbeat задач (`WHISPER_JOB_HEARTBEAT_SEC`) и watchdog `stale_failed` (`WHISPER_JOB_STALE_SEC`).
- Ключ `whisper:worker:alive` — индикатор живого worker для API и клиентов.
- Recovery при старте worker: сброс очереди и повторная постановка `queued` / `waiting_gpu` (`WHISPER_WORKER_QUEUE_FLUSH_ON_START`).
- Пометка `worker_orphaned` для задач, прерванных перезапуском worker.
- Self-heal очереди в API: атомарный LPUSH при рассинхроне `queued` и `whisper:queue`.

#### Распознавание речи

- **faster-whisper** (CTranslate2) на CUDA; модели в формате `model.bin`.
- Скрипт `download_models.py` для загрузки моделей (tiny … turbo) в volume.
- Подготовка аудио: скачивание по HTTPS, ffmpeg → mono 16 kHz WAV.
- Режим **одного файла** (`url`): mono/stereo, опциональная диаризация.
- Режим **двух каналов** (`url_rx` + `url_tx` + `call_direction`): раздельная транскрипция RX/TX с ролями Оператор/Клиент.
- Синхронизация каналов по общей mono (`url_mix`): `sync.mode` — `auto`, `manual`, `off`; fallback `none` / `use_mix` / `use_rx_tx`.
- Auto-sync: корреляция огибающих RX/TX с mix, порог `max_offset_sec`.
- Разбиение длинных сегментов Whisper по паузам между словами (`WHISPER_INTRA_SEGMENT_SPLIT_GAP_SEC`).
- Silero VAD через faster-whisper (`WHISPER_VAD_FILTER`).

#### Диаризация и роли спикеров

- Опциональная **pyannote.audio** 3.x (`WHISPER_DIARIZATION=1`, `HF_TOKEN`).
- Загрузка pyannote до начала BRPOP — снижение риска зависания CUDA при совместной работе с Whisper.
- Каталог фраз ролей (`speaker_roles_config/catalog.json`): system / operator / client по тексту начала диалога.
- Фиксированная схема pyannote 00/01/02 при отключённом каталоге.
- IVR-эвристики: реплики системы относятся к оператору.
- Ответ API: `diarized_segments`, `operator_text`, `client_text`, `formatted_text`, `role_separation`.

#### Пост-обработка

- Словарь орфографических замен (`spelling_config/dictionary.json`) с hot-reload по mtime.
- Отключение замен: `WHISPER_SPELLING_FIXES=0`.

#### Инфраструктура

- Docker Compose: **redis**, **whisper-stt**, **whisper-worker**.
- Образ на базе `nvidia/cuda:12.2.2`, PyTorch cu121, runtime nvidia.
- Persistent logs: `/logs/errors.log`, `crash-*.log`, `crashes-index.log`.
- Тома для моделей, логов, spelling и speaker_roles без пересборки образа.
- Документация: `README.md`, `docs/architecture/`.

### Changed

- (начальный релиз — предыдущих версий нет)

### Fixed

- (начальный релиз)

### Security

- Опциональный Bearer-токен для всех эндпоинтов транскрипции и jobs.
- Лимит размера скачиваемого файла (`WHISPER_MAX_DOWNLOAD_BYTES`).

---

[Unreleased]: #unreleased
[1.0.0]: #100--2026-06-09
