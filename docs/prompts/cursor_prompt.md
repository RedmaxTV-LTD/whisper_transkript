# Cursor Master Prompt

# Project: Whisper STT sidecar

## Project Goal

GPU-sidecar сервис распознавания речи для записей колл-центра.

Клиент передаёт **URL аудиозаписи** (или пару RX/TX), получает `job_id`, опрашивает статус и забирает транскрипт с ролями **Оператор / Клиент** (при dual-track или диаризации).

Система должна поддерживать:

* асинхронную очередь задач (Redis);
* распознавание **faster-whisper** на CUDA;
* mono и dual RX/TX с `call_direction`;
* синхронизацию каналов по общей mono (`url_mix` + `sync`);
* опциональную диаризацию **pyannote.audio**;
* пост-обработку текста (словарь орфографии);
* инференс ролей спикеров по каталогу фраз;
* интеграцию с n8n и внешними CRM через `dedup_key`.

---

# Main Architecture Rules

## CRITICAL RULE

```text
HTTP API must never block on GPU transcription.
```

Распознавание выполняет только **whisper-worker**. API-контейнер **не загружает** модель Whisper в память.

API не должен:

* ждать завершения транскрипции в `POST /transcribe`;
* выполнять ffmpeg / faster-whisper / pyannote в request handler;
* блокировать клиента дольше, чем нужно для записи задачи в Redis.

---

# Technology Stack

## Backend

* Python 3
* FastAPI + uvicorn
* Pydantic v2
* Redis 7 (async `redis-py`)
* faster-whisper (CTranslate2, CUDA)
* pyannote.audio 3.x (опционально)
* httpx (скачивание аудио)
* ffmpeg / ffprobe (подготовка WAV)

## Infrastructure

* Docker Compose
* NVIDIA CUDA runtime (`nvidia/cuda:12.2.2`)
* Тома: models, logs, spelling_config, speaker_roles_config

**Нет в проекте:** MySQL, Alembic, React, RQ, Celery, отдельного frontend.

---

# Required Flow

```text
Client
  ↓ POST /transcribe
whisper-stt (FastAPI, WHISPER_PROCESS_ROLE=api)
  ↓ LPUSH whisper:queue
Redis (whisper:job:*, whisper:dedup:*)
  ↓ BRPOP
whisper-worker (WHISPER_PROCESS_ROLE=worker)
  ↓ download → ffmpeg → faster-whisper [→ pyannote]
  ↓ SET job completed
Client
  ↓ GET /jobs/{job_id} | GET /jobs/by-dedup/{dedup_key}
```

---

# FORBIDDEN

```text
POST /transcribe → synchronous GPU transcription
```

```text
Loading Whisper model in API container
```

```text
Blocking I/O or GPU work inside FastAPI route handlers
```

---

# Process Roles

| Сервис | `WHISPER_PROCESS_ROLE` | Назначение |
|--------|------------------------|------------|
| whisper-stt | `api` | HTTP, Redis, dedup, job status |
| whisper-worker | `worker` | Очередь, GPU, пайплайн транскрипции |

Оба сервиса — **один Docker-образ**, разная команда запуска.

---

# Transcription Modes

## Single URL (`url`)

* mono или stereo (стereo → mono для STT);
* опционально `diarize` при `WHISPER_DIARIZATION=1`.

## Dual RX/TX (`url_rx` + `url_tx` + `call_direction`)

* `incoming`: RX = клиент, TX = оператор;
* `outgoing`: RX = оператор, TX = клиент;
* диаризация **не используется**.

## Channel sync (`url_mix` + `sync`)

* `auto` — корреляция RX/TX с mix;
* `manual` — заданные offset;
* `off` — без сдвигов;
* `fallback`: `none` | `use_mix` | `use_rx_tx`.

---

# Redis Job Model

Ключи:

* `whisper:queue` — LIST job_id;
* `whisper:job:{job_id}` — JSON `TranscribeJobRecord`;
* `whisper:dedup:{dedup_key}` → job_id;
* `whisper:worker:alive` — TTL heartbeat worker.

Статусы: `queued`, `waiting_gpu`, `downloading`, `transcribing_*`, `syncing_channels`, `merging_segments`, `completed`, `failed`, `stale_failed`.

---

# Backend Architecture Rules

Структура `app/`:

```text
main.py              — HTTP API
worker.py            — BRPOP, семафор GPU, watchdog
transcribe_pipeline.py — оркестрация пайплайна
transcribe_engine.py — faster-whisper
diarize_engine.py    — pyannote
job_store.py         — Redis
job_models.py        — модели задач
transcribe_body.py   — валидация запроса
transcribe_schemas.py — формат ответа
audio_prep.py        — download + ffmpeg
channel_sync.py      — auto-sync RX/TX
settings.py          — env config
```

Правила:

* бизнес-логика транскрипции — в pipeline/engine, не в `main.py`;
* GPU-работа — только в worker через `ThreadPoolExecutor`;
* Redis-операции — в `job_store.py`;
* конфигурация — только через `Settings` / env.

---

# API Rules

Эндпоинты:

* `POST /transcribe` — 202/200, сразу `job_id`;
* `GET /jobs/{job_id}` — статус и result;
* `GET /jobs/by-dedup/{dedup_key}` — опрос по записи;
* `GET /health` — Redis + `worker_available`.

Авторизация: опционально `Authorization: Bearer` (`WHISPER_API_TOKEN`).

OpenAPI: `/docs`, `/redoc`.

---

# Security Rules

Never commit:

* `.env`;
* `WHISPER_API_TOKEN`, `HF_TOKEN`;
* credentials JSON (например `google_key.json`).

Не логировать полные URL с секретами в query string.

---

# Docker Rules

Запуск из каталога проекта:

```bash
docker compose --env-file .env up --build -d
```

Между контейнерами — **только service names** (`redis`, не `localhost`).

После изменений в `app/` или `Dockerfile`:

```bash
cd /opt/whisper && docker compose down && docker compose up --build -d
```

Перед первым запуском — загрузить модели:

```bash
docker compose run --rm whisper-stt python3 /app/download_models.py
```

---

# Environment Rules

Вся конфигурация через `.env` / `.env.example`.

Ключевые переменные: `REDIS_URL`, `WHISPER_MODEL_PATH`, `WHISPER_MAX_CONCURRENT_JOBS`, `WHISPER_DIARIZATION`, `WHISPER_API_TOKEN`.

Hot-reload конфигов (без перезапуска): `spelling_config/dictionary.json`, `speaker_roles_config/catalog.json`.

---

# Documentation Rules

При архитектурных изменениях обновляй:

* `README.md`
* `docs/architecture/*`
* `docs/changelog/CHANGELOG.md`
* при необходимости `docs/prompts/*`

Source of truth = **код**, не устаревшие docs.

---

# Performance Rules

* `WHISPER_MAX_CONCURRENT_JOBS` ограничивает параллельные задачи (включая скачивание);
* при `WHISPER_DIARIZATION=1` pyannote загружается **до** BRPOP;
* heartbeat задач — `WHISPER_JOB_HEARTBEAT_SEC`;
* stale watchdog — `WHISPER_JOB_STALE_SEC`;
* лимит скачивания — `WHISPER_MAX_DOWNLOAD_BYTES`.

---

# Code Rules

* минимальный diff, без over-engineering;
* следовать существующим соглашениям в `app/`;
* типизация Pydantic для API;
* не дублировать логику между API и worker;
* комментарии только для неочевидной логики.

---

# Logging Rules

Persistent logs в `/logs`:

* `errors.log`
* `crash-*.log`

Логировать: создание/завершение jobs, ошибки worker, Redis failures, CUDA/diarization errors.

---

# Final Response Rules

После каждой задачи выводи:

1. Summary изменений.
2. Список изменённых файлов.
3. Обновлённую документацию (если менялась).
4. Команды для проверки.
5. Предложенный commit message (не коммитить без явной просьбы пользователя).
