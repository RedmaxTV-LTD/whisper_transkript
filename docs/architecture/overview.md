# Обзор системы

## Назначение

**Whisper STT sidecar** — сервис speech-to-text для записей телефонных разговоров колл-центра (Redmax и аналоги). Клиент передаёт URL(ы) записи, получает `job_id` / `dedup_key` и забирает готовый транскрипт с ролями **Оператор / Клиент** (при dual-track или локальной диаризации).

Типичный сценарий интеграции (n8n):

1. После появления файлов записи АТС → `POST /transcribe`.
2. Периодический опрос `GET /jobs/by-dedup/{dedup_key}`.
3. При `status=completed` — использование `result.formatted_text`, `operator_text`, `client_text`, `diarized_segments`.

Сервис **не** является UI и **не** хранит аудио постоянно: файлы скачиваются во временный каталог worker и удаляются после обработки.

Полный набор схем: [diagrams.md](./diagrams.md).

## Архитектурные принципы

| Принцип | Реализация |
|---------|------------|
| API не блокируется на STT | `POST /transcribe` только пишет в Redis; STT — в worker |
| Разделение процессов | `WHISPER_PROCESS_ROLE=api` vs `worker`; один Docker-образ |
| Гибридный бэкенд | `WHISPER_BACKEND=local` \| `openai` (переключение в `.env`) |
| Раздельный параллелизм | `WHISPER_MAX_CONCURRENT_JOBS` и `OPENAI_MAX_CONCURRENT_JOBS` |
| Идемпотентность | `dedup_key` по URL; повторный POST не плодит дубли активных задач |
| Наблюдаемость | heartbeat, статусы шагов, `client_hint`, `/logs` |
| Горячая конфигурация | spelling / speaker_roles тома без перезапуска |

### Критическое правило

```text
HTTP API must never block on GPU transcription or OpenAI calls.
```

## Процессы (Docker Compose)

| Сервис | Команда | Роль |
|--------|---------|------|
| **redis** | `redis-server --appendonly yes` | Очередь, job records, dedup, worker alive |
| **whisper-stt** | `python3 -m app` | FastAPI, `WHISPER_PROCESS_ROLE=api` |
| **whisper-worker** | `python3 -m app.worker` | BRPOP, пайплайн STT, `WHISPER_PROCESS_ROLE=worker` |

Оба Python-сервиса из одного `Dockerfile` (`nvidia/cuda:12.2.2` + PyTorch cu121 + faster-whisper). Различие — env `WHISPER_PROCESS_ROLE` и `command`.

Сеть: `whisper-net`. Между контейнерами только Docker DNS (`redis`, не `localhost`).

## Бэкенды распознавания

| `WHISPER_BACKEND` | Движок | Параллелизм | Диаризация pyannote | Модель в RAM worker |
|-------------------|--------|-------------|---------------------|---------------------|
| `local` | faster-whisper (CTranslate2, CUDA) | `WHISPER_MAX_CONCURRENT_JOBS` (часто 1) | да, если `WHISPER_DIARIZATION=1` | да |
| `openai` | OpenAI `/v1/audio/transcriptions` | `OPENAI_MAX_CONCURRENT_JOBS` (часто 8+) | нет (для ролей — dual RX/TX) | нет |

При `openai` обязателен `OPENAI_API_KEY`. Dual RX/TX уходит **двумя параллельными** HTTP-запросами; роли по `call_direction` как в local.

## Режимы обработки аудио

### 1. Один файл (`url`)

- Mono или stereo по одному HTTPS URL.
- Stereo сводится в mono 16 kHz WAV (ffmpeg).
- Local + `diarize`: pyannote + каталог ролей / схема SPEAKER_00/01/02.
- OpenAI: один запрос, без разделения ролей (`role_separation=unavailable_mono`).

### 2. Два канала (`url_rx` + `url_tx`)

- Два mono (типично `-rx.wav` / `-tx.wav`).
- Обязателен **`call_direction`**:
  - `incoming` — RX = клиент, TX = оператор;
  - `outgoing` — RX = оператор, TX = клиент.
- Диаризация не нужна: роли = каналы.
- Опционально `url_mix` + `sync` для выравнивания шкалы времени.

### 3. Синхронизация (`sync`)

| mode | Поведение |
|------|-----------|
| `off` | Без сдвигов |
| `manual` | `offset_rx_sec` / `offset_tx_sec` |
| `auto` | Корреляция огибающих RX/TX с `url_mix` |

`fallback` при недоверенных сдвигах: `none` \| `use_mix` \| `use_rx_tx`.

## HTTP API (контракт)

| Method | Path | Смысл |
|--------|------|-------|
| GET | `/health` | Redis, `worker_available`, `whisper_backend`, лимиты concurrent |
| POST | `/transcribe` | 202 новая / 200 existing по dedup |
| GET | `/jobs/{job_id}` | Статус и `result` |
| GET | `/jobs/by-dedup/{dedup_key}` | То же по ключу записи (удобно для n8n) |
| GET | `/docs` | Swagger |

Авторизация: опционально `Authorization: Bearer` (`WHISPER_API_TOKEN`).

Тело запроса валидируется в `TranscribeBody` (`app/transcribe_body.py`). Ответ задачи — `TranscribeResponse` в поле `result`.

## Технологический стек

| Слой | Технология |
|------|------------|
| HTTP | FastAPI, uvicorn, Pydantic v2 |
| Очередь | Redis 7, async redis-py |
| STT local | faster-whisper (CTranslate2) |
| STT cloud | OpenAI Audio Transcriptions (httpx) |
| Диаризация | pyannote.audio 3.x |
| Аудио | ffmpeg, ffprobe, httpx |
| VAD | Silero через faster-whisper (local) |
| Контейнер | nvidia/cuda:12.2.2, runtime nvidia |

## Внешние интеграции

- **Клиенты**: n8n, CRM, ATS, любой HTTP-клиент.
- **Аудио**: HTTPS URL (АТС, S3-compatible storage).
- **OpenAI**: при `WHISPER_BACKEND=openai`.
- **Hugging Face**: `HF_TOKEN` для gated pyannote.

## Ограничения и допущения

- Лимит скачивания: `WHISPER_MAX_DOWNLOAD_BYTES` (по умолчанию 200 MB).
- OpenAI: файл ≤ ~25 MB (`OPENAI_MAX_UPLOAD_MB`, иначе сжатие в mp3).
- Local при concurrent=1: одна задача (включая download) за раз.
- Рестарт worker → активные jobs → `failed` / `worker_orphaned`; клиент повторяет POST.
- API и worker — один `REDIS_URL`.
- Авто-overflow local→openai **не** реализован: только ручной switch в `.env`.

## Версия и документация

- Версия API в OpenAPI: см. `app/main.py` (`version`).
- История изменений: [../changelog/CHANGELOG.md](../changelog/CHANGELOG.md).
- Операционный README: [../../README.md](../../README.md).
