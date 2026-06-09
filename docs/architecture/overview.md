# Обзор системы

## Назначение

**Whisper STT sidecar** — специализированный сервис speech-to-text для записей телефонных разговоров колл-центра. Он:

1. Принимает **URL аудиофайла** (или пары RX/TX + опционально mix).
2. Ставит задачу в **очередь Redis** и сразу возвращает `job_id`.
3. Обрабатывает запись в отдельном процессе **whisper-worker** на GPU.
4. Отдаёт результат через **GET /jobs/{job_id}** или **GET /jobs/by-dedup/{dedup_key}**.

Сервис не хранит аудио постоянно: файлы скачиваются во временный каталог и удаляются после обработки.

## Архитектурные принципы

| Принцип | Реализация |
|---------|------------|
| Разделение API и вычислений | API-контейнер не загружает Whisper в память; только worker держит модель на GPU |
| Асинхронность | HTTP не блокируется на распознавании; клиент опрашивает статус |
| Идемпотентность | `dedup_key` по URL предотвращает дубли задач для одной записи |
| Наблюдаемость | Heartbeat, статусы шагов, `client_hint`, persistent logs |
| Горячая конфигурация | Словарь орфографии и каталог ролей спикеров монтируются как тома |

## Процессы (Docker Compose)

| Сервис | Команда | Роль |
|--------|---------|------|
| **redis** | `redis-server --appendonly yes` | Очередь задач, хранение job records, worker alive |
| **whisper-stt** | `python3 -m app` | FastAPI, `WHISPER_PROCESS_ROLE=api` |
| **whisper-worker** | `python3 -m app.worker` | BRPOP очереди, GPU-транскрипция, `WHISPER_PROCESS_ROLE=worker` |

Оба Python-сервиса собираются из **одного Dockerfile** (`nvidia/cuda:12.2.2` + PyTorch cu121 + faster-whisper). Различие — только переменная `WHISPER_PROCESS_ROLE` и команда запуска.

## Режимы обработки аудио

### 1. Один файл (`url`)

- Моно или стерео по одному HTTPS-URL.
- Стерео сводится в mono (16 kHz WAV) через ffmpeg.
- Опционально **диаризация** (pyannote), если `WHISPER_DIARIZATION=1`.
- Без диаризации роли «Оператор / Клиент» недоступны — единый `transcript`.

### 2. Два канала (`url_rx` + `url_tx`)

- Два отдельных mono-файла (типично `-rx.wav` / `-tx.wav`).
- Обязателен **`call_direction`**: `incoming` (RX=клиент, TX=оператор) или `outgoing` (наоборот).
- Диаризация **не используется** — роли определяются по каналу.
- Опционально **`url_mix`** и **`sync`** для выравнивания временной шкалы RX/TX.

### 3. Синхронизация каналов (`sync`)

| mode | Поведение |
|------|-----------|
| `off` | RX и TX транскрибируются независимо, без сдвигов |
| `manual` | Заданные `offset_rx_sec` / `offset_tx_sec` |
| `auto` | Корреляция огибающих RX/TX с общей mono (`url_mix`) |

При недоверенных сдвигах срабатывает **`fallback`**: `none`, `use_mix` (один транскрипт по mix) или `use_rx_tx`.

## Технологический стек

| Слой | Технология |
|------|------------|
| HTTP API | FastAPI, uvicorn, Pydantic v2 |
| Очередь | Redis 7 (async `redis-py`) |
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, GPU) |
| Диаризация | pyannote.audio 3.x (`speaker-diarization-3.1`) |
| Аудио | ffmpeg, ffprobe, httpx (скачивание) |
| VAD | Silero VAD через faster-whisper (onnxruntime) |
| Контейнер | NVIDIA CUDA 12.2, runtime nvidia |

## Внешние интеграции

- **Клиенты**: n8n, внутренние CRM/ATS, любой HTTP-клиент.
- **Источник аудио**: произвольные HTTPS URL (записи АТС, object storage).
- **Hugging Face**: токен для gated-моделей pyannote (`HF_TOKEN`).

## Ограничения и допущения

- Максимальный размер скачиваемого файла — `WHISPER_MAX_DOWNLOAD_BYTES` (по умолчанию 200 MB).
- При `WHISPER_MAX_CONCURRENT_JOBS=1` worker обрабатывает одну задачу за раз (включая этап скачивания).
- После перезапуска worker незавершённые активные задачи помечаются `failed` (`worker_orphaned`); клиент должен повторить POST.
- API и worker **обязаны** использовать один и тот же `REDIS_URL`.
