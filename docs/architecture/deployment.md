# Развёртывание

## Docker Compose

Запуск из каталога `/opt/whisper`:

```bash
docker compose --env-file .env up --build -d
```

Поднимаются три серvice: **redis**, **whisper-stt**, **whisper-worker**.

### Сеть

- `whisper-net` — bridge-сеть для redis ↔ api ↔ worker.
- DNS: `127.0.0.11` (встроенный Docker) + публичные резолверы для внешних URL аудио.

### Порты

| Сервис | Порт |
|--------|------|
| whisper-stt | `${WHISPER_HOST_PORT:-19900}:19900` |

### Тома

| Host | Container | Назначение |
|------|-----------|------------|
| `./models` | `/models` | CTranslate2-модели |
| `./logs` | `/logs` | Логи ошибок и crash dumps |
| `./spelling_config` | `/config/spelling` | `dictionary.json` |
| `./speaker_roles_config` | `/config/speaker_roles` | `catalog.json` |

### GPU

Оба Python-сервиса используют:

```yaml
runtime: nvidia
environment:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility
```

API-контейнер формально с GPU runtime, но **не загружает** Whisper; GPU нужен worker. При отсутствии GPU на хосте worker не стартует с CUDA.

## Первичная подготовка моделей

```bash
docker compose run --rm whisper-stt python3 /app/download_models.py
```

Скрипт `download_models.py` скачивает набор моделей (tiny … turbo) в подкаталоги volume. Активная модель задаётся `WHISPER_MODEL_PATH=/models/turbo`.

Принудительное обновление:

```bash
docker compose run --rm -e WHISPER_MODELS_FORCE=1 whisper-stt python3 /app/download_models.py
```

## Образ (Dockerfile)

- Base: `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`
- Системные пакеты: `ffmpeg`, `libsndfile1`, Python 3
- PyTorch: wheels cu121
- Python deps: `requirements.txt` (faster-whisper, fastapi, redis, pyannote, …)

Пересборка после изменений в `app/`:

```bash
cd /opt/whisper && docker compose down && docker compose up --build -d
```

## Переменные окружения (ключевые)

### Процессы и очередь

| Переменная | Default | Описание |
|------------|---------|----------|
| `WHISPER_PROCESS_ROLE` | `api` / `worker` | Роль процесса |
| `REDIS_URL` | `redis://redis:6379/0` | Redis |
| `WHISPER_MAX_CONCURRENT_JOBS` | `1` | Параллельные задачи на worker |
| `WHISPER_WORKER_QUEUE_FLUSH_ON_START` | `1` | Сброс очереди при старте worker |
| `WHISPER_JOB_HEARTBEAT_SEC` | `15` | Интервал heartbeat |
| `WHISPER_JOB_STALE_SEC` | `300` | Таймаут stale |
| `WHISPER_WORKER_ALIVE_TTL_SEC` | `90` | TTL ключа worker alive |
| `WHISPER_WORKER_ALIVE_REFRESH_SEC` | `25` | Период обновления alive |

### Whisper / GPU

| Переменная | Default | Описание |
|------------|---------|----------|
| `WHISPER_MODEL_PATH` | `/models/turbo` | Путь к модели |
| `WHISPER_DEVICE` | `cuda` | `cuda` или `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` | Тип вычислений CTranslate2 |
| `WHISPER_LANGUAGE` | `ru` | Язык (пусто = auto) |
| `WHISPER_BEAM_SIZE` | `5` | Beam search |
| `WHISPER_VAD_FILTER` | `1` | Silero VAD |
| `WHISPER_INTRA_SEGMENT_SPLIT_GAP_SEC` | `0.45` | Разбиение сегментов по паузам |

### Диаризация

| Переменная | Default | Описание |
|------------|---------|----------|
| `WHISPER_DIARIZATION` | `0` | Включить pyannote |
| `WHISPER_DIARIZE_DEFAULT` | `1` | Диаризация по умолчанию в запросе |
| `HF_TOKEN` | — | Токен Hugging Face |
| `PYANNOTE_PIPELINE` | `pyannote/speaker-diarization-3.1` | Модель |
| `PYANNOTE_DEVICE` | `cpu` | Устройство pyannote (часто cpu при одном GPU) |

### Безопасность и лимиты

| Переменная | Default | Описание |
|------------|---------|----------|
| `WHISPER_API_TOKEN` | — | Bearer-токен для API |
| `WHISPER_DOWNLOAD_TIMEOUT_SEC` | `300` | Таймаут скачивания |
| `WHISPER_MAX_DOWNLOAD_BYTES` | `200000000` | Лимит размера файла |

### Sync RX/TX

| Переменная | Default | Описание |
|------------|---------|----------|
| `WHISPER_SYNC_MAX_OFFSET_SEC` | `2.0` | Макс. сдвиг по умолчанию |
| `WHISPER_CHANNEL_SYNC_*` | см. settings.py | Параметры корреляции |

### Пост-обработка

| Переменная | Default | Описание |
|------------|---------|----------|
| `WHISPER_SPELLING_FIXES` | `1` | Орфографические замены |
| `WHISPER_SPELLING_DICT_PATH` | `/config/spelling/dictionary.json` | Словарь |
| `WHISPER_SPEAKER_ROLES_CATALOG` | `1` | Каталог фраз ролей |
| `WHISPER_SPEAKER_ROLES_CATALOG_PATH` | `/config/speaker_roles/catalog.json` | Путь каталога |

Полный список — в `.env.example` и [README.md](../../README.md).

## API endpoints (кратко)

| Method | Path | Описание |
|--------|------|----------|
| GET | `/health` | Redis, worker_available |
| POST | `/transcribe` | Постановка задачи (202/200) |
| GET | `/jobs/{job_id}` | Статус по UUID |
| GET | `/jobs/by-dedup/{dedup_key}` | Статус по dedup_key |
| GET | `/docs` | Swagger UI |

## Мониторинг

- **Health**: `GET /health` — `status`, `redis`, `worker_available`.
- **Логи**: `./logs/errors.log`, crash-файлы в `./logs/`.
- **Задачи**: поля `progress`, `current_step`, `heartbeat_at` в GET /jobs.

## Масштабирование

- **Горизонтально API**: несколько реплик `whisper-stt` за балансировщиком (общий Redis).
- **Worker**: один worker на GPU рекомендуется при `WHISPER_MAX_CONCURRENT_JOBS=1`; несколько worker на разных GPU — возможны, но требуют отдельных очередей или партиционирования (не реализовано в текущей версии).
- **Redis**: persistence через AOF (`appendonly yes`).

## Безопасность

- API-токен через `Authorization: Bearer`.
- Секреты (`HF_TOKEN`, `WHISPER_API_TOKEN`) — только в `.env`, не коммитить.
- Worker скачивает аудио только по URL из payload задачи (доверие к клиенту API).
