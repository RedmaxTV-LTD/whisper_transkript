# Whisper STT (GPU sidecar)

## Структура репозитория (whisper/)

| Путь | Назначение |
|------|------------|
| `app/` | FastAPI (`python -m app`), модели запроса, Redis jobs, пайплайн транскрипции |
| `app/worker.py` | Фоновый worker: `python -m app.worker` |
| `download_models.py` | Загрузка моделей **CTranslate2** в подкаталоги корня volume |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Сборка и запуск контейнера |
| `models/` | Том для **смонтированных** моделей: ожидаются **подкаталоги** с `model.bin`. Файлы `*.pt` здесь **не используются** рантаймом (см. `models/README.md`). |
| `spelling_config/dictionary.json` | Словарь пост-обработки текста (ошибка → канон); в compose монтируется на `/config/spelling/`, правки на хосте подхватываются без перезапуска. |
| `speaker_roles_config/catalog.json` | Фразы для ролей **система / оператор / клиент** по тексту начала диалога (`match_window_sec`); том → `/config/speaker_roles/`. |

**Перед первым запуском:** один раз загрузите CTranslate2 в `whisper/models/` (на хосте это `./whisper/models` при стандартном compose из корня):  
`docker compose --profile whisper run --rm whisper-stt python3 /app/download_models.py`  
По умолчанию в compose: том `./whisper/models` → `/models`, `WHISPER_MODEL_PATH=/models/turbo`.

Сервис принимает **ссылку на запись**, ставит задачу в **Redis** и обрабатывает её процессом **whisper-worker**: скачивание, **mono 16 kHz WAV**, распознавание **faster-whisper** на GPU. HTTP **POST /transcribe** сразу возвращает `job_id`; готовый текст — в **GET /jobs/{job_id}** при `status=completed`.

## Моно и роли «Оператор / Клиент»

При **одной смешанной дорожке** нельзя надёжно разделить реплики без **диаризации** или **отдельных каналов**. API возвращает единый `transcript` и `formatted_text` с заголовком «Транскрипт:»; поля `operator_text` / `client_text` остаются `null`, в `note_ru` — пояснение.

Когда появятся **rx/tx** или **стерео с раздельными каналами**, можно расширить сервис: не сводить каналы в mono и транскрибировать дорожки отдельно.

## Переменные окружения

См. `.env.example`. Ключевые:

| Переменная | Назначение |
|------------|------------|
| `WHISPER_MAX_CONCURRENT_JOBS` | Максимум одновременных задач на **worker** (семафор: слот занят и на скачивании, и на GPU). При значении **1** воркер **ждёт завершения** текущей задачи перед следующим `BRPOP` из Redis (стабильнее планировщика). |
| `WHISPER_WORKER_QUEUE_FLUSH_ON_START` | При старте **whisper-worker**: удалить `whisper:queue` и заново **LPUSH** все задачи из Redis со статусом `queued` или `waiting_gpu` (по умолчанию `1` / `true`) |
| `REDIS_URL` | Подключение Redis (в Docker по умолчанию `redis://redis:6379/0`) |
| `WHISPER_JOB_HEARTBEAT_SEC` | Интервал обновления `heartbeat_at` / `updated_at` в worker (сек) |
| `WHISPER_JOB_STALE_SEC` | Если heartbeat старше этого — задача помечается `stale_failed` (watchdog + GET) |
| `WHISPER_WORKER_ALIVE_TTL_SEC` | TTL ключа `whisper:worker:alive` (сек); API по нему понимает, что worker запущен |
| `WHISPER_WORKER_ALIVE_REFRESH_SEC` | Как часто worker продлевает ключ (сек) |
| `WHISPER_PROCESS_ROLE` | `api` — только FastAPI; `worker` — очередь и GPU (в compose задано по сервисам) |
| `WHISPER_MODEL_PATH` | Путь к каталогу модели внутри контейнера (по умолчанию в compose: `/models/turbo`) |
| `WHISPER_DEVICE` | `cuda` или `cpu` |
| `WHISPER_COMPUTE_TYPE` | например `float16`, `int8_float16` |
| `WHISPER_LANGUAGE` | Пусто — авто; иначе код (`ru`) |
| `WHISPER_API_TOKEN` | Если задан — нужен `Authorization: Bearer …` |
| `WHISPER_MODELS_DIR` | Корень для `download_models.py` (по умолчанию `/models`) |
| `WHISPER_MODELS_FORCE` | `1`/`true` — удалить и перекачать модели из списка |
| `WHISPER_HF_TOKEN` / `HF_TOKEN` | Опционально токен Hugging Face |
| `WHISPER_SPELLING_CONFIG_DIR` | Хост-каталог с `dictionary.json` (compose: `./spelling_config` → `/config/spelling`) |
| `WHISPER_SPELLING_DICT_PATH` | Путь к JSON **внутри контейнера** (по умолчанию `/config/spelling/dictionary.json` в compose) |
| `WHISPER_SPELLING_FIXES` | `0` — отключить замены |
| `WHISPER_SPEAKER_ROLES_CATALOG` | `0` — роли только по pyannote ID (00/01/02); `1` — по файлу каталога |
| `WHISPER_SPEAKER_ROLES_CATALOG_PATH` | JSON каталога **внутри контейнера** (compose: `/config/speaker_roles/catalog.json`) |
| `WHISPER_SPEAKER_ROLES_CONFIG_DIR` | Хост-каталог с `catalog.json` |

## Загрузка моделей (CTranslate2 / faster-whisper)

Модели **не** в формате OpenAI `.pt`; сервис использует **faster-whisper** и каталоги с `model.bin`.

Скрипт `download_models.py` скачивает в подкаталоги корня (например `/models/tiny`, `/models/turbo`). Укажите `WHISPER_MODEL_PATH` на нужный подкаталог.

Из каталога `whisper/` (контейнер должен видеть тот же volume, что и у сервиса):

```bash
docker compose run --rm whisper-stt python3 /app/download_models.py
# только выбранные:
docker compose run --rm whisper-stt python3 /app/download_models.py large-v3
# принудительно обновить набор по умолчанию:
docker compose run --rm -e WHISPER_MODELS_FORCE=1 whisper-stt python3 /app/download_models.py
```

Из корня репозитория с профилем `whisper`:

```bash
docker compose --profile whisper run --rm whisper-stt python3 /app/download_models.py
```

## Запуск

Из каталога `whisper/`:

```bash
docker compose --env-file .env up --build -d
```

Поднимаются три сервиса: **redis**, **whisper-stt** (API), **whisper-worker** (GPU + очередь). API не загружает модель Whisper в память; только worker.

## API

- `GET /health` — Redis, флаг **`worker_available`** (есть ли свежий ключ `whisper:worker:alive` от whisper-worker).
- `POST /transcribe` — тот же JSON, что раньше (`url` или `url_rx`/`url_tx`/`url_mix`/`sync`/…). Ответ **202** (новая задача) или **200** (уже существующая по `dedup_key`):  
  `{"job_id","dedup_key","status","existing"}`.
- `GET /jobs/{job_id}` — статус, `heartbeat_at`, по завершении поле `result` (схема как у прежнего синхронного ответа). Дополнительно: **`worker_available`**, **`retry_recommended`**, **`client_hint`**, для **`queued`** — **`in_redis_queue`** (есть ли `job_id` в `whisper:queue`; при рассинхроне API при опросе может выполнить **LPUSH**). Подсказки — при остановке worker, `stale_failed`, отсутствии задачи в очереди.
- `GET /jobs/by-dedup/{dedup_key}` — то же по стабильному ключу дедупликации (удобно для n8n без сохранения `job_id`).

После **перезапуска whisper-worker** при включённом **`WHISPER_WORKER_QUEUE_FLUSH_ON_START`** (по умолчанию да) список `whisper:queue` сбрасывается и все ожидающие задачи (`queued` / `waiting_gpu`) заново ставятся в очередь из записей в Redis. Статус **`waiting_gpu`** означает «воркер уже забрал задачу из списка» и дальше — ожидание **слота выполнения** (`WHISPER_MAX_CONCURRENT_JOBS`), при этом другая задача может ещё только **скачивать** аудио, без нагрузки на GPU. Незавершённые активные распознавания помечаются `failed` с префиксом `worker_orphaned` — клиенту нужен повторный **POST /transcribe**. При **`WHISPER_DIARIZATION=1`** worker **сначала полностью** загружает pyannote на GPU, **затем** начинает `BRPOP` очереди — иначе параллельная загрузка pyannote и faster-whisper на одном CUDA могла зависать.

### `dedup_key`

Стабильный ключ задачи: basename **без расширения** для `url_mix`, если он есть; иначе общий stem для пары `-rx`/`-tx` (например `external-…-rx.wav` и `external-…-tx.wav` → один ключ без суффиксов); иначе basename одного `url`; иначе хэш от JSON payload.

### Статусы задачи

`queued`, `waiting_gpu`, `downloading`, `transcribing_rx`, `transcribing_tx`, `transcribing_mix`, `syncing_channels`, `merging_segments`, `completed`, `failed`, `cancelled`, `stale_failed`.

### Интеграция (n8n, своя очередь и т.п.)

1. **POST** `/transcribe` с телом записи; в ответе — **`job_id`** и **`dedup_key`**. При повторном POST с теми же URL, пока задача не в терминальном статусе, вернётся **тот же** `job_id` и тот же `dedup_key`.
2. **Опрашивать статус** удобнее по **`dedup_key`**, если ваша очередь привязана к **конкретной записи/файлу**, а не к UUID: **GET** `/jobs/by-dedup/{dedup_key}` — в Redis всегда указывает на **актуальный** `job_id` для этой пары URL (после рестарта воркера, `queue reset`, повторного POST дедуп вернёт ту же задачу, но `job_id` в памяти внешней системы мог устареть).
3. В пути `dedup_key` передавайте **URL-encoded** (например `encodeURIComponent` в JS или эквивалент в n8n).
4. При `status=completed` взять `result`; при `status=failed` или `stale_failed` — поле `error` и при необходимости снова **POST /transcribe**. Повторный POST с теми же URL **не создаёт** вторую задачу, пока текущая не в `failed` / `cancelled` / `stale_failed`; после этого появится **новый** `job_id`, а **`dedup_key`** по тем же URL обычно совпадёт — опрос по **by-dedup** по-прежнему найдёт актуальную задачу.

**GET /jobs/{job_id}** остаётся корректным, если вы **надёжно сохраняете** `job_id` из ответа **POST** и не смешиваете его с другими событиями очереди.

При заданном `WHISPER_API_TOKEN` передавайте заголовок `Authorization: Bearer <token>`.
