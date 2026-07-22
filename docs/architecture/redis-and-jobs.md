# Redis и модель задач

## Ключи Redis

| Ключ | Тип | Назначение |
|------|-----|------------|
| `whisper:queue` | LIST | Очередь `job_id` (LPUSH / BRPOP) |
| `whisper:job:{job_id}` | STRING (JSON) | Полная запись `TranscribeJobRecord` |
| `whisper:dedup:{dedup_key}` | STRING | Маппинг dedup_key → актуальный job_id |
| `whisper:dedup_lock:{dedup_key}` | STRING | Кратковременная блокировка при claim |
| `whisper:worker:alive` | STRING + TTL | Heartbeat worker (наличие = worker запущен) |

## TranscribeJobRecord

```json
{
  "job_id": "uuid",
  "dedup_key": "external-7164-…",
  "status": "queued",
  "payload": { "url_rx": "…", "url_tx": "…", "call_direction": "incoming" },
  "result": null,
  "error": null,
  "created_at": "2026-06-09T…",
  "started_at": null,
  "updated_at": "2026-06-09T…",
  "completed_at": null,
  "progress": 0,
  "current_step": "",
  "heartbeat_at": null,
  "attempts": 0
}
```

## Статусы задачи

| status | Описание |
|--------|----------|
| `queued` | В очереди, ожидает BRPOP |
| `waiting_gpu` | Worker забрал задачу, ждёт слот семафора |
| `downloading` | Скачивание аудио |
| `transcribing_rx` | Распознавание канала RX |
| `transcribing_tx` | Распознавание канала TX |
| `transcribing_mix` | Распознавание mono/mix |
| `syncing_channels` | Auto-sync RX/TX |
| `merging_segments` | Слияние сегментов / диаризация |
| `completed` | Успех, `result` заполнен |
| `failed` | Ошибка в `error` |
| `cancelled` | Отменена (зарезервировано) |
| `stale_failed` | Heartbeat устарел (`WHISPER_JOB_STALE_SEC`) |

**Терминальные**: `completed`, `failed`, `cancelled`, `stale_failed`.

## Дедупликация

### Алгоритм `dedup_key`

1. Если есть `url_mix` → basename без расширения.
2. Иначе если `url_rx` + `url_tx` → общий stem (убрать `-rx`/`-tx`, longest common prefix ≥ 8 символов).
3. Иначе если один `url` → basename без расширения.
4. Иначе SHA-256 от канонического JSON payload.

### Поведение POST /transcribe

- Пока задача с тем же `dedup_key` в **активном** статусе (включая `completed`) — возвращается существующий `job_id` (HTTP 200).
- Новая задача создаётся только если предыдущая в `failed`, `cancelled` или `stale_failed` (HTTP 202).

Активные для дедупа: все не-terminal + **`completed`** (повторный POST не перезапускает успешную транскрипцию).

## Очередь и recovery

### При старте whisper-worker

Если `WHISPER_WORKER_QUEUE_FLUSH_ON_START=1` (по умолчанию):

1. Удаляется список `whisper:queue`.
2. Все задачи со статусом `queued` или `waiting_gpu` заново **LPUSH** в очередь.
3. Задачи в процессе обработки (`downloading`, `transcribing_*`, …) → `failed` с `worker_orphaned`.

### Self-heal в API

При `GET /jobs/*` для статуса `queued`: если `job_id` отсутствует в `whisper:queue`, API выполняет атомарный LPUSH (Lua-скрипт `_ENQUEUE_IF_ABSENT_LUA`).

## Watchdog

Два механизма помечают задачи как устаревшие:

1. **Worker-side** (`scan_stale_jobs` каждые 60 с): heartbeat старше `WHISPER_JOB_STALE_SEC`.
2. **API-side** (`maybe_mark_job_stale`): при каждом GET /jobs проверяется heartbeat.

Worker alive:

- Worker периодически обновляет `whisper:worker:alive` с TTL `WHISPER_WORKER_ALIVE_TTL_SEC`.
- API в `/health` и в ответах jobs возвращает `worker_available`.

## Семафор параллелизма

Параллелизм зависит от `WHISPER_BACKEND`:

| Backend | Переменная | Default |
|---------|------------|---------|
| `local` | `WHISPER_MAX_CONCURRENT_JOBS` | `1` |
| `openai` | `OPENAI_MAX_CONCURRENT_JOBS` | `8` |

Активный лимит (`settings.max_concurrent_jobs`) ограничивает одновременные задачи **на одном worker**:

- Слот занят с момента acquire до release (скачивание + STT).
- При значении `1` следующая задача остаётся в `waiting_gpu` с heartbeat «run slot busy».
- Для local это стабилизирует CUDA; для openai значение можно поднять под RPM tier аккаунта.

## Типичные сценарии сбоев

| Ситуация | Статус / поле | Действие клиента |
|----------|---------------|------------------|
| Worker остановлен | `worker_available=false`, задача в `queued` | Запустить worker, дождаться или POST снова |
| Рестарт worker mid-job | `failed`, `worker_orphaned` | POST /transcribe с тем же JSON |
| Длинный звонок | `stale_failed` | Увеличить `WHISPER_JOB_STALE_SEC` или повторить POST |
| Рассинхрон очереди | `queued`, `in_redis_queue=false` | API попытается LPUSH; иначе повторный POST |
| dedup lock busy | HTTP 503 | Кратковременно повторить POST |
