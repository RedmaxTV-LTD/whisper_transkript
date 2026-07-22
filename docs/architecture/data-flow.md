# Потоки данных

Схемы: [diagrams.md](./diagrams.md) (последовательности и state machine).

## 1. Постановка задачи (клиент → API)

```
POST /transcribe { url | url_rx+url_tx, … }
        │
        ▼
  compute_dedup_key(body)
        │
        ▼
  claim_or_get_existing_job(redis)
        │
        ├─ dedup_lock (whisper:dedup_lock:{key})
        ├─ активная задача exists → 200 + existing job_id
        └─ иначе: UUID, SET whisper:job:{id}, SET whisper:dedup:{key}, LPUSH whisper:queue
        │
        ▼
  202 { job_id, dedup_key, status: "queued" }
```

Опрос:

- `GET /jobs/{job_id}` — по UUID из POST;
- `GET /jobs/by-dedup/{dedup_key}` — предпочтительно, если внешняя система привязана к файлу записи.

При GET для `queued`: если `job_id` нет в `whisper:queue`, API делает self-heal (атомарный LPUSH). Также вызывается stale-watchdog по heartbeat.

## 2. Обработка задачи (worker)

```
BRPOP whisper:queue
        │
        ▼
  mark_job_claimed_after_brpop → waiting_gpu
        │
        ▼
  acquire semaphore (max_concurrent по WHISPER_BACKEND)
        │  heartbeat при ожидании слота
        ▼
  run_transcription_pipeline(body)
        │
        ├─ download + ffmpeg → mono 16 kHz WAV
        ├─ [dual auto] channel_sync vs mix
        │
        ├─ WHISPER_BACKEND=local
        │     └─ faster-whisper (+ pyannote для mono diarize)
        │
        └─ WHISPER_BACKEND=openai
              ├─ dual: asyncio.gather(RX, TX) → OpenAI API
              └─ mono: один запрос (без pyannote)
        │
        ▼
  build_transcribe_response → result dict
        │
        ▼
  status=completed; cleanup temp files
```

Тяжёлая работа (ffmpeg STT/OpenAI sync-клиент) — в `ThreadPoolExecutor`; asyncio обновляет статусы через `step_callback`.

При openai dual RX/TX executor sizing: `max_concurrent_jobs * 2`, чтобы два канала одной задачи шли параллельно.

## 3. Статусы шагов пайплайна

| status | current_step (пример) | progress |
|--------|----------------------|----------|
| `waiting_gpu` | waiting for run slot / run slot busy | 0–2 |
| `downloading` | downloading audio / audio files | 5–8 |
| `syncing_channels` | syncing rx/tx to mix | 15 |
| `transcribing_rx` | transcribing rx/tx channels (openai: оба сразу) | 25–30 |
| `transcribing_tx` | transcribing tx (local sequential) | 55 |
| `transcribing_mix` | mono/stereo или openai mono | 40 |
| `merging_segments` | merging / diarizing | 40–85 |
| `completed` | completed | 100 |

Статус enum `waiting_gpu` сохраняется и для openai (совместимость клиентов); в `current_step` уточняется «OpenAI» vs «GPU».

## 4. Формирование ответа

Внутренний `TranscribeResult` → `TranscribeResponse` (`app/transcribe_schemas.py`):

| Поле | Описание |
|------|----------|
| `transcript` | Полный текст |
| `operator_text` / `client_text` | Агрегаты по ролям (dual_track или diarized) |
| `formatted_text` | Текст для UI |
| `diarized_segments` | speaker, role, start/end, text |
| `role_separation` | `dual_track`, `diarized`, `unavailable_mono`, `mix_fallback`, … |
| `note_ru` | Пояснение для интеграции |
| `mix_sync_*` | Метаданные auto/manual sync |

При **mono без диаризации** (в т.ч. openai mono): `operator_text` / `client_text` = `null`.

## 5. Dual-track: маппинг каналов

```
call_direction = incoming:
  RX → client    (speaker "RX")
  TX → operator  (speaker "TX")

call_direction = outgoing:
  RX → operator
  TX → client
```

Сегменты объединяются на общей шкале (с offset sync) и сортируются по `start_sec`.

## 6. Диаризация mono (только local)

```
mono WAV
  ├─ faster-whisper → parts (+ word timestamps)
  └─ pyannote → SPEAKER_* turns
        → align_whisper_segments
        → infer_speaker_role_map (catalog.json)
        → effective_segment_role (IVR → operator)
```

Требует: `WHISPER_BACKEND=local`, `WHISPER_DIARIZATION=1`, `HF_TOKEN`, принятые условия модели на HF.

## 7. OpenAI: детали вызова

1. Подготовка WAV (как local).
2. Если размер > `OPENAI_MAX_UPLOAD_MB` — ffmpeg → mp3.
3. `POST {OPENAI_BASE_URL}/audio/transcriptions`:
   - `model`, `language`, `response_format=verbose_json`;
   - `timestamp_granularities[]=segment` (+ `word` при split по паузам).
4. Retry на 429/5xx (`OPENAI_MAX_RETRIES`).
5. Сегменты → spelling fixes → те же структуры, что local.

Ошибки в job: `openai_http_*`, `openai_upload_too_large`, `openai_missing_api_key`, …

## 8. Sequence (кратко)

См. полную sequence-диаграмму в [diagrams.md §5](./diagrams.md).
