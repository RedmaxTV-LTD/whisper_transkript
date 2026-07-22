# Схемы архитектуры

Визуальное описание Whisper STT sidecar. Диаграммы в формате Mermaid (рендерятся в GitHub, VS Code, многих Markdown-просмотрщиках).

---

## 1. Контекст системы

Кто взаимодействует с сервисом.

```mermaid
flowchart LR
  Integrator[Integrator_n8n_CRM] -->|HTTPS_JSON_API| Whisper[Whisper_STT_sidecar]
  Whisper -->|download_audio| ATS[ATS_ObjectStorage]
  Whisper -->|WHISPER_BACKEND_openai| OpenAI[OpenAI_Transcriptions]
  Whisper -->|local_diarize| HF[HuggingFace_pyannote]
```

| Участник | Роль |
|----------|------|
| Integrator (n8n, CRM) | `POST /transcribe`, опрос `GET /jobs/*` |
| Whisper STT | Очередь, скачивание, STT, ответ |
| АТС / storage | Источник HTTPS URL записей |
| OpenAI API | Облачное распознавание (опционально) |
| Hugging Face | Модели pyannote (только local + diarize) |
---

## 2. Контейнеры (Docker Compose)

Три сервиса в сети `whisper-net`. Один Docker-образ для API и worker.

```mermaid
flowchart TB
  subgraph host [Host]
    VolModels["volume_models"]
    VolLogs["volume_logs"]
    VolSpell["volume_spelling_config"]
    VolRoles["volume_speaker_roles"]
    Env[".env"]
  end

  subgraph net [whisper-net]
    Redis["redis:7-alpine\nAOF persistence"]
    API["whisper-stt\npython -m app\nrole=api\nport 19900"]
    Worker["whisper-worker\npython -m app.worker\nrole=worker"]
  end

  Env --> API
  Env --> Worker
  VolModels --> Worker
  VolLogs --> API
  VolLogs --> Worker
  VolSpell --> Worker
  VolRoles --> Worker
  API <-->|REDIS_URL| Redis
  Worker <-->|REDIS_URL| Redis
  Client[Client] -->|WHISPER_HOST_PORT| API
```

| Контейнер | Назначение | GPU |
|-----------|------------|-----|
| `redis` | Очередь `whisper:queue`, job JSON, dedup, alive | нет |
| `whisper-stt` | FastAPI: enqueue + status | runtime nvidia, модель **не** грузит |
| `whisper-worker` | BRPOP → пайплайн STT | да (local) / HTTP к OpenAI (openai) |

---

## 3. Разделение процессов API / Worker

```mermaid
flowchart TB
  subgraph apiProc [whisper-stt]
    FastAPI[FastAPI_routes]
    Claim[claim_or_get_existing_job]
    Status[GET_jobs_watchdog_heal]
  end

  subgraph redisKeys [Redis]
    Q[whisper:queue]
    Job[whisper:job:id]
    Dedup[whisper:dedup:key]
    Alive[whisper:worker:alive]
  end

  subgraph workerProc [whisper-worker]
    BRPOP[BRPOP_loop]
    Sem[Semaphore_max_concurrent]
    Pipe[run_transcription_pipeline]
    HB[heartbeat_watchdog]
  end

  FastAPI --> Claim --> Q
  Claim --> Job
  Claim --> Dedup
  Status --> Job
  Status --> Alive
  BRPOP --> Q
  BRPOP --> Sem --> Pipe
  HB --> Job
  HB --> Alive
```

---

## 4. Выбор бэкенда STT

Переключение только через `.env`: `WHISPER_BACKEND`.

```mermaid
flowchart TD
  Start[Job_dequeued] --> Prep[Download_ffmpeg_sync]
  Prep --> Backend{WHISPER_BACKEND}
  Backend -->|local| LocalPath[faster-whisper_on_GPU]
  Backend -->|openai| OpenPath[OpenAI_HTTP_API]
  LocalPath --> Diarize{diarize_enabled?}
  Diarize -->|yes_mono| Pyannote[pyannote]
  Diarize -->|no_or_dual| MergeLocal[build_result]
  Pyannote --> MergeLocal
  OpenPath --> Dual{url_rx_tx?}
  Dual -->|yes| Parallel[asyncio.gather_RX_and_TX]
  Dual -->|no| MonoOA[single_transcription]
  Parallel --> MergeOA[build_dual_track_result]
  MonoOA --> MergeOA2[TranscribeResult_no_diarize]
  MergeLocal --> Resp[build_transcribe_response]
  MergeOA --> Resp
  MergeOA2 --> Resp
  Resp --> Done[Redis_status_completed]
```

### Параллелизм

```mermaid
flowchart LR
  subgraph localBackend [local]
    L1["WHISPER_MAX_CONCURRENT_JOBS\ndefault 1"]
  end
  subgraph openaiBackend [openai]
    O1["OPENAI_MAX_CONCURRENT_JOBS\ndefault 8"]
  end
  Settings[settings.max_concurrent_jobs] --> Sem[asyncio.Semaphore]
  localBackend -.->|if_backend_local| Settings
  openaiBackend -.->|if_backend_openai| Settings
```

---

## 5. Последовательность: happy path

```mermaid
sequenceDiagram
  participant C as Client
  participant API as whisper-stt
  participant R as Redis
  participant W as whisper-worker
  participant S as Audio_HTTPS
  participant STT as Local_or_OpenAI

  C->>API: POST /transcribe
  API->>R: SET job + LPUSH queue
  API-->>C: 202 job_id dedup_key

  loop Poll
    C->>API: GET /jobs/by-dedup/key
    API->>R: GET job
    API-->>C: status progress
  end

  W->>R: BRPOP queue
  W->>R: waiting_gpu / heartbeat
  W->>S: download audio
  W->>STT: transcribe
  STT-->>W: segments text
  W->>R: completed + result
  C->>API: GET /jobs/...
  API-->>C: status=completed result
```

---

## 6. Dual-track RX/TX

```mermaid
flowchart TB
  Req[url_rx + url_tx + call_direction] --> DL[download both]
  DL --> Sync{sync.mode}
  Sync -->|off| T1[transcribe RX TX]
  Sync -->|manual| OffM[apply offset_*]
  Sync -->|auto| Corr[correlate vs url_mix]
  Corr -->|trusted| OffA[apply offsets]
  Corr -->|untrusted| FB{fallback}
  FB -->|use_mix| MixOnly[transcribe mix]
  FB -->|none_use_rx_tx| T1
  OffM --> T1
  OffA --> T1
  T1 --> Roles{call_direction}
  Roles -->|incoming| MapIn["RX=client TX=operator"]
  Roles -->|outgoing| MapOut["RX=operator TX=client"]
  MapIn --> Merge[merge sort by time]
  MapOut --> Merge
  MixOnly --> Result
  Merge --> Result[TranscribeResponse dual_track]
```

При `WHISPER_BACKEND=openai` шаги «transcribe RX/TX» выполняются **параллельно** (`asyncio.gather`).

---

## 7. Модули приложения (зависимости)

```mermaid
flowchart TB
  main[main.py] --> job_store
  main --> dedup
  main --> job_watchdog
  worker[worker.py] --> job_store
  worker --> pipeline[transcribe_pipeline]
  worker --> schemas[transcribe_schemas]
  pipeline --> audio_prep
  pipeline --> channel_sync
  pipeline --> local_eng[transcribe_engine]
  pipeline --> oa_eng[openai_engine]
  pipeline --> diarize[diarize_engine]
  local_eng --> align[align_speakers]
  local_eng --> spelling[spelling_fixes]
  local_eng --> roles_cat[speaker_roles_catalog]
  oa_eng --> spelling
  schemas --> speaker_roles
  settings[settings.py] -.-> worker
  settings -.-> pipeline
  settings -.-> main
```

---

## 8. Жизненный цикл задачи

```mermaid
stateDiagram-v2
  [*] --> queued: POST_transcribe
  queued --> waiting_gpu: BRPOP
  waiting_gpu --> downloading: sem_acquired
  downloading --> syncing_channels: dual_auto
  downloading --> transcribing_rx: dual
  downloading --> transcribing_mix: mono
  syncing_channels --> transcribing_rx
  transcribing_rx --> transcribing_tx: local_sequential
  transcribing_rx --> merging_segments: openai_parallel_done
  transcribing_tx --> merging_segments
  transcribing_mix --> merging_segments: optional_diarize
  merging_segments --> completed
  downloading --> failed
  transcribing_rx --> failed
  waiting_gpu --> stale_failed: heartbeat_timeout
  downloading --> failed: worker_orphaned_restart
  completed --> [*]
  failed --> [*]
  stale_failed --> [*]
```

---

## 9. Развёртывание и тома

```mermaid
flowchart LR
  subgraph hostFS [Host_filesystem]
    M["./whisper/models"]
    L["./logs"]
    SP["./spelling_config"]
    SR["./speaker_roles_config"]
  end
  subgraph containers [Containers]
    API2[whisper-stt]
    W2[whisper-worker]
  end
  M -->|"/models"| W2
  L -->|"/logs"| API2
  L -->|"/logs"| W2
  SP -->|"/config/spelling"| W2
  SR -->|"/config/speaker_roles"| W2
```

---

## Связанные документы

- Текстовый обзор: [overview.md](./overview.md)
- Модули: [components.md](./components.md)
- Детали потоков: [data-flow.md](./data-flow.md)
- Redis/jobs: [redis-and-jobs.md](./redis-and-jobs.md)
- Env и Compose: [deployment.md](./deployment.md)
