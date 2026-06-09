# Архитектура Whisper STT

Документация описывает GPU-sidecar сервис распознавания речи для колл-центра: асинхронная постановка задач через HTTP, обработка на **faster-whisper** (CUDA) и опциональная **pyannote**-диаризация.

## Содержание

| Документ | Описание |
|----------|----------|
| [overview.md](./overview.md) | Обзор системы, роли процессов, внешние зависимости |
| [components.md](./components.md) | Модули `app/`, конфигурация, вспомогательные каталоги |
| [data-flow.md](./data-flow.md) | Потоки данных: API → Redis → worker → результат |
| [redis-and-jobs.md](./redis-and-jobs.md) | Модель задач, статусы, дедупликация, watchdog |
| [deployment.md](./deployment.md) | Docker Compose, тома, переменные окружения, модели |

## Краткая схема

```
Клиент (n8n, CRM, …)
        │  POST /transcribe
        ▼
┌───────────────────┐     LPUSH/BRPOP      ┌────────────────────┐
│  whisper-stt      │◄────────────────────►│  Redis             │
│  (FastAPI, role=api)│   whisper:queue      │  whisper:job:*     │
└───────────────────┘                      │  whisper:dedup:*   │
        │ GET /jobs/…                      └─────────┬──────────┘
        │                                            │ BRPOP
        │                                            ▼
        │                                  ┌────────────────────┐
        │                                  │  whisper-worker    │
        │                                  │  (role=worker, GPU)│
        │                                  └─────────┬──────────┘
        │                                            │
        │         HTTPS (скачивание записей)         ▼
        └──────────────────────────────────  faster-whisper + pyannote
```

## Связанные материалы

- [README.md](../../README.md) — быстрый старт, API, интеграция с n8n
- OpenAPI: `/docs`, `/redoc` (при запущенном сервисе)
