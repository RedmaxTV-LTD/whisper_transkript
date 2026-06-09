"""Тело запроса POST /transcribe (валидация URL / RX-TX / sync)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class SyncOptionsBody(BaseModel):
    mode: Literal["auto", "manual", "off"] = Field(
        default="off",
        description="**auto** — оценка сдвига RX/TX к `url_mix`; **manual** — заданные `offset_*`; **off** — без синхронизации (без `url_mix`).",
    )
    max_offset_sec: float | None = Field(
        None,
        ge=0,
        description="Максимум |offset| в секундах по каналу; при превышении канал не доверяется (auto) или срабатывает fallback (manual). None → WHISPER_SYNC_MAX_OFFSET_SEC",
    )
    fallback: Literal["none", "use_mix", "use_rx_tx"] = Field(
        default="none",
        description="При полном сбое **auto** (оба канала недоверены) или **manual** (offset вне max): **none** — RX+TX без сдвигов; **use_mix** — один транскрипт по `url_mix`; **use_rx_tx** — явно RX+TX без сдвигов (как none по результату, иной смысл в логах/заметке).",
    )
    offset_rx_sec: float = Field(
        default=0.0,
        description="Секунды, прибавляемые к таймингам RX (только `mode: manual`)",
    )
    offset_tx_sec: float = Field(
        default=0.0,
        description="Секунды, прибавляемые к таймингам TX (только `mode: manual`)",
    )


class TranscribeBody(BaseModel):
    url: HttpUrl | None = Field(
        None,
        description="Одна запись (моно или стерео). Не используется вместе с url_rx/url_tx.",
    )
    url_rx: HttpUrl | None = Field(
        None,
        description="Моно-файл канала **RX** по HTTPS. Смысл дорожки задаётся вместе с `call_direction` (см. описание `call_direction`).",
    )
    url_tx: HttpUrl | None = Field(
        None,
        description="Моно-файл канала **TX** по HTTPS. Смысл дорожки задаётся вместе с `call_direction` (см. описание `call_direction`).",
    )
    call_direction: Literal["incoming", "outgoing"] | None = Field(
        None,
        description=(
            "Направление звонка **только при паре `url_rx` + `url_tx`** (в этом режиме поле обязательно). "
            "От него зависит сопоставление RX/TX с ролями **Клиент** и **Оператор** в ответе API.\n\n"
            "- **`incoming`** — **входящий** звонок на АТС: **RX = клиент**, **TX = оператор** "
            "(типичные имена файлов с префиксом `external-…`, суффиксы `-rx.wav` / `-tx.wav`).\n"
            "- **`outgoing`** — **исходящий** звонок с АТС: **RX = оператор**, **TX = клиент** "
            "(типичные имена с префиксом `out-…`, суффиксы `-rx.wav` / `-tx.wav`).\n\n"
            "При одном поле `url` передавать `call_direction` нельзя."
        ),
    )
    diarize: bool | None = Field(
        None,
        description="Диаризация pyannote для этого запроса; None — переменная WHISPER_DIARIZE_DEFAULT (если WHISPER_DIARIZATION=1). В режиме url_rx+url_tx не выполняется.",
    )
    url_mix: HttpUrl | None = Field(
        None,
        description=(
            "Опциональная **общая mono** того же звонка. Только с `url_rx` и `url_tx`. "
            "Для `sync.mode=auto` обязательна; для `sync.fallback=use_mix` — обязательна."
        ),
    )
    sync: SyncOptionsBody | None = Field(
        None,
        description="Параметры синхронизации RX/TX. Если указан только `url_mix`, подставляется `sync: { mode: auto }`.",
    )

    @model_validator(mode="after")
    def _url_mode(self) -> TranscribeBody:
        dual = self.url_rx is not None and self.url_tx is not None
        partial_dual = (self.url_rx is not None) ^ (self.url_tx is not None)
        if partial_dual:
            raise ValueError("Для двухканального режима укажите оба поля url_rx и url_tx")
        if dual:
            if self.call_direction is None:
                raise ValueError("Укажите call_direction: incoming или outgoing вместе с url_rx и url_tx")
            if self.url is not None:
                raise ValueError("Нельзя одновременно передавать url и пару url_rx/url_tx")
            if self.url_mix is not None and self.sync is None:
                return self.model_copy(update={"sync": SyncOptionsBody(mode="auto")})
            sync = self.sync
            if sync is not None and sync.fallback == "use_mix" and self.url_mix is None:
                raise ValueError("sync.fallback=use_mix требует url_mix")
            if sync is not None and sync.mode == "auto" and self.url_mix is None:
                raise ValueError("sync.mode=auto требует url_mix")
            if self.url_mix is not None and sync is not None and sync.mode == "off":
                raise ValueError("При sync.mode=off не передавайте url_mix (или включите sync.mode=auto|manual)")
        else:
            if self.call_direction is not None:
                raise ValueError("call_direction допустим только с url_rx и url_tx")
            if self.url_mix is not None:
                raise ValueError("url_mix допустим только вместе с url_rx и url_tx")
            if self.sync is not None:
                raise ValueError("sync допустим только вместе с url_rx и url_tx")
            if self.url is None:
                raise ValueError("Укажите url или оба url_rx и url_tx")
        return self
