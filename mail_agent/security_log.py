"""Журналы. security_events.jsonl — только метаданные инцидента, без токенов, промпта и текста ответа."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log_prompt_leak(path: Path, run_id: str, location: str) -> dict[str, Any]:
    """Одно событие SYSTEM_PROMPT_LEAK строго в формате задания: время UTC, run_id, место, причина, действие.
    Набор полей фиксирован — содержимое ответа, токен и промпт сюда не попадают."""
    if location not in ("response", "tool_call"):
        raise ValueError(location)
    event: dict[str, Any] = {
        "ts": _utc_now(),
        "event": "SYSTEM_PROMPT_LEAK",
        "run_id": run_id,
        "location": location,
        "reason": "CANARY_MATCH",
        "action": "BLOCKED",
    }
    _append_jsonl(path, event)
    return event


def log_injection_suspected(path: Path, run_id: str, email_id: str, thread_id: str, rules: tuple[str, ...]) -> dict[str, Any]:
    """Письмо с признаками prompt injection отправлено в карантин. Пишем только ID и названия правил.
    Журнал отдельный (injection_events.jsonl), чтобы security_events.jsonl содержал только события из задания."""
    event = {
        "ts": _utc_now(),
        "event": "PROMPT_INJECTION_SUSPECTED",
        "run_id": run_id,
        "email_id": email_id,
        "thread_id": thread_id,
        "rules": list(rules),
        "action": "QUARANTINED",
    }
    _append_jsonl(path, event)
    return event


def log_audit(path: Path, run_id: str, kind: str, **fields: Any) -> None:
    """Аудит прошедших проверку действий: имя инструмента, thread_id, статус. Без текстов писем."""
    _append_jsonl(path, {"ts": _utc_now(), "run_id": run_id, "kind": kind, **fields})
