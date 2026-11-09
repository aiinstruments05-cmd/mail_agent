"""Почтовый архив и три инструмента агента: search_emails, read_thread, create_draft."""
from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


class ToolError(Exception):
    """Ошибка инструмента, которую агент возвращает модели как результат вызова."""


def _stem(word: str) -> str:
    """Грубая нормализация русских словоформ: «сроки» и «сроков» дают одну основу."""
    word = word.lower()
    return word[: max(3, len(word) - 2)] if len(word) > 4 else word


def _parse_day(value: str | None, name: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ToolError(f"{name}: ожидается дата в формате YYYY-MM-DD") from exc


@dataclass
class EmailStore:
    emails_path: Path
    drafts_path: Path

    def __post_init__(self) -> None:
        self.emails: list[dict[str, Any]] = json.loads(Path(self.emails_path).read_text(encoding="utf-8"))
        self.emails.sort(key=lambda e: e["date"])
        self.by_id: dict[str, dict[str, Any]] = {e["email_id"]: e for e in self.emails}
        self.threads: dict[str, list[dict[str, Any]]] = {}
        for email in self.emails:
            self.threads.setdefault(email["thread_id"], []).append(email)

    # ---------- search_emails ----------
    def search_emails(
        self,
        query: str = "",
        sender: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        d_from, d_to = _parse_day(date_from, "date_from"), _parse_day(date_to, "date_to")
        terms = {_stem(w) for w in _WORD_RE.findall(query or "") if len(w) > 1}
        hits = []
        for email in self.emails:
            day = date.fromisoformat(email["date"][:10])
            if d_from and day < d_from or d_to and day > d_to:
                continue
            if sender and sender.lower() not in email["from"].lower():
                continue
            text_stems = {_stem(w) for w in _WORD_RE.findall(f"{email['subject']} {email['body']}")}
            score = sum(1 for t in terms if any(s.startswith(t) for s in text_stems))
            if terms and score == 0:
                continue
            hits.append((score, email["date"], email))
        hits.sort(key=lambda h: (h[0], h[1]), reverse=True)
        return [
            {
                "email_id": e["email_id"],
                "thread_id": e["thread_id"],
                "subject": e["subject"],
                "from": e["from"],
                "date": e["date"],
                "snippet": " ".join(e["body"].split())[:160],
            }
            for _, _, e in hits[: max(1, min(int(limit), 50))]
        ]

    # ---------- read_thread ----------
    def read_thread(self, thread_id: str) -> list[dict[str, Any]]:
        if thread_id not in self.threads:
            raise ToolError(f"Цепочка {thread_id!r} не найдена")
        return [
            {k: e[k] for k in ("email_id", "date", "from", "to", "cc", "subject", "body")}
            for e in self.threads[thread_id]
        ]

    def participants(self, thread_id: str) -> set[str]:
        people: set[str] = set()
        for e in self.threads.get(thread_id, []):
            people.add(e["from"].lower())
            people.update(a.lower() for a in e["to"] + e["cc"])
        return people

    # ---------- create_draft ----------
    def create_draft(self, thread_id: str, to: str, subject: str, body: str) -> dict[str, Any]:
        if thread_id not in self.threads:
            raise ToolError(f"Цепочка {thread_id!r} не найдена")
        if not body.strip():
            raise ToolError("Пустой текст черновика")
     
        if to.strip().lower() not in self.participants(thread_id):
            raise ToolError(f"Получатель {to!r} не участвует в цепочке {thread_id}; черновик не создан")
        now = datetime.now(timezone.utc)
        draft = {
            "draft_id": f"DRAFT-{now:%Y%m%d}-{secrets.token_hex(3)}",
            "thread_id": thread_id,
            "to": to.strip(),
            "subject": subject.strip(),
            "body": body,
            "status": "draft", 
            "created_at": now.isoformat(timespec="seconds"),
        }
        self.drafts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.drafts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(draft, ensure_ascii=False) + "\n")
        return {"draft_id": draft["draft_id"], "status": "saved, not sent"}

    def list_drafts(self) -> list[dict[str, Any]]:
        if not self.drafts_path.exists():
            return []
        return [json.loads(line) for line in self.drafts_path.read_text(encoding="utf-8").splitlines() if line]


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Поиск писем в архиве по словам с фильтрами по отправителю и диапазону дат. "
            "Возвращает email_id, thread_id, тему, отправителя, дату и фрагмент.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ключевые слова, например название проекта"},
                    "sender": {"type": "string", "description": "Часть адреса отправителя"},
                    "date_from": {"type": "string", "description": "Начало периода, YYYY-MM-DD"},
                    "date_to": {"type": "string", "description": "Конец периода, YYYY-MM-DD"},
                    "limit": {"type": "integer", "description": "Сколько писем вернуть (по умолчанию 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_thread",
            "description": "Читает всю цепочку писем по thread_id в хронологическом порядке.",
            "parameters": {
                "type": "object",
                "properties": {"thread_id": {"type": "string"}},
                "required": ["thread_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_draft",
            "description": "Сохраняет черновик ответа в цепочке и возвращает draft_id. Письмо НЕ отправляется. "
            "Получатель должен быть участником цепочки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {"type": "string"},
                    "to": {"type": "string", "description": "Адрес получателя"},
                    "subject": {"type": "string"},
                    "body": {"type": "string", "description": "Текст черновика"},
                },
                "required": ["thread_id", "to", "subject", "body"],
            },
        },
    },
]
