"""Общий контракт LLM-клиентов: формат ответа, протокол ChatModel и ошибка обращения к API.

Агент зависит только от этого протокола, поэтому клиенты взаимозаменяемы:
внешний API, локальная Ollama или тестовый двойник.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str 

@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             attempt: int = 0) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Сбой обращения к внешнему API. Сообщение без ключа и без содержимого запроса."""
