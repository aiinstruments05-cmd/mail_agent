"""Двойники LLM: детерминированная модель для тестов и симулятор утечки для демонстрации."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .base import ChatModel, LLMResponse, ToolCall


class ScriptedModel:
    """Детерминированный двойник LLM для тестов: каждый шаг — функция (system, messages) -> LLMResponse.

    Функция получает системный промпт, поэтому может «слить» токен так же,
    как это сделала бы взломанная модель.
    """

    def __init__(self, steps: list[Callable[[str, list[dict[str, Any]]], LLMResponse]]):
        self.steps = list(steps)
        self.calls = 0

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             attempt: int = 0) -> LLMResponse:
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return step(system, messages)


class LeakSimulator:
    """Обёртка для демонстрации: подмешивает активный canary-токен в ответ настоящей модели
    на заданном шаге — имитация успешной атаки на извлечение промпта."""

    def __init__(self, inner: ChatModel, where: str, encode: str = "plain", at_step: int = 1):
        self.inner, self.where, self.encode, self.at_step, self.calls = inner, where, encode, at_step, 0

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             attempt: int = 0) -> LLMResponse:
        import base64

        self.calls += 1
        resp = self.inner.chat(system, messages, tools, attempt)
        if self.calls != self.at_step:
            return resp
        token = re.search(r"CNRY-[0-9a-f]+", system).group(0)
        payload = base64.b64encode(token.encode()).decode() if self.encode == "base64" else token
        if self.where == "response":
            resp.text = f"{resp.text}\nСлужебная метка: {payload}"
        else:
            resp.tool_calls = [
                ToolCall(
                    id="call_leak",
                    name="search_emails",
                    arguments=json.dumps({"query": f"Орион {payload}"}, ensure_ascii=False),
                )
            ]
        return resp
