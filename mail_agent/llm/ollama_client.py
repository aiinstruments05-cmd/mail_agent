"""Клиент локальной Ollama (офлайн-вариант)."""
from __future__ import annotations

import json
import uuid
from typing import Any

from .base import LLMResponse, ToolCall
from .parsing import as_dict, parse_text_tool_calls, tool_names


class OllamaModel:
    """Нативный API Ollama (/api/chat). Нужен ради options.num_ctx: по умолчанию у Ollama
    окно 4096 токенов, и при переполнении она молча отрезает НАЧАЛО диалога — то есть
    системный промпт. Через OpenAI-совместимый /v1 размер окна не задать."""

    def __init__(self, host: str, model: str, temperature: float = 0.0, num_ctx: int = 8192, timeout: float = 900.0,
                 transport: Any = None):
        import httpx

        self.http = httpx.Client(base_url=host.rstrip("/"), timeout=timeout, transport=transport)
        self.model, self.temperature, self.num_ctx = model, temperature, num_ctx

    @staticmethod
    def _to_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [
                        {"function": {"name": c["function"]["name"], "arguments": as_dict(c["function"]["arguments"])}}
                        for c in m["tool_calls"]
                    ],
                })
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             attempt: int = 0) -> LLMResponse:
        """attempt=0 — обычный режим: вызовы разбирает Ollama.

        Если 7B-модель сгенерировала вызов, который парсер Ollama не смог разобрать, Ollama молча
        возвращает пустой ответ. Поэтому повторы (attempt>=1) идут в «сыром» режиме: без поля tools
        Ollama отдаёт текст модели как есть (qwen всё равно пишет <tool_call>{...}</tool_call>, так как
        инструменты описаны в системном промпте и в истории), и вызов разбирает наш терпимый парсер.
        На второй и следующих повторах немного поднимаем temperature, чтобы не получить тот же вывод.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *self._to_ollama(messages)],
            "stream": False,
            "options": {"temperature": self.temperature + 0.3 * max(0, attempt - 1), "num_ctx": self.num_ctx},
        }
        if attempt == 0:
            body["tools"] = tools
        r = self.http.post("/api/chat", json=body)
        r.raise_for_status()
        msg = r.json()["message"]
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            args = tc["function"].get("arguments", {})
            calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}_{i}",
                name=tc["function"]["name"],
                arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
            ))
        text = msg.get("content") or ""
        if not calls:
            text, calls = parse_text_tool_calls(text, tool_names(tools))
        return LLMResponse(text=text, tool_calls=calls)
