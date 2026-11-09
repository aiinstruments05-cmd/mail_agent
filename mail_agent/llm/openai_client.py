"""Клиент внешнего OpenAI-совместимого API (по умолчанию)."""
from __future__ import annotations

import json
import uuid
from typing import Any

from .base import LLMError, LLMResponse, ToolCall
from .parsing import parse_text_tool_calls, tool_names


class OpenAICompatibleModel:
    """Внешний OpenAI-совместимый Chat Completions API с function calling.

    Один клиент для OpenAI, OpenRouter, DeepSeek, Groq, vLLM и т. п.: отличаются только base_url,
    ключ и имя модели. Сетевые сбои, 429 и 5xx SDK повторяет сам (max_retries, экспоненциальная пауза).
    Ключ берётся из окружения/.env и никуда, кроме заголовка Authorization, не попадает.
    """

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float | None = 0.0,
                 timeout: float = 120.0, max_retries: int = 3, http_client: Any = None):
        from openai import OpenAI

        if not api_key:
            raise LLMError("Не задан ключ API: укажите MAIL_AGENT_LLM_API_KEY (или OPENAI_API_KEY) в окружении или в .env")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries,
                             http_client=http_client)
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             attempt: int = 0) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": tools,
            "tool_choice": "auto",
        }
        if self.temperature is not None:
            kwargs["temperature"] = min(self.temperature + 0.3 * attempt, 1.0)
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise LLMError("API отклонил ключ (401): проверьте MAIL_AGENT_LLM_API_KEY") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"Модель {self.model!r} или адрес API не найдены (404): проверьте "
                           "MAIL_AGENT_LLM_MODEL и MAIL_AGENT_LLM_BASE_URL") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Нет соединения с API: проверьте MAIL_AGENT_LLM_BASE_URL и сеть") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"API вернул ошибку {exc.status_code}") from exc
        if not resp.choices:
            return LLMResponse()
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            if getattr(tc, "type", "function") != "function" or tc.function is None:
                continue
            args = tc.function.arguments
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}", name=tc.function.name, arguments=args))
        text = msg.content or ""
        if not calls:
            text, calls = parse_text_tool_calls(text, tool_names(tools))
        return LLMResponse(text=text, tool_calls=calls)
