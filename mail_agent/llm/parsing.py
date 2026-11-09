"""Разбор вызовов инструментов, которые модель написала текстом, а не через function calling."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from .base import ToolCall

_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)


def _loads_tolerant(raw: str) -> Any:
    """JSON от 7B-модели: допускаем буквальные переносы строк (strict=False), обёртку ```json и мусор
    после закрывающей скобки. Ничего не «чиним» по смыслу — либо разобрали, либо None."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    for candidate in (raw, raw[: raw.rfind("}") + 1] if "}" in raw else raw):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def parse_text_tool_calls(text: str, allowed: set[str]) -> tuple[str, list[ToolCall]]:
    """Достаёт вызовы, которые модель написала текстом (<tool_call>{...}</tool_call> или голый JSON).

    Принимаются только инструменты из allowed. Разобранные блоки вырезаются из текста; всё, что
    вернула эта функция, дальше проверяет canary_guard — как и вызовы, разобранные самой Ollama.
    """
    calls: list[ToolCall] = []

    def take(obj: Any) -> bool:
        if not isinstance(obj, dict) or obj.get("name") not in allowed:
            return False
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            args = _loads_tolerant(args)
        if not isinstance(args, dict):
            return False
        calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}_{len(calls)}", obj["name"],
                              json.dumps(args, ensure_ascii=False)))
        return True

    if "<tool_call>" in text:
        rest = _TOOL_BLOCK_RE.sub(lambda m: "" if take(_loads_tolerant(m.group(1))) else m.group(0), text)
        return rest.strip(), calls
    bare = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    if bare.startswith("{") and take(_loads_tolerant(bare)):
        return "", calls
    return text, calls


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {t["function"]["name"] for t in tools}


def as_dict(arguments: str | dict) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
