"""Canary-токены и детерминированная проверка ответа модели.

Проверка выполняется кодом и не зависит от LLM. Перед выдачей ответа
пользователю и перед выполнением любого инструмента guard просматривает
весь текст ответа модели и все декодированные аргументы всех вызовов.
Совпадение — это полный активный токен как подстрока в любом из
«представлений» текста: исходном, после JSON-декодирования, после
нормализации Unicode, URL-, base64- и hex-декодирования.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

CANARY_PREFIX = "CNRY"

_B64_RE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}){12,}")
_NON_ALNUM_RE = re.compile(r"[\W_]+", re.UNICODE)


def new_canary_token() -> str:
    """Свежий токен на каждый запуск: 96 бит случайности, вероятность случайного совпадения ничтожна."""
    return f"{CANARY_PREFIX}-{secrets.token_hex(12)}"


@dataclass(frozen=True)
class GuardVerdict:
    leaked: bool
    location: str | None = None  # "response" | "tool_call"
    tool: str | None = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _try_b64(chunk: str) -> str | None:
    padded = chunk + "=" * (-len(chunk) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = decoder(padded)
        except (binascii.Error, ValueError):
            continue
        text = raw.decode("utf-8", errors="ignore")
        if text:
            return text
    return None


def text_views(text: str, depth: int = 2) -> Iterator[str]:
    """Все представления строки, в которых ищется токен."""
    yield text
    norm = _normalize(text)
    yield norm
    unquoted = urllib.parse.unquote(norm)
    if unquoted != norm:
        yield unquoted
    if depth <= 0:
        return
    for m in _B64_RE.finditer(norm):
        decoded = _try_b64(m.group(0))
        if decoded:
            yield from text_views(decoded, depth - 1)
    for m in _HEX_RE.finditer(norm):
        try:
            decoded = bytes.fromhex(m.group(0)).decode("utf-8", errors="ignore")
        except ValueError:
            continue
        if decoded:
            yield from text_views(decoded, depth - 1)


def _walk_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)
    elif obj is not None:
        yield str(obj)


def argument_views(raw_arguments: str | dict | None) -> Iterator[str]:
    """Аргументы вызова: сырой JSON, каждая декодированная строка и объект целиком."""
    if raw_arguments is None:
        return
    if isinstance(raw_arguments, str):
        yield raw_arguments
        try:
            obj = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            return
    else:
        obj = raw_arguments
    yield json.dumps(obj, ensure_ascii=False)
    yield from _walk_strings(obj)


class CanaryGuard:
    def __init__(self, active_tokens: Iterable[str]):
        self._tokens = [t for t in active_tokens if t]
        if not self._tokens:
            raise ValueError("Нужен хотя бы один активный canary-токен")
        self._lower = [t.lower() for t in self._tokens]
        self._squashed = [_NON_ALNUM_RE.sub("", t).lower() for t in self._tokens]

    def _match(self, text: str) -> bool:
        for view in text_views(text):
            low = view.lower()
            if any(t in low for t in self._lower):
                return True
            
            squashed = _NON_ALNUM_RE.sub("", low)
            if any(t in squashed for t in self._squashed):
                return True
        return False

    def inspect(self, text: str | None, tool_calls: Iterable[dict] = ()) -> GuardVerdict:
        """Проверяет ответ модели целиком. tool_calls — [{"name":..., "arguments":...}]."""
        if text and self._match(text):
            return GuardVerdict(True, "response")
        for call in tool_calls:
            name = str(call.get("name", ""))
            if self._match(name):
                return GuardVerdict(True, "tool_call", None)
            for view in argument_views(call.get("arguments")):
                if self._match(view):
                    return GuardVerdict(True, "tool_call", name)
        return GuardVerdict(False)
