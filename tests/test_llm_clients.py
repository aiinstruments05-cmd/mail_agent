"""Клиенты LLM без сети: ответы API подменяются через httpx.MockTransport."""
from __future__ import annotations

import json
import os

import httpx
import pytest

from mail_agent.config import Settings, load_dotenv
from mail_agent.llm import LLMError, OllamaModel, OpenAICompatibleModel, parse_text_tool_calls
from mail_agent.mailstore import TOOL_SCHEMAS

ALLOWED = {"search_emails", "read_thread", "create_draft"}


def _completion(message: dict) -> dict:
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}]}


def _openai(handler, temperature=0.0) -> tuple[OpenAICompatibleModel, list[dict]]:
    seen: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    model = OpenAICompatibleModel("https://api.test/v1", "sk-test", "m", temperature, max_retries=0, http_client=client)
    return model, seen


# ---------------------------------------------------------------- внешний API
def test_openai_native_tool_calls():
    msg = {"content": None, "tool_calls": [{"id": "call_1", "type": "function",
           "function": {"name": "read_thread", "arguments": '{"thread_id": "T-ORION"}'}}]}
    model, seen = _openai(lambda r: httpx.Response(200, json=_completion(msg)))
    resp = model.chat("SYS", [{"role": "user", "content": "q"}], TOOL_SCHEMAS)

    assert [(c.id, c.name, json.loads(c.arguments)) for c in resp.tool_calls] == [("call_1", "read_thread", {"thread_id": "T-ORION"})]
    body = seen[0]["body"]
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["tools"] == TOOL_SCHEMAS and body["temperature"] == 0.0
    assert seen[0]["headers"]["authorization"] == "Bearer sk-test"


def test_openai_text_tool_call_fallback_and_retry_temperature():
    text = '<tool_call>{"name": "search_emails", "arguments": {"query": "Орион"}}</tool_call>'
    model, seen = _openai(lambda r: httpx.Response(200, json=_completion({"content": text})))
    resp = model.chat("SYS", [], TOOL_SCHEMAS, attempt=2)

    assert resp.text == "" and resp.tool_calls[0].name == "search_emails"
    assert seen[0]["body"]["temperature"] == pytest.approx(0.6)


def test_openai_temperature_can_be_omitted():
    model, seen = _openai(lambda r: httpx.Response(200, json=_completion({"content": "ok"})), temperature=None)
    assert model.chat("SYS", [], TOOL_SCHEMAS).text == "ok"
    assert "temperature" not in seen[0]["body"]


@pytest.mark.parametrize("status,needle", [(401, "ключ"), (404, "не найдены"), (500, "500")])
def test_openai_errors_are_readable_and_hide_key(status, needle):
    model, _ = _openai(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(LLMError) as exc:
        model.chat("SYS", [], TOOL_SCHEMAS)
    assert needle in str(exc.value) and "sk-test" not in str(exc.value)


def test_missing_api_key_fails_fast():
    with pytest.raises(LLMError, match="ключ"):
        OpenAICompatibleModel("https://api.test/v1", "", "m")


# ---------------------------------------------------------------- настройки
def test_settings_default_to_external_api(monkeypatch):
    for k in ("MAIL_AGENT_LLM_BACKEND", "MAIL_AGENT_LLM_MODEL", "MAIL_AGENT_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert (s.llm_backend, s.llm_model, s.llm_api_key) == ("openai", "gpt-4o-mini", "")
    monkeypatch.setenv("MAIL_AGENT_LLM_BACKEND", "ollama")
    assert Settings().llm_model == "qwen2.5:7b"
    monkeypatch.setenv("MAIL_AGENT_LLM_BACKEND", "gpt")
    with pytest.raises(ValueError):
        Settings()


def test_api_key_not_in_repr(monkeypatch):
    monkeypatch.setenv("MAIL_AGENT_LLM_API_KEY", "sk-secret-123")
    assert "sk-secret-123" not in repr(Settings())


def test_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# c\nMA_TEST_A="from file"\nexport MA_TEST_B=2\nMA_TEST_C=file\n', encoding="utf-8")
    monkeypatch.setenv("MA_TEST_C", "env")
    for k in ("MA_TEST_A", "MA_TEST_B"):
        monkeypatch.delenv(k, raising=False)
    load_dotenv(env)
    try:
        assert (os.environ["MA_TEST_A"], os.environ["MA_TEST_B"], os.environ["MA_TEST_C"]) == ("from file", "2", "env")
    finally:
        os.environ.pop("MA_TEST_A", None)
        os.environ.pop("MA_TEST_B", None)


# ---------------------------------------------------------------- разбор вызовов из текста
def test_parse_text_tool_calls_variants():
    rest, calls = parse_text_tool_calls(
        'Сначала прочитаю.\n<tool_call>\n{"name": "create_draft", "arguments": {"thread_id": "T-ORION", "to": "a@b", '
        '"subject": "Re", "body": "строка1\nстрока2"}}\n</tool_call>', ALLOWED)
    assert rest == "Сначала прочитаю." and json.loads(calls[0].arguments)["body"] == "строка1\nстрока2"

    _, calls = parse_text_tool_calls('```json\n{"name": "read_thread", "parameters": {"thread_id": "T-VEGA"}}\n```', ALLOWED)
    assert json.loads(calls[0].arguments) == {"thread_id": "T-VEGA"} 
    _, calls = parse_text_tool_calls('{"name": "read_thread", "arguments": "{\\"thread_id\\": \\"T-VEGA\\"}"}', ALLOWED)
    assert json.loads(calls[0].arguments) == {"thread_id": "T-VEGA"}

    rest, calls = parse_text_tool_calls('<tool_call>{"name": "send_email", "arguments": {}}</tool_call>', ALLOWED)
    assert calls == [] and "send_email" in rest  
    _, calls = parse_text_tool_calls('<tool_call>{"name": "read_thread", "arguments": {"thread_id": "T-1"}}', ALLOWED)
    assert calls[0].name == "read_thread"  

# ---------------------------------------------------------------- Ollama (запасной вариант)
def test_ollama_retry_goes_raw_without_tools():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        text = '<tool_call>{"name": "read_thread", "arguments": {"thread_id": "T-ORION"}}</tool_call>'
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    model = OllamaModel("http://ollama.test", "qwen2.5:7b", transport=httpx.MockTransport(handler))
    model.chat("SYS", [], TOOL_SCHEMAS, attempt=0)
    resp = model.chat("SYS", [], TOOL_SCHEMAS, attempt=1)

    assert "tools" in bodies[0] and "tools" not in bodies[1]
    assert bodies[1]["options"]["num_ctx"] == 8192
    assert resp.tool_calls[0].name == "read_thread"
