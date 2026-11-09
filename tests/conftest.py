import json
import re
from pathlib import Path

import pytest

from mail_agent.agent import MailAgent
from mail_agent.config import Settings
from mail_agent.llm import LLMResponse, ScriptedModel, ToolCall
from mail_agent.mailstore import EmailStore
from mail_agent.rag import HashingEmbedder, RagIndex, load_doc_chunks, load_email_chunks

TOKEN = "CNRY-0123456789abcdef01234567"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.embedder = "hashing"
    s.drafts_path = tmp_path / "drafts.jsonl"
    s.security_log_path = tmp_path / "security_events.jsonl"
    s.injection_log_path = tmp_path / "injection_events.jsonl"
    s.audit_log_path = tmp_path / "audit.jsonl"
    return s


@pytest.fixture
def store(settings: Settings) -> EmailStore:
    return EmailStore(settings.emails_path, settings.drafts_path)


@pytest.fixture(scope="session")
def rag() -> RagIndex:
    s = Settings()
    index = RagIndex(HashingEmbedder(), qdrant_path=None) 
    index.build(load_doc_chunks(s.docs_dir) + load_email_chunks(s.emails_path))
    return index


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(args, ensure_ascii=False))


def step(text: str = "", *calls: ToolCall):
    return lambda system, messages: LLMResponse(text=text, tool_calls=list(calls))


def token_from(system: str) -> str:
    return re.search(r"CNRY-[0-9a-f]+", system).group(0)


def make_agent(settings, store, rag, steps) -> tuple[MailAgent, ScriptedModel]:
    llm = ScriptedModel(steps)
    return MailAgent(settings, llm, store, rag), llm


HAPPY_PATH = [
    step("", call("search_emails", query="Орион")),
    step("", call("read_thread", thread_id="T-ORION")),
    step(
        "",
        call(
            "create_draft",
            thread_id="T-ORION",
            to="ivan.petrov@stroymontazh.example",
            subject="Re: Проект Орион: поставка датчиков AS-200",
            body="Иван, добрый день! Срок отгрузки второй партии 20.11.2026 в силе. ...",
        ),
    ),
    step("Согласовано: партия 1 до 15.10.2026, партия 2 до 20.11.2026 [M-1003]. Условия: [DOC-DELIVERY]."),
]
