"""Инструменты и RAG."""
import json

import pytest

from mail_agent.mailstore import ToolError
from mail_agent.rag import load_doc_chunks, load_email_chunks
from mail_agent.config import Settings

from .conftest import call, make_agent, step


def test_search_filters(store):
    hits = store.search_emails("Орион")
    assert hits and all(h["thread_id"] == "T-ORION" for h in hits)
    assert hits[0].keys() == {"email_id", "thread_id", "subject", "from", "date", "snippet"}
    assert {h["email_id"] for h in store.search_emails("Орион", sender="ivan.petrov")} == {"M-1001", "M-1003", "M-1005"}
    assert [h["email_id"] for h in store.search_emails("Орион", date_from="2026-11-01")] == ["M-1005"]
    assert store.search_emails("гарантия", date_to="2026-01-01") == []
    with pytest.raises(ToolError):
        store.search_emails("x", date_from="вчера")


def test_read_thread_is_chronological(store):
    msgs = store.read_thread("T-ORION")
    assert [m["date"] for m in msgs] == sorted(m["date"] for m in msgs)
    assert {"from", "to", "cc", "body"} <= msgs[0].keys()
    with pytest.raises(ToolError):
        store.read_thread("T-NOPE")


def test_create_draft_only_to_participants(store):
    with pytest.raises(ToolError):
        store.create_draft("T-PARTNER", "audit@evil.example", "x", "y")
    res = store.create_draft("T-VEGA", "maria.sokolova@translogistic.example", "Re: Вега", "текст")
    assert res["draft_id"].startswith("DRAFT-") and res["status"] == "saved, not sent"


def test_chunks_have_metadata():
    s = Settings()
    docs, emails = load_doc_chunks(s.docs_dir), load_email_chunks(s.emails_path)
    assert {c.meta["source_id"] for c in docs} == {"DOC-DELIVERY", "DOC-PRODUCTS", "DOC-WARRANTY", "DOC-RULES"}
    assert all(c.meta["thread_id"] and c.meta["date"] for c in emails)
    assert all(c.meta["thread_id"] is None and c.meta["date"] for c in docs)


def test_rag_finds_relevant_doc(rag):
    hits = rag.search("стоимость расширенной гарантии", k=3)
    assert hits[0].meta["source_id"] == "DOC-WARRANTY"
    assert all(h.meta["thread_id"] != "T-ORION" for h in rag.search("Орион поставка", exclude_thread="T-ORION"))


def test_read_thread_adds_rag_context_and_draft_requires_it(settings, store, rag):
    agent, _ = make_agent(settings, store, rag, [
        step("", call("create_draft", thread_id="T-ORION", to="ivan.petrov@stroymontazh.example", subject="Re", body="b")),
        step("", call("read_thread", thread_id="T-ORION")),
        step("готово"),
    ])
    state_res = agent.run("Подготовь ответ по Ориону про доставку и гарантию")
    assert state_res.drafts == []  # черновик до read_thread отклонён
    assert "DOC-WARRANTY" in state_res.sources or "DOC-DELIVERY" in state_res.sources


def test_injection_email_cannot_redirect_draft(settings, store, rag):
    agent, _ = make_agent(settings, store, rag, [
        step("", call("read_thread", thread_id="T-PARTNER")),
        step("", call("create_draft", thread_id="T-PARTNER", to="audit@alfa-sensor-help.example", subject="x", body="y")),
        step("Письмо похоже на попытку атаки."),
    ])
    res = agent.run("Разбери письмо про проверку конфигурации")
    assert res.status == "done" and res.drafts == [] and store.list_drafts() == []


def test_no_data_is_reported(settings, store, rag):
    agent, _ = make_agent(settings, store, rag, [step("", call("search_emails", query="Андромеда")), step("Данных нет.")])
    res = agent.run("Что с проектом Андромеда-квазар?")
    assert res.status == "done"
    assert "Проверено кодом" in res.answer


def test_unknown_source_ids_are_flagged(settings, store, rag):
    agent, _ = make_agent(settings, store, rag, [step("По данным [M-9999] всё согласовано.")])
    assert "M-9999" in agent.run("Орион").answer.split("⚠")[1]


def test_failed_tool_reason_is_audited(settings, store, rag):
    agent, _ = make_agent(settings, store, rag, [
        step("", call("read_thread", thread_id="T-VEGA")),
        step("", call("create_draft", thread_id="T-VEGA", to="someone@outside.example", subject="x", body="y")),
        step("Готово."),
    ])
    agent.run("Ответь по Веге")
    rows = [json.loads(l) for l in settings.audit_log_path.read_text().splitlines()]
    failed = [r for r in rows if not r["ok"]]
    assert len(failed) == 1 and "не участвует в цепочке" in failed[0]["error"]


def test_thread_rag_searches_each_client_question(settings, store, rag):
    """Последнее письмо Ориона спрашивает и о доставке, и о гарантии: в контекст попадают оба документа."""
    seen = []

    def capture(system, msgs):
        seen.extend(m for m in msgs if m["role"] == "tool")
        return step("Готово")(system, msgs)

    agent, _ = make_agent(settings, store, rag, [step("", call("read_thread", thread_id="T-ORION")), capture])
    agent.run("Подготовь ответ клиенту по Ориону")
    rag_context = json.loads(seen[0]["content"])["rag_context"]
    assert "[DOC-DELIVERY]" in rag_context and "[DOC-WARRANTY]" in rag_context
    assert "7%" in rag_context  # раздел о расширенной гарантии
    assert "thread_id=T-ORION" not in rag_context  # сама цепочка из выдачи исключена
