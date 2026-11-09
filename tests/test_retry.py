"""Повтор запроса к модели, если она вернула пустой ответ (битый вызов инструмента)."""
import json

from mail_agent.agent import BLOCK_MESSAGE, RETRY_NUDGE

from .conftest import TOKEN, call, make_agent, step, token_from

EMPTY = step("")


def test_empty_reply_is_retried_with_nudge(settings, store, rag):
    seen = []

    def draft_after_nudge(system, msgs):
        seen.append(msgs[-1]["content"])
        return step("", call("create_draft", thread_id="T-ORION", to="ivan.petrov@stroymontazh.example",
                             subject="Re: Орион", body="Иван, срок 20.11.2026 в силе."))(system, msgs)

    agent, llm = make_agent(settings, store, rag, [
        step("", call("read_thread", thread_id="T-ORION")),
        EMPTY,                      
        draft_after_nudge,           
        step("Черновик готов [M-1005]."),
    ])
    res = agent.run("Подготовь черновик по Ориону", canary_token=TOKEN)
    assert res.status == "done" and len(res.drafts) == 1
    assert seen == [RETRY_NUDGE]
    assert llm.calls == 4


def test_nudge_is_not_kept_in_history(settings, store, rag):
    histories = []

    def record(reply):
        def fn(system, msgs):
            histories.append([m.get("content") for m in msgs])
            return reply(system, msgs)
        return fn

    agent, _ = make_agent(settings, store, rag, [
        EMPTY,
        record(step("", call("search_emails", query="Орион"))),
        record(step("готово")),
    ])
    agent.run("Орион", canary_token=TOKEN)
    assert histories[0][-1] == RETRY_NUDGE       
    assert RETRY_NUDGE not in histories[1]      

def test_leak_in_retry_reply_is_blocked(settings, store, rag):
    leak = lambda s, m: step("", call("search_emails", query=token_from(s)))(s, m) 
    agent, llm = make_agent(settings, store, rag, [EMPTY, leak, step("не должно выполниться")])
    res = agent.run("Орион", canary_token=TOKEN)
    assert res.status == "blocked" and res.answer == BLOCK_MESSAGE
    assert llm.calls == 2
    events = [json.loads(l) for l in settings.security_log_path.read_text().splitlines()]
    assert [e["location"] for e in events] == ["tool_call"]
    assert TOKEN not in settings.security_log_path.read_text()


def test_retries_are_bounded(settings, store, rag):
    agent, llm = make_agent(settings, store, rag, [EMPTY])
    res = agent.run("Орион", canary_token=TOKEN)
    assert llm.calls == 1 + settings.empty_retries
    assert res.status == "done" and "Модель не вернула текст ответа" in res.answer


def test_missing_draft_triggers_one_nudge(settings, store, rag):
    agent, llm = make_agent(settings, store, rag, [
        step("", call("read_thread", thread_id="T-ORION")),
        step("Черновик готов, draft_id: 123456"),         
        step("", call("create_draft", thread_id="T-ORION", to="ivan.petrov@stroymontazh.example",
                      subject="Re: Орион", body="Иван, срок 20.11.2026 в силе.")),
        step("Готово."),
    ])
    res = agent.run("Подготовь черновик ответа по Ориону", canary_token=TOKEN)
    assert res.status == "done" and len(res.drafts) == 1 and llm.calls == 4


def test_draft_nudge_happens_once_and_fake_id_is_flagged(settings, store, rag):
    agent, llm = make_agent(settings, store, rag, [step("draft_id: 123456")])
    res = agent.run("Подготовь черновик", canary_token=TOKEN)
    assert llm.calls == 2  
    assert "черновик не сохранён" in res.answer


def test_no_nudge_when_draft_not_requested(settings, store, rag):
    agent, llm = make_agent(settings, store, rag, [step("Сводка по Ориону.")])
    agent.run("Какие сроки согласованы по Ориону?", canary_token=TOKEN)
    assert llm.calls == 1


def test_leak_after_draft_nudge_is_blocked(settings, store, rag):
    leak = lambda s, m: step(f"вот: {token_from(s)}")(s, m)  
    agent, _ = make_agent(settings, store, rag, [step("Готово, без черновика"), leak])
    res = agent.run("Подготовь черновик", canary_token=TOKEN)
    assert res.status == "blocked"
    assert [json.loads(l)["location"] for l in settings.security_log_path.read_text().splitlines()] == ["response"]
