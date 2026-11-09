"""Командная строка: index | ask | search | drafts."""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .agent import MailAgent
from .config import Settings, load_dotenv
from .llm import LLMError, LeakSimulator, OllamaModel, OpenAICompatibleModel
from .mailstore import EmailStore
from .rag import RagIndex, format_hits, load_doc_chunks, load_email_chunks, make_embedder


def build_rag(settings: Settings, force: bool = False) -> tuple[RagIndex, int]:
    rag = RagIndex(make_embedder(settings.embedder, settings.e5_model), settings.qdrant_path)
    chunks = load_doc_chunks(settings.docs_dir) + load_email_chunks(settings.emails_path)
    return rag, rag.build(chunks, force=force)


def make_llm(settings: Settings):
    if settings.llm_backend == "ollama":
        return OllamaModel(settings.ollama_host, settings.llm_model, settings.llm_temperature or 0.0, settings.num_ctx)
    return OpenAICompatibleModel(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                                 settings.llm_temperature, settings.llm_timeout, settings.llm_max_retries)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mail_agent", description="Почтовый агент с RAG и canary-защитой")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="Построить/перестроить векторный индекс").add_argument("--force", action="store_true")

    s = sub.add_parser("search", help="Проверить RAG-поиск без LLM")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)

    a = sub.add_parser("ask", help="Задать вопрос агенту")
    a.add_argument("query")
    a.add_argument("--model", help="Имя модели (по умолчанию gpt-4o-mini, для Ollama — qwen2.5:7b)")
    a.add_argument("--backend", choices=["openai", "ollama"], help="openai — внешний API (по умолчанию), ollama — локально")
    a.add_argument("--simulate-leak", choices=["response", "tool_call"],
                   help="Демонстрация: подмешать активный canary-токен в ответ модели")
    a.add_argument("--leak-encoding", choices=["plain", "base64"], default="plain")
    a.add_argument("--json", action="store_true", help="Вывести результат в JSON")

    sub.add_parser("drafts", help="Показать сохранённые черновики")

    args = p.parse_args(argv)
    load_dotenv()
    if getattr(args, "backend", None):
        os.environ["MAIL_AGENT_LLM_BACKEND"] = args.backend
    settings = Settings()

    if args.cmd == "index":
        rag, n = build_rag(settings, force=args.force)
        rag.close()
        print(f"Проиндексировано фрагментов: {n}" if n else "Индекс актуален")
        return 0

    if args.cmd == "search":
        rag, _ = build_rag(settings)
        print(format_hits(rag.search(args.query, k=args.k)))
        rag.close()
        return 0

    store = EmailStore(settings.emails_path, settings.drafts_path)
    if args.cmd == "drafts":
        for d in store.list_drafts():
            print(json.dumps(d, ensure_ascii=False, indent=2))
        return 0

    if args.model:
        settings.llm_model = args.model
    try:
        llm = make_llm(settings)
    except LLMError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    if args.simulate_leak:
        llm = LeakSimulator(llm, where=args.simulate_leak, encode=args.leak_encoding)
    rag, _ = build_rag(settings)
    try:
        result = MailAgent(settings, llm, store, rag).run(args.query)
    except LLMError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    finally:
        rag.close()
    if args.json:
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    else:
        print(f"[run_id={result.run_id} status={result.status}]\n")
        print(result.answer)
    return 0 if result.status in ("done", "blocked") else 1


if __name__ == "__main__":
    sys.exit(main())
