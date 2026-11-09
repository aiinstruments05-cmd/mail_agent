"""Настройки агента. Всё переопределяется переменными окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Минимальный разбор .env (KEY=VALUE, # комментарии). Уже заданные переменные окружения
    не перезаписываются. Ключ API хранится только здесь или в окружении — не в коде и не в промпте."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DEFAULT_MODELS = {"openai": "gpt-4o-mini", "ollama": "qwen2.5:7b"}


@dataclass
class Settings:
   
    llm_backend: str = field(default_factory=lambda: _env("MAIL_AGENT_LLM_BACKEND", "openai"))
    llm_base_url: str = field(default_factory=lambda: _env("MAIL_AGENT_LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(
        default_factory=lambda: _env("MAIL_AGENT_LLM_API_KEY", "") or _env("OPENAI_API_KEY", ""), repr=False
    )
    llm_model: str = field(default_factory=lambda: _env("MAIL_AGENT_LLM_MODEL", ""))
    llm_temperature: float | None = field(default_factory=lambda: _float_or_none(_env("MAIL_AGENT_LLM_TEMPERATURE", "0")))
    llm_timeout: float = field(default_factory=lambda: float(_env("MAIL_AGENT_LLM_TIMEOUT", "120")))
    llm_max_retries: int = field(default_factory=lambda: int(_env("MAIL_AGENT_LLM_MAX_RETRIES", "3")))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST_URL", "http://localhost:11434"))
    num_ctx: int = field(default_factory=lambda: int(_env("MAIL_AGENT_NUM_CTX", "8192")))

    
    embedder: str = field(default_factory=lambda: _env("MAIL_AGENT_EMBEDDER", "e5"))  
    e5_model: str = field(default_factory=lambda: _env("MAIL_AGENT_E5_MODEL", "intfloat/multilingual-e5-small"))
    top_k: int = 5

   
    emails_path: Path = ROOT / "data" / "emails.json"
    docs_dir: Path = ROOT / "data" / "docs"
    prompt_path: Path = ROOT / "prompts" / "system_prompt.md"
    qdrant_path: Path = field(default_factory=lambda: Path(_env("MAIL_AGENT_QDRANT_PATH", str(ROOT / "var" / "qdrant"))))
    drafts_path: Path = field(default_factory=lambda: Path(_env("MAIL_AGENT_DRAFTS_PATH", str(ROOT / "var" / "drafts.jsonl"))))
    security_log_path: Path = field(
        default_factory=lambda: Path(_env("MAIL_AGENT_SECURITY_LOG", str(ROOT / "logs" / "security_events.jsonl")))
    )
    injection_log_path: Path = field(
        default_factory=lambda: Path(_env("MAIL_AGENT_INJECTION_LOG", str(ROOT / "logs" / "injection_events.jsonl")))
    )
    audit_log_path: Path = field(default_factory=lambda: Path(_env("MAIL_AGENT_AUDIT_LOG", str(ROOT / "logs" / "audit.jsonl"))))

    manager_name: str = "Анна Смирнова"
    manager_email: str = "anna.smirnova@alfa-sensor.example"
    company: str = "ООО «Альфа-Сенсор»"
    max_steps: int = 10
    empty_retries: int = 2  

    def __post_init__(self) -> None:
        if self.llm_backend not in DEFAULT_MODELS:
            raise ValueError(f"MAIL_AGENT_LLM_BACKEND: ожидается openai или ollama, получено {self.llm_backend!r}")
        if not self.llm_model:
            self.llm_model = DEFAULT_MODELS[self.llm_backend]


def _float_or_none(value: str) -> float | None:
    """Пустая строка — не передавать temperature вовсе (некоторые модели её не принимают)."""
    return float(value) if value.strip() else None
