"""LLM-клиенты. Публичный интерфейс пакета — импорт из mail_agent.llm."""
from .base import ChatModel, LLMError, LLMResponse, ToolCall
from .fakes import LeakSimulator, ScriptedModel
from .ollama_client import OllamaModel
from .openai_client import OpenAICompatibleModel
from .parsing import parse_text_tool_calls

__all__ = [
    "ChatModel", "LLMError", "LLMResponse", "ToolCall",
    "OpenAICompatibleModel", "OllamaModel", "ScriptedModel", "LeakSimulator", "parse_text_tool_calls",
]
