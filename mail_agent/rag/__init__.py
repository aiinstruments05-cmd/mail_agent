"""RAG: документы и письма → фрагменты → эмбеддинги → Qdrant → top-k. Публичный интерфейс — mail_agent.rag."""
from .chunking import Chunk, load_doc_chunks, load_email_chunks, split_text
from .embedders import E5Embedder, Embedder, HashingEmbedder, make_embedder
from .store import Hit, RagIndex, format_hits

__all__ = [
    "Chunk", "load_doc_chunks", "load_email_chunks", "split_text",
    "Embedder", "E5Embedder", "HashingEmbedder", "make_embedder",
    "Hit", "RagIndex", "format_hits",
]
