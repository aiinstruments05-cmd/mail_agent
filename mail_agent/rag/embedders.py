"""Модели эмбеддингов: E5 (основная) и офлайн-эмбеддер на хешах (для тестов)."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int
    min_score: float 

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class E5Embedder:
    """intfloat/multilingual-e5-small: мультиязычная модель, хорошо работает с русским.

    E5 обучена с префиксами «query: » и «passage: » — без них качество падает.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.min_score = 0.80  
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0]


class HashingEmbedder:
    """Офлайн-запасной вариант без скачивания моделей (используется в тестах).

    Символьные 3–5-граммы слов, хешированные в вектор фиксированной длины, log-TF, L2.
    Это лексический, а не семантический поиск, но словоформы русского языка он ловит.
    """

    def __init__(self, dim: int = 4096):
        self.name = f"hashing-{dim}"
        self.dim = dim
        self.min_score = 0.08

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            w = f"<{word}>"
            grams = [w[i : i + n] for n in (3, 4, 5) for i in range(len(w) - n + 1)] or [w]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0
        v = np.log1p(v)
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_embedder(kind: str, e5_model: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    return E5Embedder(e5_model)
