"""Векторное хранилище на Qdrant: индексация, поиск top-k и форматирование фрагментов для модели."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk
from .embedders import Embedder


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    meta: dict[str, Any]


class RagIndex:
    COLLECTION = "mail_rag"

    def __init__(self, embedder: Embedder, qdrant_path: Path | None):
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.client = QdrantClient(location=":memory:") if qdrant_path is None else QdrantClient(path=str(qdrant_path))

    def close(self) -> None:
        self.client.close()

    def _index_signature(self, chunks: list[Chunk]) -> str:
        h = hashlib.sha256(self.embedder.name.encode())
        for c in chunks:
            h.update(c.chunk_id.encode())
            h.update(c.text.encode())
            h.update(json.dumps(c.meta, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    def build(self, chunks: list[Chunk], force: bool = False) -> int:
        from qdrant_client import models

        sig = self._index_signature(chunks)
        if not force and self.client.collection_exists(self.COLLECTION) and self._stored_signature() == sig:
            return 0
        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(
            self.COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        vectors = self.embedder.embed_passages([c.text for c in chunks])
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=np.asarray(vec, dtype=np.float32).tolist(),
                payload={"chunk_id": c.chunk_id, "text": c.text, **c.meta},
            )
            for c, vec in zip(chunks, vectors)
        ]
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, "__signature__")),
                vector=[0.0] * (self.embedder.dim - 1) + [1.0],
                payload={"kind": "__meta__", "signature": sig},
            )
        )
        self.client.upsert(self.COLLECTION, points=points)
        return len(chunks)

    def _stored_signature(self) -> str | None:
        got = self.client.retrieve(self.COLLECTION, ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, "__signature__"))])
        return got[0].payload.get("signature") if got else None

    def search(
        self,
        query: str,
        k: int = 5,
        kind: str | None = None,
        exclude_thread: str | None = None,
    ) -> list[Hit]:
        from qdrant_client import models

        must = [models.FieldCondition(key="kind", match=models.MatchAny(any=[kind] if kind else ["doc", "email"]))]
        must_not = [models.FieldCondition(key="suspicious", match=models.MatchValue(value=True))]
        if exclude_thread:
            must_not.append(models.FieldCondition(key="thread_id", match=models.MatchValue(value=exclude_thread)))
        res = self.client.query_points(
            self.COLLECTION,
            query=np.asarray(self.embedder.embed_query(query), dtype=np.float32).tolist(),
            query_filter=models.Filter(must=must, must_not=must_not),
            limit=k,
            with_payload=True,
        )
        hits = []
        for p in res.points:
            if p.score < self.embedder.min_score:
                continue
            payload = dict(p.payload or {})
            hits.append(Hit(payload.pop("chunk_id"), float(p.score), payload.pop("text"), payload))
        return hits


def format_hits(hits: list[Hit]) -> str:
    """Фрагменты для модели: у каждого ID источника, тип, дата и thread_id для писем."""
    if not hits:
        return "Релевантных фрагментов не найдено."
    blocks = []
    for h in hits:
        m = h.meta
        head = f"[{m['source_id']}] тип={m['kind']} дата={m.get('date')}"
        if m.get("thread_id"):
            head += f" thread_id={m['thread_id']}"
        head += f" фрагмент={h.chunk_id} релевантность={h.score:.2f}"
        blocks.append(f"{head}\n{h.text}")
    return "\n\n".join(blocks)
