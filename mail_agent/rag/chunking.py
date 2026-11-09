"""Загрузка справочных документов и писем и разбиение на фрагменты с метаданными."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..injection import detect_email


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: dict[str, Any]  


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if text.startswith("---"):
        _, head, text = text.split("---", 2)
        for line in head.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, text.strip()


def split_text(text: str, max_chars: int = 700, overlap: int = 120) -> list[str]:
    """Режем по абзацам; слишком длинный абзац - окнами с перекрытием."""
    parts: list[str] = []
    buf = ""
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        if len(buf) + len(para) + 2 <= max_chars:
            buf = f"{buf}\n\n{para}" if buf else para
            continue
        if buf:
            parts.append(buf)
        while len(para) > max_chars:
            parts.append(para[:max_chars])
            para = para[max_chars - overlap :]
        buf = para
    if buf:
        parts.append(buf)
    return parts


def load_doc_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        source_id = meta.get("id", path.stem)
        sections = re.split(r"\n(?=## )", body)
        n = 0
        for section in sections:
            section = re.sub(r"^# .*\n", "", section).strip()
            if not section:
                continue
            for piece in split_text(section):
                n += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_id}#{n}",
                        text=f"{meta.get('title', '')}. {piece}",
                        meta={"source_id": source_id, "kind": "doc", "thread_id": None,
                              "date": meta.get("date"), "title": meta.get("title"), "suspicious": False},
                    )
                )
    return chunks


def load_email_chunks(emails_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for e in json.loads(Path(emails_path).read_text(encoding="utf-8")):
        for i, piece in enumerate(split_text(e["body"]), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{e['email_id']}#{i}",
                    text=f"Тема: {e['subject']}\nОт: {e['from']}\n{piece}",
                    meta={"source_id": e["email_id"], "kind": "email", "thread_id": e["thread_id"],
                          "date": e["date"][:10], "title": e["subject"],
                          "suspicious": detect_email(e).suspicious},
                )
            )
    return chunks
