"""Knowledge Agent (pipeline stage 6) — retrieval over project metadata.

Design deviation from the earlier plan, stated openly: at ~150 chunks, a
vector database (Chroma) is machinery without benefit. A numpy cosine index
over sentence-transformer embeddings is simpler, dependency-lighter, and
identical in result at this scale. The paper still honestly calls this RAG:
embed, retrieve top-k, ground the generation.

The embedder is injectable so tests run without torch/sentence-transformers.
Default: all-MiniLM-L6-v2, lazily imported on first use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]  # (n texts) -> (n, d) float array


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


def default_embedder() -> EmbedFn:
    from sentence_transformers import SentenceTransformer  # lazy: heavy import
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True))

    return embed


class KnowledgeBase:
    """Embed once, retrieve by cosine similarity. Save/load as npz+jsonl."""

    def __init__(self, chunks: list[Chunk], embed_fn: Optional[EmbedFn] = None):
        if not chunks:
            raise ValueError("empty corpus")
        self.chunks = chunks
        self.embed = embed_fn or default_embedder()
        vecs = self.embed([c.text for c in chunks]).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.vecs = vecs / np.maximum(norms, 1e-9)

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        q = self.embed([query]).astype(np.float32)[0]
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        scores = self.vecs @ q
        order = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in order]

    # -- persistence (so the dashboard doesn't re-embed on every start) ----
    def save(self, dirpath: str | Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vecs.npy", self.vecs)
        with open(d / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps({"id": c.id, "text": c.text, "meta": c.meta}) + "\n")

    @classmethod
    def load(cls, dirpath: str | Path,
             embed_fn: Optional[EmbedFn] = None) -> "KnowledgeBase":
        d = Path(dirpath)
        kb = cls.__new__(cls)
        kb.vecs = np.load(d / "vecs.npy")
        kb.chunks = [Chunk(**json.loads(line))
                     for line in open(d / "chunks.jsonl")]
        kb.embed = embed_fn or default_embedder()
        return kb
