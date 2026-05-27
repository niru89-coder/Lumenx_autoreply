"""ChromaDB-backed retriever over the LLM Wiki.

Uses Chroma's default embedding function (ONNX-backed all-MiniLM-L6-v2)
so we don't pay Anthropic for retrieval. Persisted to data/wiki_chroma/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from agent.config import REPO_ROOT
from agent.llm_wiki.builder import WikiChunk

CHROMA_DIR = REPO_ROOT / "data" / "wiki_chroma"
COLLECTION_NAME = "lumenx_wiki"


@dataclass
class WikiHit:
    chunk_id: str
    text: str
    title: str
    product_id: str | None
    section: str
    source_path: str
    distance: float | None  # lower = closer (Chroma uses cosine distance)


class WikiRetriever:
    def __init__(
        self,
        chroma_path: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection_name = collection_name

    def _collection(self) -> Any:
        return self._client.get_or_create_collection(name=self._collection_name)

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass

    def index(self, chunks: list[WikiChunk]) -> int:
        if not chunks:
            return 0
        coll = self._collection()
        coll.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "product_id": c.product_id or "_company",
                    "section": c.section,
                    "title": c.title,
                    "source_path": c.source_path,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def query(self, q: str, k: int = 3) -> list[WikiHit]:
        coll = self._collection()
        res = coll.query(query_texts=[q], n_results=k)
        ids = res["ids"][0] if res.get("ids") else []
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else [None] * len(ids)

        hits: list[WikiHit] = []
        for i in range(len(ids)):
            m = metas[i] or {}
            hits.append(
                WikiHit(
                    chunk_id=ids[i],
                    text=docs[i],
                    title=str(m.get("title", "")),
                    product_id=(
                        None if m.get("product_id") == "_company" else m.get("product_id")
                    ),
                    section=str(m.get("section", "")),
                    source_path=str(m.get("source_path", "")),
                    distance=dists[i],
                )
            )
        return hits
