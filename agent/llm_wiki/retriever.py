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

    # Intents for which a company-wide chunk is almost always relevant and
    # would otherwise be drowned by 20 lexically-similar product chunks.
    # (Diagnosed in Phase 1 verification; see PLAN.md Phase 3.)
    _COMPANY_BIAS_INTENTS: set[str] = {
        "pricing",
        "discount",
        "cancellation",
        "billing",
    }

    def _query_raw(
        self,
        q: str,
        n: int,
        where: dict[str, Any] | None = None,
    ) -> list[WikiHit]:
        if n <= 0:
            return []
        coll = self._collection()
        kwargs: dict[str, Any] = {"query_texts": [q], "n_results": n}
        if where is not None:
            kwargs["where"] = where
        res = coll.query(**kwargs)
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

    def query(self, q: str, k: int = 3) -> list[WikiHit]:
        """Unfiltered vector query. Use for diagnostics / generic lookups."""
        return self._query_raw(q, n=k)

    def query_intent_aware(
        self,
        q: str,
        intent: str,
        product_id: str | None = None,
        k: int = 5,
        k_company: int = 2,
        k_product: int = 2,
    ) -> list[WikiHit]:
        """Intent-aware + product-aware retrieval.

        Reserves slots so neither company-wide chunks nor the thread's own
        product can get smothered by lexically similar competitors:

        - For company-policy intents (pricing / discount / cancellation /
          billing), reserve up to `k_company` slots for company-wide chunks
          (Phase 1 verification finding).
        - When a thread's `product_id` is known, reserve up to `k_product`
          slots for that product's chunks (Phase 3 verification finding —
          a follow-up message that drops the product name made BillSplit's
          own pricing miss the top-5 for a BillSplit thread).

        Remaining slots fill from open vector search. All hits are de-duped
        by chunk_id and finally re-sorted by distance so the strongest
        actual matches still rank first.
        """
        bias_company = intent in self._COMPANY_BIAS_INTENTS
        slots_reserved = (k_company if bias_company else 0) + (
            k_product if product_id else 0
        )
        k_open = max(0, k - slots_reserved)

        all_hits: list[WikiHit] = []
        if bias_company:
            all_hits += self._query_raw(
                q, n=k_company, where={"product_id": "_company"}
            )
        if product_id:
            all_hits += self._query_raw(
                q, n=k_product, where={"product_id": product_id}
            )
        if k_open > 0:
            all_hits += self._query_raw(q, n=k_open)

        seen: set[str] = set()
        merged: list[WikiHit] = []
        for h in all_hits:
            if h.chunk_id in seen:
                continue
            seen.add(h.chunk_id)
            merged.append(h)
        merged.sort(key=lambda h: h.distance if h.distance is not None else float("inf"))
        return merged[:k]
