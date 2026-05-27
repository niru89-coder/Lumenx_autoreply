"""Q&A pool seeded from LumenX's export of past human-resolved threads.

This is a stand-in for the Phase 5 feedback log. Same retrieval interface;
Phase 5 will repoint it at approved drafts from `human_actions` instead of
the export. Until then, this pool gives the drafter at least one similar
already-resolved case to anchor on.

Schema:
  - one entry per thread: first customer message paired with first admin
    reply. Multi-turn extensions are out of scope for now.
  - embedded text is the CUSTOMER MESSAGE so vector search finds questions
    that look like the new one.
  - the admin REPLY is stored in metadata so a retrieved hit carries both
    halves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from agent.config import REPO_ROOT

QA_CHROMA_DIR = REPO_ROOT / "data" / "qa_chroma"
COLLECTION_NAME = "lumenx_historical_qa"
EXPORT_PATH = REPO_ROOT / "data" / "raw" / "export.json"


@dataclass
class QAEntry:
    id: str
    question: str
    answer: str
    intent: str
    product_id: str
    thread_id: str
    created_at: str


@dataclass
class QAHit:
    entry: QAEntry
    distance: float | None


def _first_messages(thread: dict[str, Any]) -> tuple[dict | None, dict | None]:
    """Return (first_customer_msg, first_admin_reply_after_it) or (None, None)."""
    msgs = thread.get("messages") or []
    customer = next((m for m in msgs if m.get("role") == "customer" and (m.get("text") or "").strip()), None)
    if customer is None:
        return None, None
    cust_ts = customer.get("ts", "")
    admin = next(
        (
            m for m in msgs
            if m.get("role") == "admin"
            and (m.get("text") or "").strip()
            and m.get("ts", "") >= cust_ts
        ),
        None,
    )
    return customer, admin


def entries_from_export(export_path: Path = EXPORT_PATH) -> list[QAEntry]:
    raw = json.loads(export_path.read_text(encoding="utf-8"))
    entries: list[QAEntry] = []
    for thread in raw.get("threads", []):
        cust, admin = _first_messages(thread)
        if not cust or not admin:
            continue
        entries.append(
            QAEntry(
                id=f"qa-{thread['id']}",
                question=cust["text"].strip(),
                answer=admin["text"].strip(),
                intent=str(thread.get("intent", "") or ""),
                product_id=str(thread.get("product_id", "") or ""),
                thread_id=str(thread["id"]),
                created_at=str(thread.get("created_at", "") or ""),
            )
        )
    return entries


class HistoricalQAPool:
    def __init__(
        self,
        chroma_path: Path = QA_CHROMA_DIR,
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

    def index(self, entries: list[QAEntry]) -> int:
        if not entries:
            return 0
        coll = self._collection()
        coll.upsert(
            ids=[e.id for e in entries],
            documents=[e.question for e in entries],
            metadatas=[
                {
                    "answer": e.answer,
                    "intent": e.intent,
                    "product_id": e.product_id,
                    "thread_id": e.thread_id,
                    "created_at": e.created_at,
                }
                for e in entries
            ],
        )
        return len(entries)

    def query(
        self,
        q: str,
        k: int = 3,
        *,
        intent: str | None = None,
        product_id: str | None = None,
    ) -> list[QAHit]:
        """Top-k by question similarity. Optional intent/product filters bias
        toward more relevant past cases when we know the current intent."""
        coll = self._collection()
        # Build a Chroma where filter. Chroma requires either a single key OR
        # an explicit $and for multiple keys.
        where: dict[str, Any] | None = None
        clauses: list[dict[str, Any]] = []
        if intent:
            clauses.append({"intent": intent})
        if product_id:
            clauses.append({"product_id": product_id})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        kwargs: dict[str, Any] = {"query_texts": [q], "n_results": k}
        if where is not None:
            kwargs["where"] = where

        res = coll.query(**kwargs)
        ids = res["ids"][0] if res.get("ids") else []
        docs = res["documents"][0] if res.get("documents") else []
        metas = res["metadatas"][0] if res.get("metadatas") else []
        dists = res["distances"][0] if res.get("distances") else [None] * len(ids)

        hits: list[QAHit] = []
        for i in range(len(ids)):
            m = metas[i] or {}
            hits.append(
                QAHit(
                    entry=QAEntry(
                        id=ids[i],
                        question=docs[i],
                        answer=str(m.get("answer", "")),
                        intent=str(m.get("intent", "")),
                        product_id=str(m.get("product_id", "")),
                        thread_id=str(m.get("thread_id", "")),
                        created_at=str(m.get("created_at", "")),
                    ),
                    distance=dists[i],
                )
            )
        return hits
