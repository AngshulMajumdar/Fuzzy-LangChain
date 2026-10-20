from __future__ import annotations

from typing import Any

from .core import DocumentRecord, FuzzyIndex

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - package remains importable without LangChain
    Document = None
    BaseRetriever = object
    ConfigDict = dict


class FuzzyRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter around :class:`FuzzyIndex`."""

    if BaseRetriever is not object:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    k: int = 4
    candidate_pool: int | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
        if Document is None:
            raise RuntimeError("langchain-core is required for FuzzyRetriever")
        hits = self.index.search(query, k=self.k, candidate_pool=self.candidate_pool)
        docs = []
        for h in hits:
            md = dict(h.document.metadata)
            md.update(
                {
                    "id": h.document.id,
                    "title": h.document.title,
                    "source": h.document.source,
                    "raw_relevance": h.raw_relevance,
                    "fuzzy_score": h.fuzzy_score,
                    "authority": h.authority,
                    "freshness": h.freshness,
                    "reliability": h.reliability,
                }
            )
            docs.append(Document(page_content=h.document.text, metadata=md))
        return docs


def to_records(documents: list[Any]) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for i, d in enumerate(documents):
        md = dict(getattr(d, "metadata", {}) or {})
        records.append(
            DocumentRecord(
                id=str(md.get("id", f"doc-{i}")),
                text=str(getattr(d, "page_content", "")),
                title=str(md.get("title", "")),
                source=str(md.get("source", "")),
                metadata=md,
            )
        )
    return records
