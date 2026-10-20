from __future__ import annotations

from typing import Any, Callable

from .core import ExtractiveGroundedGenerator, FuzzyIndex, FuzzyRetrievalConfig
from .retriever import FuzzyRetriever


class LangChainFuzzyRAG:
    """End-to-end LangChain RAG using fuzzy evidence control.

    The optional ``generator`` receives ``(question, documents)`` and may be any
    callable backed by an LLM. Without one, a deterministic grounded sentence
    synthesizer is used for offline evaluation and zero-key deployments.
    """

    def __init__(self, index: FuzzyIndex, k: int = 4, generator: Callable[[str, list[Any]], str] | None = None):
        self.index = index
        self.retriever = FuzzyRetriever(index=index, k=k)
        self.k = k
        self.generator = generator
        self.extractive = ExtractiveGroundedGenerator()

    def invoke(self, question: str) -> dict[str, Any]:
        docs = self.retriever.invoke(question)
        # Crucially, decision logic uses the LangChain retriever result rather than
        # silently issuing a second retrieval outside the framework path.
        from .core import DocumentRecord, FuzzyEvidence

        hits: list[FuzzyEvidence] = []
        for d in docs:
            md = dict(d.metadata or {})
            record = DocumentRecord(
                id=str(md.get("id", "unknown")),
                text=d.page_content,
                title=str(md.get("title", "")),
                source=str(md.get("source", "")),
                metadata=md,
            )
            hits.append(
                FuzzyEvidence(
                    document=record,
                    raw_relevance=float(md.get("raw_relevance", 0.0)),
                    fuzzy_score=float(md.get("fuzzy_score", 0.0)),
                    authority=float(md.get("authority", 0.5)),
                    freshness=float(md.get("freshness", 0.5)),
                    reliability=float(md.get("reliability", 0.5)),
                    memberships={},
                )
            )
        decision = self.index.decision(hits)
        if decision["route"] == "expand":
            expanded_retriever = FuzzyRetriever(index=self.index, k=max(self.k * 2, 6), candidate_pool=self.index.config.extended_pool)
            docs = expanded_retriever.invoke(question)
            hits = []
            for d in docs:
                md = dict(d.metadata or {})
                record = DocumentRecord(
                    id=str(md.get("id", "unknown")), text=d.page_content,
                    title=str(md.get("title", "")), source=str(md.get("source", "")), metadata=md
                )
                hits.append(FuzzyEvidence(
                    record, float(md.get("raw_relevance", 0.0)), float(md.get("fuzzy_score", 0.0)),
                    float(md.get("authority", 0.5)), float(md.get("freshness", 0.5)),
                    float(md.get("reliability", 0.5)), {}
                ))
            decision = self.index.decision(hits)
            if decision["route"] == "expand":
                decision = {**decision, "route": "abstain", "reason": "ambiguous_after_expansion"}
        if decision["route"] != "generate":
            answer, citations = self.extractive.abstain_text, []
        elif self.generator is not None:
            answer = str(self.generator(question, docs))
            citations = [str(d.metadata.get("id")) for d in docs[:1]]
        else:
            answer, citations = self.extractive.generate(question, hits)
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "confidence": float(decision["confidence"]),
            "margin": float(decision["margin"]),
            "decision": decision["route"],
            "reason": decision["reason"],
            "evidence": [dict(d.metadata) for d in docs],
        }

    def as_runnable(self):
        try:
            from langchain_core.runnables import RunnableLambda
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("langchain-core is required for as_runnable()") from exc
        return RunnableLambda(lambda x: self.invoke(x["question"] if isinstance(x, dict) else str(x)))
