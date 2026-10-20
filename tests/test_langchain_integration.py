import importlib.util
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None, reason="langchain-core not installed")


def test_real_base_retriever_and_runnable():
    from langchain_core.retrievers import BaseRetriever
    from fuzzy_langchain import DocumentRecord, FuzzyIndex
    from fuzzy_langchain.retriever import FuzzyRetriever
    from fuzzy_langchain.rag import LangChainFuzzyRAG

    docs = [
        DocumentRecord("old", "The current portal is OldDesk.", metadata={"authority":0.4,"freshness":0.1,"reliability":0.5}),
        DocumentRecord("new", "The current portal is PeopleHub.", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
    ]
    retriever = FuzzyRetriever(index=FuzzyIndex(docs), k=2)
    assert isinstance(retriever, BaseRetriever)
    out = LangChainFuzzyRAG(retriever.index, k=2).as_runnable().invoke({"question":"What is the current portal?"})
    assert "PeopleHub" in out["answer"]
