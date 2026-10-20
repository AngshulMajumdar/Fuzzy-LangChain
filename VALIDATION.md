# Validation

Validated on Python 3.13.5.

- Repository test suite: **7/7 passed**.
- Core full-RAG benchmark: **8/8 clean + 18/18 conflict/stale/missing cases passed**.
- LangChain full-RAG benchmark: **8/8 clean + 18/18 conflict/stale/missing cases passed**.
- `FuzzyRetriever` validated as a real `langchain_core.retrievers.BaseRetriever` subclass against the supplied **langchain-core 1.6.11** source.
- End-to-end LangChain execution validated through `RunnableLambda`.
- Wheel build with local build tooling: passed.

The sandbox cannot resolve PyPI. For validation against the supplied LangChain source archive, minimal **test-only** compatibility modules were supplied for unavailable peripheral dependencies such as LangSmith tracing. They are not part of this repository and do not replace the retriever, Runnable, fuzzy controller, RAG engine, or benchmark.
