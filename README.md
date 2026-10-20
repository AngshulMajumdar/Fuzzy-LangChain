# Fuzzy LangChain RAG

**Fuzzy LangChain RAG** adds an interpretable fuzzy decision layer to LangChain retrieval-augmented generation. Evidence is evaluated jointly by **relevance, source authority, freshness, and reliability**. A Sugeno fuzzy controller then decides whether to answer, widen retrieval, or abstain.

## Architecture

`documents -> candidate retrieval -> fuzzy evidence scoring -> optional retrieval expansion -> grounded answer + citation OR abstention`

The LangChain adapter is a real `BaseRetriever`; the end-to-end path can be exposed as a `RunnableLambda`. The RAG logic does not throw away uncertainty immediately after top-k retrieval.

## Why fuzzy orchestration?

Binary thresholds are brittle when evidence quality is continuous. A source may be highly relevant but stale, fresh but weakly authoritative, or trustworthy while only moderately relevant. Fuzzy memberships preserve those partial truths long enough for the RAG system to make an explicit decision.

The included controller uses low/medium/high memberships and transparent Sugeno rules. A trusted current source can defeat a lexically stronger stale source; equal high-quality contradictions trigger expanded retrieval and then abstention if ambiguity remains.

## Full RAG benchmark

The benchmark executes the complete path: ingestion, retrieval, fuzzy assessment, optional expanded retrieval, grounded answer synthesis, citation, and abstention.

| Suite | Fuzzy RAG | Plain TF-IDF RAG |
|---|---:|---:|
| Clean end-to-end accuracy | **1.000** | 1.000 |
| Conflict/stale/missing end-to-end accuracy | **1.000** | 0.278 |
| Conflict Recall@1 | **1.000** | 0.250 |
| Conflict MRR | **1.000** | 0.611 |
| Conflict abstention accuracy | **1.000** | 0.333 |
| Grounding accuracy | **1.000** | 1.000 |

Raw case-level results are in `artifacts/full_rag_results.json`. Framework-path results are in `artifacts/framework_full_rag_results.json`.

## Quick start

```python
from fuzzy_langchain import DocumentRecord, FuzzyIndex
from fuzzy_langchain.rag import LangChainFuzzyRAG

docs = [
    DocumentRecord(
        "official",
        "The current travel portal is TripDesk.",
        source="official-handbook",
        metadata={"authority": 0.98, "freshness": 1.0, "reliability": 0.95},
    )
]

index = FuzzyIndex(docs)
rag = LangChainFuzzyRAG(index)
print(rag.invoke("Which portal is used for travel?"))
```

## Reproduce

```bash
pip install -e .
python evaluation/rag_benchmark.py
python evaluation/framework_rag_benchmark.py
pytest -q
```

This GitHub repository is the **full RAG implementation**: source, benchmarks, tests, examples, raw results, and validation notes. Hugging Face deployment is intentionally split into a separate API-only package.
