from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from evaluation.rag_benchmark import CLEAN_DOCS, CLEAN_CASES, CONFLICT_DOCS, CONFLICT_CASES
from fuzzy_langchain.core import FuzzyIndex, ExtractiveGroundedGenerator
from fuzzy_langchain.rag import LangChainFuzzyRAG


def _run_suite(docs, cases):
    rag = LangChainFuzzyRAG(FuzzyIndex(docs), k=4)
    rows=[]
    for c in cases:
        out=rag.as_runnable().invoke({"question":c["q"]})
        evidence=out.get("evidence",[])
        top_id = str(evidence[0].get("id")) if evidence else None
        if c["abstain"]:
            correct = out["answer"] == ExtractiveGroundedGenerator.abstain_text
        else:
            correct = str(c["gold"]).lower() in out["answer"].lower()
        rows.append({"id":c["id"],"top_doc":top_id,"answer":out["answer"],"decision":out["decision"],"correct":correct,"citations":out["citations"]})
    return {
        "n":len(rows),
        "accuracy":sum(r["correct"] for r in rows)/len(rows),
        "rows":rows,
    }


def run_framework_benchmark(output_path: str|Path|None=None):
    result={
        "framework":"langchain-core",
        "pipeline":"BaseRetriever -> fuzzy decision/optional expanded retrieval -> RunnableLambda -> grounded generation/abstention",
        "clean":_run_suite(CLEAN_DOCS,CLEAN_CASES),
        "conflict":_run_suite(CONFLICT_DOCS,CONFLICT_CASES),
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

if __name__=="__main__": print(json.dumps(run_framework_benchmark(),indent=2))
