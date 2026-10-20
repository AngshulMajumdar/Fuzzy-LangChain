from fuzzy_langchain import DocumentRecord, FuzzyRAGEngine

docs = [
    DocumentRecord("old", "The current portal is OldDesk.", source="cache", metadata={"authority":0.4,"freshness":0.1,"reliability":0.5}),
    DocumentRecord("new", "The current portal is PeopleHub.", source="handbook", metadata={"authority":0.98,"freshness":1.0,"reliability":0.95}),
]
rag = FuzzyRAGEngine(docs)
print(rag.query("What is the current portal?"))
