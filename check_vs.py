from src.services.vector_store import VectorStore
vs = VectorStore()
print('Count:', vs.count())
print('Sources:', vs.list_sources())
if vs.count() > 0:
    results = vs.similarity_search_with_score('how to install', k=5)
    for doc, score in results:
        print(f'Score: {score:.4f} | Source: {doc.metadata.get("source")} | Section: {doc.metadata.get("section")}')
        print(f'  Content: {doc.page_content[:200]}...')
        print()