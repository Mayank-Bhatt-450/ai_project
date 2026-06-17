
import hashlib
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from config import Settings

settings=Settings()
class VectorStore:
    def __init__(self):
        print(settings.chroma_dir)
        self.client=chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name='knowledge_base',
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    def add_chunks(self, chunks):
        if not chunks:
            return 0

        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            text = chunk["text"]
            meta = chunk["metadata"]
            content_key = f"{meta.get('source')}::{meta.get('section')}::{text}"
            chunk_id = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
            ids.append(chunk_id)
            documents.append(text)

            clean_meta = {
                key: (value if value is not None else "") for key, value in meta.items()
                }
            metadatas.append(clean_meta)

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)
    
    def query(self, query_text, top_k = 5) :
        top_k = top_k 
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self.collection.count()),
        )

        response = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            response.append({"text": doc, "metadata": meta, "distance": dist})
        return response
    
    def list_sources(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted(
            {m.get("source", "unknown") for m in all_meta}
            )

    def delete_source(self, source: str) -> int:
        existing = self.collection.get(where={"source": source}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()