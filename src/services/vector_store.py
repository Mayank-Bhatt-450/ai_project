import hashlib
import logging
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings

config = Settings()
logger = logging.getLogger(__name__)


def _build_embeddings():
    from langchain_ollama import OllamaEmbeddings
    kwargs = {"model": config.ollama_embedding_model}
    if config.ollama_base_url:
        kwargs["base_url"] = config.ollama_base_url
    return OllamaEmbeddings(**kwargs)


class VectorStore:
    def __init__(self):
        self._embeddings = _build_embeddings()
        self._client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self._store = Chroma(
            client=self._client,
            collection_name="knowledge_base",
            embedding_function=self._embeddings,
            collection_configuration={"hnsw": {"space": "cosine"}},
        )

    
    def add_documents(self, docs) :
        ids, texts, metas = [], [], []
        for doc in docs:
            h = hashlib.sha256(
                f"{doc.metadata.get('source')}::{doc.metadata.get('section')}::{doc.page_content}".encode()
            ).hexdigest()
            ids.append(h)
            texts.append(doc.page_content)
            metas.append({k: (v if v is not None else "") for k, v in doc.metadata.items()})

        if not ids:
            return 0

        self._store.add_texts(texts=texts, metadatas=metas, ids=ids)
        logger.info("Added %d chunks to vector store", len(ids))
        return len(ids)

    
    def similarity_search_with_score(
        self, query: str, k: int = None
    ):
        """
        Returns (Document, similarity_score) pairs where score ∈ [0, 1].
        Higher score = more similar. Uses cosine similarity via LangChain's
        relevance score normalization: relevance = 1 - cosine_distance.
        """
        k = k or 5
        if self.count() == 0:
            logger.warning("similarity_search called but vector store is empty")
            return []

        results = self._store.similarity_search_with_relevance_scores(query, k=min(k, self.count()))


        for doc, score in results:
            logger.debug(
                "score=%.4f source=%s section=%s",
                score,
                doc.metadata.get("source"),
                doc.metadata.get("section"),
            )
        return results

    
    def list_sources(self) :
        col = self._client.get_collection("knowledge_base")
        if col.count() == 0:
            return []
        all_meta = col.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "unknown") for m in all_meta})

    def delete_source(self, source: str):
        col = self._client.get_collection("knowledge_base")
        result = col.get(where={"source": source}, include=[])
        ids = result.get("ids", [])
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def count(self):
        try:
            return self._client.get_collection("knowledge_base").count()
        except Exception:
            return 0