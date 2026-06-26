import logging
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.vector_store import VectorStore
from data_ingestors import text_ingestor
from agents.rag_agent import RAGAgent
from services.memory import Memory
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.DEBUG) 



_store  = VectorStore()
_memory = Memory()
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await RAGAgent.create(vectorstore=_store, memory=_memory)
    yield

app = FastAPI(title="RAG Agent API", lifespan=lifespan)
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class FileIngestRequest(BaseModel):
    path: str


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        resp = await _agent.answer(req.user_id, req.session_id, req.message)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "answer": resp.answer,
        "grounded": resp.grounded,
        "citations": [asdict(c) for c in resp.citations],
    }


@app.post("/ingest/file")
def ingest_file(req: FileIngestRequest):
    try:
        docs = text_ingestor.process_contenets(req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    added = _store.add_documents(docs)
    return {"added_chunks": added, "total_chunks": _store.count()}


@app.get("/sources")
def sources():
    return {"sources": _store.list_sources()}

@app.get("/debug/scores")
def debug_scores(query: str, k: int = 5):
    results = _store.similarity_search_with_score(query, k=k)
    return {
        "query": query,
        "threshold": _agent.vectorstore._store._collection.configuration,
        "results": [
            {
                "score": round(score, 4),
                "source": doc.metadata.get("source"),
                "section": doc.metadata.get("section"),
                "snippet": doc.page_content[:200],
            }
            for doc, score in results
        ],
    }