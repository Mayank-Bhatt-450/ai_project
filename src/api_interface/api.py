import logging
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.vector_store import VectorStore
from data_ingestors import text_ingestor
from agents.rag_agent import RAGAgent
from services.memory import Memory
from contextlib import asynccontextmanager
from agents.query_rephraser import QueryRephraserService 

logging.basicConfig(level=logging.DEBUG) 



_store  = VectorStore()
_memory = Memory()

_query_rephraser = QueryRephraserService() 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await RAGAgent.create(vectorstore=_store, memory=_memory)
    _query_rephraser.rephrase("warm up")
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
        
        if not req.message.strip().startswith('#'):
            rephrase_result = _query_rephraser.rephrase(req.message)
            message_to_process = rephrase_result["rephrased"]
        else:
            
            message_to_process = req.message
        resp = await _agent.answer(req.user_id, req.session_id, message_to_process)
        print('##################message_to_process=',message_to_process)
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

@app.get("/usage/{user_id}")
def get_token_usage(user_id: str):
    """Return cumulative token usage for a user, broken down by model."""
    return _memory.get_token_usage(user_id)


@app.get("/chats/{user_id}/sessions")
def list_sessions(user_id: str):
    """List all sessions for a user with message counts and timestamps."""
    return {"user_id": user_id, "sessions": _memory.list_sessions(user_id)}


@app.get("/chats/{user_id}")
def get_chat_history(
    user_id: str,
    session_id: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Retrieve paginated chat messages for a user.
    Optionally filter by session_id. Supports limit/offset pagination.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    return _memory.get_chat_history(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )