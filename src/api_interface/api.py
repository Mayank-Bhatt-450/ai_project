
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.vector_store import VectorStore
from data_ingestors import text_ingestor

app = FastAPI(title="RAG Agent API")


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class FileIngestRequest(BaseModel):
    path: str


store =VectorStore()
@app.post("/chat")
def chat(req: ChatRequest):
    pass


@app.post("/ingest/file")
def ingest_file(req: FileIngestRequest):
    try:
        chunks = text_ingestor.process_file(req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    added = store.add_chunks(chunks)
    
    return {"added_chunks": added, "total_chunks": store.count()}

@app.get("/sources")
def sources():
    return {"sources":store.list_sources()}
