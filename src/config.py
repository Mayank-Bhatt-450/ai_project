from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data: Path = Field(
        default="./data",
        description="Data location for AI"
    )
    chroma_dir: Path = Field(
        default=Path(__file__).resolve().parent / "chroma_db",
        description="DB for storing vector embeddings"
    )
    memory_db_path: Path = Field(
        default=Path(__file__).resolve().parent / "db.sql",
        description="SQLite DB for conversation memory"
    )
    recent_history_turns: int = Field(
        default=5,
        description="Recent history turns to include in context"
    )

    ollama_base_url: str = Field(
        default="",
        description="Ollama base URL (leave empty to use Ollama's default localhost)"
    )
    ollama_model: str = Field(
        default="gemma4:latest",
        description="Ollama chat model name"
    )
    ollama_request_timeout: int = Field(
        default=120,   
        description="Timeout in seconds for Ollama requests"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name"
    )

    relevance_similarity_threshold: float = Field(
        default=0.30,
        description="Minimum cosine similarity [0,1] for a chunk to be used in an answer"
    )

    llm_retry_max_attempts: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls"
    )
    llm_retry_min_wait: float = Field(
        default=2.0,
        description="Minimum wait in seconds between LLM retries (exponential backoff)"
    )
    llm_retry_max_wait: float = Field(
        default=30.0,
        description="Maximum wait in seconds between LLM retries (exponential backoff)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = ""
        extra = "ignore"