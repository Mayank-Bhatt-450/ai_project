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
        description="DB for storing data"
    )

    class Config:
        env_prefix = ""