
from pathlib import Path
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from services import chunk_creator

def _get_splitter(chunk_size: int = None, overlap: int = None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or 1000,
        chunk_overlap=overlap or 200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _enumerate_chunks(docs: list[Document]) -> list[Document]:
    """Add a human-readable 'section' field to each chunk for citations."""
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "")
        loc = f"page {page}, " if page else ""
        doc.metadata["section"] = f"{loc}chunk {i}"
    return docs


def load_text_or_markdown(path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    ftype = "markdown" if path.suffix.lower() in (".md", ".markdown") else "text"
    doc = Document(
        page_content=_clean(raw),
        metadata={"source": path.name, "type": ftype, "page": ""},
    )
    splitter = _get_splitter()
    return _enumerate_chunks(splitter.split_documents([doc]))

def load_raw_text(data):
    doc = Document(
        page_content=_clean(data),
        metadata={"source": 'from chat', "type": 'chat', "page": ""},
    )
    splitter = _get_splitter()
    return _enumerate_chunks(splitter.split_documents([doc]))

def process_contenets(path: str | Path='',process_text_flag:bool=False,data=''):
    """Dispatch a local file to the right loader based on extension."""
    if not process_text_flag:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md", ".markdown"):
            return load_text_or_markdown(path)
        raise ValueError(
            f"Unsupported file type '{suffix}'."
        )
    else:
        return load_raw_text(data)
