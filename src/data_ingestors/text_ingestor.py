
from pathlib import Path



from services import chunk_creator


def load_text_or_markdown(path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    file_type = "markdown" if path.suffix.lower() in (".md", ".markdown") else "text"
    response = []
    for i, chunk in enumerate(chunk_creator.chunk_by_sentences(raw)):
        response.append({
            "text": chunk,
            "metadata": {
                "source": path.name,
                "type": file_type,
                "page": None,
                "section": f"chunk {i + 1}",
            },
        })
    return response


def process_file(path: str | Path) -> list[dict]:
    """Dispatch a local file to the right loader based on extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        return load_text_or_markdown(path)
    raise ValueError(
        f"Unsupported file type '{suffix}'."
    )
