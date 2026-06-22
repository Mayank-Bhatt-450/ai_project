
from typing import  Literal
import nltk
from nltk.tokenize import sent_tokenize
# Download NLTK data once 
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

Strategy = Literal["word", "token", "sentence"]

def chunk_by_sentences(text, max_words = 250):
    
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        sent_len = len(sentence.split())
        if current and current_len + sent_len > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks