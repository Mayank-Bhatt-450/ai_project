import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import Settings
from services.memory import Memory
from services.vector_store import VectorStore
from agents.services.ollama_client_maker import OllamaClient
from data_ingestors import text_ingestor
config = Settings()
logger = logging.getLogger(__name__)

# BUG FIX: NOT_FOUND_MESSAGE must be a plain single-line string with no

NOT_FOUND_MESSAGE = "I could not find relevant information in the provided sources."

SYSTEM_TEMPLATE = """You are a precise research assistant. You answer ONLY \
using the numbered SOURCE excerpts below. No other knowledge may be used.

Rules — no exceptions:
• Every factual claim must cite its source with a bracketed number, e.g. [1] or [2][3].
• If the sources do not fully answer the question, say so explicitly.
• If NONE of the sources are relevant, respond with exactly:
  "{not_found}"
• Never invent, guess, or extrapolate beyond what the sources say.

SOURCES:
{context}"""

HUMAN_TEMPLATE = "{question}"


@dataclass
class Citation:
    index: int
    source: str
    section: str
    page: Any
    snippet: str
    score: float


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True


def _build_llm():
    return OllamaClient().get_llm_client()


def format_docs_for_prompt(docs_and_scores: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        blocks.append(
            f"[{i}] (source: {meta.get('source', 'unknown')}, "
            f"{meta.get('section', '')})\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def docs_to_citations(docs_and_scores: list[tuple[Document, float]]) -> list[Citation]:
    return [
        Citation(
            index=i,
            source=doc.metadata.get("source", "unknown"),
            section=doc.metadata.get("section", ""),
            page=doc.metadata.get("page") or None,
            snippet=doc.page_content[:280] + ("..." if len(doc.page_content) > 280 else ""),
            score=round(score, 4),
        )
        for i, (doc, score) in enumerate(docs_and_scores, start=1)
    ]


class RAGAgent:
    def __init__(self, vectorstore: VectorStore = None, memory: Memory = None):
        self.vectorstore = vectorstore or VectorStore()
        self.memory = memory or Memory()
        self._llm = _build_llm()
        self._chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("human", HUMAN_TEMPLATE),
        ])
        return RunnablePassthrough() | prompt | self._llm | StrOutputParser()

    def answer(self, user_id: str, session_id: str, query: str) -> RAGResponse:
        
        if query.strip().lower().startswith("#remember:"):
            fact = query.split(":", 1)[-1].strip()
            self.memory.add_message(user_id, session_id, "user", query)
            ack = f"Got it, I'll remember: {fact}"
            self.memory.add_message(user_id, session_id, "assistant", ack)
            docs=text_ingestor.process_contenets(process_text_flag=True,data=fact)
            self.vectorstore.add_documents(docs)
            return RAGResponse(answer=ack, citations=[], grounded=True)

        # Retrieve and score
        docs_and_scores = self.vectorstore.similarity_search_with_score(query)
        relevant = [
            (doc, score) for doc, score in docs_and_scores
            if score >= config.relevance_similarity_threshold
        ]

        logger.info(
            "query=%r  retrieved=%d  above_threshold=%d  threshold=%.2f",
            query, len(docs_and_scores), len(relevant), config.relevance_similarity_threshold,
        )

        if not relevant:
            self.memory.add_message(user_id, session_id, "user", query)
            self.memory.add_message(user_id, session_id, "assistant", NOT_FOUND_MESSAGE)
            return RAGResponse(answer=NOT_FOUND_MESSAGE, citations=[], grounded=False)

        context = format_docs_for_prompt(relevant)
        citations = docs_to_citations(relevant)

        history = self.memory.get_recent_history(user_id, limit=config.recent_history_turns)
        preferences = self.memory.get_preferences(user_id)

        history_text = ""
        if preferences:
            history_text += "User preferences: " + "; ".join(preferences.values()) + "\n\n"
        if history:
            history_text += "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in history
            )

        question = f"{history_text}\n\nUser: {query}".strip() if history_text else query

        answer_text: str = self._chain.invoke({
            "context": context,
            "not_found": NOT_FOUND_MESSAGE,
            "question": question,
        })
        answer_text = answer_text.strip()
        grounded = NOT_FOUND_MESSAGE not in answer_text

        self.memory.add_message(user_id, session_id, "user", query)
        self.memory.add_message(user_id, session_id, "assistant", answer_text)

        return RAGResponse(
            answer=answer_text,
            citations=citations if grounded else [],
            grounded=grounded,
        )

    def format_response(self, response: RAGResponse) -> str:
        output = [response.answer]
        if response.citations:
            output.append("\nSources:")
            for c in response.citations:
                loc = f"page {c.page}" if c.page else c.section
                output.append(f"  [{c.index}] {c.source} ({loc}) [{c.score * 100:.0f}% match]")
                output.append(f'      "{c.snippet}"')
        return "\n".join(output)