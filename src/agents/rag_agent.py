import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import Settings
from services.memory import Memory
from services.vector_store import VectorStore
from agents.services.ollama_client_maker import OllamaClient
from data_ingestors import text_ingestor

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


config = Settings()
logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldnot find relevant information in the provided sources."
#8. Do NOT show tool details or traceback only response.
SYSTEM_TEMPLATE = """
You are a precise assistant with access to external MCP tools and a knowledge base.

You have two ways to answer user requests.

## 1. User Commands (Highest Priority)

If the user's message starts with the character '#', you MUST treat it as a command.

Examples:
- #timesheet
- #timesheet --showmore
- #leave
- #calendar

For these messages:

1. Immediately call the `execute_user_commands` tool.
2. Pass the user command unchanged as the tool input.
3. Return the tool's response directly to the user.
4. Do NOT answer from the knowledge base.
5. Do NOT explain what the tool does.
6. Do NOT summarize, modify, or fabricate the tool's output.
7. Only if the tool reports that the command is unknown or cannot be executed should you inform the user accordingly.


## 2. Knowledge Base

If the user's message does NOT begin with '#', answer using ONLY the numbered SOURCE excerpts provided below.

Rules:
- Every factual statement must cite its source using bracketed citations, for example: [1] or [2][3].
- Never use outside knowledge.
- Never fabricate facts.
- If the sources do not fully answer the question, explicitly state that the available information is incomplete.
- If none of the sources are relevant, respond with exactly:
  "{not_found}"

SOURCES:
{context}
"""

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


def _build_llm(tools):
    return OllamaClient(tools).get_llm_client()


def format_docs_for_prompt(docs_and_scores):
    blocks = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        blocks.append(
            f"[{i}] (source: {meta.get('source', 'unknown')}, "
            f"{meta.get('section', '')})\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def docs_to_citations(docs_and_scores):
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


def _make_retry_chain(chain, max_attempts: int, min_wait: float, max_wait: float):
    """Wrap an async ainvoke call with tenacity retry logic."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _invoke(payload):
        return await chain.ainvoke(payload)

    return _invoke


def _extract_token_usage(answer: dict) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for msg in answer.get("messages", []):
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
    return input_tokens, output_tokens


class RAGAgent:
    def __init__(self, vectorstore: VectorStore = None, memory: Memory = None):
        self.vectorstore = vectorstore or VectorStore()
        self.memory = memory or Memory()
        self.client = MultiServerMCPClient(
        {
            "time": {
                "transport": "sse",
                "url": "http://localhost:4040/sse"
            }
        }
    )
        
        self._chain = None
        self._llm =None
    async def _init_llm(self):
        self.tools= await self.client.get_tools()
        self._llm = _build_llm(self.tools)
        self._chain = self._build_chain()
        self._invoke = _make_retry_chain(
            self._chain,
            max_attempts=config.llm_retry_max_attempts,
            min_wait=config.llm_retry_min_wait,
            max_wait=config.llm_retry_max_wait,
        )

    def _build_chain(self):
        return create_agent(
            model=self._llm,
            tools=self.tools,
        )

    async def answer(self, user_id, session_id, query):
        is_command = query.strip().startswith("#")
        if query.strip().lower().startswith("#remember:"):
            fact = query.split(":", 1)[-1].strip()
            self.memory.add_message(user_id, session_id, "user", query)
            ack = f"Got it, I'll remember: {fact}"
            self.memory.add_message(user_id, session_id, "assistant", ack)
            docs=text_ingestor.process_contenets(process_text_flag=True,data=fact)
            self.vectorstore.add_documents(docs)
            return RAGResponse(answer=ack, citations=[], grounded=True)
        context=''
        citations=[]
        if not is_command:
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
        system_msg = SYSTEM_TEMPLATE.format(
            context=context, 
            not_found=NOT_FOUND_MESSAGE
        )

        messages = [
            ("system", system_msg),
            ("user", question) 
        ]
        answer = await self._invoke({
            "messages": messages
        })
        answer_text = ""
        
        # Loop backwards through the agent's memory
        for msg in reversed(answer["messages"]):
            print(msg.content)
            if msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                answer_text = ' '.join([i['text']for i in msg.content])
                break
                
            if msg.type == "tool" and msg.content:
                answer_text = ' '.join([i['text']for i in msg.content])
                break

        # Record token usage for this user
        input_tokens, output_tokens = _extract_token_usage(answer)
        if input_tokens or output_tokens:
            self.memory.record_token_usage(
                user_id=user_id,
                session_id=session_id,
                model=config.ollama_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            logger.info(
                "token_usage user=%s input=%d output=%d total=%d",
                user_id, input_tokens, output_tokens, input_tokens + output_tokens,
            )

        grounded = NOT_FOUND_MESSAGE not in answer_text

        self.memory.add_message(user_id, session_id, "user", query)
        self.memory.add_message(user_id, session_id, "assistant", answer_text)

        return RAGResponse(
            answer=answer_text,
            citations=citations if grounded else [],
            grounded=grounded,
        )

    @classmethod
    async def create(cls, vectorstore=None, memory=None):
        self = cls(vectorstore, memory)
        await self._init_llm()
        return self