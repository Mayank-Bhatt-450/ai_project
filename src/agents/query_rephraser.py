from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.services.ollama_client_maker import OllamaClient
from agents.prompt_templates.query_rephrase_prompt import QUERY_REPHRASER_SYSTEM_PROMPT


class QueryRephraserService:
    def __init__(self):
        self.llm = OllamaClient(tools=None, model="qwen2.5:1.5b").get_llm_client()

        self.rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_REPHRASER_SYSTEM_PROMPT),
            ("human", "{query}")
        ])

        self.chain = self.rephrase_prompt | self.llm | StrOutputParser()

    def should_rephrase(self, query: str) -> bool:
        """Determine if rephrasing is needed"""
        query = query.strip()

        if query.startswith('#'):
            return False

        if len(query) < 5:
            return False

        if query.endswith('?') and len(query.split()) > 3:
            return False

        return True

    def rephrase(self, query: str) -> dict:
        """Rephrase the query for better retrieval"""
        if not self.should_rephrase(query):
            return {
                "original": query,
                "rephrased": query,
                "was_rephrased": False
            }

        try:
            rephrased = self.chain.invoke({"query": query})
            rephrased = rephrased.strip()

            return {
                "original": query,
                "rephrased": rephrased,
                "was_rephrased": rephrased != query
            }
        except Exception as e:
            return {
                "original": query,
                "rephrased": query,
                "was_rephrased": False,
                "error": str(e)
            }