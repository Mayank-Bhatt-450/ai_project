from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.services.ollama_client_maker import OllamaClient

class QueryRephraserService:
    def __init__(self):
        self.llm = OllamaClient(tools=None, model="qwen2.5:1.5b").get_llm_client()
        
        self.rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query optimization assistant. 
Your task is to rephrase user queries to improve retrieval from a knowledge base.

Rules:
- Keep the original intent and meaning exactly the same
- Expand abbreviations and acronyms if obvious
- Make vague queries more specific if context is clear
- Fix grammatical issues that might hurt search
- Do NOT add information not present in the original query
- Do NOT change commands (anything starting with #)
- Return ONLY the rephrased query, nothing else
- If the query is already clear and well-formed, return it unchanged

Examples:
Input: "wat is timsheet"
Output: "What is a timesheet?"

Input: "show me proj alpha docs"
Output: "Show me documentation for Project Alpha"

Input: "#timesheet"
Output: "#timesheet"

Input: "rahul brothers"
Output: "How many brothers does Rahul have?"
"""),
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