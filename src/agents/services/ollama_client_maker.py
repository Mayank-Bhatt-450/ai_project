
import requests
from langchain_ollama import ChatOllama
from  config import Settings
config=Settings()

class OllamaClient:
    def __init__(self, tools=[],base_url: str = None, model: str = None):
        print((base_url , config.ollama_base_url))
        self.base_url = self.base_url = (
                                            (base_url or config.ollama_base_url).rstrip("/")
                                            if (base_url or config.ollama_base_url)
                                            else None
                                        )
        self.model = model or config.ollama_model
        self.tools=tools


    def get_llm_client(self):
        if self.base_url:
            return ChatOllama(
            base_url=self.base_url,
            model=self.model ,
            num_predict=10000,
            tools=self.tools,
        )
        else:
            return ChatOllama(model=self.model)
