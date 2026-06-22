
import requests
from langchain_ollama import ChatOllama
from  config import Settings
config=Settings()

class OllamaClient:
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.model = model or config.ollama_model


    def get_llm_client(self):
        if self.base_url:
            return ChatOllama(
            base_url=self.base_url,
            model=self.model ,
            num_predict=10000,
        )
        else:
            return ChatOllama(model=self.model)
