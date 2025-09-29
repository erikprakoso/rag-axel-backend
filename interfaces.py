from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBInterface(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str], metadata: List[dict]) -> None:
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int) -> List[Dict]:
        pass

class EmbeddingInterface(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass