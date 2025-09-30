from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Document(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float


class RAGResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    question: str
    conversation_id: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    qdrant_status: str
    ollama_status: str


# Tambahkan model untuk conversation context
class ConversationContext(BaseModel):
    conversation_id: Optional[str] = None
    previous_messages: List[Dict] = []
    max_history: int = 5


class Query(BaseModel):
    question: str
    top_k: Optional[int] = 3
    conversation_id: Optional[str] = None


class ConversationContext:
    def __init__(self, conversation_id: str, messages: List[Dict] = None):
        self.conversation_id = conversation_id
        self.messages = messages or []
