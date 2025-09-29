from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Document(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class Query(BaseModel):
    question: str
    top_k: Optional[int] = 3


class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float


class RAGResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    question: str


class HealthCheck(BaseModel):
    status: str
    qdrant_status: str
    ollama_status: str
