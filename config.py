from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any

class Settings(BaseSettings):
    # Vector DB Config
    VECTOR_DB_TYPE: str = "qdrant"  # atau "chroma", "pinecone", dll
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # Embedding Config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # LLM Config
    LLM_PROVIDER: str = "ollama"  # atau "openai", "anthropic", dll
    LLM_MODEL: str = "llama3:8b"
    
    # Processing Config
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ✅ Cara baru di Pydantic v2
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
