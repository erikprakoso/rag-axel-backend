from interfaces import VectorDBInterface, EmbeddingInterface
from config import settings
from typing import List, Dict
import uuid

class SentenceTransformerEmbedder(EmbeddingInterface):
    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or settings.EMBEDDING_MODEL)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]

class QdrantVectorDB(VectorDBInterface):
    def __init__(self, embedder: EmbeddingInterface, collection_name: str = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(
            host=settings.QDRANT_HOST, 
            port=settings.QDRANT_PORT
        )
        self.embedder = embedder
        self.collection_name = collection_name or "knowledge_base"
        self._create_collection()
    
    def _create_collection(self):
        # Existing implementation...
        pass
    
    def add_documents(self, documents: List[str], metadata: List[dict]) -> None:
        embeddings = self.embedder.embed_documents(documents)
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": doc, "metadata": meta}
            )
            for doc, embedding, meta in zip(documents, embeddings, metadata)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query: str, limit: int = 3) -> List[Dict]:
        query_embedding = self.embedder.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {
                "text": result.payload["text"],
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            }
            for result in results
        ]

# Factory untuk flexibility
class VectorDBFactory:
    @staticmethod
    def create_vector_db(db_type: str = None, **kwargs) -> VectorDBInterface:
        db_type = db_type or settings.VECTOR_DB_TYPE
        
        embedder = SentenceTransformerEmbedder()
        
        if db_type == "qdrant":
            return QdrantVectorDB(embedder, **kwargs)
        # Tambahkan implementasi lain (Chroma, Pinecone, etc)
        else:
            raise ValueError(f"Unsupported vector DB: {db_type}")

# Global instance dengan config
vector_db = VectorDBFactory.create_vector_db()