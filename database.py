from interfaces import VectorDBInterface, EmbeddingInterface
from config import settings
from typing import List, Dict
import uuid
from qdrant_client.models import Distance, PointStruct, VectorParams, MatchValue, FieldCondition, Filter


class SentenceTransformerEmbedder(EmbeddingInterface):
    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name or settings.EMBEDDING_MODEL)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]


class QdrantVectorDB(VectorDBInterface):
    def __init__(self, embedder: EmbeddingInterface, collection_name: str = None, similarity_threshold: float = 0.5):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.embedder = embedder
        self.collection_name = collection_name or "axel_base_knowledge"
        self.similarity_threshold = similarity_threshold
        self._create_collection()

    def _create_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=Distance.COSINE
                )
            )

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

    def search(self, query: str, limit: int = 3, domain: str = None) -> List[Dict]:
        print("Query:", query)
        query_embedding = self.embedder.embed_query(query)

        query_filter = None
        if domain:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.domain",
                        match=MatchValue(value=domain)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter
        )

        # Filter berdasarkan similarity threshold
        filtered_results = [
            {
                "text": result.payload["text"],
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            }
            for result in results
            # Hanya ambil yang similarity tinggi
            if result.score >= self.similarity_threshold
        ]

        print(
            f"Found {len(results)} results, {len(filtered_results)} after threshold filtering")
        return filtered_results


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
