from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os

from database import vector_db, VectorDBFactory
from models import Document, Query, RAGResponse, HealthCheck, SearchResult
from utils import format_sources, generate_response_with_ollama, check_ollama_health
from document_processors import DocumentProcessorFactory

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation dengan FastAPI, Qdrant, dan Ollama",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    return {"message": "RAG System API is running!"}


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Check system health"""
    try:
        qdrant_info = vector_db.get_collection_info()
        qdrant_status = "healthy"
    except:
        qdrant_status = "unhealthy"

    ollama_status = check_ollama_health()

    return HealthCheck(
        status="healthy" if qdrant_status == "healthy" and ollama_status == "healthy" else "degraded",
        qdrant_status=qdrant_status,
        ollama_status=ollama_status
    )


@app.post("/documents", tags=["Documents"])
async def add_documents(documents: List[Document], collection: str = None):
    """Add documents to specified or default collection"""
    try:
        db_instance = vector_db
        if collection:
            # Buat instance baru untuk collection berbeda
            db_instance = VectorDBFactory.create_vector_db(collection_name=collection)
        
        texts = [doc.text for doc in documents]
        metadata = [doc.metadata for doc in documents]
        
        db_instance.add_documents(texts, metadata)
        return {"message": f"Added {len(documents)} documents to {collection or 'default'} collection"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", tags=["Documents"])
async def upload_file(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection: str = None
):
    """Upload various file types"""
    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1].lower()
        processor = DocumentProcessorFactory.get_processor(file_extension)
        
        # Process file
        with open(f"temp_{file.filename}", "wb") as f:
            content = await file.read()
            f.write(content)
        
        chunks = processor.process(f"temp_{file.filename}", chunk_size, chunk_overlap)
        
        # Clean up
        os.remove(f"temp_{file.filename}")
        
        # Add to vector DB
        documents = [Document(text=chunk) for chunk in chunks]
        await add_documents(documents, collection)
        
        return {"message": f"Processed {len(chunks)} chunks from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult], tags=["Search"])
async def search_documents(query: Query):
    """Search for similar documents"""
    try:
        results = vector_db.search(query.question, query.top_k)
        return format_sources(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching: {str(e)}")


@app.post("/ask", response_model=RAGResponse, tags=["RAG"])
async def ask_question(query: Query):
    """Ask a question using RAG"""
    try:
        # 1. Retrieve relevant documents
        results = vector_db.search(query.question, query.top_k)

        if not results:
            return RAGResponse(
                answer="Tidak ditemukan informasi yang relevan untuk pertanyaan ini.",
                sources=[],
                question=query.question
            )

        # 2. Prepare context
        context = "\n\n".join(
            [f"Source {i+1}:\n{result['text']}" for i, result in enumerate(results)])

        # 3. Generate response
        answer = generate_response_with_ollama(context, query.question)

        return RAGResponse(
            answer=answer,
            sources=format_sources(results),
            question=query.question
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/collection-info", tags=["Admin"])
async def get_collection_info():
    """Get information about the vector database collection"""
    try:
        info = vector_db.get_collection_info()
        return {
            "collection_name": info.config.params.vectors.size,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.vectors_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting collection info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
