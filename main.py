from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import os

from database import vector_db, VectorDBFactory
from models import Document, Query, RAGResponse, HealthCheck, SearchResult
from utils import format_sources, generate_response_with_ollama, check_ollama_health, build_enhanced_query, generate_stream_response_with_ollama
from document_processors import DocumentProcessorFactory
from conversation_manager import conversation_manager

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation dengan FastAPI, Qdrant, dan Ollama dengan Conversation Context",
    version="1.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task untuk cleanup


@app.on_event("startup")
async def startup_event():
    import asyncio

    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # setiap jam
            cleaned = conversation_manager.cleanup_expired_conversations()
            if cleaned > 0:
                print(f"Cleaned up {cleaned} expired conversations")

    asyncio.create_task(periodic_cleanup())


@app.get("/", tags=["Health"])
async def root():
    return {"message": "RAG System API with Conversation Context is running!"}


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
            db_instance = VectorDBFactory.create_vector_db(
                collection_name=collection)

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
        file_extension = os.path.splitext(file.filename)[1].lower()
        processor = DocumentProcessorFactory.get_processor(file_extension)

        with open(f"temp_{file.filename}", "wb") as f:
            content = await file.read()
            f.write(content)

        chunks = processor.process(
            f"temp_{file.filename}", chunk_size, chunk_overlap)

        os.remove(f"temp_{file.filename}")

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


from fastapi.responses import StreamingResponse
import json
from models import StreamResponse, StreamOption

@app.post("/ask", response_model=RAGResponse, tags=["RAG"])
async def ask_question(query: Query, background_tasks: BackgroundTasks):
    """Ask a question using RAG dengan conversation context internal"""
    try:
        # Handle conversation ID
        conversation_id = query.conversation_id

        # Validasi conversation ID
        if conversation_id and not conversation_manager.conversation_exists(conversation_id):
            conversation_id = None

        # Buat conversation baru jika tidak ada ID yang valid
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation()

        # Dapatkan conversation history secara internal
        conversation_history = conversation_manager.get_recent_context(
            conversation_id, max_messages=3)

        # Enhance query secara internal menggunakan history
        enhanced_query = build_enhanced_query(
            query.question, conversation_history)

        print(f"Original query: {query.question}")
        print(f"Enhanced query: {enhanced_query}")
        print(f"Conversation ID: {conversation_id}")
        print(f"Stream: {query.stream}")

        # 1. Retrieve relevant documents dengan enhanced query
        results = vector_db.search(enhanced_query, query.top_k)

        print(f"Search results: {len(results)} documents found")

        # Handle no relevant documents
        if not results:
            response = RAGResponse(
                answer="Maaf, saya tidak menemukan informasi yang relevan tentang pertanyaan Anda di dokumentasi API Telkom.",
                sources=[],
                question=query.question,
                conversation_id=conversation_id
            )

            background_tasks.add_task(
                conversation_manager.add_message,
                conversation_id,
                "user",
                query.question,
                {"has_relevant_docs": False}
            )
            background_tasks.add_task(
                conversation_manager.add_message,
                conversation_id,
                "assistant",
                response.answer,
                {"has_relevant_docs": False}
            )

            return response

        # 2. Cek similarity score
        max_score = max([result["score"] for result in results])
        if max_score < 0.3:
            response = RAGResponse(
                answer="Maaf, informasi yang saya miliki tidak cukup relevan untuk menjawab pertanyaan tersebut.",
                sources=format_sources(results),
                question=query.question,
                conversation_id=conversation_id
            )

            background_tasks.add_task(
                conversation_manager.add_message,
                conversation_id,
                "user",
                query.question,
                {"has_relevant_docs": False, "max_score": max_score}
            )
            background_tasks.add_task(
                conversation_manager.add_message,
                conversation_id,
                "assistant",
                response.answer,
                {"has_relevant_docs": False, "max_score": max_score}
            )

            return response

        # 3. Prepare context
        context = "\n\n".join(
            [f"Source {i+1} (relevansi: {result['score']:.2f}):\n{result['text']}"
             for i, result in enumerate(results)])

        # 4. Handle streaming vs non-streaming
        if query.stream == StreamOption.TRUE:
            return await handle_streaming_response(
                context, query.question, conversation_history, 
                results, conversation_id, background_tasks
            )
        else:
            return await handle_normal_response(
                context, query.question, conversation_history,
                results, conversation_id, background_tasks
            )

    except Exception as e:
        print(f"Error in ask_question: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}")

async def handle_normal_response(context: str, question: str, conversation_history: List[Dict],
                               results: List, conversation_id: str, background_tasks: BackgroundTasks):
    """Handle non-streaming response"""
    answer = generate_response_with_ollama(
        context, question, conversation_history)

    response = RAGResponse(
        answer=answer,
        sources=format_sources(results),
        question=question,
        conversation_id=conversation_id
    )

    # Simpan ke conversation history
    max_score = max([result["score"] for result in results])
    background_tasks.add_task(
        conversation_manager.add_message,
        conversation_id,
        "user",
        question,
        {"has_relevant_docs": True, "doc_count": len(results), "max_score": max_score}
    )
    background_tasks.add_task(
        conversation_manager.add_message,
        conversation_id,
        "assistant",
        answer,
        {"has_relevant_docs": True, "sources_count": len(results)}
    )

    return response

async def handle_streaming_response(context: str, question: str, conversation_history: List[Dict],
                                  results: List, conversation_id: str, background_tasks: BackgroundTasks):
    """Handle streaming response"""
    
    async def generate_stream():
        full_answer = ""
        try:
            # Generate streaming response
            stream = generate_stream_response_with_ollama(
                context, question, conversation_history
            )
            
            # Stream tokens
            for token in stream:
                full_answer += token
                yield f"data: {json.dumps({'token': token, 'conversation_id': conversation_id, 'is_final': False})}\n\n"
            
            # Final message
            yield f"data: {json.dumps({'token': '', 'conversation_id': conversation_id, 'is_final': True})}\n\n"
            
        except Exception as e:
            error_msg = "Error: Sistem sedang gangguan."
            yield f"data: {json.dumps({'token': error_msg, 'conversation_id': conversation_id, 'is_final': True})}\n\n"
            full_answer = error_msg
        
        finally:
            # Save to conversation history in background
            max_score = max([result["score"] for result in results])
            background_tasks.add_task(
                conversation_manager.add_message,
                conversation_id,
                "user",
                question,
                {"has_relevant_docs": True, "doc_count": len(results), "max_score": max_score}
            )
            background_tasks.add_task(
                conversation_manager.add_message,
                conversation_id,
                "assistant",
                full_answer,
                {"has_relevant_docs": True, "sources_count": len(results)}
            )

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.get("/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(conversation_id: str):
    """Get conversation history (untuk debugging/admin purposes)"""
    try:
        if not conversation_manager.conversation_exists(conversation_id):
            raise HTTPException(
                status_code=404, detail="Conversation not found")

        history = conversation_manager.get_conversation_history(
            conversation_id)
        return {
            "conversation_id": conversation_id,
            "message_count": len(history),
            "messages": history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation(conversation_id: str):
    """Delete conversation"""
    try:
        if conversation_manager.conversation_exists(conversation_id):
            # Dalam implementasi sebenarnya, kita akan hapus dari storage
            # Untuk sekarang, kita set sebagai expired
            conversation_manager.cleanup_expired_conversations()
            return {"message": f"Conversation {conversation_id} deleted"}
        else:
            raise HTTPException(
                status_code=404, detail="Conversation not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
