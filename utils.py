import ollama
from typing import List, Dict


def generate_response_with_ollama(context: str, question: str, model: str = "llama3:8b") -> str:
    """
    Generate response using Ollama LLM with RAG context
    """
    prompt = f"""Berdasarkan konteks berikut, jawablah pertanyaan dengan jelas dan tepat.
    
KONTEKS:
{context}

PERTANYAAN: {question}

JAWABAN:"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        )
        return response['response']
    except Exception as e:
        return f"Error generating response: {str(e)}"


def check_ollama_health() -> str:
    """Check if Ollama is running and models are available"""
    try:
        models = ollama.list()
        llama_models = [model for model in models['models']
                        if 'llama3' in model['name']]
        if llama_models:
            return "healthy"
        return "no llama model found"
    except:
        return "unhealthy"


def format_sources(sources: List[Dict]) -> List[Dict]:
    """Format sources for response"""
    return [
        {
            "text": source["text"],
            "metadata": source.get("metadata") or {},
            "score": round(source["score"], 4)
        }
        for source in sources
    ]