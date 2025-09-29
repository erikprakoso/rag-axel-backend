import ollama
from typing import List, Dict


def generate_response_with_ollama(context: str, question: str, model: str = "llama3:8b") -> str:
    """
    Generate response as AXEL - Direct and to the point
    """
    prompt = f"""Berdasarkan informasi berikut:

{context}

Jawab pertanyaan ini dengan singkat dan langsung ke intinya: {question}

Jawaban:"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Lebih rendah untuk konsistensi
                "top_p": 0.9,
                "num_ctx": 4096,
                "system": "Anda adalah AXEL, asisten dokumentasi API Telkom. Jawab pertanyaan dengan SINGKAT, PADAT, langsung ke inti. Hindari salam pembuka, penutup, atau kata-kata tidak perlu. Fokus pada informasi teknis yang diminta."
            }
        )
        return response['response'].strip()
    except Exception as e:
        return "Error: Sistem sedang gangguan."


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
