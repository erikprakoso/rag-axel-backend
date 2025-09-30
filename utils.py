import ollama
from typing import Generator, List, Dict


def generate_response_with_ollama(context: str, question: str, conversation_history: List[Dict] = None, model: str = "llama3:8b") -> str:
    """
    Generate response as AXEL - Direct and to the point
    """
    history_context = ""
    if conversation_history:
        history_context = "\n\nRiwayat Percakapan Terkait:\n"
        for msg in conversation_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_context += f"{role}: {msg.get('content', '')}\n"

    prompt = f"""Berdasarkan informasi dokumentasi berikut:

{context}
{history_context}

Jawab pertanyaan ini dengan singkat dan langsung ke intinya: {question}

Jawaban:"""

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096,
                "system": "Anda adalah AXEL, asisten dokumentasi API Telkom. Jawab pertanyaan dengan SINGKAT, PADAT, langsung ke inti. Hindari salam pembuka, penutup, atau kata-kata tidak perlu. Fokus pada informasi teknis yang diminta."
            }
        )
        return response['response'].strip()
    except Exception as e:
        return "Error: Sistem sedang gangguan."

def generate_stream_response_with_ollama(context: str, question: str, conversation_history: List[Dict] = None, model: str = "llama3:8b") -> Generator[str, None, None]:
    """
    Generate streaming response as AXEL
    """
    history_context = ""
    if conversation_history:
        history_context = "\n\nRiwayat Percakapan Terkait:\n"
        for msg in conversation_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_context += f"{role}: {msg.get('content', '')}\n"

    prompt = f"""Berdasarkan informasi dokumentasi berikut:

{context}
{history_context}

Jawab pertanyaan ini dengan singkat dan langsung ke intinya: {question}

Jawaban:"""

    try:
        stream = ollama.generate(
            model=model,
            prompt=prompt,
            stream=True,  # Enable streaming
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096,
                "system": "Anda adalah AXEL, asisten dokumentasi API Telkom. Jawab pertanyaan dengan SINGKAT, PADAT, langsung ke inti. Hindari salam pembuka, penutup, atau kata-kata tidak perlu. Fokus pada informasi teknis yang diminta."
            }
        )
        
        for chunk in stream:
            if 'response' in chunk:
                yield chunk['response']
                
    except Exception as e:
        yield "Error: Sistem sedang gangguan."

def build_enhanced_query(question: str, conversation_history: List[Dict]) -> str:
    """
    Build enhanced query menggunakan conversation history (internal)
    """
    if not conversation_history:
        return question
    
    # Ambil pesan user terakhir untuk konteks
    recent_user_messages = [
        msg.get("content", "") for msg in conversation_history[-2:] 
        if msg.get("role") == "user"
    ]
    
    if len(recent_user_messages) > 1:
        # Jika ada multiple user messages, tambahkan konteks
        context = " ".join(recent_user_messages[:-1])  # Exclude current question
        return f"{question} [Konteks: {context}]"
    
    return question


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
