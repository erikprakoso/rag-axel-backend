from abc import ABC, abstractmethod
from typing import List
import os

class DocumentProcessor(ABC):
    @abstractmethod
    def process(self, file_path: str) -> List[str]:
        pass

class TextProcessor(DocumentProcessor):
    def process(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Implement better chunking
        return self._split_text(text, chunk_size, chunk_overlap)
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        # Implement recursive text splitter atau gunakan library
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            chunks.append(text[start:end])
            start = end - chunk_overlap
        return chunks

class PDFProcessor(DocumentProcessor):
    def process(self, file_path: str, **kwargs) -> List[str]:
        # Implement PDF processing dengan PyPDF2 atau pdfplumber
        try:
            import PyPDF2
            chunks = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        chunks.append(text)
            return chunks
        except ImportError:
            raise ImportError("PyPDF2 required for PDF processing")

class DocumentProcessorFactory:
    @staticmethod
    def get_processor(file_extension: str) -> DocumentProcessor:
        processors = {
            '.txt': TextProcessor(),
            '.pdf': PDFProcessor(),
            # Tambahkan processor lain
        }
        
        if file_extension not in processors:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return processors[file_extension]