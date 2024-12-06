import os
from typing import List, Dict
from pypdf import PdfReader
import logging

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize DocumentProcessor with configurable chunk parameters.
        
        Args:
            chunk_size (int): Size of text chunks to create
            chunk_overlap (int): Overlap between text chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract text from a PDF file, handling multi-page documents.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            List[str]: Extracted text from the PDF
        """
        try:
            reader = PdfReader(pdf_path)
            full_text = []
            
            for page in reader.pages:
                full_text.append(page.extract_text())
            
            return full_text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text (str): Input text to chunk
        
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = text_splitter.split_text(text)
        
        return [
            {
                "text": chunk,
                "metadata": {
                    "source": "pdf_chunk",
                    "chunk_id": idx
                }
            } for idx, chunk in enumerate(chunks)
        ]
