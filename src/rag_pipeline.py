# src/rag_pipeline.py
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .api_service import GeminiService
import logging
import yaml
import os

class RAGPipeline:
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.logger = logging.getLogger(__name__)
        
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService(
            self.config['embedding']['model_name']
        )
        self.vector_store = VectorStore(
            embedding_dim=self.config['embedding']['dimension'],
            index_path=self.config['vectorstore']['index_path']
        )
        self.gemini_service = GeminiService()
    
    def process_documents(self, uploaded_files):
        """
        Process PDF documents from uploaded Streamlit files
        
        Args:
            uploaded_files (list): List of uploaded file objects
        """
        try:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with open(uploaded_file.name, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                pdf_texts = self.document_processor.extract_text_from_pdf(uploaded_file.name)
                
                for text in pdf_texts:
                    chunks = self.document_processor.chunk_text(text)
                    
                    chunk_texts = [chunk['text'] for chunk in chunks]
                    embeddings = self.embedding_service.generate_embeddings(chunk_texts)
                    
                    self.vector_store.add_embeddings(embeddings, chunks)
                
                # Clean up temporary file
                os.remove(uploaded_file.name)
        
        except Exception as e:
            self.logger.error(f"Document processing error: {e}")
    
    # src/rag_pipeline.py
    def retrieve_and_generate(self, query: str):
        try:
            query_embedding = self.embedding_service.generate_embeddings([query])[0]
            
            distances, indices, retrieved_metadata = self.vector_store.search(
                query_embedding, 
                top_k=self.config['vectorstore']['top_k_retrieval']
            )
            
            # Use retrieved metadata for context
            if len(retrieved_metadata) > 0:
                context = " ".join([
                    f"Chunk {meta.get('metadata', {}).get('chunk_id', 'N/A')}: {meta.get('text', '')}" 
                    for meta in retrieved_metadata
                ])
            else:
                context = "No relevant context found."
            
            response = self.gemini_service.generate_response(context, query)
            
            return response
        except Exception as e:
            self.logger.error(f"RAG pipeline error: {str(e)}")
            return "Query processing failed."