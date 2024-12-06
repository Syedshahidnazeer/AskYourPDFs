from sentence_transformers import SentenceTransformer
import numpy as np
import logging

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name (str): Name of the embedding model
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for given texts.
        
        Args:
            texts (list): List of text chunks
        
        Returns:
            np.ndarray: Embedding vectors
        """
        try:
            return self.model.encode(texts, show_progress_bar=False)
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return np.array([])
