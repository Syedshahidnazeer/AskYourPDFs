import faiss
import numpy as np
import os
import logging
from typing import List, Dict

class VectorStore:
    def __init__(self, embedding_dim: int, index_path: str = "./vectorstore"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(index_path, exist_ok=True)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []  # Store associated metadata
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        try:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.metadata.extend(metadata)
        except Exception as e:
            self.logger.error(f"Error adding embeddings: {e}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        try:
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            
            # Retrieve associated metadata
            retrieved_metadata = [
                self.metadata[idx] for idx in indices[0] 
                if 0 <= idx < len(self.metadata)
            ]
            
            return distances, indices, retrieved_metadata
        except Exception as e:
            self.logger.error(f"Error searching embeddings: {e}")
            return np.array([]), np.array([]), []
