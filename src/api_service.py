import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

class GeminiService:
    def __init__(self):
        """
        Initialize Gemini API service with authentication.
        """
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("Gemini API Key not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, context: str, query: str, max_tokens: int = 1024):
        """
        Generate response using Gemini API with context.
        
        Args:
            context (str): Retrieved context for augmentation
            query (str): User's original query
            max_tokens (int): Maximum tokens for response
        
        Returns:
            str: Generated response
        """
        try:
            prompt = f"Context: {context}\n\nQuery: {query}\n\nResponse:"
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."
