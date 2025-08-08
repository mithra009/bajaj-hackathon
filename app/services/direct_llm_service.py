"""
Direct LLM Service - A simplified version of LLM service that processes documents and queries with minimal preprocessing.
"""
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded Gemini API keys
GEMINI_API_KEYS = [
    "AIzaSyC0KEbkvN6zBcR-RguvpZFSppWViQK1Id4",
    "AIzaSyCy81UdmFJaNRY0Y8YPKMSJT3zpideLzG8",
    "AIzaSyC0kZIHetPNcRkA9MY0nncqiqdtBi7TzAM",
    "AIzaSyBAdlPvCwXXDZyvQJ6mXVhxyrz20vJMul8",
    "AIzaSyA3wADP1tAbXwFJ6lB9hj4SM1piMast9hI",
    "AIzaSyDF6BuUFYc3jSEKLKv2Nsr3v8MISJ6j0V8",
    "AIzaSyAHG_guIGql9JG5NaBiRQpHmEQ9O09Dfoo",
    "AIzaSyBXVE_Zo_XsjvilpzVjugIe3wg9ZWe62vM",
    "AIzaSyBCPf3_VWZBQ4tJPGE8fSM9MBXV70ccPLw",
    "AIzaSyDR9Xw3WtwqlN2uB8SNMog9wpVfXtr7L9I"
]

class DirectLLMService:
    def __init__(self):
        """Initialize the DirectLLMService with Gemini API keys."""
        self.gemini_api_keys = GEMINI_API_KEYS
        self.current_key_index = 0
        self.model_name = "gemini-2.0-flash"
        
    def _get_next_api_key(self) -> str:
        """Get the next API key in a round-robin fashion."""
        key = self.gemini_api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_api_keys)
        return key
    
    async def process_queries(self, document_url: str, queries: List[str]) -> Dict[str, str]:
        """
        Process queries by directly passing the URL to Gemini.
        
        Args:
            document_url: URL of the document to process
            queries: List of queries to process
            
        Returns:
            Dictionary mapping query numbers to answers
        """
        try:
            logger.info(f"Processing document URL: {document_url}")
            logger.info(f"Number of queries: {len(queries)}")
            
            # Get API key and configure Gemini
            api_key = self._get_next_api_key()
            logger.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
            genai.configure(api_key=api_key)
            
            # Initialize the model
            model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized model: {self.model_name}")
            
            # Process each query
            results = {}
            for i, query in enumerate(queries, 1):
                try:
                    logger.info(f"Processing query {i}: {query}")
                    
                    # Create the system prompt with the URL
                    system_prompt = (
                        "You are a helpful assistant that answers questions based on the provided document. "
                        f"The document can be accessed at this URL: {document_url}\n"
                        "Please analyze the content at this URL and answer the following question. "
                        "If the answer cannot be found in the document, say 'Not found in document'.\n\n"
                        f"Question: {query}\nAnswer:"
                    )
                    
                    logger.debug(f"System prompt: {system_prompt}")
                    
                    # Generate response
                    logger.info("Sending request to Gemini...")
                    response = await model.generate_content_async(system_prompt)
                    
                    # Log the raw response
                    logger.info(f"Received response from Gemini: {response}")
                    
                    # Extract text from response
                    response_text = response.text if hasattr(response, 'text') else str(response)
                    logger.info(f"Extracted response text: {response_text[:200]}..." if len(response_text) > 200 else f"Extracted response text: {response_text}")
                    
                    # Store the result
                    results[str(i)] = response_text.strip()
                    
                except Exception as e:
                    error_msg = f"Error processing query {i}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    results[str(i)] = error_msg[:200]
                    
                logger.info(f"Results so far: {results}")
            
            # Convert results to the expected format
            answers = [results.get(str(i+1), "No response generated") for i in range(len(queries))]
            logger.info(f"Returning answers: {answers}")
            return answers
            
        except Exception as e:
            logger.error(f"Error in process_queries: {e}", exc_info=True)
            return [f"Error: {str(e)[:200]}" for _ in queries]

# Create a singleton instance for the application
direct_llm_service = DirectLLMService()

# Example usage:
# response = await direct_llm_service.process_queries(
#     document_url="https://example.com/document.pdf",
#     queries=["What is this document about?", "What are the key points?"]
# )
