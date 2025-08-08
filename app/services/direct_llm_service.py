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
                    
                    # Create a more detailed system prompt
                    system_prompt = (
                        "You are an expert at extracting information from documents. "
                        "I will provide you with a document URL and a question. "
                        "Your task is to carefully analyze the document and provide a precise answer.\n\n"
                        f"DOCUMENT URL: {document_url}\n"
                        f"QUESTION: {query}\n\n"
                        "INSTRUCTIONS:\n"
                        "1. First, access and read the document carefully.\n"
                        "2. For the given question, provide a direct and concise answer within 250 characters.\n"
                        "3. If the answer is not explicitly in the document, use your knowledge to provide the most likely answer.\n"
                        "4. For flight information, look for sections like 'Flight Details', 'Itinerary', or similar.\n"
                        "5. Format your response as plain text without any additional explanations or markdown.\n\n"
                        "ANSWER:"
                    )
                    
                    logger.debug(f"System prompt: {system_prompt}")
                    
                    # Try multiple approaches to get a good response
                    response_text = None
                    
                    # First attempt: Try with the full prompt
                    try:
                        logger.info("Attempt 1: Sending request to Gemini with full prompt...")
                        response = await model.generate_content_async(system_prompt)
                        response_text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
                        logger.info(f"Response from Gemini (Attempt 1): {response_text[:200]}...")
                        
                        # If first response indicates no answer, try a different approach
                        if not response_text or 'not found' in response_text.lower() or 'no information' in response_text.lower():
                            raise ValueError("Initial response indicated no answer found")
                            
                    except Exception as e1:
                        logger.warning(f"First attempt failed, trying alternative approach: {str(e1)}")
                        
                        # Second attempt: Try with a simpler, more direct prompt
                        try:
                            simple_prompt = (
                                f"Look at this document: {document_url}\n"
                                f"Answer this question concisely: {query}\n"
                                "If you can't find the answer, make your best guess.\n"
                                "Answer (just the answer, no extra text):"
                            )
                            logger.info("Attempt 2: Sending simplified request to Gemini...")
                            response = await model.generate_content_async(simple_prompt)
                            response_text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
                            logger.info(f"Response from Gemini (Attempt 2): {response_text[:200]}...")
                            
                        except Exception as e2:
                            logger.error(f"Second attempt also failed: {str(e2)}")
                            response_text = "I couldn't find that information in the document."
                    
                    # If we still don't have a good response, use a fallback
                    if not response_text or len(response_text) < 2:
                        response_text = "Information not available in the document."
                    
                    # Store the final result
                    results[str(i)] = response_text
                    
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
