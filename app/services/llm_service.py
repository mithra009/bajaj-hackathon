import os
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import httpx
import requests
import time
import traceback
import json
import random
import logging
import fitz  # PyMuPDF
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .query_logger import query_logger

# List of API keys
API_KEYS = [
    "AIzaSyD1wpr6HXQzG67TopO5xIThzyQ1rxt85us",
    "AIzaSyAw1xER-y-EpXVgg2DCQr_GLNBS1dlgDGo",
    "AIzaSyDRafUeLPLv7wxqVrxZeetl5hGJoz39ax0",
    "AIzaSyD2S-t1eQw-eLV-dplK7UR8i40k5oKRVGs",
    "AIzaSyB-9VDWC3-6QGI3wQAie22f2OyIo06zTcg",
    "AIzaSyDVuQGygWyeo2J40anesm3aWLQK5vmjGeM",
    "AIzaSyCp1waEadzMh4p1HKmHr7GinZqzgJgFMDM",
    "AIzaSyDzBLii6fraXMxwFsu9teJ8qPwpPZP33dE",
    "AIzaSyCMcQUU-GrklfWQe9qs2pV3sh6dGNIOpE8"
]

# Import configuration
from app.config import MODEL_NAME, MAX_TOKENS, MAX_QUERIES_PER_BATCH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initializes the LLMService with a random API key from the list."""
        if not API_KEYS:
            raise ValueError("No API keys provided in the API_KEYS list")
            
        if not API_KEYS:
            raise ValueError("No API keys available")
            
        self.api_key = random.choice(API_KEYS)
        self.model_name = MODEL_NAME
        self.max_tokens = MAX_TOKENS
        self.executor = ThreadPoolExecutor(max_workers=5)  # For parallel processing
        logger.info(f"Initializing LLMService with model: {self.model_name}")
        self._setup_genai()
        
    def _download_and_extract_text(self, url: str) -> str:
        """Download PDF and extract text synchronously in a thread."""
        try:
            # Use requests to download the PDF
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract text using fitz
            full_text = []
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                for page in doc:
                    full_text.append(page.get_text())
            
            return "\n".join(full_text)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    async def get_document_text(self, document_link: str) -> Tuple[bool, str]:
        """Download document and extract text.
        
        Args:
            document_link: URL of the document to process
            
        Returns:
            Tuple of (success: bool, text_or_error: str)
        """
        if not self._is_valid_url(document_link):
            return False, "Invalid document URL"
            
        logger.info(f"Downloading and processing document from: {document_link}")
        
        try:
            # Run the synchronous download and extraction in a thread
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.executor,
                lambda: self._download_and_extract_text(document_link)
            )
            
            if not text.strip():
                return False, "No text could be extracted from the document"
                
            logger.info(f"Successfully extracted {len(text)} characters from document")
            return True, text
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    


    def _setup_genai(self, used_keys=None):
        """Configures the Gemini API client with the current API key.
        
        Args:
            used_keys: Set of API keys that have already been tried
        """
        used_keys = used_keys or set()
        
        if not self.api_key:
            available_keys = [k for k in API_KEYS if k not in used_keys]
            if not available_keys:
                raise ValueError("No available API keys left to try")
            self.api_key = random.choice(available_keys)
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            return True
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error with API key {self.api_key[:10]}...: {error_msg}")
            
            # Mark this key as used
            used_keys.add(self.api_key)
            
            # Try a different API key if available
            available_keys = [k for k in API_KEYS if k not in used_keys]
            if available_keys:
                self.api_key = random.choice(available_keys)
                logger.info(f"Retrying with API key: {self.api_key[:10]}...")
                return self._setup_genai(used_keys)
                
            # If we get here, no more keys to try
            raise ValueError("All API keys have been exhausted. Please add more keys or try again later.")

    async def _call_llm(self, prompt: str, retry_count: int = 0, used_keys: set = None) -> str:
        """Calls the LLM with the given prompt and returns the response.
        
        Args:
            prompt: The prompt to send to the LLM
            retry_count: Number of retry attempts so far
            used_keys: Set of API keys that have already been tried
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        used_keys = used_keys or set()
        max_retries = len(API_KEYS) * 2  # Allow trying each key twice
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error calling LLM: {error_msg}")
            
            # Check if this is a rate limit or quota error
            is_rate_limit = any(keyword in error_msg for keyword in [
                "rate limit", "quota", "limit exceeded", 
                "quota exceeded", "429", "resource exhausted"
            ])
            
            if is_rate_limit and retry_count < max_retries:
                logger.info(f"Rate limit hit. Retrying with a different API key (attempt {retry_count + 1}/{max_retries})...")
                # Mark current key as used
                used_keys.add(self.api_key)
                # Get a new key
                available_keys = [k for k in API_KEYS if k not in used_keys]
                if available_keys:
                    self.api_key = random.choice(available_keys)
                    logger.info(f"Switching to API key: {self.api_key[:10]}...")
                    # Re-initialize with new key
                    self._setup_genai(used_keys)
                    # Retry the request
                    return await self._call_llm(prompt, retry_count + 1, used_keys)
            
            # If we get here, either it's not a rate limit error or we've exhausted retries
            raise

    def _is_valid_url(self, url: str) -> bool:
        """Checks if a URL string is well-formed."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _is_document_accessible(self, url: str) -> bool:
        """Checks if a document is accessible via an HTTP HEAD request."""
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _prepare_prompt(self, queries: List[str], document_text: str) -> str:
        """Prepares the structured prompt for the LLM with extracted document text."""
        doc_preview = document_text
            
        prompt_parts = [
            "DOCUMENT CONTEXT (partial):\n",
            "----------------------------------------\n",
            doc_preview,
            "\n----------------------------------------\n\n",
            "INSTRUCTIONS:\n",
            "1. Answer all questions based on the document content above.\n",
            "2. If the answer is in the document, provide a clear, concise response (under 1000 characters).\n",
            "3. If the answer is not in the document, provide a brief and general answer.\n",
            "4. Do not mention that the document does not contain relevant content.\n\n",
            "QUESTIONS TO ANSWER:\n"
        ]
        
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}\n")
        
        prompt_parts.append("\n===== YOUR RESPONSES =====\n")
        prompt_parts.append("Format your responses like this for each question:\n")
        
        for i in range(1, len(queries) + 1):
            prompt_parts.append(f"Answer {i}: [Your answer to question {i}]\n")
        
        return "".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, queries: List[str]) -> Dict[str, str]:
        """Parses the raw text response from the LLM into a structured dictionary."""
        responses = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for i, query in enumerate(queries, 1):
            query_num = i
            answer_prefix = f"Answer {query_num}:"
            found_answer = "I couldn't find a specific answer to this question in the document."
            
            for line in lines:
                if line.startswith(answer_prefix):
                    found_answer = line[len(answer_prefix):].strip()
                    # Clean up quotes if they wrap the entire answer
                    if (found_answer.startswith('"') and found_answer.endswith('"')) or \
                       (found_answer.startswith("'") and found_answer.endswith("'")):
                        found_answer = found_answer[1:-1]
                    break
            responses[f"Query {query_num}"] = found_answer
            
        return responses

    async def _process_batch_with_key(self, queries: List[str], document_text: str, metadata: Dict[str, Any], api_key: str) -> Dict[str, str]:
        """Process a batch of queries with a specific API key using extracted document text."""
        try:
            # Create a new instance with the specified API key
            temp_service = LLMService()
            temp_service.api_key = api_key
            temp_service._setup_genai()
            
            # Process all queries in this batch in one go with the extracted text
            prompt = temp_service._prepare_prompt(queries, document_text)
            
            logger.info(f"Processing batch {metadata.get('batch_num', '?')}/{metadata.get('total_batches', '?')} "
                        f"with {len(queries)} queries using API key ...{api_key[-4:]}")
            
            response = await temp_service.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": 0.1
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            response_text = response.text
            return self._parse_llm_response(response_text, queries)
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Return default responses for this batch
            return {f"Query {i+1}": "I couldn't find a specific answer to this question in the document." 
                    for i in range(len(queries))}
    
    async def _process_batch(self, queries: List[str], document_text: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Process a single batch of queries (up to MAX_QUERIES_PER_BATCH) with extracted text."""
        # Use the current API key for this batch
        return await self._process_batch_with_key(queries, document_text, metadata, self.api_key)

    async def generate_response(self, queries: List[str], document_link: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Validates input, downloads and extracts text from PDF, and processes queries in parallel batches.
        
        Args:
            queries: List of questions to ask about the document
            document_link: URL of the PDF document to query
            metadata: Additional metadata to store with the query log
            
        Returns:
            Dictionary mapping query numbers to their responses
        """
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            if not queries:
                raise ValueError("At least one query is required")
            
            # --- START FIX ---
            # Run the synchronous accessibility check in a thread to avoid blocking.
            loop = asyncio.get_event_loop()
            is_accessible = await loop.run_in_executor(
                self.executor,
                lambda: self._is_document_accessible(document_link)
            )
            # --- END FIX ---
                
            # Download and extract text from PDF
            success, document_text = await self.get_document_text(document_link)
            if not success:
                raise ValueError(f"Failed to process document: {document_text}")
                
            logger.info(f"Successfully extracted text from PDF (length: {len(document_text)} chars)")
            
            # Prepare metadata for logging
            query_metadata = {
                "model": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "num_queries": len(queries),
                "document_accessible": is_accessible, # The variable is now defined
                "processing_mode": "batched" if len(queries) > MAX_QUERIES_PER_BATCH else "single",
                **metadata
            }
            
            logger.info(f"\n=== PROCESSING {len(queries)} QUERIES ===")
            logger.info(f"Document: {document_link}")
            
            # Split queries into batches if needed
            query_batches = [queries[i:i + MAX_QUERIES_PER_BATCH] 
                             for i in range(0, len(queries), MAX_QUERIES_PER_BATCH)]
            
            # Create a list to store batch tasks with their assigned API keys
            batch_tasks = []
            used_keys = set()
            
            # Assign a unique API key to each batch
            for batch_num, batch in enumerate(query_batches, 1):
                # Get an API key that hasn't been used yet, or cycle through if we've used them all
                available_keys = [k for k in API_KEYS if k not in used_keys]
                if not available_keys:
                    # If we've used all keys, clear the set and start over
                    used_keys.clear()
                    available_keys = API_KEYS.copy()
                
                batch_key = random.choice(available_keys)
                used_keys.add(batch_key)
                
                batch_metadata = {
                    **query_metadata,
                    "batch_num": batch_num,
                    "total_batches": len(query_batches),
                    "api_key_used": f"...{batch_key[-4:]}"  # Log last 4 chars for tracking
                }
                
                # Create a task for this batch with its own LLMService instance
                task = self._process_batch_with_key(
                    batch, 
                    document_text,  # Pass extracted text instead of link
                    batch_metadata, 
                    batch_key
                )
                batch_tasks.append(task)
            
            # Process all batches in parallel
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine all responses
            combined_responses = {}
            query_counter = 1
            
            for batch_idx, batch_response in enumerate(batch_responses):
                batch = query_batches[batch_idx] if batch_idx < len(query_batches) else []
                
                if isinstance(batch_response, dict):
                    # Add successful batch responses
                    for i in range(len(batch)):
                        q_num = f"Query {i+1}"  # The original query number within the batch
                        if q_num in batch_response:
                            combined_responses[f"Query {query_counter}"] = batch_response[q_num]
                        else:
                            combined_responses[f"Query {query_counter}"] = \
                                "I couldn't find a specific answer to this question in the document."
                        query_counter += 1
                else:
                    # Handle failed batch with default responses
                    for _ in range(len(batch)):
                        if query_counter <= len(queries):
                            combined_responses[f"Query {query_counter}"] = \
                                "I couldn't find a specific answer to this question in the document."
                            query_counter += 1
            
            # Log the successful processing
            try:
                log_id = query_logger.log_query(
                    document_link=document_link,
                    queries=queries,
                    responses=combined_responses,
                    metadata={
                        **query_metadata,
                        "processing_time_seconds": time.time() - start_time,
                        "num_batches": len(query_batches),
                        "response_character_count": sum(len(str(r)) for r in combined_responses.values())
                    }
                )
                logger.info(f"Successfully processed all queries. Log ID: {log_id}")
            except Exception as e:
                logger.error(f"Failed to log query: {str(e)}")
                logger.error(traceback.format_exc())
            
            return combined_responses

        except Exception as e:
            logger.error(f"\n=== ERROR OCCURRED IN LLM_SERVICE ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("=== STACK TRACE ===")
            logger.error(traceback.format_exc())
            
            # Return default responses for all queries
            return {f"Query {i+1}": f"Error processing request: {str(e)}" 
                   for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()