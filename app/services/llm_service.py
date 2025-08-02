import os
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import httpx
import requests
import time
import traceback
import random
import json
import logging
import fitz  # PyMuPDF
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .query_logger import query_logger

# List of API keys
API_KEYS = [
    "AIzaSyDAa0dhdGFViUi21NQi7N_6ke2ycpiNark",
    "AIzaSyAcLhSQoDkl21TU-KSt-gQeYBnsdB6Z6Us",
    "AIzaSyDSuF6W5nFHNvkFA1WsJNsuc9XSSeSN8zk",
    "AIzaSyB1z571_km2zmFnS_knLBwe0FW-Vj--NUU",
    "AIzaSyDhfuCmWvCrFOyVMxXuyV2Cmum-BvBlUG8",
    "AIzaSyATLstXN7EbDJllawshb4Ma0MFlKHLfpXc",
    "AIzaSyCL_84MAG8kNHnw6m9Xycb1H8FweXwhkTQ",
    "AIzaSyB7RJZSsvjK2bPXPQBm683DdoXhUFNgQiI",
    "AIzaSyCSbnkVBRFTLOpNZMP7R55OTLwhLLggQfk",
    "AIzaSyBiKM-oux_CFUS3N6OINzFDTSLV43oyyYM"


    

   
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
        """
        Synchronously downloads a PDF from the given URL and extracts its text.
        
        Args:
            url: The URL of the PDF to download
            
        Returns:
            Extracted text from the PDF, or empty string if extraction fails
        """
        try:
            # Download the PDF
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract text from the PDF
            full_text = []
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                for page in doc:
                    full_text.append(page.get_text())
            
            return "\n".join(full_text)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF {url}: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    async def get_document_text(self, document_link: str) -> Tuple[bool, str]:
        """
        Asynchronously downloads and extracts text from a document.
        
        Args:
            document_link: URL of the document to process
            
        Returns:
            Tuple of (success, text_or_error_message)
        """
        if not self._is_valid_url(document_link):
            return False, "Invalid document URL"
            
        logger.info(f"Downloading and processing document from: {document_link}")
        
        try:
            # Run the synchronous download and extraction in a thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                self.executor,
                lambda: self._download_and_extract_text(document_link)
            )
            
            if not text.strip():
                return False, "No text could be extracted from the document"
                
            logger.info(f"Successfully extracted {len(text)} characters from document")
            # Log first 500 characters of the extracted text for debugging
            sample_text = text[:500].replace('\n', ' ').strip()
            logger.info(f"Extracted text sample (first 500 chars): {sample_text}...")
            return True, text
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg

    def _setup_genai(self, used_keys=None, max_retries=3):
        """Configures the Gemini API client with the current API key.
        
        Args:
            used_keys: Set of API keys that have already been tried
            max_retries: Maximum number of retry attempts with different API keys
        """
        used_keys = used_keys or set()
        
        for attempt in range(max_retries):
            try:
                if not self.api_key or self.api_key in used_keys:
                    available_keys = [k for k in API_KEYS if k not in used_keys]
                    if not available_keys:
                        raise ValueError("No available API keys left to try")
                    self.api_key = random.choice(available_keys)
                
                logger.info(f"Configuring Gemini API with key: {self.api_key[:10]}...")
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Successfully configured Gemini API with model: {self.model_name}")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Attempt {attempt + 1} failed with API key {self.api_key[:10]}...: {error_msg}")
                
                # Mark this key as used
                used_keys.add(self.api_key)
                
                # Check if we have more keys to try
                available_keys = [k for k in API_KEYS if k not in used_keys]
                if not available_keys:
                    raise ValueError("All API keys have been exhausted. Please add more keys or try again later.")
                
                # Try next key if available
                self.api_key = random.choice(available_keys)
                logger.info(f"Retrying with API key: {self.api_key[:10]}...")
        
        # If we get here, we've exceeded max retries
        raise ValueError(f"Failed to initialize Gemini API after {max_retries} attempts. Please check your API keys and try again.")

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

    async def _is_document_accessible(self, url: str) -> bool:
        """Checks if a document is accessible via an async HTTP HEAD request."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.head(url, timeout=10, follow_redirects=True)
                return response.status_code == 200
        except httpx.RequestError:
            return False

    def _count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the given text."""
        # Rough estimate: 1 token ~= 4 characters for English text
        return (len(text) + 3) // 4

    def _prepare_prompt(self, queries: List[str], document_text: str) -> Tuple[str, int]:
        """Prepares the structured prompt for the LLM with extracted document text.
        
        Returns:
            Tuple of (prompt_text, total_tokens)
        """
        # Count tokens for document text
        doc_tokens = self._count_tokens(document_text)
        
        # Prepare base prompt parts
        base_prompt = (
            "You are a helpful AI assistant that answers questions based on the provided document context.\n"
            "Answer the following questions based on the document content below.\n"
            "If the answer cannot be found in the document, respond with 'The document does not provide specific details on this matter.'\n\n"
            "QUESTIONS TO ANSWER:\n"
            "DOCUMENT CONTEXT:\n"
            "----------------------------------------\n"
        )
        base_tokens = self._count_tokens(base_prompt)
        
        # Calculate total tokens for the prompt
        queries_text = "\n".join(queries)
        queries_tokens = self._count_tokens(queries_text)
        
        # Build the full prompt
        prompt_parts = [
            base_prompt,
            document_text,
            "\n\nQUESTIONS:\n",
            queries_text
        ]
        
        total_tokens = base_tokens + doc_tokens + queries_tokens + 20  # Add buffer for formatting
        
        # Log token counts
        logger.info(f"Token counts - Document: {doc_tokens}, Queries: {queries_tokens}, Total: {total_tokens}")
        
        for i in range(1, len(queries) + 1):
            prompt_parts.append(f"Answer {i}: [Your answer to question {i}]\n")
            
        prompt_text = "".join(prompt_parts)
        return prompt_text, total_tokens

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
        """Process a batch of queries with a specific API key."""
        try:
            # Create a new instance with the specified API key
            service = LLMService()
            service.api_key = api_key
            service._setup_genai()
            
            # Prepare the prompt with all queries and document text
            prompt, token_count = self._prepare_prompt(queries, document_text)
            logger.info(f"Processing batch {metadata.get('batch_num', '?')}/{metadata.get('total_batches', '?')} with {len(queries)} queries, {token_count} tokens using API key ...{api_key[-4:]}")
            
            # Call the LLM with safety settings
            response = await service.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Get the response text
            response_text = response.text
            logger.debug(f"LLM response for batch {metadata.get('batch_num')}: {response_text[:200]}...")
            
            # Parse the response into individual answers
            return self._parse_llm_response(response_text, queries)
            
        except Exception as e:
            logger.error(f"Error in _process_batch_with_key: {str(e)}")
            logger.error(traceback.format_exc())
            # Return error responses for all queries in this batch
            return {f"Query {i+1}": f"Error processing request: {str(e)}" 
                   for i in range(len(queries))}

    async def _process_batch(self, queries: List[str], document_text: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Process a single batch of queries (up to MAX_QUERIES_PER_BATCH)."""
        # Use the current API key for this batch
        return await self._process_batch_with_key(queries, document_text, metadata, self.api_key)

    async def generate_response(self, queries: List[str], document_link: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Validates input, checks document accessibility, extracts PDF text, and processes queries in parallel batches.
        
        Args:
            queries: List of questions to ask about the document
            document_link: URL of the PDF document to process
            metadata: Additional metadata to store with the query log
            
        Returns:
            Dictionary mapping query numbers to their answers
        """
        metadata = metadata or {}
        
        try:
            if not queries:
                raise ValueError("At least one query is required")
            if not document_link or not self._is_valid_url(document_link):
                raise ValueError("A valid document link is required")

            # Check if document is accessible
            is_accessible = await self._is_document_accessible(document_link)
            if not is_accessible:
                raise ValueError(f"Document at {document_link} is not accessible or not found")
            
            # Download and extract text from PDF
            success, document_text = await self.get_document_text(document_link)
            if not success:
                raise ValueError(f"Failed to process document: {document_text}")
            
            # Prepare metadata for logging
            query_metadata = {
                "document_link": document_link,
                "timestamp": datetime.utcnow().isoformat(),
                "num_queries": len(queries),
                "document_accessible": is_accessible,
                "processing_mode": "batched" if len(queries) > MAX_QUERIES_PER_BATCH else "single",
                **metadata
            }
            
            # Log the query
            log_id = query_logger.log_query(queries, document_link, query_metadata)
            query_metadata["log_id"] = log_id
            
            # Split queries into batches if needed
            query_batches = [queries[i:i + MAX_QUERIES_PER_BATCH] 
                          for i in range(0, len(queries), MAX_QUERIES_PER_BATCH)]
            
            # Create a list to store batch tasks with their assigned API keys
            batch_tasks = []
            used_keys = set()
            
            for batch_num, batch in enumerate(query_batches, 1):
                # Get an available API key that hasn't been used in this request
                available_keys = [k for k in API_KEYS if k not in used_keys]
                if not available_keys:
                    # If we've used all keys, clear the set and start reusing them
                    used_keys.clear()
                    available_keys = API_KEYS.copy()
                
                batch_key = random.choice(available_keys)
                used_keys.add(batch_key)
                
                # Create metadata for this batch
                batch_metadata = {
                    **query_metadata,
                    "batch_num": batch_num,
                    "total_batches": len(query_batches),
                    "queries_in_batch": len(batch),
                    "api_key_used": f"...{batch_key[-4:]}"
                }
                
                # Create a task for this batch with its own LLMService instance
                task = self._process_batch_with_key(
                    batch, 
                    document_text,  # Pass the extracted text
                    batch_metadata, 
                    batch_key
                )
                batch_tasks.append(task)
            
            # Process all batches in parallel
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine all responses
            combined_responses = {}
            for response in batch_responses:
                if isinstance(response, Exception):
                    logger.error(f"Error in batch processing: {str(response)}")
                    continue
                combined_responses.update(response)
            
            # Add log ID to the combined responses
            if log_id:
                combined_responses["log_id"] = log_id
            
            # Log the successful processing
            try:
                start_time = time.time()
                query_logger.log_query(
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