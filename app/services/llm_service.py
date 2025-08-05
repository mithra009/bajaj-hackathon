import os
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple, Union, Type
from urllib.parse import urlparse
import httpx
import requests
import time
import traceback
import random
import json
import logging
import fitz  # PyMuPDF
import re 
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from sklearn.metrics.pairwise import cosine_similarity
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

# Define custom exceptions for better error handling
class RateLimitError(Exception):
    """Raised when the API rate limit is exceeded"""
    pass

class APIError(Exception):
    """Base class for other API-related exceptions"""
    pass

class APITimeoutError(APIError):
    """Raised when the API request times out"""
    pass

class APIConnectionError(APIError):
    """Raised when there's a connection error with the API"""
    pass
from .query_logger import query_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to store the last used key index
KEY_INDEX_FILE = Path("/app/data/api_key_index.json")

# Hardcoded Gemini API keys
GEMINI_KEYS = [
    "AIzaSyCtV8k4-J_4lmwE4aL4dRQsSt63Iq-gAmo",
    "AIzaSyBdUY5Vn0ZbbgqLlURsaCJNzpV1CiFwKeE",
    "AIzaSyBNsBJr0Nnh0jWS37ZiX9g7gvyls5SFLpM",
    "AIzaSyAIEMcIJrils1DXLweKai6T6Dz2agzQF-0",
    "AIzaSyAWnC5B0F3MvmTE5rUQY3Uh27BXh2KM4iU",
    "AIzaSyDYiPfw55zlRZgWJKrYWdYzqyt1wueB-kE"
]

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Configure OpenAI client
openai.api_key = OPENAI_API_KEY

def get_next_key_index():
    """Get the next key index to use, with persistence."""
    try:
        if KEY_INDEX_FILE.exists():
            with open(KEY_INDEX_FILE, 'r') as f:
                data = json.load(f)
                last_index = data.get('last_index', -1)
        else:
            last_index = -1
            
        next_index = (last_index + 1) % len(GEMINI_KEYS)
        
        # Save the next index for future use
        with open(KEY_INDEX_FILE, 'w') as f:
            json.dump({'last_index': next_index}, f)
            
        return next_index
    except Exception as e:
        logger.error(f"Error managing key index: {str(e)}")
        return 0  # Fallback to first key

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
        """Initializes the LLMService with API key rotation."""
        self.gemini_api_keys = [
            "AIzaSyCtV8k4-J_4lmwE4aL4dRQsSt63Iq-gAmo",
            "AIzaSyBdUY5Vn0ZbbgqLlURsaCJNzpV1CiFwKeE",
            "AIzaSyBNsBJr0Nnh0jWS37ZiX9g7gvyls5SFLpM",
            "AIzaSyAIEMcIJrils1DXLweKai6T6Dz2agzQF-0",
            "AIzaSyAWnC5B0F3MvmTE5rUQY3Uh27BXh2KM4iU",
            "AIzaSyDYiPfw55zlRZgWJKrYWdYzqyt1wueB-kE"
        ]
        self.current_key_index = 0
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        self.model_name = "gemini-2.5-flash-lite"
        self.max_tokens = 8196
        self.executor = ThreadPoolExecutor(max_workers=5)  # For parallel processing
        logger.info("Initializing LLMService with Gemini Flash Lite model")
        self._setup_genai()
        
    def _get_next_api_key(self):
        """Get the next Gemini API key using persistent rotation."""
        key_index = get_next_key_index()
        return GEMINI_KEYS[key_index]
    
    def _setup_genai(self):
        """Setup the GenAI client with the next available API key."""
        self.api_key = self._get_next_api_key()
        logger.info(f"Using API key (last 5 chars): {self.api_key[-5:]}")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
    def _prepare_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Prepares the structured prompt for the LLM with relevant context chunks.
        
        Args:
            query: The user's query
            context_chunks: List of relevant text chunks from the document
            
        Returns:
            Formatted prompt string
        """
        context = "\n\n---\n".join(context_chunks)
        prompt = f"""You are a helpful insurance assistant. Use the following policy information to answer the question.

--- Policy Document Extract ---
{context}
--- End Extract ---

Now answer the following question *clearly and concisely*. If necessary, you can add some general industry-standard medical insurance knowledge — but prioritize using the extract:

*Question:* {query}
"""
        return prompt

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the given text."""
        # Rough estimate: 1 token ~= 4 characters for English text
        return (len(text) + 3) // 4

    def _download_pdf(self, url: str) -> bytes:
        """
        Download a PDF from the given URL.
        
        Args:
            url: URL of the PDF to download
            
        Returns:
            PDF content as bytes
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            raise

    def extract_chunks(self, pdf_bytes: bytes, max_chars: int = 1000) -> List[str]:
        """
        Extract text chunks from PDF bytes.
        
        Args:
            pdf_bytes: PDF content as bytes
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            chunks = []
            for page in doc:
                text = page.get_text()
                text = text.strip().replace('\n', ' ')
                for i in range(0, len(text), max_chars):
                    chunk = text[i:i + max_chars]
                    if len(chunk.split()) > 5:  # Skip very small chunks
                        chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError, APIConnectionError)),
        reraise=True
    )
    async def _get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text chunks to embed
            model: OpenAI embedding model to use
            
        Returns:
            List of embeddings
        
        Raises:
            RateLimitError: If the API rate limit is exceeded
            APIError: For other API-related errors
            APITimeoutError: If the request times out
            APIConnectionError: If there's a connection error
        """
        try:
            # Use the async client for better performance
            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}")
            raise RateLimitError(str(e))
        except openai.APITimeoutError as e:
            logger.warning(f"API request timed out: {str(e)}")
            raise APITimeoutError(str(e))
        except openai.APIConnectionError as e:
            logger.warning(f"API connection error: {str(e)}")
            raise APIConnectionError(str(e))
        except openai.APIError as e:
            logger.warning(f"API error: {str(e)}")
            raise APIError(str(e))
        except Exception as e:
            logger.error(f"Unexpected error getting embeddings: {str(e)}")
            raise APIError(f"Failed to get embeddings: {str(e)}")

    def _find_top_chunks(self, query: str, query_emb: List[float], chunk_embs: List[List[float]], 
                        chunks: List[str], top_k: int = 5) -> List[str]:
        """
        Find the top-k most relevant chunks for a query.
        
        Args:
            query: The query text
            query_emb: Embedding of the query
            chunk_embs: List of chunk embeddings
            chunks: List of text chunks
            top_k: Number of top chunks to return (default: 5)
            
        Returns:
            List of top-k most relevant chunks
        """
        try:
            sims = cosine_similarity([query_emb], chunk_embs)[0]
            top_idxs = sims.argsort()[-top_k:][::-1]
            return [chunks[i] for i in top_idxs]
        except Exception as e:
            logger.error(f"Error finding top chunks: {str(e)}")
            return chunks[:top_k]  # Return first k chunks in case of error

    async def _download_and_extract_text(self, url: str):
        """
        Asynchronously downloads a PDF from the given URL, extracts text chunks and generates embeddings.
        
        Args:
            url: The URL of the PDF to download
            
        Returns:
            Tuple of (chunks, embeddings) or (None, None) if extraction fails
        """
        try:
            # Download the PDF
            pdf_bytes = await asyncio.to_thread(self._download_pdf, url)
            if not pdf_bytes:
                logger.error(f"Failed to download PDF from {url}")
                return None, None
                
            # Extract text chunks
            chunks = await asyncio.to_thread(self.extract_chunks, pdf_bytes)
            if not chunks:
                logger.error(f"No text could be extracted from PDF at {url}")
                return None, None
                
            # Generate embeddings for each chunk
            embeddings = await self._get_embeddings(chunks)
            
            if not embeddings or len(embeddings) != len(chunks):
                logger.error(f"Failed to generate embeddings for chunks from {url}")
                return None, None
                
            return chunks, embeddings
            
        except Exception as e:
            logger.error(f"Error in _download_and_extract_text: {str(e)}", exc_info=True)
            return None, None
    
    async def _process_chunk_with_semaphore(self, chunk: str, semaphore: asyncio.Semaphore) -> Optional[List[float]]:
        """Process a chunk with a semaphore to limit concurrency."""
        async with semaphore:
            try:
                # Get embedding for the chunk
                embeddings = await self._get_embeddings([chunk])
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                return [0.0] * 1536  # Return zero vector if no embeddings
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                return [0.0] * 1536  # Return zero vector on error

    async def process_document(self, document_link: str):
        """
        Asynchronously processes a document: downloads, extracts chunks, and generates embeddings.
        Uses asyncio for concurrent processing of chunks.
        
        Args:
            document_link: URL of the document to process
            
        Returns:
            Tuple of (success, result_dict_or_error_message)
            On success, result_dict contains:
            - 'chunks': List of text chunks
            - 'embeddings': List of chunk embeddings
        """
        if not self._is_valid_url(document_link):
            logger.error(f"Invalid document URL: {document_link}")
            return False, f"Invalid document URL: {document_link}"
            
        try:
            # Download PDF and extract text chunks
            chunks, embeddings = await self._download_and_extract_text(document_link)
            
            if not chunks or not embeddings:
                error_msg = f"Failed to process document: {document_link}"
                logger.error(error_msg)
                return False, error_msg
                
            return True, {
                'chunks': chunks,
                'embeddings': embeddings
            }
            
        except Exception as e:
            error_msg = f"Error processing document {document_link}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    async def _process_single_query(
        self, 
        query: str, 
        query_idx: int, 
        chunks: List[str], 
        chunk_embeddings: List[List[float]],
        semaphore: asyncio.Semaphore
    ) -> Tuple[int, str]:
        """
        Process a single query asynchronously with its own API key.
        
        Args:
            query: The query to process
            query_idx: Index of the query
            chunks: List of document chunks
            chunk_embeddings: Embeddings for the document chunks
            semaphore: Semaphore to limit concurrency
            
        Returns:
            Tuple of (query_index, response)
        """
        async with semaphore:
            try:
                # Get query embedding
                query_emb = (await self._get_embeddings([query]))[0]
                
                # Find most relevant chunks
                top_chunks = self._find_top_chunks(query, query_emb, chunk_embeddings, chunks)
                
                # Prepare prompt with context
                prompt = self._prepare_prompt(query, top_chunks)
                
                # Call the LLM
                response = await self._call_llm(prompt)
                
                return query_idx, response
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                return query_idx, f"Error processing query: {str(e)}"

    async def _call_llm(self, prompt: str, retry_count: int = 0, used_keys: set = None) -> str:
        """Calls the Gemini Flash Lite model with the given prompt and returns the response.
        Uses round-robin API key rotation and handles rate limiting.
        
        Args:
            prompt: The prompt to send to the LLM
            retry_count: Number of retry attempts so far
            used_keys: Set of API keys that have already been tried
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        if used_keys is None:
            used_keys = set()
            
        if len(used_keys) >= len(self.gemini_api_keys):
            raise Exception("All API keys have been exhausted")
            
        # Get next available API key
        current_key = None
        for key in self.gemini_api_keys:
            if key not in used_keys:
                current_key = key
                used_keys.add(key)
                break
                
        if not current_key:
            raise Exception("No available API keys")
            
        try:
            # Configure Gemini with the current API key
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(self.model_name)
            
            # Generate content with error handling
            response = await model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": self.max_tokens,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Extract and return the response text
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'parts'):
                return ' '.join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            else:
                raise ValueError("Unexpected response format from Gemini API")
                
        except Exception as e:
            logger.warning(f"API call failed with key ending in {current_key[-4:]}: {str(e)}")
            # If rate limited or other API error, try with next key
            if retry_count < len(self.gemini_api_keys) - 1:
                return await self._call_llm(prompt, retry_count + 1, used_keys)
            raise Exception(f"All API keys failed. Last error: {str(e)}")

    async def _process_single_query(
        self, 
        query: str, 
        query_idx: int, 
        chunks: List[str], 
        chunk_embeddings: List[List[float]],
        semaphore: asyncio.Semaphore
    ) -> Tuple[int, str]:
        """
        Process a single query asynchronously with its own API key.
        
        Args:
            query: The query to process
            query_idx: Index of the query
            chunks: List of document chunks
            chunk_embeddings: Embeddings for the document chunks
            semaphore: Semaphore to limit concurrency
            
        Returns:
            Tuple of (query_index, response)
        """
        async with semaphore:
            try:
                # Get query embedding
                query_emb = (await self._get_embeddings([query]))[0]
                
                # Find most relevant chunks
                top_chunks = self._find_top_chunks(query, query_emb, chunk_embeddings, chunks)
                
                # Prepare and send prompt to LLM
                prompt = self._prepare_prompt(query, top_chunks)
                response = await self._call_llm(prompt)
                
                return query_idx, response
                
            except Exception as e:
                logger.error(f"Error processing query {query_idx + 1}: {str(e)}", exc_info=True)
                return query_idx, f"Error processing query: {str(e)}"

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """
        Process multiple queries against a document in parallel with minimal latency.
        
        Args:
            queries: List of queries to process
            document_link: URL of the document to process
            
        Returns:
            Dictionary mapping query numbers to their answers
        """
        if not queries:
            return {}
            
        # Process the document first
        success, result = await self.process_document(document_link)
        if not success:
            return {str(i+1): f"Error: {result}" for i in range(len(queries))}
            
        chunks = result['chunks']
        chunk_embeddings = result['embeddings']
        
        # Create a semaphore to limit concurrency to the number of available API keys
        semaphore = asyncio.Semaphore(len(GEMINI_KEYS))
        
        # Process all queries concurrently
        tasks = []
        for idx, query in enumerate(queries):
            task = asyncio.create_task(
                self._process_single_query(query, idx, chunks, chunk_embeddings, semaphore)
            )
            tasks.append(task)
        
        # Gather all results
        results = {}
        for task in asyncio.as_completed(tasks):
            try:
                idx, response = await task
                results[str(idx + 1)] = response
            except Exception as e:
                logger.error(f"Error in query task: {str(e)}")
        
        return results

    async def process_single_query(self, query: str, document_link: str) -> str:
        """
        Process a single query against a document.
        
        Args:
            query: The query to process
            document_link: URL of the document to process
            
        Returns:
            The answer to the query
        """
        results = await self.process_queries([query], document_link)
        return results.get('1', 'No response generated')
        
        # Calculate total tokens for the prompt
        queries_text = "\n".join(queries)
        queries_tokens = self._count_tokens(queries_text)
        
        prompt_parts = [
            base_prompt,
            "\n\nQUESTIONS:\n",
            queries_text,
            document_text
            
        ]
        
        total_tokens = base_tokens + doc_tokens + queries_tokens + 20  # Add buffer for formatting
        
        # Log token counts
        logger.info(f"Token counts - Document: {doc_tokens}, Queries: {queries_tokens}, Total: {total_tokens}")
        
        # Add the questions to the prompt
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"Question {i}: {query}\n")
            
        prompt_parts.append("\nPlease provide your answers in the following format:\n")
        for i in range(1, len(queries) + 1):
            prompt_parts.append(f"Answer {i}: [Your answer to question {i}]\n")
            
        prompt_text = "".join(prompt_parts)
        return prompt_text, total_tokens


    def _parse_llm_response(self, response_text: str, queries: List[str]) -> Dict[str, str]:
        """Parses the raw text response from the LLM into a structured dictionary."""
        responses = {}
        num_queries = len(queries)

        # Initialize all queries with a default "not found" message
        # Use both original query and "Query X" format for compatibility
        for i, query in enumerate(queries, 1):
            responses[query] = "I couldn't find a specific answer to this question in the document."
            responses[f"Query {i}"] = "I couldn't find a specific answer to this question in the document."

        # Use a regular expression to split the response by "Answer X:"
        # This is much more flexible than checking line by line.
        # (?i) makes it case-insensitive. \s* handles variable spaces.
        pattern = r'(?i)Answer\s*\d+:'
        answer_sections = re.split(pattern, response_text)

        if len(answer_sections) > 1:
            # The first element is anything before "Answer 1:", so we discard it.
            parsed_answers = answer_sections[1:]
            
            for i, answer_text in enumerate(parsed_answers):
                if i < num_queries:
                    # Map the parsed answer to both original query and "Query X" format
                    original_query = queries[i]
                    clean_answer = answer_text.strip()
                    responses[original_query] = clean_answer
                    responses[f"Query {i+1}"] = clean_answer
                    
        # This handles the edge case where the AI gives a single block of text
        # without any "Answer X:" formatting.
        elif len(queries) == 1 and response_text.strip():
            clean_answer = response_text.strip()
            responses[queries[0]] = clean_answer
            responses["Query 1"] = clean_answer
                
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
            logger.info(f"LLM response for batch {metadata.get('batch_num')}: {response_text[:500]}...")
            
            # Parse the response into individual answers
            parsed_responses = self._parse_llm_response(response_text, queries)
            logger.info(f"Parsed responses: {parsed_responses}")
            return parsed_responses
            
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
            
            # Log document text length for debugging
            logger.info(f"Extracted document text length: {len(document_text)} characters")
            logger.info(f"Document text sample: {document_text[:200]}...")
            
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