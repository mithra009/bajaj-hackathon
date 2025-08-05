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
    "AIzaSyBt3uYlNWBM0jAixwK-IP4kd1iYnEz5ekE",
    "AIzaSyBxsA2DJGaG0foTkQF07bnYK-2Nnw3PU08",
    "AIzaSyC8gk0hfmDGBoBdOO3vPlafql6h-9qRPEA",
    "AIzaSyBq_2OA0beLrqgnNY6tLylXWuva9Oe8duI",
    "AIzaSyCsqLzRq2sN8MoMJlwzvvZVV0VddRiScY8",
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
        self.model_name = "gemini-1.5-flash-latest"
        self.max_tokens = 8196
        self.executor = ThreadPoolExecutor(max_workers=5)  # For parallel processing
        logger.info("Initializing LLMService with Gemini Flash model")
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

    async def _get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text chunks to embed
            model: OpenAI embedding model to use
            
        Returns:
            List of embeddings
        
        Raises:
            APIError: For API-related errors
        """
        try:
            # Use the async client for better performance
            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except openai.APITimeoutError as e:
            logger.warning(f"API request timed out: {str(e)}")
            raise APITimeoutError(str(e))
        except openai.APIConnectionError as e:
            logger.warning(f"API connection error: {str(e)}")
            raise APIConnectionError(str(e))
        except openai.APIStatusError as e:
            logger.warning(f"API status error: {str(e)}")
            raise APIError(str(e))
        except openai.OpenAIError as e:
            logger.warning(f"OpenAI API error: {str(e)}")
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

    async def _call_llm(self, prompt: str, retry_count: int = 0, used_keys: set = None) -> str:
        """Calls the Gemini model with the given prompt and returns the response.
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
                query_emb_list = await self._get_embeddings([query])
                if not query_emb_list:
                    raise ValueError("Failed to get embedding for the query.")
                query_emb = query_emb_list[0]
                
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

# Singleton instance for the application
llm_service = LLMService()