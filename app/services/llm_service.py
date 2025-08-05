import os
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import requests
import time
import traceback
import json
import logging
import fitz  # PyMuPDF
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Path to store the last used key index
KEY_INDEX_FILE = Path("/app/data/api_key_index.json")

# Gemini API keys
GEMINI_KEYS = [
    "AIzaSyBO54qUNvonF9dGr2Tx0zCmPZYAWF2dAPA",
    "AIzaSyAhrjt-117-d0x5hmDn03_UWBtzgh46FHI",
    "AIzaSyB9wLm1gln5Xw-earmB9RF-i1g5nl9QAEk",
    
]

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
        
        with open(KEY_INDEX_FILE, 'w') as f:
            json.dump({'last_index': next_index}, f)
            
        return next_index
    except Exception as e:
        logger.error(f"Error managing key index: {str(e)}")
        return 0  # Fallback to first key

# Import configuration from your app
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
        """Initializes the LLMService."""
        self.gemini_api_keys = GEMINI_KEYS
        self.model_name = "gemini-1.5-flash-latest"
        self.max_tokens = 8196
        self.executor = ThreadPoolExecutor(max_workers=5)
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using model: {self.model_name}")
        self._setup_genai()
        
    def _get_next_api_key(self) -> str:
        """Get the next Gemini API key using persistent rotation."""
        key_index = get_next_key_index()
        return self.gemini_api_keys[key_index]
    
    def _setup_genai(self):
        """Setup the GenAI client with an API key."""
        api_key = self._get_next_api_key()
        logger.info(f"Configuring GenAI with API key (last 5 chars): ...{api_key[-5:]}")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
    def _prepare_prompt(self, query: str, full_context: str) -> str:
        """
        Prepares the prompt for the LLM with the full document context.
        """
        prompt = f"""You are a smart assistant helping a user extract information from a policy PDF.

--- Policy Document Extract ---
{full_context}
--- End Extract ---

Please answer the following question clearly, concisely, and truthfully. Use the document context when possible. If the question is unrelated to the document (e.g., about general knowledge, coding, or famous people), provide a general answer. Do not say "Not available".

The response should be a single paragraph within 700 characters.

Question: {query}
"""
        return prompt

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _download_pdf(self, url: str) -> bytes:
        """Downloads a PDF from the given URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            raise

    def extract_full_text(self, pdf_bytes: bytes, max_chars_per_chunk: int = 1000) -> str:
        """Extracts and joins all text chunks from PDF bytes."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            chunks = []
            for page in doc:
                text = page.get_text().strip().replace('\n', ' ')
                for i in range(0, len(text), max_chars_per_chunk):
                    chunk = text[i:i + max_chars_per_chunk]
                    if len(chunk.split()) > 5:
                        chunks.append(chunk)
            return "\n\n".join(chunks)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    async def _download_and_extract(self, url: str) -> Optional[str]:
        """Asynchronously downloads a PDF and extracts its full text."""
        try:
            pdf_bytes = await asyncio.to_thread(self._download_pdf, url)
            if not pdf_bytes:
                logger.error(f"Failed to download PDF from {url}")
                return None
            
            full_text = await asyncio.to_thread(self.extract_full_text, pdf_bytes)
            if not full_text:
                logger.error(f"No text could be extracted from PDF at {url}")
                return None
                
            return full_text
        except Exception as e:
            logger.error(f"Error in _download_and_extract: {str(e)}", exc_info=True)
            return None

    async def _call_llm(self, prompt: str, retry_count: int = 0, used_keys: set = None) -> str:
        """Calls the Gemini model with API key rotation and retry logic."""
        if used_keys is None:
            used_keys = set()
            
        if len(used_keys) >= len(self.gemini_api_keys):
            raise Exception("All API keys have been exhausted")
            
        current_key = next((key for key in self.gemini_api_keys if key not in used_keys), None)
            
        if not current_key:
            raise Exception("No available API keys")
        
        used_keys.add(current_key)

        try:
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(self.model_name)
            
            response = await model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.95,
                    "top_k": 40
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            if hasattr(response, 'text'):
                return response.text.strip()
            elif response.candidates and response.candidates[0].content.parts:
                return "".join(part.text for part in response.candidates[0].content.parts).strip()
            else:
                raise ValueError("Unexpected or empty response format from Gemini API")
                
        except Exception as e:
            logger.warning(f"API call failed with key ...{current_key[-4:]}: {str(e)}")
            if retry_count < len(self.gemini_api_keys) - 1:
                return await self._call_llm(prompt, retry_count + 1, used_keys)
            raise Exception(f"All API keys failed. Last error: {str(e)}")

    async def _process_single_query(
        self, 
        query: str, 
        query_idx: int, 
        full_context: str,
        semaphore: asyncio.Semaphore
    ) -> Tuple[int, str]:
        """Processes a single query using the full document context."""
        async with semaphore:
            try:
                prompt = self._prepare_prompt(query, full_context)
                response = await self._call_llm(prompt)
                return query_idx, response
            except Exception as e:
                logger.error(f"Error processing query {query_idx + 1}: {str(e)}", exc_info=True)
                return query_idx, f"Error processing query: {str(e)}"

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Processes multiple queries against a document in parallel."""
        if not queries:
            return {}
            
        # Download and extract the full text from the document
        full_context = await self._download_and_extract(document_link)
        if not full_context:
            error_msg = f"Failed to process document: {document_link}"
            return {str(i+1): error_msg for i in range(len(queries))}
            
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(len(self.gemini_api_keys))
        
        # Process all queries concurrently
        tasks = [
            asyncio.create_task(
                self._process_single_query(query, idx, full_context, semaphore)
            )
            for idx, query in enumerate(queries)
        ]
        
        # Gather all results
        results = {}
        for task in asyncio.as_completed(tasks):
            try:
                idx, response = await task
                results[str(idx + 1)] = response
            except Exception as e:
                logger.error(f"Error in a query task: {str(e)}")
        
        return results

# Singleton instance for the application
llm_service = LLMService()