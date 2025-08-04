import os
import asyncio
import openai
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
import re 
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import os
from dotenv import load_dotenv
from .query_logger import query_logger

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Configure OpenAI client
openai.api_key = OPENAI_API_KEY

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
        """Initializes the LLMService with the API key from environment variable."""
        self.model_name = MODEL_NAME
        self.max_tokens = MAX_TOKENS
        self.executor = ThreadPoolExecutor(max_workers=5)  # For parallel processing
        logger.info(f"Initializing LLMService with model: {self.model_name}")
        self._setup_openai()
        
    def _setup_openai(self):
        """Setup the OpenAI client."""
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
        
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



    async def _call_llm(self, prompt: str, retry_count: int = 0, max_retries: int = 3) -> str:
        """Calls the OpenAI API with the given prompt and returns the response.
        
        Args:
            prompt: The prompt to send to the LLM
            retry_count: Number of retry attempts so far
            max_retries: Maximum number of retry attempts
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.7
                )
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error calling OpenAI API: {error_msg}")
            
            # Check if this is a rate limit or quota error
            is_rate_limit = any(keyword in error_msg for keyword in [
                "rate limit", "quota", "limit exceeded", 
                "quota exceeded", "429", "resource exhausted"
            ])
            
            if is_rate_limit and retry_count < max_retries:
                wait_time = (2 ** retry_count) + (random.randint(0, 1000) / 1000)  # Exponential backoff with jitter
                logger.info(f"Rate limit hit. Retrying in {wait_time:.2f} seconds (attempt {retry_count + 1}/{max_retries})...")
                await asyncio.sleep(wait_time)
                return await self._call_llm(prompt, retry_count + 1, max_retries)
            
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
            "Response format: Answer {question number}: response\n"
            "Answer the following questions based on the document content within 700 characters. Follow the format strictly.\n"
            "If the answer cannot be found in the document, respond generally from your knowledge.\n\n"
        )
        base_tokens = self._count_tokens(base_prompt)
        
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
    async def _process_batch(self, queries: List[str], document_text: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Process a single batch of queries (up to MAX_QUERIES_PER_BATCH)."""
        try:
            # Prepare the prompt with all queries and document text
            prompt, token_count = self._prepare_prompt(queries, document_text)
            logger.info(f"Processing batch {metadata.get('batch_num', '?')}/{metadata.get('total_batches', '?')} with {len(queries)} queries, {token_count} tokens")
            
            # Call the OpenAI API using the main client
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document text."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.7
                )
            )
            
            # Get the response text
            response_text = response.choices[0].message.content
            logger.info(f"LLM response for batch {metadata.get('batch_num')}: {response_text[:500]}...")
            
            # Parse the response into individual answers
            parsed_responses = self._parse_llm_response(response_text, queries)
            logger.info(f"Parsed responses: {parsed_responses}")
            return parsed_responses
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(traceback.format_exc())
            # Return error responses for all queries in this batch
            return {f"Query {i+1}": f"Error processing request: {str(e)}" 
                   for i in range(len(queries))}

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
            
            # Create a list to store batch tasks
            batch_tasks = []
            
            for batch_num, batch in enumerate(query_batches, 1):
                # Create metadata for this batch
                batch_metadata = {
                    **query_metadata,
                    "batch_num": batch_num,
                    "total_batches": len(query_batches),
                    "queries_in_batch": len(batch)
                }
                
                # Create a task for this batch
                task = self._process_batch(
                    batch, 
                    document_text,
                    batch_metadata
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