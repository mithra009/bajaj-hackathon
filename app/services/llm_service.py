import os
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import httpx
import time
import traceback
import json
import random
import logging
import hashlib
from datetime import datetime
from pathlib import Path
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

class OptimizedLLMService:
    def __init__(self):
        """Initializes the LLMService with optimizations for better performance."""
        if not API_KEYS:
            raise ValueError("No API keys provided in the API_KEYS list")
            
        self.api_key = random.choice(API_KEYS)
        self.model_name = MODEL_NAME
        self.max_tokens = MAX_TOKENS
        
        # Performance optimizations
        self._http_client = None
        self._response_cache = {}
        self._document_accessibility_cache = {}
        self._key_usage_tracker = {key: {"last_used": 0, "error_count": 0} for key in API_KEYS}
        
        logger.info(f"Initializing OptimizedLLMService with model: {self.model_name}")
        self._setup_genai()

    async def _get_http_client(self):
        """Get or create a reusable HTTP client with optimized settings."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(5.0),  # Reduced timeout for faster failure detection
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return self._http_client

    def _get_cache_key(self, queries: List[str], document_link: str) -> str:
        """Generate a cache key for the queries and document combination."""
        combined = f"{document_link}:{':'.join(sorted(queries))}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _calculate_optimal_batch_size(self, queries: List[str]) -> int:
        """Calculate optimal batch size based on query complexity."""
        if not queries:
            return MAX_QUERIES_PER_BATCH
            
        avg_query_length = sum(len(q) for q in queries) / len(queries)
        if avg_query_length > 200:  # Long queries
            return max(3, MAX_QUERIES_PER_BATCH // 2)
        elif avg_query_length < 50:  # Short queries
            return min(MAX_QUERIES_PER_BATCH * 2, 20)
        return MAX_QUERIES_PER_BATCH

    def _get_best_available_key(self, used_keys: set) -> str:
        """Select the best API key based on usage patterns and error rates."""
        available_keys = [k for k in API_KEYS if k not in used_keys]
        if not available_keys:
            # Reset if all keys have been used
            return random.choice(API_KEYS)
        
        # Prefer keys with lower error rates and longer time since last use
        current_time = time.time()
        best_key = min(available_keys, 
                      key=lambda k: (
                          self._key_usage_tracker[k]["error_count"], 
                          -(current_time - self._key_usage_tracker[k]["last_used"])
                      ))
        
        # Update usage tracking
        self._key_usage_tracker[best_key]["last_used"] = current_time
        return best_key

    def _setup_genai(self, used_keys=None):
        """Configures the Gemini API client with the current API key."""
        used_keys = used_keys or set()
        
        if not self.api_key:
            self.api_key = self._get_best_available_key(used_keys)
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            return True
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Error with API key {self.api_key[:10]}...: {error_msg}")
            
            # Update error tracking
            self._key_usage_tracker[self.api_key]["error_count"] += 1
            used_keys.add(self.api_key)
            
            # Try a different API key if available
            available_keys = [k for k in API_KEYS if k not in used_keys]
            if available_keys:
                self.api_key = self._get_best_available_key(used_keys)
                logger.info(f"Retrying with API key: {self.api_key[:10]}...")
                return self._setup_genai(used_keys)
                
            raise ValueError("All API keys have been exhausted. Please add more keys or try again later.")

    def _is_valid_url(self, url: str) -> bool:
        """Checks if a URL string is well-formed."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    async def _is_document_accessible(self, url: str, use_cache: bool = True) -> bool:
        """Checks if a document is accessible with caching for performance."""
        if use_cache and url in self._document_accessibility_cache:
            cache_time, result = self._document_accessibility_cache[url]
            # Cache for 5 minutes
            if time.time() - cache_time < 300:
                return result
        
        try:
            client = await self._get_http_client()
            response = await client.head(url, timeout=5.0, follow_redirects=True)
            result = response.status_code == 200
            
            if use_cache:
                self._document_accessibility_cache[url] = (time.time(), result)
            
            return result
        except (httpx.RequestError, httpx.TimeoutException):
            if use_cache:
                self._document_accessibility_cache[url] = (time.time(), False)
            return False

    def _prepare_optimized_prompt(self, queries: List[str], document_link: str) -> str:
        """Prepares an optimized, more concise prompt for faster processing."""
        prompt_parts = [
            f"Document: {document_link}\n\n",
            "Answer each question concisely (<500 chars each) based on the document. If not in document, give brief general answer.\n",
            "Format: Answer N: [response]\n\n"
        ]
        
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}\n")
            
        return "".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, queries: List[str]) -> Dict[str, str]:
        """Parses the raw text response from the LLM into a structured dictionary."""
        responses = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for i, query in enumerate(queries, 1):
            answer_prefix = f"Answer {i}:"
            found_answer = "I couldn't find a specific answer to this question in the document."
            
            for line in lines:
                if line.startswith(answer_prefix):
                    found_answer = line[len(answer_prefix):].strip()
                    # Clean up quotes if they wrap the entire answer
                    if (found_answer.startswith('"') and found_answer.endswith('"')) or \
                       (found_answer.startswith("'") and found_answer.endswith("'")):
                        found_answer = found_answer[1:-1]
                    break
            responses[f"Query {i}"] = found_answer
            
        return responses

    async def _process_batch_with_key(self, queries: List[str], document_link: str, 
                                    metadata: Dict[str, Any], api_key: str) -> Dict[str, str]:
        """Process a batch of queries with a specific API key."""
        try:
            # Create a new instance with the specified API key
            temp_service = OptimizedLLMService()
            temp_service.api_key = api_key
            temp_service._setup_genai()
            
            # Use optimized prompt
            prompt = temp_service._prepare_optimized_prompt(queries, document_link)
            
            logger.info(f"Processing batch {metadata.get('batch_num', '?')}/{metadata.get('total_batches', '?')} "
                      f"with {len(queries)} queries using API key ...{api_key[-4:]}")
            
            # Optimized generation config for faster responses
            response = await temp_service.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": min(self.max_tokens, 800),  # Reduced for faster processing
                    "temperature": 0.0,  # Deterministic responses are faster
                    "top_p": 0.8,       # Reduce search space
                    "top_k": 20         # Further limit token selection
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
            # Update error tracking
            if api_key in self._key_usage_tracker:
                self._key_usage_tracker[api_key]["error_count"] += 1
            
            # Return default responses for this batch
            return {f"Query {i+1}": "I couldn't find a specific answer to this question in the document." 
                    for i in range(len(queries))}

    async def generate_response(self, queries: List[str], document_link: str, 
                              metadata: Dict[str, Any] = None, use_cache: bool = True,
                              skip_accessibility_check: bool = False) -> Dict[str, str]:
        """
        Optimized version of generate_response with caching and performance improvements.
        """
        start_time = time.time()
        metadata = metadata or {}
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(queries, document_link)
            if cache_key in self._response_cache:
                cache_time, cached_response = self._response_cache[cache_key]
                # Cache for 10 minutes
                if time.time() - cache_time < 600:
                    logger.info("Returning cached response")
                    return cached_response
        
        try:
            if not queries:
                raise ValueError("At least one query is required")
            if not document_link or not self._is_valid_url(document_link):
                raise ValueError("A valid document link is required")

            # Optional accessibility check for performance
            if not skip_accessibility_check:
                is_accessible = await self._is_document_accessible(document_link, use_cache=True)
                if not is_accessible:
                    raise ValueError(f"Document at {document_link} is not accessible or not found")
            else:
                is_accessible = True  # Assume accessible when skipping check
            
            # Calculate optimal batch size
            optimal_batch_size = self._calculate_optimal_batch_size(queries)
            
            # Prepare metadata for logging
            query_metadata = {
                "model": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "num_queries": len(queries),
                "document_accessible": is_accessible,
                "optimal_batch_size": optimal_batch_size,
                "processing_mode": "batched" if len(queries) > optimal_batch_size else "single",
                **metadata
            }
            
            logger.info(f"\n=== PROCESSING {len(queries)} QUERIES (Optimized) ===")
            logger.info(f"Document: {document_link}")
            logger.info(f"Batch size: {optimal_batch_size}")
            
            # Split queries into optimized batches
            query_batches = [queries[i:i + optimal_batch_size] 
                          for i in range(0, len(queries), optimal_batch_size)]
            
            # Create batch tasks with smart API key assignment
            batch_tasks = []
            used_keys = set()
            
            for batch_num, batch in enumerate(query_batches, 1):
                batch_key = self._get_best_available_key(used_keys)
                used_keys.add(batch_key)
                
                batch_metadata = {
                    **query_metadata,
                    "batch_num": batch_num,
                    "total_batches": len(query_batches),
                    "api_key_used": f"...{batch_key[-4:]}"
                }
                
                task = self._process_batch_with_key(batch, document_link, batch_metadata, batch_key)
                batch_tasks.append(task)
            
            # Process all batches in parallel
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine all responses
            combined_responses = {}
            query_counter = 1
            
            for batch_idx, batch_response in enumerate(batch_responses):
                batch = query_batches[batch_idx] if batch_idx < len(query_batches) else []
                
                if isinstance(batch_response, dict):
                    for i in range(len(batch)):
                        q_num = f"Query {i+1}"
                        if q_num in batch_response:
                            combined_responses[f"Query {query_counter}"] = batch_response[q_num]
                        else:
                            combined_responses[f"Query {query_counter}"] = \
                                "I couldn't find a specific answer to this question in the document."
                        query_counter += 1
                else:
                    for _ in range(len(batch)):
                        if query_counter <= len(queries):
                            combined_responses[f"Query {query_counter}"] = \
                                "I couldn't find a specific answer to this question in the document."
                            query_counter += 1
            
            # Cache the response if enabled
            if use_cache:
                self._response_cache[cache_key] = (time.time(), combined_responses)
                # Limit cache size to prevent memory issues
                if len(self._response_cache) > 100:
                    # Remove oldest entries
                    oldest_key = min(self._response_cache.keys(), 
                                   key=lambda k: self._response_cache[k][0])
                    del self._response_cache[oldest_key]
            
            # Log the successful processing
            try:
                processing_time = time.time() - start_time
                log_id = query_logger.log_query(
                    document_link=document_link,
                    queries=queries,
                    responses=combined_responses,
                    metadata={
                        **query_metadata,
                        "processing_time_seconds": processing_time,
                        "num_batches": len(query_batches),
                        "response_character_count": sum(len(str(r)) for r in combined_responses.values()),
                        "optimizations_used": ["caching", "smart_batching", "connection_pooling", "optimized_prompts"]
                    }
                )
                logger.info(f"Successfully processed all queries in {processing_time:.2f}s. Log ID: {log_id}")
            except Exception as e:
                logger.error(f"Failed to log query: {str(e)}")
            
            return combined_responses

        except Exception as e:
            logger.error(f"\n=== ERROR OCCURRED IN OPTIMIZED_LLM_SERVICE ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("=== STACK TRACE ===")
            logger.error(traceback.format_exc())
            
            return {f"Query {i+1}": f"Error processing request: {str(e)}" 
                   for i in range(len(queries))}

    async def cleanup(self):
        """Cleanup resources when shutting down."""
        if self._http_client:
            await self._http_client.aclose()

# Singleton instance for the application
optimized_llm_service = OptimizedLLMService()