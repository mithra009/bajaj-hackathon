import os
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import requests
import time
import traceback
import random
import json
import logging
import re
import fitz  # PyMuPDF
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from typing import List, Dict, Any, Optional, Tuple, Union

# Load environment variables
load_dotenv()

# Define custom exceptions
class APIError(Exception):
    """Base class for other API-related exceptions"""
    pass

# --- STATEFUL KEY ROTATION SETUP ---
KEY_INDEX_FILE = Path("/app/data/api_key_index.json")

def get_next_key_index(num_keys: int) -> int:
    """Reads the last used index from a file, increments it, and saves it back."""
    try:
        KEY_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        if KEY_INDEX_FILE.exists():
            with open(KEY_INDEX_FILE, 'r') as f:
                data = json.load(f)
                last_index = data.get('last_index', -1)
        else:
            last_index = -1
        
        next_index = (last_index + 1) % num_keys
        
        with open(KEY_INDEX_FILE, 'w') as f:
            json.dump({'last_index': next_index}, f)
        return next_index
    except Exception as e:
        logging.error(f"Error managing key index file: {e}")
        return 0

# --- API KEY CONFIGURATION ---
GEMINI_KEYS = [
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initializes the LLMService."""
        self.gemini_api_keys = GEMINI_KEYS
        self.model_name = "gemini-2.0-flash"
        self.embedding_model = "text-embedding-3-small"
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set. It is required for embeddings.")
        self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        # Configuration
        self.max_tokens = 8196
        self.max_embedding_batch_size = 100  # Number of texts to process in a single batch
        self.max_tokens_per_batch = 200000  # Conservative limit below the 300k token limit
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using generation model: {self.model_name}")
        logger.info(f"Using OpenAI embedding model: {self.embedding_model}")
        logger.info(f"Max embedding batch size: {self.max_embedding_batch_size} texts or {self.max_tokens_per_batch} tokens")

    def _prepare_rag_prompt_for_all_queries(self, queries_with_context: List[Tuple[str, List[str]]]) -> str:
        """Prepares a single prompt containing all queries and their respective relevant chunks."""
        prompt_parts = [
            "You are a helpful assistant providing clear, concise answers from the given context.",
            "IMPORTANT: Respond in simple paragraphs without markdown formatting.",
            "For each question, use ONLY the provided context to answer. If the context doesn't contain the answer, provide a general response.",
            "\n--- QUESTIONS AND CONTEXT ---\n"
        ]

        for i, (query, context_chunks) in enumerate(queries_with_context, 1):
            context_str = "\n".join(context_chunks)
            prompt_parts.append(f"QUESTION {i}: {query}")
            prompt_parts.append(f"CONTEXT: {context_str}")
            prompt_parts.append("")

        prompt_parts.append(
            "Provide clear, concise answers for each question in order. "
            "Start each answer with the question number followed by a colon. "
            "Keep each answer under 700 characters. "
            "Example:\n"
            "1: [Answer to question 1]\n"
            "2: [Answer to question 2]"
        )
        return "\n".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, num_queries: int) -> Dict[str, str]:
        """Parses the LLM's response into a dictionary, handling various formats."""
        responses = {}
        # Try different patterns to extract answers
        patterns = [
            r"^\s*(\d+)[:\.]\s*(.*?)(?=\n\s*\d+[:.]|\Z)",  # Number: or Number. followed by text until next number or end
            r"^\s*(\d+)[\)]\s*(.*?)(?=\n\s*\d+[\)]|\Z)",  # Number) followed by text
            r"^\s*(\d+)\s*[-]\s*(.*?)(?=\n\s*\d+\s*[-]|\Z)"  # Number - text
        ]
        
        parsed_answers = {}
        for pattern in patterns:
            if not parsed_answers:  # Only try next pattern if previous ones didn't work
                matches = re.finditer(pattern, response_text, re.MULTILINE | re.DOTALL)
                parsed_answers = {int(m.group(1)): m.group(2).strip() for m in matches}
        
        # Fallback: Split by lines and try to match any line starting with a number
        if not parsed_answers:
            for line in response_text.split('\n'):
                match = re.match(r"^\s*(\d+)[:\.\s]+(.*)", line)
                if match:
                    parsed_answers[int(match.group(1))] = match.group(2).strip()
        
        # Generate responses with better fallback messages
        for i in range(num_queries):
            query_num = i + 1
            answer = parsed_answers.get(query_num)
            if not answer or len(answer) < 5:  # Very short answer might be invalid
                answer = "The information for this question could not be determined from the provided context."
            responses[str(query_num)] = answer
            
        return responses

    def _estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        # Rough estimate: 1 token ~= 4 chars in English
        return max(1, len(text) // 4)

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using OpenAI's model with batching."""
        if not texts:
            return []

        all_embeddings = []
        current_batch = []
        current_batch_tokens = 0
        
        # First, sort texts by length to optimize batching
        sorted_texts = sorted(enumerate(texts), key=lambda x: len(x[1]), reverse=True)
        
        for idx, text in sorted_texts:
            text_tokens = self._estimate_token_count(text)
            
            # If a single text is too large, split it into chunks
            if text_tokens > self.max_tokens_per_batch:
                logger.warning(f"Text at index {idx} is too large ({text_tokens} tokens). Splitting into chunks...")
                chunks = [text[i:i + self.max_tokens_per_batch * 4] for i in range(0, len(text), self.max_tokens_per_batch * 4)]
                chunk_embeddings = await self._process_batches(chunks, [idx] * len(chunks))
                # Average the chunk embeddings for the final text embedding
                if chunk_embeddings:
                    all_embeddings.append((idx, np.mean(chunk_embeddings, axis=0).tolist()))
                continue
                
            # Check if adding this text would exceed batch limits
            if (current_batch and 
                (len(current_batch) >= self.max_embedding_batch_size or 
                 current_batch_tokens + text_tokens > self.max_tokens_per_batch)):
                # Process the current batch
                batch_embeddings = await self._process_batch(current_batch)
                all_embeddings.extend(batch_embeddings)
                current_batch = []
                current_batch_tokens = 0
                
            current_batch.append((idx, text))
            current_batch_tokens += text_tokens
        
        # Process any remaining texts in the last batch
        if current_batch:
            batch_embeddings = await self._process_batch(current_batch)
            all_embeddings.extend(batch_embeddings)
        
        # Sort embeddings by original index and return just the vectors
        all_embeddings.sort(key=lambda x: x[0])
        return [emb for idx, emb in all_embeddings]
    
    async def _process_batch(self, batch: List[Tuple[int, str]]) -> List[Tuple[int, List[float]]]:
        """Process a single batch of texts and return embeddings with original indices."""
        try:
            # Extract just the texts for the API call
            texts = [text for idx, text in batch]
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            # Return list of (original_index, embedding) tuples
            return [(batch[i][0], item.embedding) 
                   for i, item in enumerate(response.data)]
        except Exception as e:
            logger.error(f"Error processing batch of size {len(batch)}: {e}")
            # Return empty embeddings for failed batch
            return [(idx, [0.0] * 1536) for idx, _ in batch]
            
    async def _process_batches(self, texts: List[str], indices: List[int]) -> List[List[float]]:
        """Process multiple batches of texts and return all embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), self.max_embedding_batch_size):
            batch_texts = texts[i:i + self.max_embedding_batch_size]
            batch_indices = indices[i:i + self.max_embedding_batch_size]
            batch = list(zip(batch_indices, batch_texts))
            batch_embeddings = await self._process_batch(batch)
            all_embeddings.extend([emb for idx, emb in batch_embeddings])
        return all_embeddings

    def _find_top_chunks(self, query_emb: List[float], chunk_embs: np.ndarray, chunks: List[str], top_k: int = 7) -> List[str]:
        """Finds the top-k most relevant chunks for a query using cosine similarity."""
        sims = cosine_similarity([query_emb], chunk_embs)[0]
        top_idxs = sims.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_idxs]

    def _download_pdf(self, url: str) -> bytes:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    def extract_chunks(self, pdf_bytes: bytes, max_chars: int = 1000) -> List[str]:
        """Extract text chunks from PDF with better handling of large documents."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            chunks = []
            total_chars = 0
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if not text:
                    continue
                    
                # Split into paragraphs first to maintain some structure
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                
                for para in paragraphs:
                    # If paragraph is too long, split it into smaller chunks
                    if len(para) > max_chars * 2:  # More aggressive splitting for very long paragraphs
                        words = para.split()
                        current_chunk = []
                        current_length = 0
                        
                        for word in words:
                            word_length = len(word) + 1  # +1 for space
                            if current_length + word_length > max_chars and current_chunk:
                                chunks.append(' '.join(current_chunk))
                                current_chunk = [word]
                                current_length = word_length
                            else:
                                current_chunk.append(word)
                                current_length += word_length
                        
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                    else:
                        chunks.append(para)
                    
                    total_chars += len(para)
                    if total_chars > 1000000:  # Limit total characters to process
                        logger.warning(f"Reached maximum document size limit at page {page_num + 1}")
                        return chunks
            
            logger.info(f"Extracted {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise APIError(f"Failed to extract text from PDF: {e}")

    async def _call_llm_with_single_key(self, prompt: str, api_key: str) -> str:
        """Calls the Gemini model with a single, specific API key."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.1,  # Lower temperature for more focused answers
                    "max_output_tokens": 2048,  # Reduced from max to speed up response
                    "top_p": 0.9,
                    "top_k": 40
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                model.generate_content_async(prompt),
                timeout=30.0  # 30 second timeout
            )
            return response.text.strip() if response.text else "No response generated"
        except Exception as e:
            logger.error(f"API call failed with key ...{api_key[-4:]}: {e}")
            raise Exception(f"Gemini API call failed: {e}")

    def _is_pdf_url(self, url: str) -> bool:
        """Check if the URL points to a PDF file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith('.pdf')

    async def _process_direct_url_queries(self, queries: List[str], url: str, api_key: str) -> Dict[str, str]:
        """Process queries by directly sending URL to Gemini without downloading content."""
        try:
            logger.info("\nProcessing URL directly with Gemini...")
            start_time = time.time()
            
            # Prepare prompt with URL and queries
            prompt = (
                "You are a helpful assistant that answers questions based on the provided URL.\n"
                "IMPORTANT: Respond in simple paragraphs without markdown formatting.\n\n"
                f"URL: {url}\n\n"
                "QUESTIONS:\n"
            )
            
            for i, query in enumerate(queries, 1):
                prompt += f"{i}. {query}\n"
                
            prompt += ("\n"
                     "Provide clear, concise answers for each question in order. "
                     "Start each answer with the question number followed by a colon. "
                     "Keep each answer under 500 characters.")
            
            # Call the LLM with the URL and queries
            logger.info("\nCalling Gemini API with URL and queries...")
            llm_response_text = await self._call_llm_with_single_key(prompt, api_key)
            
            # Parse the response
            final_responses = self._parse_llm_response(llm_response_text, len(queries))
            
            logger.info(f"URL processing completed in {time.time() - start_time:.2f}s")
            return final_responses
            
        except Exception as e:
            logger.error(f"Error in _process_direct_url_queries: {e}")
            raise Exception(f"Gemini API call failed: {e}")

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Processes all queries in a single RAG-based batch.
        
        Args:
            queries: List of questions to be answered
            document_link: URL of the document to process
            
        Returns:
            Dict mapping question numbers to answers
        """
        start_time = time.time()
        
        if not queries:
            return {}

        try:
            # Log document information in a structured format
            logger.info(f"\n{'='*120}")
            logger.info("PROCESSING NEW REQUEST")
            logger.info(f"{'='*120}")
            logger.info(f"DOCUMENT_URL: {document_link}")
            logger.info(f"QUERY_COUNT: {len(queries)}")
            logger.info("QUERIES:")
            for i, query in enumerate(queries, 1):
                logger.info(f"  {i}. {query}")
            logger.info(f"{'='*120}")
            
            # 1. Select the next Gemini API key for this entire request
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key_for_request = self.gemini_api_keys[key_index]
            logger.info(f"Using Gemini key index: {key_index} (...{api_key_for_request[-4:]})")

            # Check if URL is a PDF or direct URL
            if not self._is_pdf_url(document_link):
                logger.info("Detected non-PDF URL, processing directly with Gemini...")
                return await self._process_direct_url_queries(queries, document_link, api_key_for_request)

            # 2. Download and chunk document
            logger.info("\nDownloading and processing document...")
            download_start = time.time()
            pdf_bytes = await asyncio.to_thread(self._download_pdf, document_link)
            doc_chunks = await asyncio.to_thread(self.extract_chunks, pdf_bytes)
            if not doc_chunks:
                raise ValueError("Failed to extract text chunks from the document.")
            logger.info(f"Document processed in {time.time() - download_start:.2f}s | Chunks: {len(doc_chunks)}")

            # 3. RAG Pre-processing using OpenAI Embeddings
            logger.info("\nGenerating embeddings...")
            embed_start = time.time()
            doc_embeddings = np.array(await self._get_embeddings(doc_chunks))
            query_embeddings = await self._get_embeddings(queries)
            logger.info(f"Embeddings generated in {time.time() - embed_start:.2f}s")
            
            # Log each query and its processing
            queries_with_context = []
            logger.info("\nProcessing queries...")
            for i, (query, query_emb) in enumerate(zip(queries, query_embeddings), 1):
                # Find top 7 relevant chunks for this query
                relevant_chunks = self._find_top_chunks(query_emb, doc_embeddings, doc_chunks, top_k=7)
                queries_with_context.append((query, relevant_chunks))
                logger.info(f"\nQuery {i}:"
                          f"\n  Question: {query}"
                          f"\n  Context Chunks: {len(relevant_chunks)}")

            # 4. Prepare a single prompt with all queries and their relevant chunks
            logger.info("\nGenerating prompt...")
            prompt = self._prepare_rag_prompt_for_all_queries(queries_with_context)
            
            # 5. Make a single API call to Gemini
            logger.info("\nCalling Gemini API...")
            llm_start = time.time()
            llm_response_text = await self._call_llm_with_single_key(prompt, api_key_for_request)
            llm_time = time.time() - llm_start
            logger.info(f"LLM API call completed in {llm_time:.2f}s")

            # 6. Parse the single response
            final_responses = self._parse_llm_response(llm_response_text, len(queries))
            
            # Log final responses and timing in a structured format
            total_time = time.time() - start_time
            logger.info(f"\n{'='*120}")
            logger.info("QUERY PROCESSING COMPLETE")
            logger.info(f"{'='*120}")
            logger.info("DOCUMENT_PROCESSED: " + ("SUCCESS" if final_responses else "FAILED"))
            logger.info(f"DOCUMENT_URL: {document_link}")
            logger.info(f"TOTAL_PROCESSING_TIME: {total_time:.2f}s")
            logger.info(f"LLM_API_TIME: {llm_time:.2f}s ({llm_time/total_time*100:.1f}% of total)")
            
            # Log each query and its answer
            logger.info("\nQUERY_RESULTS:")
            for i, (query, answer) in enumerate(zip(queries, final_responses.values()), 1):
                logger.info(f"\nQUERY_{i}:")
                logger.info(f"  QUESTION: {query}")
                logger.info(f"  ANSWER: {answer}")
                logger.info(f"  STATUS: PROCESSED")
            
            logger.info(f"\n{'='*120}")
            logger.info(f"END OF PROCESSING")
            logger.info(f"{'='*120}\n")
            
            return final_responses

        except Exception as e:
            logger.error(f"Critical error in process_queries: {e}", exc_info=True)
            return {str(i+1): f"Failed to process request: {e}" for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()