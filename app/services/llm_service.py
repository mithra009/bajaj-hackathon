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

        # Optimized configuration for maximum speed and accuracy
        self.max_tokens = 8196
        self.max_embedding_tokens_per_request = 2800000  # Near OpenAI's 300k limit with buffer
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using generation model: {self.model_name}")
        logger.info(f"Using OpenAI embedding model: {self.embedding_model}")

    def _prepare_individual_query_prompt(self, query: str, context_chunks: List[str]) -> str:
        """Prepares a focused prompt for a single query with relevant context."""
        context_str = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks[:5])])
        
        prompt = f"""You are an expert insurance policy analyst. Answer the question based  on the provided context from the document.

CONTEXT FROM INSURANCE POLICY:
{context_str}

QUESTION: {query}

INSTRUCTIONS:
- Answer directly and specifically based on the context provided
- If the context contains the information, provide a detailed answer
- Reference specific policy terms, clauses, or procedures mentioned in the context
- If some parts of the question cannot be answered from the context, Answer from your own knowledge generally, do not answer that you cannot find content.
- Keep the answer comprehensive but concise (under 800 characters) in a single paragraph
- Do not use markdown formatting

ANSWER:"""
        
        return prompt

    def _estimate_token_count(self, text: str) -> int:
        """Accurate token estimation for text-embedding-3-small."""
        # More precise estimate: ~4 characters per token for English text
        # Add 10% buffer for safety
        return int((len(text) / 4) * 1.1)

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Token-optimized embedding generation with maximum batch efficiency."""
        if not texts:
            return []

        try:
            MAX_TOKENS_PER_REQUEST = 2800000  # Leave buffer from 300k limit
            all_embeddings = []
            current_batch = []
            current_batch_tokens = 0
            
            logger.info(f"Processing {len(texts)} texts for embeddings...")
            
            for i, text in enumerate(texts):
                text_tokens = self._estimate_token_count(text)
                
                # Handle extremely large single texts (rare edge case)
                if text_tokens > MAX_TOKENS_PER_REQUEST:
                    logger.warning(f"Text {i} exceeds token limit ({text_tokens} tokens), truncating...")
                    # Truncate to fit within limits (roughly 280k * 4 = 1.12M chars)
                    truncated_text = text[:11200000]
                    text_tokens = self._estimate_token_count(truncated_text)
                    text = truncated_text
                
                # If adding this text exceeds token limit, process current batch
                if current_batch and (current_batch_tokens + text_tokens > MAX_TOKENS_PER_REQUEST):
                    logger.info(f"Processing batch with {len(current_batch)} texts ({current_batch_tokens:,} tokens)")
                    batch_embeddings = await self._process_embedding_batch(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Start new batch
                    current_batch = [text]
                    current_batch_tokens = text_tokens
                else:
                    current_batch.append(text)
                    current_batch_tokens += text_tokens
            
            # Process final batch
            if current_batch:
                logger.info(f"Processing final batch with {len(current_batch)} texts ({current_batch_tokens:,} tokens)")
                batch_embeddings = await self._process_embedding_batch(current_batch)
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Critical error in embedding generation: {e}")
            return [[0.0] * 1536] * len(texts)

    async def _process_embedding_batch(self, batch_texts: List[str]) -> List[List[float]]:
        """Process a single optimized batch with retry logic."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                embeddings = [item.embedding for item in response.data]
                
                elapsed = time.time() - start_time
                logger.info(f"Batch processed in {elapsed:.2f}s ({len(batch_texts)} texts)")
                
                return embeddings
                
            except Exception as e:
                wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                logger.warning(f"Embedding batch attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for batch")
                    # Return zero embeddings for completely failed batch
                    return [[0.0] * 1536] * len(batch_texts)
                else:
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

    def _find_top_chunks_optimized(self, query_emb: List[float], chunk_embs: np.ndarray, chunks: List[str], top_k: int = 8) -> List[str]:
        """Optimized chunk selection with better relevance scoring."""
        if len(chunk_embs) == 0:
            return chunks[:top_k] if chunks else []
            
        try:
            # Calculate cosine similarity
            sims = cosine_similarity([query_emb], chunk_embs)[0]
            
            # Get top indices
            top_idxs = sims.argsort()[-top_k:][::-1]
            
            # Filter out chunks with very low similarity (< 0.1)
            relevant_chunks = []
            for idx in top_idxs:
                if sims[idx] > 0.1:  # Minimum relevance threshold
                    relevant_chunks.append(chunks[idx])
            
            # If no relevant chunks found, return top chunks anyway
            if not relevant_chunks:
                relevant_chunks = [chunks[i] for i in top_idxs[:3]]
                
            return relevant_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Error in chunk selection: {e}")
            return chunks[:top_k] if chunks else []

    def extract_chunks_optimized(self, pdf_bytes: bytes, max_chars: int = 1200) -> List[str]:
        """Optimized text extraction with aggressive chunking for large documents."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            chunks = []
            
            # For very large documents, use more aggressive chunking
            total_pages = len(doc)
            if total_pages > 100:
                max_chars = 1500  # Larger chunks for big documents
                logger.info(f"Large document detected ({total_pages} pages), using larger chunks")
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if not text:
                    continue
                
                # Aggressive text cleaning for better embeddings
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\-\%\$]', ' ', text)  # Keep important punctuation
                
                # For large documents, use paragraph-based chunking for efficiency
                if total_pages > 50:
                    # Split by double newlines (paragraphs) first
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    current_chunk = ""
                    for para in paragraphs:
                        if len(current_chunk) + len(para) + 2 > max_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = para
                            else:
                                # Single paragraph too long, split by sentences
                                sentences = re.split(r'[.!?]+', para)
                                temp_chunk = ""
                                for sentence in sentences:
                                    sentence = sentence.strip()
                                    if not sentence:
                                        continue
                                    if len(temp_chunk) + len(sentence) + 2 > max_chars:
                                        if temp_chunk:
                                            chunks.append(temp_chunk.strip())
                                            temp_chunk = sentence
                                    else:
                                        temp_chunk += ". " + sentence if temp_chunk else sentence
                                if temp_chunk:
                                    current_chunk = temp_chunk
                        else:
                            current_chunk += "\n\n" + para if current_chunk else para
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                else:
                    # Original sentence-based chunking for smaller documents
                    sentences = re.split(r'[.!?]+', text)
                    
                    current_chunk = ""
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if len(current_chunk) + len(sentence) + 2 > max_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                # Single sentence too long, split by words
                                words = sentence.split()
                                temp_chunk = ""
                                for word in words:
                                    if len(temp_chunk) + len(word) + 1 > max_chars:
                                        if temp_chunk:
                                            chunks.append(temp_chunk.strip())
                                            temp_chunk = word
                                        else:
                                            chunks.append(word)  # Very long single word
                                    else:
                                        temp_chunk += " " + word if temp_chunk else word
                                if temp_chunk:
                                    current_chunk = temp_chunk
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                # Memory and performance optimization for very large documents
                if len(chunks) > 1000:  # Increased limit for large docs
                    logger.warning(f"Reached chunk limit at page {page_num + 1}")
                    break
            
            # Remove very short chunks (less than 100 characters for large docs)
            min_chunk_size = 100 if total_pages > 100 else 50
            chunks = [chunk for chunk in chunks if len(chunk) > min_chunk_size]
            
            logger.info(f"Extracted {len(chunks)} optimized chunks from {total_pages}-page document")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise APIError(f"Failed to extract text from PDF: {e}")

    async def _call_llm_optimized(self, prompt: str, api_key: str) -> str:
        """Optimized LLM call with better error handling and faster response."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.2,  # Balanced for accuracy and creativity
                    "max_output_tokens": 1024,  # Optimized for response speed
                    "top_p": 0.9,
                    "top_k": 30
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            # Reduced timeout for faster processing
            response = await asyncio.wait_for(
                model.generate_content_async(prompt),
                timeout=20.0
            )
            return response.text.strip() if response.text else "Unable to generate response"
            
        except asyncio.TimeoutError:
            logger.error(f"API call timeout with key ...{api_key[-4:]}")
            raise Exception("Request timeout - please try again")
        except Exception as e:
            logger.error(f"API call failed with key ...{api_key[-4:]}: {e}")
            raise Exception(f"Gemini API call failed: {str(e)[:100]}")

    def _download_pdf_optimized(self, url: str) -> bytes:
        """Optimized PDF download with better error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*'
            }
            
            response = requests.get(url, timeout=15, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"Unexpected content type: {content_type}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise APIError(f"Failed to download PDF: {e}")

    def _is_pdf_url(self, url: str) -> bool:
        """Enhanced PDF URL detection."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check file extension
        if path.endswith('.pdf'):
            return True
            
        # Check for PDF indicators in URL
        pdf_indicators = ['pdf', 'document', 'attachment']
        return any(indicator in url.lower() for indicator in pdf_indicators)

    async def process_queries_parallel(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Process queries in parallel for maximum speed while maintaining accuracy."""
        start_time = time.time()
        
        if not queries:
            return {}

        try:
            logger.info(f"\n{'='*100}")
            logger.info(f"PROCESSING {len(queries)} QUERIES IN PARALLEL")
            logger.info(f"DOCUMENT: {document_link}")
            logger.info(f"{'='*100}")

            # 1. Select API key for this request
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key = self.gemini_api_keys[key_index]
            logger.info(f"Using Gemini key index: {key_index}")

            # 2. Download and process document
            logger.info("Downloading and processing document...")
            download_start = time.time()
            pdf_bytes = await asyncio.to_thread(self._download_pdf_optimized, document_link)
            doc_chunks = await asyncio.to_thread(self.extract_chunks_optimized, pdf_bytes)
            
            if not doc_chunks:
                raise ValueError("No text extracted from document")
                
            logger.info(f"Document processed in {time.time() - download_start:.2f}s | Chunks: {len(doc_chunks)}")

            # 3. Generate embeddings with maximum efficiency
            logger.info("Generating embeddings with token-optimized batching...")
            embed_start = time.time()
            
            # Estimate total tokens to optimize strategy
            total_doc_tokens = sum(self._estimate_token_count(chunk) for chunk in doc_chunks)
            total_query_tokens = sum(self._estimate_token_count(q) for q in queries)
            
            logger.info(f"Document tokens: {total_doc_tokens:,}, Query tokens: {total_query_tokens:,}")
            
            # Process embeddings concurrently if both fit within limits, otherwise sequentially
            if total_doc_tokens + total_query_tokens < self.max_embedding_tokens_per_request:
                logger.info("Processing document and queries in single combined batch")
                all_texts = doc_chunks + queries
                all_embeddings = await self._get_embeddings_batch(all_texts)
                doc_embeddings = np.array(all_embeddings[:len(doc_chunks)])
                query_embeddings = all_embeddings[len(doc_chunks):]
            else:
                logger.info("Processing document and queries in separate optimized batches")
                # Process concurrently but in separate batches
                doc_emb_task = asyncio.create_task(self._get_embeddings_batch(doc_chunks))
                query_emb_task = asyncio.create_task(self._get_embeddings_batch(queries))
                
                doc_embeddings, query_embeddings = await asyncio.gather(doc_emb_task, query_emb_task)
                doc_embeddings = np.array(doc_embeddings)
            
            embed_time = time.time() - embed_start
            logger.info(f"Embeddings generated in {embed_time:.2f}s")

            # 4. Process queries in parallel batches
            logger.info("Processing queries...")
            llm_start = time.time()
            
            # Process queries in batches of 5 for optimal balance of speed and accuracy
            batch_size = 5
            final_responses = {}
            
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i + batch_size]
                batch_query_embeddings = query_embeddings[i:i + batch_size]
                
                # Create tasks for parallel processing
                tasks = []
                for j, (query, query_emb) in enumerate(zip(batch_queries, batch_query_embeddings)):
                    # Find relevant chunks for this query
                    relevant_chunks = self._find_top_chunks_optimized(
                        query_emb, doc_embeddings, doc_chunks, top_k=5
                    )
                    
                    # Create prompt for this specific query
                    prompt = self._prepare_individual_query_prompt(query, relevant_chunks)
                    
                    # Create async task for this query
                    task = asyncio.create_task(self._call_llm_optimized(prompt, api_key))
                    tasks.append((i + j + 1, task))
                
                # Execute batch in parallel
                batch_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                # Process results
                for (query_num, _), result in zip(tasks, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Query {query_num} failed: {result}")
                        final_responses[str(query_num)] = f"Error processing query: {str(result)[:100]}"
                    else:
                        final_responses[str(query_num)] = result
                
                # Brief pause between batches to avoid rate limits
                if i + batch_size < len(queries):
                    await asyncio.sleep(0.5)
            
            llm_time = time.time() - llm_start
            total_time = time.time() - start_time
            
            logger.info(f"\n{'='*100}")
            logger.info(f"PARALLEL PROCESSING COMPLETE")
            logger.info(f"TOTAL_TIME: {total_time:.2f}s")
            logger.info(f"LLM_TIME: {llm_time:.2f}s")
            logger.info(f"QUERIES_PROCESSED: {len(final_responses)}")
            logger.info(f"AVERAGE_TIME_PER_QUERY: {llm_time/len(queries):.2f}s")
            logger.info(f"{'='*100}")
            
            return final_responses

        except Exception as e:
            logger.error(f"Critical error in parallel processing: {e}", exc_info=True)
            return {str(i+1): f"Processing failed: {str(e)[:100]}" for i in range(len(queries))}

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Main entry point - uses parallel processing for optimal performance."""
        return await self.process_queries_parallel(queries, document_link)

# Singleton instance for the application
llm_service = LLMService()