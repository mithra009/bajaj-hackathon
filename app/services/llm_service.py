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

        # Batch processing configuration
        self.max_batch_size = 10  # Maximum queries per batch
        self.max_tokens = 8196
        self.max_embedding_tokens_per_request = 2800000
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using generation model: {self.model_name}")
        logger.info(f"Using OpenAI embedding model: {self.embedding_model}")
        logger.info(f"Batch processing: max {self.max_batch_size} queries per batch")

    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image"""
        try:
            # Check common image hosting domains
            image_domains = ['ibb.co', 'imgur.com', 'postimg.cc', 'imageban.ru', 'imageshack.us']
            parsed = urlparse(url.lower())
            
            if any(domain in parsed.netloc for domain in image_domains):
                return True
                
            # Check file extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
            return any(url.lower().endswith(ext) for ext in image_extensions)
            
        except Exception:
            return False

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf') or 'pdf' in url.lower()
        
    def _get_url_type(self, url: str) -> str:
        """Determine the type of content at the given URL."""
        try:
            # Check common file extensions first
            url_lower = url.lower()
            if any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                return 'image'
            elif url_lower.endswith('.pdf'):
                return 'pdf'
            elif any(ext in url_lower for ext in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']):
                return 'document'
            
            # If extension check is inconclusive, try HEAD request
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                content_type = response.headers.get('content-type', '').lower()
                
                if 'image' in content_type:
                    return 'image'
                elif 'pdf' in content_type:
                    return 'pdf'
                elif any(doc_type in content_type for doc_type in ['word', 'excel', 'powerpoint', 'msword', 'spreadsheet', 'presentation']):
                    return 'document'
                elif 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                    return 'webpage'
                else:
                    return 'unknown'
            except:
                return 'unknown'
                
        except Exception as e:
            logger.warning(f"Error determining URL type for {url}: {e}")
            return 'unknown'

    def _prepare_batch_query_prompt(self, queries_with_context: List[Tuple[int, str, List[str]]]) -> str:
        """Prepares a batch prompt for multiple queries with their relevant contexts."""
        
        # Build the batch prompt
        prompt_parts = [
            "You are an expert insurance policy analyst. Answer multiple questions based on the provided contexts from the document.",
            "",
            "INSTRUCTIONS:",
            "- Answer each question directly and specifically based on its provided context",
            "- Reference specific policy terms, clauses, or procedures mentioned in the context", 
            "- If context contains the information, provide detailed answers",
            "- If some parts cannot be answered from context, use your general knowledge",
            "- Keep each answer comprehensive but concise (under 500 characters)",
            "- Do not use markdown formatting",
            "- Format your response as: ANSWER_[NUMBER]: [your answer]",
            "",
            "QUESTIONS AND CONTEXTS:",
            ""
        ]
        
        for query_num, query, context_chunks in queries_with_context:
            prompt_parts.append(f"QUESTION_{query_num}: {query}")
            prompt_parts.append("CONTEXT:")
            for i, chunk in enumerate(context_chunks[:7]):  # Limit to top 3 chunks per query
                prompt_parts.append(f"  Context {i+1}: {chunk}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "RESPONSES:",
            "Please provide answers in the format ANSWER_[NUMBER]: [your answer]",
            ""
        ])
        
        return "\n".join(prompt_parts)

    def _parse_batch_response(self, response_text: str, query_numbers: List[int]) -> Dict[str, str]:
        """Parse the batch response to extract individual answers."""
        answers = {}
        
        try:
            # Split response by lines and look for ANSWER_ patterns
            lines = response_text.split('\n')
            current_answer = ""
            current_num = None
            
            for line in lines:
                line = line.strip()
                
                # Check if line starts with ANSWER_
                answer_match = re.match(r'ANSWER[_\s]*(\d+)[:\s]*(.+)', line, re.IGNORECASE)
                if answer_match:
                    # Save previous answer if exists
                    if current_num is not None and current_answer.strip():
                        answers[str(current_num)] = current_answer.strip()
                    
                    # Start new answer
                    current_num = int(answer_match.group(1))
                    current_answer = answer_match.group(2).strip()
                elif current_num is not None and line:
                    # Continue building current answer
                    current_answer += " " + line
            
            # Save the last answer
            if current_num is not None and current_answer.strip():
                answers[str(current_num)] = current_answer.strip()
            
            # Fallback parsing if the above doesn't work well
            if not answers:
                # Try to extract answers by splitting on ANSWER_ keywords
                answer_blocks = re.split(r'ANSWER[_\s]*\d+[:\s]*', response_text, flags=re.IGNORECASE)
                if len(answer_blocks) > 1:  # First block is usually empty
                    for i, block in enumerate(answer_blocks[1:], 1):
                        if i <= len(query_numbers):
                            answer = block.strip().split('\n')[0] if block.strip() else "No answer provided"
                            answers[str(query_numbers[i-1])] = answer[:500]  # Limit length
            
            # Ensure all query numbers have answers
            for num in query_numbers:
                if str(num) not in answers:
                    answers[str(num)] = "Unable to extract answer from batch response"
                    
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            # Fallback: split response roughly by number of queries
            parts = response_text.split('\n\n') if '\n\n' in response_text else [response_text]
            for i, num in enumerate(query_numbers):
                if i < len(parts):
                    answers[str(num)] = parts[i]
                else:
                    answers[str(num)] = "Unable to parse answer from batch response"
        
        return answers

    def _estimate_token_count(self, text: str) -> int:
        """Accurate token estimation for text-embedding-3-small."""
        return int((len(text) / 4) * 1.1)

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Token-optimized embedding generation with maximum batch efficiency."""
        if not texts:
            return []

        try:
            MAX_TOKENS_PER_REQUEST = 2800000
            all_embeddings = []
            current_batch = []
            current_batch_tokens = 0
            
            logger.info(f"Processing {len(texts)} texts for embeddings...")
            
            for i, text in enumerate(texts):
                text_tokens = self._estimate_token_count(text)
                
                if text_tokens > MAX_TOKENS_PER_REQUEST:
                    logger.warning(f"Text {i} exceeds token limit ({text_tokens} tokens), truncating...")
                    truncated_text = text[:11200000]
                    text_tokens = self._estimate_token_count(truncated_text)
                    text = truncated_text
                
                if current_batch and (current_batch_tokens + text_tokens > MAX_TOKENS_PER_REQUEST):
                    logger.info(f"Processing batch with {len(current_batch)} texts ({current_batch_tokens:,} tokens)")
                    batch_embeddings = await self._process_embedding_batch(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    
                    current_batch = [text]
                    current_batch_tokens = text_tokens
                else:
                    current_batch.append(text)
                    current_batch_tokens += text_tokens
            
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
                wait_time = (2 ** attempt) * 0.5
                logger.warning(f"Embedding batch attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for batch")
                    return [[0.0] * 1536] * len(batch_texts)
                else:
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

    def _find_top_chunks_optimized(self, query_emb: List[float], chunk_embs: np.ndarray, chunks: List[str], top_k: int = 5) -> List[str]:
        """Optimized chunk selection with better relevance scoring."""
        if len(chunk_embs) == 0:
            return chunks[:top_k] if chunks else []
            
        try:
            sims = cosine_similarity([query_emb], chunk_embs)[0]
            top_idxs = sims.argsort()[-top_k:][::-1]
            
            relevant_chunks = []
            for idx in top_idxs:
                if sims[idx] > 0.1:
                    relevant_chunks.append(chunks[idx])
            
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
            
            total_pages = len(doc)
            if total_pages > 100:
                max_chars = 1500
                logger.info(f"Large document detected ({total_pages} pages), using larger chunks")
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if not text:
                    continue
                
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\-\%\$]', ' ', text)
                
                if total_pages > 50:
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    current_chunk = ""
                    for para in paragraphs:
                        if len(current_chunk) + len(para) + 2 > max_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = para
                            else:
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
                                words = sentence.split()
                                temp_chunk = ""
                                for word in words:
                                    if len(temp_chunk) + len(word) + 1 > max_chars:
                                        if temp_chunk:
                                            chunks.append(temp_chunk.strip())
                                            temp_chunk = word
                                        else:
                                            chunks.append(word)
                                    else:
                                        temp_chunk += " " + word if temp_chunk else word
                                if temp_chunk:
                                    current_chunk = temp_chunk
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                if len(chunks) > 1000:
                    logger.warning(f"Reached chunk limit at page {page_num + 1}")
                    break
            
            min_chunk_size = 100 if total_pages > 100 else 50
            chunks = [chunk for chunk in chunks if len(chunk) > min_chunk_size]
            
            logger.info(f"Extracted {len(chunks)} optimized chunks from {total_pages}-page document")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise APIError(f"Failed to extract text from PDF: {e}")

    async def _call_llm_batch(self, prompt: str, api_key: str, timeout: float = 30.0) -> str:
        """Enhanced LLM call for batch processing with increased timeout."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 4096,  # Increased for batch responses
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

            response = await asyncio.wait_for(
                model.generate_content_async(prompt),
                timeout=timeout
            )
            return response.text.strip() if response.text else "Unable to generate batch response"
            
        except asyncio.TimeoutError:
            logger.error(f"Batch API call timeout with key ...{api_key[-4:]}")
            raise Exception(f"Batch request timeout after {timeout}s - please try again")
        except Exception as e:
            logger.error(f"Batch API call failed with key ...{api_key[-4:]}: {e}")
            raise Exception(f"Gemini batch API call failed: {str(e)[:100]}")

    def _download_pdf_optimized(self, url: str) -> bytes:
        """Optimized PDF download with better error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*'
            }
            
            response = requests.get(url, timeout=15, headers=headers, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"Unexpected content type: {content_type}")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise APIError(f"Failed to download PDF: {e}")

    async def process_queries_with_batch_processing(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Process queries using batch processing - multiple queries per API call."""
        start_time = time.time()
        
        if not queries:
            return {}

        try:
            logger.info(f"\n{'='*100}")
            logger.info(f"BATCH PROCESSING {len(queries)} QUERIES")
            logger.info(f"BATCH SIZE: {self.max_batch_size}")
            logger.info(f"DOCUMENT: {document_link}")
            logger.info(f"{'='*100}")

            # 1. Download and process document
            logger.info("Downloading and processing document...")
            download_start = time.time()
            pdf_bytes = await asyncio.to_thread(self._download_pdf_optimized, document_link)
            doc_chunks = await asyncio.to_thread(self.extract_chunks_optimized, pdf_bytes)
            
            if not doc_chunks:
                raise ValueError("No text extracted from document")
                
            logger.info(f"Document processed in {time.time() - download_start:.2f}s | Chunks: {len(doc_chunks)}")

            # 2. Generate embeddings
            logger.info("Generating embeddings...")
            embed_start = time.time()
            
            total_doc_tokens = sum(self._estimate_token_count(chunk) for chunk in doc_chunks)
            total_query_tokens = sum(self._estimate_token_count(q) for q in queries)
            
            logger.info(f"Document tokens: {total_doc_tokens:,}, Query tokens: {total_query_tokens:,}")
            
            if total_doc_tokens + total_query_tokens < self.max_embedding_tokens_per_request:
                all_texts = doc_chunks + queries
                all_embeddings = await self._get_embeddings_batch(all_texts)
                doc_embeddings = np.array(all_embeddings[:len(doc_chunks)])
                query_embeddings = all_embeddings[len(doc_chunks):]
            else:
                doc_emb_task = asyncio.create_task(self._get_embeddings_batch(doc_chunks))
                query_emb_task = asyncio.create_task(self._get_embeddings_batch(queries))
                
                doc_embeddings, query_embeddings = await asyncio.gather(doc_emb_task, query_emb_task)
                doc_embeddings = np.array(doc_embeddings)
            
            embed_time = time.time() - embed_start
            logger.info(f"Embeddings generated in {embed_time:.2f}s")

            # 3. Process queries in batches
            logger.info("Processing queries in batches...")
            batch_start = time.time()
            
            final_responses = {}
            num_batches = (len(queries) + self.max_batch_size - 1) // self.max_batch_size
            
            # Process batches in parallel
            batch_tasks = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.max_batch_size
                end_idx = min(start_idx + self.max_batch_size, len(queries))
                
                batch_queries = queries[start_idx:end_idx]
                batch_query_embeddings = query_embeddings[start_idx:end_idx]
                batch_query_numbers = list(range(start_idx + 1, end_idx + 1))
                
                # Prepare queries with context for this batch
                queries_with_context = []
                for i, (query, query_emb) in enumerate(zip(batch_queries, batch_query_embeddings)):
                    relevant_chunks = self._find_top_chunks_optimized(
                        query_emb, doc_embeddings, doc_chunks, top_k=3  # Reduced for batch processing
                    )
                    queries_with_context.append((batch_query_numbers[i], query, relevant_chunks))
                
                # Create batch task
                batch_task = asyncio.create_task(self._process_batch(
                    queries_with_context, batch_query_numbers, batch_idx + 1
                ))
                batch_tasks.append(batch_task)
            
            # Execute all batches in parallel
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed: {result}")
                    # Add error responses for failed batch
                    continue
                else:
                    final_responses.update(result)
            
            batch_time = time.time() - batch_start
            total_time = time.time() - start_time
            
            logger.info(f"\n{'='*100}")
            logger.info(f"BATCH PROCESSING COMPLETE")
            logger.info(f"TOTAL_TIME: {total_time:.2f}s")
            logger.info(f"BATCH_TIME: {batch_time:.2f}s") 
            logger.info(f"BATCHES_PROCESSED: {num_batches}")
            logger.info(f"QUERIES_PROCESSED: {len(final_responses)}")
            logger.info(f"AVERAGE_TIME_PER_BATCH: {batch_time/num_batches:.2f}s")
            logger.info(f"{'='*100}")
            
            return final_responses

        except Exception as e:
            logger.error(f"Critical error in batch processing: {e}", exc_info=True)
            return {str(i+1): f"Processing failed: {str(e)[:100]}" for i in range(len(queries))}

    async def _prepare_direct_url_prompt(self, queries: List[str], url: str, url_type: str) -> str:
        """Prepare a prompt for direct URL processing."""
        url_type_descriptions = {
            'image': 'image',
            'document': 'document (Word, Excel, PowerPoint, etc.)',
            'webpage': 'webpage',
            'unknown': 'URL content'
        }
        
        content_type = url_type_descriptions.get(url_type, 'URL content')
        
        prompt_parts = [
            f"You are an expert analyst. Answer the following questions based on the {content_type} at this URL: {url}",
            "",
            "INSTRUCTIONS:",
            "- Answer each question directly and specifically",
            "- If the content is not accessible, state that clearly",
            "- Keep answers concise but complete (under 500 characters each)",
            "- Format your response as: ANSWER_[NUMBER]: [your answer]",
            "",
            "QUESTIONS:",
            ""
        ]
        
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}")
        
        prompt_parts.extend([
            "",
            "RESPONSES:",
            "Provide your answers in the format ANSWER_[NUMBER]: [your answer]"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _process_direct_url(self, queries: List[str], url: str, url_type: str) -> Dict[str, str]:
        """Process queries by sending them directly to Gemini with the URL."""
        try:
            # Get API key
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key = self.gemini_api_keys[key_index]
            
            # Prepare the prompt
            prompt = await self._prepare_direct_url_prompt(queries, url, url_type)
            
            # Call Gemini API
            response_text = await self._call_llm_batch(prompt, api_key)
            
            # Parse the response
            query_numbers = list(range(1, len(queries) + 1))
            parsed_answers = self._parse_batch_response(response_text, query_numbers)
            
            # Ensure all queries have answers
            results = {}
            for i, query in enumerate(queries, 1):
                results[str(i)] = parsed_answers.get(str(i), f"Could not process answer for query {i}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing direct URL: {e}")
            return {str(i+1): f"Error processing {url_type} URL: {str(e)[:100]}" for i in range(len(queries))}
    
    async def _process_batch(self, queries_with_context: List[Tuple[int, str, List[str]]], 
                           query_numbers: List[int], batch_num: int) -> Dict[str, str]:
        """Process a single batch of queries."""
        try:
            logger.info(f"Processing batch {batch_num} with {len(queries_with_context)} queries")
            
            # Get API key for this batch
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key = self.gemini_api_keys[key_index]
            
            # Create batch prompt
            batch_prompt = self._prepare_batch_query_prompt(queries_with_context)
            
            # Call LLM with batch prompt
            response_text = await self._call_llm_batch(batch_prompt, api_key)
            
            # Parse batch response
            batch_responses = self._parse_batch_response(response_text, query_numbers)
            
            logger.info(f"Batch {batch_num} completed successfully")
            return batch_responses
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            # Return error responses for this batch
            return {str(num): f"Batch processing error: {str(e)[:100]}" for num in query_numbers}

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """
        Main entry point for processing queries.
        
        For PDFs: Uses batch processing with chunking and embeddings.
        For other URLs: Sends directly to Gemini with appropriate context.
        """
        # Log the document URL and queries at the start
        logger.info(f"\n{'='*100}")
        logger.info(f"PROCESSING NEW REQUEST")
        logger.info(f"{'='*100}")
        logger.info(f"DOCUMENT_URL: {document_link}")
        logger.info(f"QUERY_COUNT: {len(queries)}")
        for i, query in enumerate(queries, 1):
            logger.info(f"QUERY_{i}: {query}")
        logger.info(f"{'='*100}\n")
        
        start_time = time.time()
        try:
            # Determine URL type
            url_type = self._get_url_type(document_link)
            logger.info(f"Detected URL type: {url_type}")
            
            if url_type == 'pdf':
                # Use batch processing for PDFs
                logger.info("Processing as PDF with chunking and embeddings...")
                results = await self.process_queries_with_batch_processing(queries, document_link)
            else:
                # For non-PDF URLs, process directly
                logger.info(f"Processing as {url_type} URL with direct Gemini call...")
                results = await self._process_direct_url(queries, document_link, url_type)
            
            # Log the responses
            logger.info(f"\n{'='*100}")
            logger.info("RESPONSES GENERATED")
            logger.info(f"{'='*100}")
            for i, (query, response) in enumerate(zip(queries, results.values()), 1):
                logger.info(f"\nQUERY_{i}: {query}")
                logger.info(f"RESPONSE_{i}: {response}")
            
            total_time = time.time() - start_time
            logger.info(f"\nProcessing completed in {total_time:.2f} seconds")
            logger.info(f"{'='*100}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing queries: {str(e)}", exc_info=True)
            # Return error responses for all queries
            return {str(i+1): f"Error processing request: {str(e)[:200]}" for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()