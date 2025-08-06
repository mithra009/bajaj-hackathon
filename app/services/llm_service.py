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
        self.max_batch_size = 5  # Reduced batch size for better parsing
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

    def _prepare_batch_query_prompt(self, queries_with_context: List[Tuple[int, str, List[str]]]) -> str:
        """Improved batch prompt with clearer formatting requirements."""
        
        prompt_parts = [
            "You are an expert document analyst. Answer multiple questions based on the provided contexts.",
            "",
            "CRITICAL FORMATTING INSTRUCTIONS:",
            "- You must respond with a valid JSON object only",
            "- Use this exact format: {\"answers\": [\"answer1\", \"answer2\", \"answer3\"]}",
            "- Each answer should be a complete string within quotes",
            "- Do not include question numbers in answers",
            "- Do not use any other formatting or explanations outside the JSON",
            "- Keep answers concise but complete (under 300 words each)",
            "",
            "QUESTIONS AND CONTEXTS:",
            ""
        ]
        
        for query_num, query, context_chunks in queries_with_context:
            prompt_parts.append(f"Question {query_num}: {query}")
            prompt_parts.append("Context:")
            for i, chunk in enumerate(context_chunks[:3]):
                prompt_parts.append(f"  - {chunk}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "RESPONSE FORMAT:",
            "Respond only with a JSON object in this exact format:",
            "{\"answers\": [\"answer for question 1\", \"answer for question 2\", \"answer for question 3\"]}",
            "",
            "Each answer should directly address its corresponding question based on the context provided.",
            ""
        ])
        
        return "\n".join(prompt_parts)

    def _prepare_url_batch_query_prompt(self, batch_queries: List[str], query_numbers: List[int], document_url: str) -> str:
        """Improved URL batch prompt with better formatting."""
        
        prompt_parts = [
            "You are an expert document analyst. Answer multiple questions based on the document at the provided URL.",
            "",
            f"DOCUMENT URL: {document_url}",
            "",
            "CRITICAL FORMATTING INSTRUCTIONS:",
            "- You must respond with a valid JSON object only",
            "- Use this exact format: {\"answers\": [\"answer1\", \"answer2\", \"answer3\"]}",
            "- Each answer should be a complete string within quotes",
            "- Do not include question numbers in answers",
            "- Do not use any other formatting or explanations outside the JSON",
            "- Keep answers concise but complete (under 300 words each)",
            "",
            "QUESTIONS:",
            ""
        ]
        
        for i, query in enumerate(batch_queries):
            prompt_parts.append(f"Question {i+1}: {query}")
        
        prompt_parts.extend([
            "",
            "RESPONSE FORMAT:",
            "Access the document at the URL and respond only with a JSON object:",
            "{\"answers\": [\"answer for question 1\", \"answer for question 2\", \"answer for question 3\"]}",
            "",
            "Each answer should directly address its corresponding question based on the document content.",
            ""
        ])
        
        return "\n".join(prompt_parts)

    def _prepare_image_batch_query_prompt(self, batch_queries: List[str], query_numbers: List[int], image_url: str) -> str:
        """New method for handling image URLs."""
        
        prompt_parts = [
            "You are an expert image and document analyst. Answer multiple questions based on the image at the provided URL.",
            "",
            f"IMAGE URL: {image_url}",
            "",
            "CRITICAL FORMATTING INSTRUCTIONS:",
            "- You must respond with a valid JSON object only",
            "- Use this exact format: {\"answers\": [\"answer1\", \"answer2\", \"answer3\"]}",
            "- Each answer should be a complete string within quotes",
            "- Do not include question numbers in answers",
            "- Do not use any other formatting or explanations outside the JSON",
            "- Keep answers concise but complete (under 300 words each)",
            "",
            "QUESTIONS:",
            ""
        ]
        
        for i, query in enumerate(batch_queries):
            prompt_parts.append(f"Question {i+1}: {query}")
        
        prompt_parts.extend([
            "",
            "RESPONSE FORMAT:",
            "Analyze the image at the URL and respond only with a JSON object:",
            "{\"answers\": [\"answer for question 1\", \"answer for question 2\", \"answer for question 3\"]}",
            "",
            "Each answer should directly address its corresponding question based on the image content.",
            ""
        ])
        
        return "\n".join(prompt_parts)

    def _clean_and_extract_json(self, response_text: str) -> Dict[str, Any]:
        """Extract and clean JSON from response text."""
        try:
            # Remove any markdown code blocks
            response_text = re.sub(r'```(?:json)?', '', response_text)
            response_text = response_text.strip()
            
            # Find JSON object boundaries
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                
                # Clean up common issues
                json_text = json_text.replace('\\"', '"')
                json_text = json_text.replace("'", '"')
                json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                
                return json.loads(json_text)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract JSON: {e}")
            return None

    def _parse_batch_response(self, response_text: str, query_numbers: List[int]) -> Dict[str, str]:
        """Improved batch response parsing with better JSON handling."""
        answers = {}
        
        try:
            logger.info(f"Parsing response for {len(query_numbers)} queries")
            logger.debug(f"Raw response: {response_text[:500]}...")
            
            # Try to extract JSON first
            parsed_json = self._clean_and_extract_json(response_text)
            
            if parsed_json and 'answers' in parsed_json:
                answers_list = parsed_json['answers']
                logger.info(f"Successfully parsed JSON with {len(answers_list)} answers")
                
                for i, answer in enumerate(answers_list):
                    if i < len(query_numbers):
                        if isinstance(answer, str) and answer.strip():
                            # Clean the answer text
                            clean_answer = answer.strip()
                            clean_answer = re.sub(r'\s+', ' ', clean_answer)
                            clean_answer = clean_answer.replace('\\"', '"').replace("\\'", "'")
                            answers[str(query_numbers[i])] = clean_answer
                        else:
                            answers[str(query_numbers[i])] = "No valid answer provided"
                
                # Ensure all queries have answers
                for num in query_numbers:
                    if str(num) not in answers:
                        answers[str(num)] = "Answer not found in response"
                
                return answers
            
            # Fallback parsing methods
            logger.warning("JSON parsing failed, trying alternative methods")
            
            # Try to find answers in array format
            array_match = re.search(r'\["([^"]*)"(?:,\s*"([^"]*)")*\]', response_text)
            if array_match:
                found_answers = re.findall(r'"([^"]*)"', array_match.group(0))
                for i, answer in enumerate(found_answers):
                    if i < len(query_numbers):
                        answers[str(query_numbers[i])] = answer.strip()
            
            # Final fallback: split by common delimiters
            if not answers:
                # Try splitting by numbered patterns
                parts = re.split(r'(?:ANSWER[_\s]*\d+[:\s]*|Question\s*\d+[:\s]*)', response_text, flags=re.IGNORECASE)
                parts = [p.strip() for p in parts if p.strip()]
                
                for i, part in enumerate(parts[:len(query_numbers)]):
                    if i < len(query_numbers):
                        clean_part = re.sub(r'\s+', ' ', part)[:500]  # Limit length
                        answers[str(query_numbers[i])] = clean_part
            
            # Ensure all query numbers have answers
            for num in query_numbers:
                if str(num) not in answers:
                    answers[str(num)] = "Unable to parse answer from response"
                    
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            # Emergency fallback
            for i, num in enumerate(query_numbers):
                answers[str(num)] = f"Parsing error: {str(e)[:100]}"
        
        logger.info(f"Final parsed answers count: {len(answers)}")
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

    async def _call_llm_batch(self, prompt: str, api_key: str, timeout: float = 45.0) -> str:
        """Enhanced LLM call for batch processing with increased timeout and better settings."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.1,  # Lower temperature for more consistent formatting
                    "max_output_tokens": 4096,
                    "top_p": 0.8,
                    "top_k": 20
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
            
            result = response.text.strip() if response.text else ""
            logger.info(f"LLM response length: {len(result)} characters")
            return result
            
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
        """Process queries using batch processing with improved URL type detection."""
        start_time = time.time()
        
        if not queries:
            return {}

        try:
            logger.info(f"\n{'='*100}")
            logger.info(f"BATCH PROCESSING {len(queries)} QUERIES")
            logger.info(f"BATCH SIZE: {self.max_batch_size}")
            logger.info(f"DOCUMENT: {document_link}")
            logger.info(f"{'='*100}")

            # Determine document type
            is_pdf = self._is_pdf_url(document_link)
            is_image = self._is_image_url(document_link)
            
            logger.info(f"Document type - PDF: {is_pdf}, Image: {is_image}")

            doc_chunks = []
            doc_embeddings = np.array([])
            
            if is_pdf:
                try:
                    # Process PDF document
                    logger.info("Processing PDF document...")
                    download_start = time.time()
                    pdf_bytes = await asyncio.to_thread(self._download_pdf_optimized, document_link)
                    doc_chunks = await asyncio.to_thread(self.extract_chunks_optimized, pdf_bytes)
                    
                    if not doc_chunks:
                        raise ValueError("No text extracted from PDF document")
                        
                    logger.info(f"PDF processed in {time.time() - download_start:.2f}s | Chunks: {len(doc_chunks)}")

                    # Generate embeddings for PDF content
                    logger.info("Generating embeddings for PDF...")
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
                    
                except Exception as pdf_error:
                    logger.warning(f"Failed to process as PDF: {pdf_error}")
                    logger.info("Switching to direct URL processing...")
                    is_pdf = False
            
            if not is_pdf:
                # For non-PDF documents (including images), just get query embeddings
                logger.info("Processing as direct URL/image")
                query_embeddings = await self._get_embeddings_batch(queries)

            # Process queries in batches
            logger.info("Processing queries in batches...")
            batch_start = time.time()
            
            final_responses = {}
            num_batches = (len(queries) + self.max_batch_size - 1) // self.max_batch_size
            
            batch_tasks = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.max_batch_size
                end_idx = min(start_idx + self.max_batch_size, len(queries))
                
                batch_queries = queries[start_idx:end_idx]
                batch_query_numbers = list(range(start_idx + 1, end_idx + 1))
                
                if is_pdf:
                    # PDF processing with context
                    batch_query_embeddings = query_embeddings[start_idx:end_idx]
                    queries_with_context = []
                    for i, (query, query_emb) in enumerate(zip(batch_queries, batch_query_embeddings)):
                        relevant_chunks = self._find_top_chunks_optimized(
                            query_emb, doc_embeddings, doc_chunks, top_k=3
                        )
                        queries_with_context.append((batch_query_numbers[i], query, relevant_chunks))
                    
                    batch_task = asyncio.create_task(self._process_batch(
                        queries_with_context, batch_query_numbers, batch_idx + 1
                    ))
                elif is_image:
                    # Image processing
                    batch_task = asyncio.create_task(self._process_image_batch(
                        batch_queries, batch_query_numbers, document_link, batch_idx + 1
                    ))
                else:
                    # URL processing
                    batch_task = asyncio.create_task(self._process_url_batch(
                        batch_queries, batch_query_numbers, document_link, batch_idx + 1
                    ))
                
                batch_tasks.append(batch_task)
            
            # Execute all batches in parallel
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Combine results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed: {result}")
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

    async def _process_batch(self, queries_with_context: List[Tuple[int, str, List[str]]], 
                           query_numbers: List[int], batch_num: int) -> Dict[str, str]:
        """Process a single batch of queries with PDF context."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing PDF batch {batch_num} (attempt {attempt + 1}) with {len(queries_with_context)} queries")
                
                # Get API key for this batch
                key_index = get_next_key_index(len(self.gemini_api_keys))
                api_key = self.gemini_api_keys[key_index]
                
                # Create batch prompt
                batch_prompt = self._prepare_batch_query_prompt(queries_with_context)
                
                # Call LLM with batch prompt
                response_text = await self._call_llm_batch(batch_prompt, api_key)
                
                # Parse batch response
                batch_responses = self._parse_batch_response(response_text, query_numbers)
                
                logger.info(f"PDF batch {batch_num} completed successfully")
                return batch_responses
                
            except Exception as e:
                logger.error(f"Error processing PDF batch {batch_num} attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Final attempt failed - return error responses
                    return {str(num): f"PDF batch processing error: {str(e)[:100]}" for num in query_numbers}
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _process_url_batch(self, batch_queries: List[str], query_numbers: List[int], 
                               document_url: str, batch_num: int) -> Dict[str, str]:
        """Process a batch of queries with direct URL (non-PDF documents)."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing URL batch {batch_num} (attempt {attempt + 1}) with {len(batch_queries)} queries")
                
                # Get API key for this batch
                key_index = get_next_key_index(len(self.gemini_api_keys))
                api_key = self.gemini_api_keys[key_index]
                
                # Create batch prompt for URL processing
                batch_prompt = self._prepare_url_batch_query_prompt(batch_queries, query_numbers, document_url)
                
                # Call LLM with batch prompt
                response_text = await self._call_llm_batch(batch_prompt, api_key)
                
                # Parse batch response
                batch_responses = self._parse_batch_response(response_text, query_numbers)
                
                logger.info(f"URL batch {batch_num} completed successfully")
                return batch_responses
                
            except Exception as e:
                logger.error(f"Error processing URL batch {batch_num} attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {str(num): f"URL batch processing error: {str(e)[:100]}" for num in query_numbers}
                else:
                    await asyncio.sleep(2 ** attempt)

    async def _process_image_batch(self, batch_queries: List[str], query_numbers: List[int], 
                                 image_url: str, batch_num: int) -> Dict[str, str]:
        """Process a batch of queries with image URL."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing image batch {batch_num} (attempt {attempt + 1}) with {len(batch_queries)} queries")
                
                # Get API key for this batch
                key_index = get_next_key_index(len(self.gemini_api_keys))
                api_key = self.gemini_api_keys[key_index]
                
                # Create batch prompt for image processing
                batch_prompt = self._prepare_image_batch_query_prompt(batch_queries, query_numbers, image_url)
                
                # Call LLM with batch prompt
                response_text = await self._call_llm_batch(batch_prompt, api_key)
                
                # Parse batch response
                batch_responses = self._parse_batch_response(response_text, query_numbers)
                
                logger.info(f"Image batch {batch_num} completed successfully")
                return batch_responses
                
            except Exception as e:
                logger.error(f"Error processing image batch {batch_num} attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return {str(num): f"Image batch processing error: {str(e)[:100]}" for num in query_numbers}
                else:
                    await asyncio.sleep(2 ** attempt)

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Main entry point - uses batch processing for optimal performance."""
        result = await self.process_queries_with_batch_processing(queries, document_link)
        
        # Convert the result to the expected format
        answers_list = []
        for i in range(1, len(queries) + 1):
            answer = result.get(str(i), "No answer available")
            answers_list.append(answer)
        
        # Return in the expected format
        return {"answers": answers_list}

# Singleton instance for the application
llm_service = LLMService()