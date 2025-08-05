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
    "AIzaSyDRwSnA5trSGHSucMLUa5Yo_y43Q5bJ0wg",
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
        self.model_name = "gemini-2.5-flash-lits"
        self.embedding_model = "text-embedding-3-small"
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set. It is required for embeddings.")
        self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        self.max_tokens = 8196
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using generation model: {self.model_name}")
        logger.info(f"Using OpenAI embedding model: {self.embedding_model}")

    def _prepare_rag_prompt_for_all_queries(self, queries_with_context: List[Tuple[str, List[str]]]) -> str:
        """Prepares a single prompt containing all queries and their respective relevant chunks."""
        prompt_parts = [
            "You are a smart assistant helping a user extract information from a policy PDF.",
            "For each question below, use ONLY the provided 'Relevant Context' to answer clearly, concisely, and truthfully.",
            "If a question is unrelated to the document, provide a general answer. Do not say 'Not available'.",
            "\n--- BATCH OF QUESTIONS WITH RELEVANT CONTEXT ---\n"
        ]

        for i, (query, context_chunks) in enumerate(queries_with_context, 1):
            context_str = "\n---\n".join(context_chunks)
            prompt_parts.append(f"### Query {i}: {query}")
            prompt_parts.append(f"--- Relevant Context for Query {i} ---\n{context_str}")
            prompt_parts.append(f"--- End Context for Query {i} ---\n")

        prompt_parts.append(
            "Now, provide the answers for all queries above within 1200 characters in a numbered list corresponding to the query number. For example:\n"
            "1. [Your answer to Query 1]\n"
            "2. [Your answer to Query 2]\n"
        )
        return "\n".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, num_queries: int) -> Dict[str, str]:
        """Parses the LLM's numbered list response into a dictionary."""
        responses = {}
        answer_lines = re.findall(r"^\s*(\d+)\.\s*(.*)", response_text, re.MULTILINE)
        parsed_answers = {int(num): text.strip() for num, text in answer_lines}

        for i in range(num_queries):
            query_num = i + 1
            answer = parsed_answers.get(query_num, f"Error: No answer found for question {query_num} in the response.")
            responses[str(query_num)] = answer
            
        return responses

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using OpenAI's model."""
        try:
            response = await self.openai_client.embeddings.create(model=self.embedding_model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise APIError(f"Failed to get OpenAI embeddings: {e}")

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
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page in doc:
            text = page.get_text("text")
            text = text.strip().replace('\n', ' ')
            for i in range(0, len(text), max_chars):
                chunk = text[i:i + max_chars]
                if len(chunk.split()) > 5:
                    chunks.append(chunk)
        return chunks

    async def _call_llm_with_single_key(self, prompt: str, api_key: str) -> str:
        """Calls the Gemini model with a single, specific API key."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name)
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = await model.generate_content_async(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": self.max_tokens},
                safety_settings=safety_settings
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"API call failed with key ...{api_key[-4:]}: {e}")
            raise Exception(f"Gemini API call failed: {e}")

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Processes all queries in a single RAG-based batch."""
        if not queries:
            return {}

        try:
            # 1. Select the next Gemini API key for this entire request
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key_for_request = self.gemini_api_keys[key_index]
            logger.info(f"Processing request with Gemini key index {key_index} (...{api_key_for_request[-4:]})")

            # 2. Download and chunk document
            pdf_bytes = await asyncio.to_thread(self._download_pdf, document_link)
            doc_chunks = await asyncio.to_thread(self.extract_chunks, pdf_bytes)
            if not doc_chunks:
                raise ValueError("Failed to extract text chunks from the document.")

            # 3. RAG Pre-processing using OpenAI Embeddings
            doc_embeddings = np.array(await self._get_embeddings(doc_chunks))
            query_embeddings = await self._get_embeddings(queries)
            
            queries_with_context = []
            for i, query in enumerate(queries):
                query_emb = query_embeddings[i]
                # Find top 7 relevant chunks for this query
                relevant_chunks = self._find_top_chunks(query_emb, doc_embeddings, doc_chunks, top_k=7)
                queries_with_context.append((query, relevant_chunks))

            # 4. Prepare a single prompt with all queries and their relevant chunks
            prompt = self._prepare_rag_prompt_for_all_queries(queries_with_context)

            # 5. Make a single API call to Gemini
            llm_response_text = await self._call_llm_with_single_key(prompt, api_key_for_request)

            # 6. Parse the single response
            final_responses = self._parse_llm_response(llm_response_text, len(queries))
            
            return final_responses

        except Exception as e:
            logger.error(f"Critical error in process_queries: {e}", exc_info=True)
            return {str(i+1): f"Failed to process request: {e}" for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()