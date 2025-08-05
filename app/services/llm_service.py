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
from google.generativeai.types import HarmCategory, HarmBlockThreshold, generation
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

# --- API KEY CONFIGURATION ---
# Gemini API keys for generation
GEMINI_KEYS = [
    "AIzaSyDRwSnA5trSGHSucMLUa5Yo_y43Q5bJ0wg",
    "AIzaSyAHG_guIGql9JG5NaBiRQpHmEQ9O09Dfoo",
    "AIzaSyBXVE_Zo_XsjvilpzVjugIe3wg9ZWe62vM",
    "AIzaSyBCPf3_VWZBQ4tJPGE8fSM9MBXV70ccPLw",
    "AIzaSyDR9Xw3WtwqlN2uB8SNMog9wpVfXtr7L9I"
]

# OpenAI API key for embeddings
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
        self.model_name = "gemini-2.5-flash-lite"
        self.embedding_model = "text-embedding-3-small"
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set. It is required for embeddings.")
        self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        self.max_tokens = 8196
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using generation model: {self.model_name}")
        logger.info(f"Using OpenAI embedding model: {self.embedding_model}")

    def _prepare_prompt_for_batch(self, batch_data: List[Tuple[str, int, List[str]]]) -> str:
        """Prepares a single prompt for a batch of queries, each with its own context."""
        prompt_parts = [
            "You are a smart assistant helping a user extract information from a policy PDF.",
            "For each question below, use the provided 'Relevant Context' to answer clearly, concisely, and truthfully.",
            "If a question is unrelated to the document, provide a general answer. Do not say 'Not available'.",
            "The response for each question should be a single paragraph within 700 characters.",
            "\n--- BATCH OF QUESTIONS ---\n"
        ]

        for i, (query, original_index, context_chunks) in enumerate(batch_data, 1):
            context_str = "\n---\n".join(context_chunks)
            prompt_parts.append(f"### Query {i} (Original Index: {original_index}): {query}")
            prompt_parts.append(f"--- Relevant Context for Query {i} ---\n{context_str}")
            prompt_parts.append(f"--- End Context for Query {i} ---\n")

        prompt_parts.append(
            "Now, provide the answers for all queries above in a numbered list, corresponding to the query number in this batch (Query 1, Query 2, etc.). For example:\n"
            "1. [Your answer to Query 1]\n"
            "2. [Your answer to Query 2]\n"
        )
        return "\n".join(prompt_parts)

    def _parse_llm_batch_response(self, response_text: str, batch_data: List[Tuple[str, int, List[str]]]) -> Dict[str, str]:
        """Parses the LLM's numbered list response into a dictionary."""
        responses = {}
        answer_lines = re.findall(r"^\s*(\d+)\.\s*(.*)", response_text, re.MULTILINE)
        parsed_answers = {int(num): text.strip() for num, text in answer_lines}

        for i, (query, original_index, chunks) in enumerate(batch_data, 1):
            answer = parsed_answers.get(i, f"Error: No answer found for question {i} in the batch response.")
            responses[str(original_index + 1)] = answer
            
        return responses

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using OpenAI's model."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise APIError(f"Failed to get OpenAI embeddings: {e}")

    def _find_top_chunks(self, query_emb: List[float], chunk_embs: np.ndarray, chunks: List[str], top_k: int = 3) -> List[str]:
        """Finds the top-k most relevant chunks for a query using cosine similarity."""
        sims = cosine_similarity([query_emb], chunk_embs)[0]
        top_idxs = sims.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_idxs]

    def _is_valid_url(self, url: str) -> bool:
        return all(list(urlparse(url))[:2])

    def _download_pdf(self, url: str) -> bytes:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    def extract_chunks(self, pdf_bytes: bytes, max_chars: int = 1000) -> List[str]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chunks = []
        for page in doc:
            text = page.get_text().strip().replace('\n', ' ')
            for i in range(0, len(text), max_chars):
                chunk = text[i:i + max_chars]
                if len(chunk.split()) > 5:
                    chunks.append(chunk)
        return chunks

    async def _call_llm(self, prompt: str) -> str:
        """Calls the Gemini model with stateless, randomized API key rotation."""
        shuffled_keys = self.gemini_api_keys.copy()
        random.shuffle(shuffled_keys)
        last_error = None

        for key in shuffled_keys:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(self.model_name)
                
                # --- FIX: Correctly defined safety settings to prevent blocking ---
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }

                response = await model.generate_content_async(
                    prompt,
                    generation_config={"temperature": 0.3, "max_output_tokens": self.max_tokens},
                    safety_settings=safety_settings
                )
                return response.text.strip()
            
            # --- IMPROVEMENT: More specific error handling for blocked prompts ---
            except generation.BlockedPromptError as e:
                logger.error(f"Request was blocked by Gemini API with key ...{key[-4:]}. Reason: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"API call failed with key ...{key[-4:]}: {e}")
                last_error = e

        raise Exception(f"All API keys failed. Last error: {last_error}")

    async def _process_batch(self, batch_data: List[Tuple[str, int, List[str]]], semaphore: asyncio.Semaphore) -> Dict[str, str]:
        """Processes a single batch of queries."""
        async with semaphore:
            try:
                prompt = self._prepare_prompt_for_batch(batch_data)
                llm_response_text = await self._call_llm(prompt)
                return self._parse_llm_batch_response(llm_response_text, batch_data)
            except Exception as e:
                logger.error(f"Error processing a batch: {e}", exc_info=True)
                return {str(original_index + 1): f"Error processing batch: {e}" for query, original_index, chunks in batch_data}

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """Processes multiple queries against a document in batches using RAG with OpenAI embeddings."""
        if not queries:
            return {}

        try:
            pdf_bytes = await asyncio.to_thread(self._download_pdf, document_link)
            doc_chunks = await asyncio.to_thread(self.extract_chunks, pdf_bytes)
            if not doc_chunks:
                raise ValueError("Failed to extract text chunks from the document.")

            doc_embeddings = np.array(await self._get_embeddings(doc_chunks))
            query_embeddings = await self._get_embeddings(queries)

            query_data_with_context = []
            for i, query in enumerate(queries):
                query_emb = query_embeddings[i]
                relevant_chunks = self._find_top_chunks(query_emb, doc_embeddings, doc_chunks)
                query_data_with_context.append((query, i, relevant_chunks))

            batch_size = 15
            batches = [query_data_with_context[i:i + batch_size] for i in range(0, len(query_data_with_context), batch_size)]
            
            semaphore = asyncio.Semaphore(len(self.gemini_api_keys))
            batch_tasks = [self._process_batch(batch, semaphore) for batch in batches]
            batch_results = await asyncio.gather(*batch_tasks)

            final_responses = {}
            for result_dict in batch_results:
                final_responses.update(result_dict)
            
            return final_responses

        except Exception as e:
            logger.error(f"Critical error in process_queries: {e}", exc_info=True)
            return {str(i+1): f"Failed to process request: {e}" for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()