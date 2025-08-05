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

# Load environment variables
load_dotenv()

# --- STATEFUL KEY ROTATION SETUP ---
KEY_INDEX_FILE = Path("/app/data/api_key_index.json")

def get_next_key_index(num_keys: int) -> int:
    """
    Reads the last used index from a file, increments it, and saves it back.
    This provides stateful, sequential key rotation across requests.
    """
    try:
        # Create the data directory if it doesn't exist
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
        # Fallback to the first key in case of file system errors
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
        self.max_tokens = 8196
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info(f"Initialized with {len(self.gemini_api_keys)} Gemini API keys")
        logger.info(f"Using generation model: {self.model_name}")

    def _prepare_prompt_for_all_queries(self, queries: List[str], full_context: str) -> str:
        """Prepares a single prompt containing the full context and all questions."""
        
        question_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])

        prompt = f"""You are a smart assistant helping a user extract information from a policy PDF.

--- Policy Document ---
{full_context}
--- End Document ---

Please answer all of the following questions clearly and concisely based on the document provided. If a question is unrelated to the document, provide a general answer.

--- Questions ---
{question_block}
--- End Questions ---

Now, provide the answers within 700 characters in a numbered list corresponding to the question number. For example:
1. [Your answer to question 1]
2. [Your answer to question 2]
"""
        return prompt

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

    def _is_valid_url(self, url: str) -> bool:
        return all(list(urlparse(url))[:2])

    def _download_pdf(self, url: str) -> bytes:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    def extract_full_text(self, pdf_bytes: bytes, max_chars: int = 1000) -> str:
        """Extracts and joins all text chunks from PDF bytes into a single string."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []
        for page in doc:
            all_text.append(page.get_text("text"))
        return "\n\n".join(all_text)

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
                generation_config={"temperature": 0.1, "max_output_tokens": self.max_tokens},
                safety_settings=safety_settings
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"API call failed with key ...{api_key[-4:]}: {e}")
            raise Exception(f"Gemini API call failed: {e}")

    async def process_queries(self, queries: List[str], document_link: str) -> Dict[str, str]:
        """
        Processes all queries in a single batch with stateful key rotation per request.
        """
        if not queries:
            return {}

        try:
            # 1. Select the next API key for this entire request
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key_for_request = self.gemini_api_keys[key_index]
            logger.info(f"Processing request with API key index {key_index} (...{api_key_for_request[-4:]})")

            # 2. Download and extract full document text
            pdf_bytes = await asyncio.to_thread(self._download_pdf, document_link)
            full_context = await asyncio.to_thread(self.extract_full_text, pdf_bytes)
            if not full_context:
                raise ValueError("Failed to extract text from the document.")

            # 3. Prepare a single prompt for all questions
            prompt = self._prepare_prompt_for_all_queries(queries, full_context)

            # 4. Make a single API call
            llm_response_text = await self._call_llm_with_single_key(prompt, api_key_for_request)

            # 5. Parse the response
            final_responses = self._parse_llm_response(llm_response_text, len(queries))
            
            return final_responses

        except Exception as e:
            logger.error(f"Critical error in process_queries: {e}", exc_info=True)
            return {str(i+1): f"Failed to process request: {e}" for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()