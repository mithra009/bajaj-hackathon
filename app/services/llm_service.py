import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import httpx
import traceback
import json
import random
import logging
from datetime import datetime

from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .query_logger import query_logger

# Hardcoded list of Google Gemini API keys
# The service will use one of these keys to communicate with the Gemini API.
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

# Model configuration
MODEL_NAME = "gemini-2.5-flash"
MAX_TOKENS = 8192

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initializes the LLMService with a random API key from the hardcoded list."""
        if not API_KEYS:
            raise ValueError("The hardcoded API_KEYS list is empty.")
            
        self.api_key = random.choice(API_KEYS)
        self.model_name = MODEL_NAME
        self.max_tokens = MAX_TOKENS
        logger.info(f"Initializing LLMService with model: {self.model_name}")
        self._setup_genai()

    def _setup_genai(self, used_keys=None):
        """Configures the Gemini API client, rotating keys if necessary."""
        used_keys = used_keys or set()
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            logger.error(f"Error with API key ...{self.api_key[-4:]}: {e}")
            used_keys.add(self.api_key)
            
            available_keys = [k for k in API_KEYS if k not in used_keys]
            if not available_keys:
                raise ValueError("All hardcoded API keys have failed.")
            
            self.api_key = random.choice(available_keys)
            logger.info(f"Retrying with a new API key: ...{self.api_key[-4:]}")
            self._setup_genai(used_keys)

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
        except httpx.RequestError as e:
            logger.warning(f"Document at {url} is not accessible. Error: {e}")
            return False

    def _prepare_prompt(self, queries: List[str], document_link: str) -> str:
        """Prepares the structured prompt for the LLM."""
        prompt_parts = [
            "Answer all questions based on the document provided at the link. If the answer is in the document, provide a clear, concise response. If it is not in the document, state that the information is not available in the document.",
            f"Document Link: {document_link}\n\n",
            "===== QUESTIONS TO ANSWER =====\n"
        ]
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}\n")
        
        prompt_parts.append("\n===== YOUR RESPONSES =====\n")
        prompt_parts.append("Provide your responses in the following format, with each answer on a new line:\n")
        
        for i in range(1, len(queries) + 1):
            prompt_parts.append(f"Answer {i}: [Your answer to question {i}]\n")
            
        return "".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, num_queries: int) -> Dict[str, str]:
        """Parses the raw text response from the LLM into a structured dictionary."""
        responses = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for i in range(1, num_queries + 1):
            answer_prefix = f"Answer {i}:"
            found_answer = "The model did not provide a specific answer for this question."
            
            for line in lines:
                if line.startswith(answer_prefix):
                    found_answer = line[len(answer_prefix):].strip()
                    break
            responses[f"Query {i}"] = found_answer
            
        return responses

    async def generate_response(self, queries: List[str], document_link: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Main function to generate a response from the LLM.
        """
        if not queries:
            raise ValueError("At least one query is required")
        if not document_link or not self._is_valid_url(document_link):
            raise ValueError("A valid document link is required")

        if not await self._is_document_accessible(document_link):
            raise ValueError(f"Document at {document_link} is not accessible or not found.")

        prompt = self._prepare_prompt(queries, document_link)
        
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": 0.7
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            response_text = response.text
            responses = self._parse_llm_response(response_text, len(queries))
            
            query_metadata = {
                "model": self.model_name,
                "api_key_used": f"...{self.api_key[-4:]}",
                **(metadata or {})
            }
            
            log_id = query_logger.log_query(
                document_link=document_link,
                queries=queries,
                responses=responses,
                metadata=query_metadata
            )
            responses["log_id"] = log_id
            
            return responses

        except Exception as e:
            logger.error(f"Error during LLM response generation: {e}", exc_info=True)
            raise

llm_service = LLMService()