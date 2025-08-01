import os
import google.generativeai as genai
from typing import List, Dict, Any
from urllib.parse import urlparse
import httpx
import time
import traceback
import json
from datetime import datetime
from pathlib import Path
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .query_logger import query_logger

class LLMService:
    def __init__(self):
        """Initializes the LLMService."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self._setup_genai()

    def _setup_genai(self):
        """Configures the Gemini API client."""
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

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

    def _prepare_prompt(self, queries: List[str], document_link: str) -> str:
        """Prepares the structured prompt for the LLM."""
        prompt_parts = [
            "Answer all questions based on the provided insurance policy document. If the answer is in the document, give a clear, concise response in under 1000 characters. If it is not in the document, then provide a brief and general answer."
            f"Document Link: {document_link}\n\n",
            "===== QUESTIONS TO ANSWER =====\n"
        ]
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}\n")
        
        prompt_parts.append("\n===== YOUR RESPONSES =====\n")
        prompt_parts.append("Please provide your responses in the following format for each question:\n")
        
        for i in range(1, len(queries) + 1):
            prompt_parts.append(f"Answer {i}: [Your answer to question {i}]\n")
            
        return "".join(prompt_parts)

    def _parse_llm_response(self, response_text: str, queries: List[str]) -> Dict[str, str]:
        """Parses the raw text response from the LLM into a structured dictionary."""
        responses = {}
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        for i, query in enumerate(queries, 1):
            query_num = i
            answer_prefix = f"Answer {query_num}:"
            found_answer = "I couldn't find a specific answer to this question in the document."
            
            for line in lines:
                if line.startswith(answer_prefix):
                    found_answer = line[len(answer_prefix):].strip()
                    # Clean up quotes if they wrap the entire answer
                    if (found_answer.startswith('"') and found_answer.endswith('"')) or \
                       (found_answer.startswith("'") and found_answer.endswith("'")):
                        found_answer = found_answer[1:-1]
                    break
            responses[f"Query {query_num}"] = found_answer
            
        return responses

    async def generate_response(self, queries: List[str], document_link: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Validates input, checks document accessibility, calls the LLM, and parses the response.
        Logs the query and response for future reference.
        
        Args:
            queries: List of questions to ask about the document
            document_link: URL of the document to query
            metadata: Additional metadata to store with the query log
            
        Returns:
            Dictionary mapping query numbers to their responses
        """
        try:
            if not queries:
                raise ValueError("At least one query is required")
            if not document_link or not self._is_valid_url(document_link):
                raise ValueError("A valid document link is required")

            is_accessible = await self._is_document_accessible(document_link)
            if not is_accessible:
                raise ValueError(f"Document at {document_link} is not accessible or not found")

            prompt = self._prepare_prompt(queries, document_link)
            
            # Log the query before processing
            query_metadata = {
                "model": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "num_queries": len(queries),
                "document_accessible": is_accessible,
                **(metadata or {})
            }
            
            print(f"\n=== QUERY LOGGING ===")
            print(f"Document: {document_link}")
            print(f"Number of queries: {len(queries)}")
            print(f"Metadata: {json.dumps(query_metadata, indent=2)}")

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

            # Process the response
            response_text = response.text
            responses = self._parse_llm_response(response_text, queries)
            
            # Log the successful query and response
            try:
                log_id = query_logger.log_query(
                    document_link=document_link,
                    queries=queries,
                    responses=responses,
                    metadata={
                        **query_metadata,
                        "response_character_count": sum(len(str(r)) for r in responses.values())
                    }
                )
                print(f"Query logged successfully with ID: {log_id}")
            except Exception as e:
                print(f"Warning: Failed to log query: {str(e)}")
                import traceback
                traceback.print_exc()
            
            return responses

        except Exception as e:
            print(f"\n=== ERROR OCCURRED IN LLM_SERVICE ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("=== STACK TRACE ===")
            traceback.print_exc()
            print("=== END OF ERROR ===")
            raise

# Singleton instance for the application
llm_service = LLMService()