import os
import sys
import asyncio
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple, Callable
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
import io
import tempfile
from PIL import Image
import openpyxl
import docx
import zipfile
from pptx import Presentation
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import re

@dataclass
class TextSplitter:
    """Recursive text splitter that splits text into chunks based on separators."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: Optional[List[str]] = None
    length_function: Callable[[str], int] = len
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive splitting on separators."""
        final_chunks = []
        self._split_text_recursive(text, self.separators, final_chunks)
        return final_chunks
    
    def _split_text_recursive(self, text: str, separators: List[str], final_chunks: List[str]) -> None:
        """Recursively split text into chunks."""
        # Get the current separator
        separator = separators[0] if separators else ""
        
        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Merge the splits, now splitting them
        good_splits = []
        current_chunk = ""
        
        for s in splits:
            if separator and separator not in self.separators[-1]:
                s = s + separator
                
            if len(current_chunk) + len(s) < self.chunk_size:
                current_chunk += s
            else:
                if current_chunk:
                    good_splits.append(current_chunk.strip())
                current_chunk = s
        
        if current_chunk:
            good_splits.append(current_chunk.strip())
        
        # If we have more separators to try, recurse
        if len(separators) > 1:
            new_separators = separators[1:]
            new_splits = []
            for s in good_splits:
                if len(s) > self.chunk_size + self.chunk_overlap:
                    new_splits.extend(self._split_text_recursive(s, new_separators, final_chunks))
                else:
                    new_splits.append(s)
            good_splits = new_splits
        
        # Add the final chunks
        for s in good_splits:
            if len(s) > 0:
                final_chunks.append(s)

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

    def _get_file_type(self, url: str) -> str:
        """Determine file type from URL."""
        url_lower = url.lower()
        
        if any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
            return 'image'
        elif '.pdf' in url_lower:
            return 'pdf'
        elif any(ext in url_lower for ext in ['.xlsx', '.xls']):
            return 'excel'
        elif any(ext in url_lower for ext in ['.docx', '.doc']):
            return 'word'
        elif any(ext in url_lower for ext in ['.pptx', '.ppt']):
            return 'powerpoint'
        elif '.zip' in url_lower:
            return 'zip'
        else:
            return 'unknown'

    def _download_file(self, url: str) -> bytes:
        """Download file from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise APIError(f"Failed to download file: {e}")

    async def _upload_to_gemini(self, file_bytes: bytes, mime_type: str, api_key: str) -> Any:
        """Upload file to Gemini and return file object."""
        try:
            genai.configure(api_key=api_key)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Upload to Gemini
                uploaded_file = genai.upload_file(temp_file_path, mime_type=mime_type)
                
                # Wait for processing
                while uploaded_file.state.name == "PROCESSING":
                    await asyncio.sleep(1)
                    uploaded_file = genai.get_file(uploaded_file.name)
                
                if uploaded_file.state.name == "FAILED":
                    raise Exception(f"File processing failed: {uploaded_file.state}")
                
                return uploaded_file
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error uploading to Gemini: {e}")
            raise Exception(f"Failed to upload file to Gemini: {e}")

    def _extract_text_from_excel(self, file_bytes: bytes) -> str:
        """Extract text from Excel file with detailed logging."""
        try:
            logger.info("Starting Excel file processing...")
            start_time = time.time()
            
            # Load workbook
            workbook = openpyxl.load_workbook(io.BytesIO(file_bytes))
            sheet_count = len(workbook.sheetnames)
            logger.info(f"Loaded Excel file with {sheet_count} sheets")
            
            extracted_text = []
            total_rows = 0
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_rows = sheet.max_row
                total_rows += sheet_rows
                
                sheet_header = f"\n--- Sheet: {sheet_name} ({sheet_rows} rows) ---"
                extracted_text.append(sheet_header)
                logger.debug(f"Processing sheet: {sheet_name} with {sheet_rows} rows")
                
                # Get headers if they exist
                headers = []
                if sheet.max_row > 0:
                    headers = [str(cell.value) if cell.value else f"Column{idx+1}" 
                             for idx, cell in enumerate(sheet[1])]
                
                # Add header row if exists
                if headers:
                    extracted_text.append(" | ".join(headers))
                    start_row = 2
                else:
                    start_row = 1
                
                # Process data rows
                for row in sheet.iter_rows(min_row=start_row, values_only=True):
                    row_text = []
                    for cell in row:
                        if cell is not None:
                            cell_text = str(cell).strip()
                            if cell_text:  # Only include non-empty cells
                                row_text.append(cell_text)
                    if row_text:  # Only add non-empty rows
                        extracted_text.append(" | ".join(row_text))
            
            processing_time = time.time() - start_time
            result = "\n".join(extracted_text)
            
            logger.info(
                f"Excel processing completed in {processing_time:.2f}s. "
                f"Extracted {total_rows} total rows from {sheet_count} sheets. "
                f"Total text length: {len(result)} characters"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing Excel file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

    def _extract_text_from_word(self, file_bytes: bytes) -> str:
        """Extract text from Word document."""
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting Word text: {e}")
            raise Exception(f"Failed to extract text from Word document: {e}")

    def _extract_text_from_powerpoint(self, file_bytes: bytes) -> str:
        """
        Extract text from PowerPoint presentation with detailed error handling.
        Returns extracted text or raises an exception with detailed error information.
        """
        try:
            logger.info("Starting PowerPoint text extraction...")
            start_time = time.time()
            
            # Create a temporary file to help with debugging
            with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                # Try to load the presentation
                prs = Presentation(temp_path)
                total_slides = len(prs.slides)
                logger.info(f"Successfully loaded PowerPoint with {total_slides} slides")
                
                full_text = []
                slides_processed = 0
                
                for i, slide in enumerate(prs.slides, 1):
                    try:
                        slide_text = [f"--- Slide {i} ---"]
                        
                        # Get slide title if exists
                        if slide.shapes.title and slide.shapes.title.text.strip():
                            title = slide.shapes.title.text.strip()
                            slide_text.append(f"Title: {title}")
                        
                        # Get all text from shapes
                        for shape in slide.shapes:
                            try:
                                if hasattr(shape, "text") and shape.text and shape.text.strip():
                                    # Skip title text as we already have it
                                    if not (hasattr(slide.shapes, 'title') and shape == slide.shapes.title):
                                        text = shape.text.strip()
                                        if text:  # Only add non-empty text
                                            slide_text.append(text)
                                
                                # Handle tables if present
                                if shape.has_table:
                                    table_text = []
                                    for row in shape.table.rows:
                                        row_text = []
                                        for cell in row.cells:
                                            if cell.text and cell.text.strip():
                                                row_text.append(cell.text.strip())
                                        if row_text:
                                            table_text.append(" | ".join(row_text))
                                    if table_text:
                                        slide_text.append("Table:\n" + "\n".join(table_text))
                                        
                            except Exception as shape_error:
                                logger.warning(f"Error processing shape in slide {i}: {shape_error}")
                                continue
                        
                        # Add speaker notes if any
                        try:
                            if slide.has_notes_slide and slide.notes_slide.notes_text_frame and slide.notes_slide.notes_text_frame.text.strip():
                                notes = slide.notes_slide.notes_text_frame.text.strip()
                                slide_text.append(f"Notes: {notes}")
                        except Exception as notes_error:
                            logger.warning(f"Error reading notes for slide {i}: {notes_error}")
                        
                        full_text.append("\n".join(slide_text))
                        slides_processed += 1
                        
                    except Exception as slide_error:
                        logger.error(f"Error processing slide {i}: {slide_error}")
                        full_text.append(f"--- Slide {i} [Error processing slide] ---")
                
                result = "\n\n".join(full_text)
                processing_time = time.time() - start_time
                
                logger.info(
                    f"PowerPoint processing completed in {processing_time:.2f}s. "
                    f"Processed {slides_processed}/{total_slides} slides. "
                    f"Extracted {len(result)} characters."
                )
                
                if not result.strip():
                    raise ValueError("No text content could be extracted from the PowerPoint file")
                    
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up temporary file: {cleanup_error}")
            
        except Exception as e:
            error_msg = f"Failed to extract text from PowerPoint: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

    def _extract_text_from_zip(self, file_bytes: bytes) -> str:
        """Extract text from ZIP file contents."""
        try:
            extracted_content = []
            
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.endswith('/'):
                        continue  # Skip directories
                    
                    try:
                        file_content = zip_file.read(file_info.filename)
                        file_type = self._get_file_type(file_info.filename)
                        
                        extracted_content.append(f"\n--- File: {file_info.filename} ---")
                        
                        if file_type == 'pdf':
                            # Extract PDF text
                            doc = fitz.open(stream=file_content, filetype="pdf")
                            for page in doc:
                                text = page.get_text()
                                if text.strip():
                                    extracted_content.append(text)
                            doc.close()
                            
                        elif file_type == 'word':
                            text = self._extract_text_from_word(file_content)
                            extracted_content.append(text)
                            
                        elif file_type == 'excel':
                            text = self._extract_text_from_excel(file_content)
                            extracted_content.append(text)
                            
                        elif file_type == 'powerpoint':
                            text = self._extract_text_from_powerpoint(file_content)
                            extracted_content.append(text)
                            
                        else:
                            # Try to decode as text
                            try:
                                text = file_content.decode('utf-8')
                                extracted_content.append(text)
                            except:
                                extracted_content.append(f"[Binary file - {len(file_content)} bytes]")
                                
                    except Exception as e:
                        extracted_content.append(f"[Error processing {file_info.filename}: {e}]")
            
            return "\n".join(extracted_content)
            
        except Exception as e:
            logger.error(f"Error extracting ZIP contents: {e}")
            raise Exception(f"Failed to extract ZIP contents: {e}")

    async def _process_file_with_gemini(self, queries: List[str], file_bytes: bytes, 
                                      file_type: str, api_key: str) -> Dict[str, str]:
        """Process file using Gemini's file upload capability."""
        try:
            logger.info(f"Processing {file_type} file with Gemini upload...")
            
            # Determine MIME type
            mime_types = {
                'image': 'image/png',
                'pdf': 'application/pdf',
                'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'word': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'powerpoint': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
            }
            
            mime_type = mime_types.get(file_type, 'application/octet-stream')
            
            # Upload file to Gemini
            uploaded_file = await self._upload_to_gemini(file_bytes, mime_type, api_key)
            
            # Prepare prompt
            prompt = self._prepare_file_analysis_prompt(queries)
            
            # Generate response
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
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
            
            response = await model.generate_content_async([uploaded_file, prompt])
            
            # Clean up uploaded file
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
            
            # Parse response
            query_numbers = list(range(1, len(queries) + 1))
            return self._parse_batch_response(response.text, query_numbers)
            
        except Exception as e:
            logger.error(f"Error processing file with Gemini: {e}")
            raise Exception(f"Gemini file processing failed: {e}")

    async def _process_text_based_file(self, queries: List[str], file_bytes: bytes, 
                                     file_type: str, api_key: str) -> Dict[str, str]:
        """Process text-based files by extracting content first."""
        try:
            logger.info(f"Processing {file_type} by extracting text content...")
            
            # Extract text based on file type
            if file_type == 'excel':
                extracted_text = self._extract_text_from_excel(file_bytes)
            elif file_type == 'word':
                extracted_text = self._extract_text_from_word(file_bytes)
            elif file_type == 'zip':
                extracted_text = self._extract_text_from_zip(file_bytes)
            elif file_type == 'pdf':
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                extracted_text = ""
                for page in doc:
                    extracted_text += page.get_text() + "\n"
                doc.close()
            elif file_type == 'powerpoint':
                extracted_text = self._extract_text_from_powerpoint(file_bytes)
            else:
                # Try to decode as text
                extracted_text = file_bytes.decode('utf-8', errors='ignore')
            
            if not extracted_text.strip():
                raise Exception("No text content could be extracted from the file")
            
            # Prepare prompt with extracted text
            prompt = self._prepare_text_analysis_prompt(queries, extracted_text, file_type)
            
            # Generate response
            response_text = await self._call_llm_batch(prompt, api_key, timeout=45.0)
            
            # Parse response
            query_numbers = list(range(1, len(queries) + 1))
            return self._parse_batch_response(response_text, query_numbers)
            
        except Exception as e:
            logger.error(f"Error processing text-based file: {e}")
            raise Exception(f"Text extraction failed: {e}")

    def _prepare_file_analysis_prompt(self, queries: List[str]) -> str:
        """Prepare prompt for file analysis with Gemini upload."""
        prompt_parts = [
            "You are an expert health insurance policy analyst. Your task is to carefully analyze the provided policy document and answer the following questions.",
            "",
            "DOCUMENT TYPE: This is a health insurance policy document that may be in PowerPoint format. Pay special attention to:",
            "- All slides and their content (titles, bullet points, tables, charts)",
            "- Speaker notes and slide notes (often contain important details)",
            "- Fine print, footnotes, and disclaimers",
            "- Any appendices, reference slides, or additional materials",
            "- Policy numbers, coverage details, terms and conditions",
            "",
            "INSTRUCTIONS:",
            "1. Read and analyze the ENTIRE document carefully before answering any questions.",
            "2. For each question, provide a comprehensive response (200-300 characters) with specific references.",
            "3. Include slide numbers, section headers, or specific locations where the information was found.",
            "4. For numerical values (limits, sub-limits, waiting periods), provide exact figures from the document.",
            "5. If a question has multiple parts, address each part clearly in your response.",
            "6. For coverage details, be specific about what is included and any exclusions.",
            "7. If you're unsure about an answer, provide the most relevant information you can find.",
            "",
            "EXAMPLE QUERIES AND ANSWERS:",
            "Question: What types of hospitalization expenses are covered, and what are the limits for room and room expenses?",
            "Answer: Covers medical expenses including room rent up to ₹5,000/day and ICU charges up to ₹10,000/day. (Slide 5)",
            "",
            "Question: What is domiciliary hospitalization, and what are its key exclusions?",
            "Answer: Domiciliary hospitalization is home treatment due to unavailability of hospital beds or patient condition, excluding treatments like asthma, bronchitis, epilepsy, etc. (Slide 7)",
            "",
            "Question: What are the benefits and limits of telemedicine and maternity coverage under this policy?",
            "Answer: Telemedicine is covered up to ₹2,000 per policy year; maternity coverage is not included. (Slide 9)",
            "",
            "Question: What specialized treatments are covered, and what are their sub-limits?",
            "Answer: Covers specialized treatments like robotic surgeries and stem cell therapy with sub-limits (e.g., ₹1,00,000 for deep brain stimulation). (Slide 11)",
            "",
            "Question: What are the waiting periods for pre-existing diseases and specified diseases or procedures?",
            "Answer: 48 months for pre-existing diseases and 24 months for specified conditions like cataract, hernia, etc. (Slide 13)"
            "",
            "QUESTIONS:"
        ]
        
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}")
            
        prompt_parts.extend([
            "",
            "RESPONSE FORMAT:",
            "For each question, provide your answer in the format: ANSWER_[NUMBER]: [your answer]",
            "- Be specific and reference slide numbers or sections (e.g., 'As per Slide 15, this is covered...')",
            "- For coverage details, mention any sub-limits or conditions",
            "- If information is spread across multiple slides, consolidate it into a comprehensive answer",
            "- If the exact information isn't available, provide the most relevant details you can find",
            "",
            "EXAMPLE RESPONSES:",
            "GOOD: 'ANSWER_1: The policy covers hospitalization expenses up to ₹5,00,000 per year (Slide 8). Room rent is limited to 1% of sum insured per day (Slide 9). Pre-existing diseases have a 3-year waiting period (Slide 14).' ",
            "",
            "NOT ACCEPTABLE: 'ANSWER_1: The document doesn't specify coverage details. Please refer to the full policy document.'",
            "",
            "Now analyze the document thoroughly and provide your responses:"
        ])
        
        return "\n".join(prompt_parts)

    async def _process_image_with_gemini(self, image_url: str, queries: List[str]) -> List[str]:
        """Process image using Gemini's vision capabilities with detailed instructions for table reading."""
        try:
            # Prepare the detailed prompt with explicit instructions for reading tables
            prompt_parts = [
                "You are an expert at reading and interpreting tables from images. Your task is to carefully analyze the table in the image and answer the following questions based SOLELY on the visible data.",
                "",
                "CRITICAL INSTRUCTIONS:",
                "1. Carefully examine the ENTIRE table, including all headers, rows, and columns.",
                "2. For each question, provide the answer based ONLY on the data visible in the table.",
                "3. If the table contains the information but you're uncertain about the exact value, make your best attempt to read it.",
                "4. For numerical values, provide the exact numbers as they appear in the table.",
                "5. If a question is about a specific sum insured amount, find the corresponding row in the table.",
                "6. For questions about daily limits or coverage, look for the relevant column in the table.",
                "7. Format your response as: ANSWER_[NUMBER]: [your answer] with each answer on a new line.",
                "8. DO NOT say the table doesn't contain the information if you can see relevant data - make your best effort to answer.",
                "9.If the image contributes the content to answer only few queries, answer them in that context, and the remaining answer from your knowledge strictly, do not mention fie doesnot contain information",
                "",
                "IMPORTANT: The table appears to have the following structure:",
                "- First column: Sum Insured amounts (like 4 Lakhs, 8 Lakhs, etc.)",
                "- Other columns: Different types of coverage/limits (like Room, Boarding, Nursing, ICU, etc.)",
                "- Rows represent different sum insured amounts",
                "- Cells contain the coverage/limit amounts",
                "",
                "QUESTIONS:"
            ]
            
            # Add each question with a number
            for i, query in enumerate(queries, 1):
                prompt_parts.append(f"{i}. {query}")
                
            prompt_parts.extend([
                "",
                "Please provide your answers based on the table data. If you can see the information in the table, provide the exact values. If the information is not in the table, say 'The table does not contain this information.'"
            ])
            
            # Generate response
            genai.configure(api_key=self.gemini_api_keys[0])
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
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
            
            response = await model.generate_content_async([prompt_parts])
            
            # Parse response
            query_numbers = list(range(1, len(queries) + 1))
            return self._parse_batch_response(response.text, query_numbers)
            
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {e}")
            raise Exception(f"Gemini image processing failed: {e}")

    def _prepare_text_analysis_prompt(self, queries: List[str], extracted_text: str, file_type: str) -> str:
        """Prepare prompt for text-based analysis with policy-specific instructions."""
        prompt_parts = [
            f"You are an expert {file_type.upper()} policy analyzer. Answer questions based on the document content with specific section references.",
            "",
            "EXAMPLE QUERY AND ANSWER:",
            "Query: 'I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?'",
            "Answer: 'Yes, under Arogya Sanjeevani Policy under Clause 10.8'",
            "",
            "CRITICAL INSTRUCTIONS:",
            "1. First search the document for the answer, including section numbers in your response.",
            "2. If not found, provide a concise answer based on standard policy terms.",
            "3. Keep responses around 250 characters, focusing on key details and section references.",
            "4. For claim-related queries, mention relevant clauses (e.g., 'Under Clause 10.8...')",
            "5. Format: ANSWER_[NUMBER]: [concise answer with section references]",
            "",
            f"DOCUMENT CONTENT:",
            "=" * 50,
            extracted_text[:15000],  # Limit content size
            "=" * 50,
            "",
            "QUESTIONS:",
            ""
        ]
        
        for i, query in enumerate(queries, 1):
            prompt_parts.append(f"{i}. {query}")
        
        prompt_parts.extend([
            "",
            "RESPONSE GUIDELINES:",
            "- Reference specific sections (e.g., 'As per Section 4.2...' or 'Under Clause 10.8...')",
            "- For claim amounts, specify coverage limits and conditions",
            "- Keep answers concise but informative (around 250 characters)",
            "- If exact section isn't found, provide the most relevant information available"
        ])
        
        return "\n".join(prompt_parts)

    def _prepare_batch_query_prompt(self, queries_with_context: List[Tuple[int, str, List[str]]]) -> str:
        """Prepares a batch prompt for multiple policy-related queries with context."""
        
        # Build the batch prompt with policy-specific instructions
        prompt_parts = [
            "You are an expert health insurance policy analyst. Answer questions based on the provided policy context or standard policy knowledge.",
            "",
            "EXAMPLE QUERY AND ANSWER:",
            "Query: 'I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?'",
            "Answer: 'Yes, under Arogya Sanjeevani Policy under Clause 10.8'",
            "",
            "CRITICAL INSTRUCTIONS:",
            "1. For each question, search the provided context for relevant policy sections.",
            "2. If found, provide a concise answer (around 250 chars) with section references.",
            "3. If not found, provide a general answer based on standard policy terms.",
            "4. For claim-related queries, mention relevant clauses (e.g., 'Under Clause 10.8...')",
            "5. Format: ANSWER_[NUMBER]: [concise answer with section references]",
            "",
            "QUESTIONS AND POLICY CONTEXTS:",
            ""
        ]
        
        for query_num, query, context_chunks in queries_with_context:
            prompt_parts.append(f"QUESTION_{query_num}: {query}")
            prompt_parts.append("RELEVANT POLICY SECTIONS:")
            if context_chunks:
                for i, chunk in enumerate(context_chunks[:5]):  # Top 5 most relevant policy sections
                    prompt_parts.append(f"- {chunk}")
            else:
                prompt_parts.append("- General policy knowledge")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "RESPONSE GUIDELINES:",
            "1. Reference specific sections when possible (e.g., 'As per Section 4.2...' or 'Under Clause 10.8...')",
            "2. For claim amounts, specify coverage limits and conditions",
            "3. Keep answers concise (around 250 characters) but informative",
            "4. For partial claims, mention portability benefits under Clause 10.8",
            "5. If exact section isn't found, provide the most relevant information available",
            "",
            "Format: ANSWER_[NUMBER]: [concise answer with section references]"
        ])
        
        return "\n".join(prompt_parts)

    async def _call_llm_batch(self, prompt: str, api_key: str, timeout: float = 30.0) -> str:
        """Enhanced LLM call for batch processing with increased timeout."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 4096,
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
            return response.text.strip() if response.text else "Unable to generate response"
            
        except asyncio.TimeoutError:
            logger.error(f"API call timeout with key ...{api_key[-4:]}")
            raise Exception(f"Request timeout after {timeout}s - please try again")
        except Exception as e:
            logger.error(f"API call failed with key ...{api_key[-4:]}: {e}")
            raise Exception(f"Gemini API call failed: {str(e)[:100]}")

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
                            answers[str(query_numbers[i-1])] = answer[:1000]  # Limit length
            
            # Ensure all query numbers have answers
            for num in query_numbers:
                if str(num) not in answers:
                    answers[str(num)] = "Unable to extract answer from response"
                    
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            # Fallback: split response roughly by number of queries
            parts = response_text.split('\n\n') if '\n\n' in response_text else [response_text]
            for i, num in enumerate(query_numbers):
                if i < len(parts):
                    answers[str(num)] = parts[i]
                else:
                    answers[str(num)] = "Unable to parse answer from response"
        
        return answers

    async def _process_non_pdf_file(self, queries: List[str], url: str) -> Dict[str, str]:
        """Process non-PDF files (images, Excel, Word, PowerPoint, etc.)."""
        try:
            file_type = self._get_file_type(url)
            
            # Get API key
            key_index = get_next_key_index(len(self.gemini_api_keys))
            api_key = self.gemini_api_keys[key_index]
            
            logger.info(f"Processing {file_type} file: {url}")
            
            # For PPTX files, send URL directly to Gemini
            if file_type == 'powerpoint':
                try:
                    # Configure Gemini
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        self.model_name,
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 4096,
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
                    
                    # Prepare prompt with URL
                    prompt = self._prepare_file_analysis_prompt(queries)
                    message = f"Please analyze the presentation at this URL and answer the following questions.\nURL: {url}\n\n{prompt}"
                    
                    # Generate response
                    response = await model.generate_content_async(message)
                    
                    # Parse response
                    query_numbers = list(range(1, len(queries) + 1))
                    return self._parse_batch_response(response.text, query_numbers)
                    
                except Exception as e:
                    logger.error(f"Error processing PPTX with direct URL: {e}", exc_info=True)
                    # Fall back to local extraction if direct URL fails
                    logger.info("Falling back to local PPTX extraction...")
                    file_bytes = await asyncio.to_thread(self._download_file, url)
                    extracted_text = self._extract_text_from_powerpoint(file_bytes)
                    if extracted_text.strip():
                        prompt = self._prepare_text_analysis_prompt(queries, extracted_text, file_type)
                        response = await self._call_llm_batch(prompt, api_key)
                        return self._parse_batch_response(response, list(range(1, len(queries) + 1)))
                    raise
            
            # For images, use Gemini's vision capabilities
            elif file_type == 'image':
                file_bytes = await asyncio.to_thread(self._download_file, url)
                try:
                    # Convert downloaded bytes into a PIL image object
                    image = Image.open(io.BytesIO(file_bytes))

                    # Configure Gemini
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        self.model_name,
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 4096,
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

                    # Create a detailed prompt for accurate image analysis, especially for tables and math problems
                    prompt_parts = [
                        "You are an expert at reading text, numbers, and mathematical expressions from images. Your task is to carefully analyze the provided image and answer the following questions based STRICTLY on the visible data.",
                        "",
                        "CRITICAL INSTRUCTIONS FOR MATHEMATICAL EXPRESSIONS:",
                        "1. For each question asking for a calculation (e.g., 'What is X+Y?'), do the following:",
                        "   - If the exact calculation is shown in the image, report it EXACTLY as shown, even if it's incorrect.",
                        "   - If the image shows the expression but not the answer,then answer correctly from your own knowledge",
                        "   - If the expression is not shown at all, answer the expression correctly from your knowledge.",
                        "2. DO NOT perform any calculations or correct any mathematical errors you see in the image.",
                        "3. If you see text that looks like a math problem, report it EXACTLY as written, including any typos or errors.",
                        "4. If the image contains a table with numbers, read them exactly as they appear, even if they seem incorrect.",
                        "5. Format your response STRICTLY as: ANSWER_[NUMBER]: [your answer]",
                        "6. Each answer must be on a new line.",
                        "",
                        "QUESTIONS ABOUT THE IMAGE CONTENT:"
                    ]
                    for i, query in enumerate(queries, 1):
                        prompt_parts.append(f"{i}. {query}")
                    
                    final_prompt = "\n".join(prompt_parts)

                    # Generate content with both the prompt and the image data
                    response = await model.generate_content_async([final_prompt, image])

                    # Parse the structured response
                    query_numbers = list(range(1, len(queries) + 1))
                    return self._parse_batch_response(response.text, query_numbers)

                except Exception as e:
                    logger.error(f"Error processing image with Gemini Vision: {e}", exc_info=True)
                    return {str(i+1): f"Error processing image: {str(e)[:200]}" for i in range(len(queries))}
            
            # For PDFs, try local extraction first, then fallback to Gemini upload
            elif file_type == 'pdf':
                try:
                    file_bytes = await asyncio.to_thread(self._download_file, url)
                    # First try local text extraction
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    extracted_text = ""
                    for page in doc:
                        extracted_text += page.get_text() + "\n"
                    doc.close()
                    
                    if extracted_text.strip():
                        prompt = self._prepare_text_analysis_prompt(queries, extracted_text, file_type)
                        response = await self._call_llm_batch(prompt, api_key)
                        return self._parse_batch_response(response, list(range(1, len(queries) + 1)))
                    
                    # Fallback to Gemini upload if local extraction fails
                    return await self._process_file_with_gemini(queries, file_bytes, file_type, api_key)
                    
                except Exception as e:
                    logger.error(f"Error processing PDF: {e}", exc_info=True)
                    return {str(i+1): f"Error processing PDF: {str(e)[:200]}" for i in range(len(queries))}
            
            # For unknown file types, try direct processing
            else:
                logger.info("Processing as unknown file with direct processing...")
                try:
                    # Download the file first
                    file_bytes = await asyncio.to_thread(self._download_file, url)
                    if not file_bytes:
                        return {str(i+1): "Error: No content could be retrieved from the URL" for i in range(len(queries))}
                    
                    # Try processing with Gemini
                    return await self._process_file_with_gemini(queries, file_bytes, file_type, api_key)
                        
                except Exception as e:
                    logger.error(f"Error processing unknown file: {e}", exc_info=True)
                    return {str(i+1): f"Error processing file: {str(e)[:200]}" for i in range(len(queries))}
                try:
                    file_bytes = await asyncio.to_thread(self._download_file, url)
                    # First try local text extraction
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    extracted_text = ""
                    for page in doc:
                        extracted_text += page.get_text() + "\n"
                    doc.close()
                    
                    if extracted_text.strip():
                        prompt = self._prepare_text_analysis_prompt(queries, extracted_text, file_type)
                        response = await self._call_llm_batch(prompt, api_key)
                        return self._parse_batch_response(response, list(range(1, len(queries) + 1)))
                    
                    # If no text was extracted, fall through to Gemini upload
                    logger.info("No text extracted from PDF, trying Gemini upload...")
                    
                except Exception as e:
                    logger.warning(f"Local PDF processing failed, trying Gemini upload: {e}")
                
                # Try Gemini upload as fallback
                try:
                    return await self._process_file_with_gemini(queries, file_bytes, file_type, api_key)
                except Exception as e:
                    logger.error(f"Gemini upload failed for {file_type}: {e}")
                    raise Exception(f"Failed to process {file_type} file: {e}")
            
            # For other file types, use the text-based processing
            return await self._process_text_based_file(queries, file_bytes, file_type, api_key)
            
        except Exception as e:
            logger.error(f"Error processing non-PDF file: {e}")
            error_msg = f"Error processing {self._get_file_type(url)} file: {str(e)[:200]}"
            return {str(i+1): error_msg for i in range(len(queries))}

    # [Include all the existing PDF processing methods here - they remain unchanged]
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

    def _find_top_chunks_optimized(self, query_emb: List[float], chunk_embs: np.ndarray, chunks: List[str], top_k: int = 7) -> List[str]:
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
        For other file types: Processes directly with appropriate handling.
        """
        start_time = time.time()
        logger.info(f"\n{'='*100}")
        logger.info(f"PROCESSING DOCUMENT: {document_link}")
        logger.info(f"QUERIES: {json.dumps(queries, indent=2)}")
        logger.info(f"START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*100}\n")
        
        # Log system information for debugging
        logger.debug(f"System Info - Python: {sys.version}")
        logger.debug(f"Dependencies - openpyxl: {openpyxl.__version__}, python-docx: {docx.__version__}")
        
        # Log memory usage
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.debug(f"Memory Usage - RSS: {mem_info.rss / 1024 / 1024:.2f}MB, "
                       f"VMS: {mem_info.vms / 1024 / 1024:.2f}MB")
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        
        try:
            # Determine file type
            file_type = self._get_file_type(document_link)
            logger.info(f"Detected file type: {file_type}")
            
            if file_type == 'pdf':
                # Use batch processing for PDFs
                logger.info("Processing as PDF with chunking and embeddings...")
                results = await self.process_queries_with_batch_processing(queries, document_link)
            else:
                # For non-PDF files, process directly
                logger.info(f"Processing as {file_type} file with direct processing...")
                results = await self._process_non_pdf_file(queries, document_link)
            
            # Calculate processing time
            total_time = time.time() - start_time
            
            # Log detailed response information
            logger.info(f"\n{'='*100}")
            logger.info("PROCESSING SUMMARY")
            logger.info(f"{'='*100}")
            logger.info(f"DOCUMENT_URL: {document_link}")
            logger.info(f"FILE_TYPE: {file_type.upper()}")
            logger.info(f"PROCESSING_MODE: {'BATCH' if file_type == 'pdf' else 'DIRECT'}")
            logger.info(f"START_TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            logger.info(f"END_TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"TOTAL_PROCESSING_TIME: {total_time:.2f} seconds")
            logger.info(f"QUERY_COUNT: {len(queries)}")
            
            # Log detailed query responses
            logger.info(f"\n{'='*100}")
            logger.info("DETAILED RESPONSES")
            logger.info(f"{'='*100}")
            
            for i, (query, response) in enumerate(zip(queries, results.values()), 1):
                # Truncate very long responses for logging
                response_preview = str(response)
                if len(response_preview) > 500:
                    response_preview = response_preview[:500] + "... [truncated]"
                
                logger.info(f"\nQUERY_{i}:")
                logger.info(f"  {query}")
                logger.info(f"RESPONSE_{i} (Length: {len(str(response))} chars):")
                logger.info(f"  {response_preview}")
            
            # Log memory usage at the end
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                logger.info(f"\n{'='*50}")
                logger.info("SYSTEM RESOURCES AT COMPLETION:")
                logger.info(f"  Memory RSS: {mem_info.rss / 1024 / 1024:.2f} MB")
                logger.info(f"  Memory VMS: {mem_info.vms / 1024 / 1024:.2f} MB")
                logger.info(f"  CPU Usage: {process.cpu_percent()}%")
            except ImportError:
                logger.info("\nSystem resource monitoring not available (psutil not installed)")
            
            logger.info(f"{'='*100}\n")
            
            # Ensure all queries have responses
            final_results = {}
            for i in range(1, len(queries) + 1):
                response = results.get(str(i), "No response generated")
                final_results[str(i)] = response
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error processing queries: {str(e)}", exc_info=True)
            # Return error responses for all queries
            return {str(i+1): f"Error processing request: {str(e)[:200]}" for i in range(len(queries))}

# Singleton instance for the application
llm_service = LLMService()