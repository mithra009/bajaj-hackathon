"""
Direct LLM Router - Provides endpoints for direct LLM processing without preprocessing.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any
import logging
from pydantic import BaseModel, HttpUrl, Field
from ...services.direct_llm_service import direct_llm_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    """Request model for direct LLM queries."""
    documents: str  # Single document URL as string
    questions: List[str]  # List of questions

@router.post("/hackrx/run", response_model=Dict[str, Any])
async def process_direct_queries(request: Request, query_request: QueryRequest):
    """
    Process queries directly with minimal preprocessing.
    
    Args:
        request: The FastAPI request object
        query_request: The query request containing document URL and questions
        
    Returns:
        Dictionary containing the answers to the queries
    """
    try:
        logger.info(f"Processing direct LLM request for document: {query_request.documents}")
        
        # Process the queries
        response = await direct_llm_service.process_queries(
            document_url=query_request.documents,
            queries=query_request.questions
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in direct LLM processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
