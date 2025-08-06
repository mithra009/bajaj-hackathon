import os
import logging
import time
import json
import uvicorn
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Type

from fastapi import FastAPI, HTTPException, Depends, status, Request, Body, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, HttpUrl, Field, validator
from dotenv import load_dotenv

# Define RateLimitError for compatibility
class RateLimitError(Exception):
    """Raised when the API request hits a rate limit."""
    pass

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document Query API",
    description="API for querying documents using LLMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
security = HTTPBearer()
# Use the provided authentication token
API_KEY = "fd53cda9e372cc74319d047c60acdcc06e62e7e5550a92d842c425b82df84e4d"

# Log the first and last 4 characters of the key for verification
logger.info(f"Using API key: {API_KEY[:4]}...{API_KEY[-4:]}")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify the provided API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Import services after app creation
from app.services import llm_service, query_logger

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the document to query")
    questions: List[str] = Field(..., min_items=1, max_items=200, description="List of questions to ask about the document")
    
    class Config:
        schema_extra = {
            "example": {
                "documents": "https://example.com/insurance-policy.pdf",
                "questions": [
                    "What is the policy coverage for emergency hospitalization?",
                    "What is the claim submission process?"
                ]
            }
        }

class QueryResponse(BaseModel):
    answers: List[str]

class QueryLogResponse(BaseModel):
    id: str
    timestamp: str
    document_link: str
    queries: List[str]
    responses: Dict[str, str]
    metadata: Dict[str, Any]

class QueryLogsResponse(BaseModel):
    logs: List[QueryLogResponse]

@app.get("/hackrx/logs", response_model=QueryLogsResponse, dependencies=[Depends(verify_token)])
async def get_query_logs(limit: int = 100):
    """
    Retrieve the most recent query logs.
    
    Args:
        limit: Maximum number of logs to return (default: 100)
    """
    try:
        logs = query_logger.get_logs(limit=limit)
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve logs: {str(e)}"
        )

@app.post("/hackrx/run", response_model=QueryResponse, summary="Query a document with multiple questions")
async def query_document(
    request: Request,
    query_data: QueryRequest = Body(...),
    background_tasks: BackgroundTasks = None,
    _: str = Depends(verify_token)
):
    """
    Query a document with multiple questions and get responses with minimal latency.
    
    This endpoint processes multiple queries in parallel using available API keys
    and returns responses as soon as they're available.
    """
    start_time = time.time()
    
    try:
        # Process queries in parallel
        results_dict = await llm_service.process_queries(
            queries=query_data.questions,
            document_link=str(query_data.documents)
        )
        
        # Sort results by query index ("1", "2", ...) and extract answers
        sorted_answers = [results_dict[str(i)] for i in sorted(results_dict.keys(), key=int)]
        
        # Prepare the final response object
        response_data = QueryResponse(answers=sorted_answers)
        
        # Log the successful query
        if background_tasks:
            background_tasks.add_task(
                query_logger.log_query,
                document_link=str(query_data.documents),
                queries=query_data.questions,
                responses=results_dict,
                metadata={
                    "processing_time": time.time() - start_time,
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            )
        
        return response_data
        
    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": "API rate limit exceeded. Please try again later.",
                "details": str(e)
            }
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during API call: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Service temporarily unavailable. Please try again later.",
                "details": str(e)
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred while processing your request.",
                "details": str(e) if app.debug else None
            }
        )

@app.get("/")
async def root():
    return {
        "message": "LLM Query API is running",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "document_query": {
                "method": "POST",
                "path": "/hackrx/run",
                "description": "Query documents with a list of questions"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Check API health status"
            }
        },
        "documentation": "Add /docs to the URL for interactive API documentation"
    }

@app.get("/health", include_in_schema=False)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        Dict with status and timestamp
    """
    try:
        # Basic health check - verify we can import and initialize the LLM service
        from app.services.llm_service import llm_service
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "document-query-api",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)