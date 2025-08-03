from fastapi import FastAPI, HTTPException, Depends, status, Request, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Security
security = HTTPBearer()
API_KEY = "fd53cda9e372cc74319d047c60acdcc06e62e7e5550a92d842c425b82df84e4d"

app = FastAPI(
    title="LLM Query API",
    version="1.0.0",
    description="API for querying documents using Google's Gemini LLM"
)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

from app.services.llm_service import llm_service
from app.services.query_logger import query_logger

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

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

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def query_document(
    request: Request,
    query_data: QueryRequest = Body(...)
):
    """
    Query the LLM with a list of questions and a document link using embedding-based retrieval.
    """
    logger.info("="*50)
    logger.info("=== NEW REQUEST RECEIVED ===")
    logger.info(f"Document: {query_data.documents}")
    logger.info(f"Number of questions: {len(query_data.questions)}")
    
    try:
        start_time = time.time()
        
        # Log the request details
        logger.info("=== REQUEST DETAILS ===")
        logger.info(f"Document URL: {query_data.documents}")
        logger.info("Questions:")
        for i, question in enumerate(query_data.questions, 1):
            logger.info(f"  {i}. {question}")
        
        # Call the LLM service with additional metadata
        logger.info("=== CALLING LLM SERVICE WITH EMBEDDING RETRIEVAL ===")
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        responses = await llm_service.generate_response(
            queries=query_data.questions,
            document_link=query_data.documents,
            metadata={
                "source_ip": client_host,
                "user_agent": user_agent,
                "embedding_retrieval": True
            }
        )
        
        # Process the responses
        logger.info("=== PROCESSING RESPONSES ===")
        answers = []
        for i in range(1, len(query_data.questions) + 1):
            answer = responses.get(f"Query {i}", "No answer found")
            answers.append(answer)
            logger.info(f"Answer {i}: {answer[:100]}..." if len(answer) > 100 else f"Answer {i}: {answer}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"=== REQUEST COMPLETED ===")
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        logger.info("="*50)
        
        # Return the answers along with the log ID
        return {
            "answers": answers,
            "log_id": responses.get("log_id"),
            "processing_time": processing_time,
            "embedding_retrieval": True
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"=== ERROR ===")
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        logger.error("="*50)
        # Return empty answers array on error
        return {"answers": []}



@app.get("/")
async def root():
    return {
        "message": "LLM Query API with Embedding-Based Retrieval is running",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "embedding_retrieval": True,
            "model": "all-MiniLM-L6-v2",
            "chunk_size": 1200,
            "top_k_chunks": 5
        },
        "endpoints": {
            "document_query": {
                "method": "POST",
                "path": "/hackrx/run",
                "description": "Query documents with semantic search using embeddings"
            },
            "logs": {
                "method": "GET",
                "path": "/hackrx/logs",
                "description": "Retrieve query logs"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Check API health status"
            }
        },
        "documentation": "Add /docs to the URL for interactive API documentation"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)