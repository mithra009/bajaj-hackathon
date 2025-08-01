import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status, Request, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists with proper permissions
logs_dir = Path("/app/logs")
logs_dir.mkdir(exist_ok=True, mode=0o777)

# Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEYS", "").split(",")[0]  # Use first API key for authentication

app = FastAPI(
    title="LLM Query API",
    version="1.0.0",
    description="API for querying documents using Google's Gemini LLM"
)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided API key"""
    if not API_KEY or credentials.credentials != API_KEY:
        logger.warning("Invalid authentication attempt")
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
    Query the LLM with a list of questions and a document link.
    """
    print("\n" + "="*50)
    print("=== NEW REQUEST RECEIVED ===")
    print(f"Document: {query_data.documents}")
    print(f"Number of questions: {len(query_data.questions)}")
    
    try:
        start_time = time.time()
        
        # Log the request details
        print("\n=== REQUEST DETAILS ===")
        print(f"Document URL: {query_data.documents}")
        print("\nQuestions:")
        for i, question in enumerate(query_data.questions, 1):
            print(f"  {i}. {question}")
        
        # Call the LLM service with additional metadata
        print("\n=== CALLING LLM SERVICE ===")
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        responses = await llm_service.generate_response(
            queries=query_data.questions,
            document_link=query_data.documents,
            metadata={
                "source_ip": client_host,
                "user_agent": user_agent
            }
        )
        
        # Process the responses
        print("\n=== PROCESSING RESPONSES ===")
        answers = []
        for i in range(1, len(query_data.questions) + 1):
            answer = responses.get(f"Query {i}", "No answer found")
            answers.append(answer)
            print(f"Answer {i}: {answer[:100]}..." if len(answer) > 100 else f"Answer {i}: {answer}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\n=== REQUEST COMPLETED ===")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print("="*50 + "\n")
        
        # Return the answers along with the log ID
        return {
            "answers": answers,
            "log_id": responses.get("log_id")
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"\n=== ERROR ===\n{error_msg}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        # Return empty answers array on error
        return {"answers": []}



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

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting server on {host}:{port} with log level: {log_level}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True
    )
