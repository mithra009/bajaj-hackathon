import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status, Request, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables for configuration
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

# Ensure logs directory exists
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True, mode=0o775)

# --- Security and Authentication ---
security = HTTPBearer()
# Use the first key from the comma-separated environment variable for API authentication.
# This MUST be set in your deployment environment.
AUTH_API_KEY = os.getenv("API_KEYS", "").split(",")[0]

app = FastAPI(
    title="LLM Query API",
    version="1.0.0",
    description="API for querying documents using Google's Gemini LLM"
)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided API key against the one from the environment."""
    if not AUTH_API_KEY or credentials.credentials != AUTH_API_KEY:
        logger.warning("Invalid authentication attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Import services after setting up config
from app.services.llm_service import llm_service
from app.services.query_logger import query_logger

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    log_id: str | None = None

class QueryLogResponse(BaseModel):
    id: str
    timestamp: str
    document_link: str
    queries: List[str]
    responses: Dict[str, str]
    metadata: Dict[str, Any]

class QueryLogsResponse(BaseModel):
    logs: List[QueryLogResponse]


# --- API Endpoints ---
@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def query_document(request: Request, query_data: QueryRequest = Body(...)):
    """
    Query the LLM with a list of questions and a document link.
    """
    logger.info(f"New query request received for document: {query_data.documents}")
    try:
        start_time = time.time()
        
        request_metadata = {
            "source_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # Call the LLM service
        responses = await llm_service.generate_response(
            queries=query_data.questions,
            document_link=query_data.documents,
            metadata=request_metadata
        )
        
        # Process responses
        log_id = responses.pop("log_id", None)
        answers = [responses.get(f"Query {i+1}", "No answer found.") for i in range(len(query_data.questions))]

        processing_time = time.time() - start_time
        logger.info(f"Request processed in {processing_time:.2f} seconds. Log ID: {log_id}")
        
        return {"answers": answers, "log_id": log_id}
        
    except ValueError as e:
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.critical(f"An unexpected error occurred during query processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred.")

@app.get("/hackrx/logs", response_model=QueryLogsResponse, dependencies=[Depends(verify_token)])
async def get_query_logs(limit: int = 100):
    """
    Retrieve the most recent query logs.
    """
    try:
        logs = query_logger.get_logs(limit=limit)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve logs.")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting server on {host}:{port} with log level {log_level}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True
    )