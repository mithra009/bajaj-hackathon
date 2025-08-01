from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
from datetime import datetime

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

@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_token)])
async def query_document(request: QueryRequest):
    """
    Query the LLM with a list of questions and a document link.
    """
    print("\n" + "="*50)
    print("=== NEW REQUEST RECEIVED ===")
    print(f"Document: {request.documents}")
    print(f"Number of questions: {len(request.questions)}")
    
    try:
        start_time = time.time()
        
        # Log the request details
        print("\n=== REQUEST DETAILS ===")
        print(f"Document URL: {request.documents}")
        print("\nQuestions:")
        for i, question in enumerate(request.questions, 1):
            print(f"  {i}. {question}")
        
        # Call the LLM service
        print("\n=== CALLING LLM SERVICE ===")
        responses = await llm_service.generate_response(
            queries=request.questions,
            document_link=request.documents
        )
        
        # Process the responses
        print("\n=== PROCESSING RESPONSES ===")
        answers = []
        for i in range(1, len(request.questions) + 1):
            answer = responses.get(f"Query {i}", "No answer found")
            answers.append(answer)
            print(f"Answer {i}: {answer[:100]}..." if len(answer) > 100 else f"Answer {i}: {answer}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\n=== REQUEST COMPLETED ===")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print("="*50 + "\n")
        
        # Return just the answers without the error field
        return {"answers": answers}
        
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
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
