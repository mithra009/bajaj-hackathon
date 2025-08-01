import os
from typing import List
from pydantic import BaseSettings, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").lower()
    
    # Authentication
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    
    # LLM Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4096"))
    
    @validator('API_KEYS', pre=True)
    def validate_api_keys(cls, v):
        if not v:
            raise ValueError("At least one API key must be provided in the API_KEYS environment variable")
        return v

# Create settings instance
settings = Settings()
