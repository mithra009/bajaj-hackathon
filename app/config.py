"""
Configuration settings for the LLM service.

This module contains all the configuration parameters for the LLM service,
including model settings, API configurations, and performance parameters.
"""
import os
import warnings
from typing import List, Optional, Dict, Any

# Import BaseSettings from pydantic v1
from pydantic import BaseModel, validator
from pydantic.networks import HttpUrl

class Settings(BaseModel):
    """Application settings and configurations."""
    
    # Application settings
    APP_NAME: str = "Document Query API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    
    # Security
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Model configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash-latest")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", " 8196"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    TOP_P: float = float(os.getenv("TOP_P", "0.95"))
    TOP_K: int = int(os.getenv("TOP_K", "40"))
    
    # Rate limiting
    MAX_CONCURRENT_QUERIES: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "10"))
    MAX_QUERIES_PER_BATCH: int = int(os.getenv("MAX_QUERIES_PER_BATCH", "5"))
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Timeouts (in seconds)
    HTTP_TIMEOUT: int = int(os.getenv("HTTP_TIMEOUT", "30"))
    MODEL_TIMEOUT: int = int(os.getenv("MODEL_TIMEOUT", "60"))
    
    # Caching
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", str(24 * 60 * 60)))  # 24 hours
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File paths
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    
    # Validation
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("OPENAI_API_KEY")
    def check_openai_key(cls, v):
        if not v and os.getenv("ENVIRONMENT") != "test":
            import warnings
            warnings.warn(
                "No OpenAI API key provided. The service will not be able to generate embeddings."
            )
        return v

# Create settings instance
settings = Settings()

# For backward compatibility
MODEL_NAME = settings.MODEL_NAME
MAX_TOKENS = settings.MAX_TOKENS
MAX_QUERIES_PER_BATCH = settings.MAX_CONCURRENT_QUERIES