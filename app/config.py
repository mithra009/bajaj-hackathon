"""
Configuration settings for the LLM service.

This module contains all the configuration parameters for the LLM service,
including model settings, API configurations, and performance parameters.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, HttpUrl, validator

class Settings(BaseSettings):
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
    MODEL_NAME: str = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    MAX_TOKENS: int = 8192
    MAX_CONTEXT_LENGTH: int = 10000  # Leave some room for the prompt
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.95
    TOP_K: int = 40
    
    # API settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Rate limiting and concurrency
    # Note: Gemini API keys are now managed directly in the LLMService
    MAX_CONCURRENT_QUERIES: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "10"))
    MAX_CONCURRENT_EMBEDDINGS: int = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "5"))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Timeouts (in seconds)
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))
    EMBEDDING_TIMEOUT: int = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
    DOWNLOAD_TIMEOUT: int = int(os.getenv("DOWNLOAD_TIMEOUT", "120"))
    
    # Document processing
    MAX_DOCUMENT_SIZE_MB: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "20"))  # 20MB max
    CHUNK_SIZE: int = 1000  # Characters per chunk
    CHUNK_OVERLAP: int = 200  # Characters overlap between chunks
    
    # Caching
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", str(24 * 60 * 60)))  # 24 hours
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
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