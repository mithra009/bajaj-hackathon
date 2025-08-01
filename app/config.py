import os
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

class Settings:
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").lower()
    
    # LLM Configuration (can still be set via environment)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))

# Create a single settings instance for the app to use
settings = Settings()