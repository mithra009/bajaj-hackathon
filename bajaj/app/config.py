"""Configuration settings for the LLM service."""
import os

# Model configuration
MODEL_NAME = "gemini-2.5-flash-lite"
MAX_TOKENS = 2048  # Increased for better response quality
MAX_QUERIES_PER_BATCH = 15  # Maximum number of queries to process in a single API call

# Embedding configuration
# Using a smaller model that's faster to load
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800  # Reduced chunk size for better performance
CHUNK_OVERLAP = 100  # Reduced overlap
TOP_K_CHUNKS = 3  # Reduced number of chunks to retrieve

# Processing configuration
MAX_WORKERS = 2  # Reduced number of worker threads to save memory
CACHE_TIMEOUT = 3600  # Cache timeout in seconds (1 hour)

# Model download configuration
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/models")  # Directory to cache models
MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", "300"))  # 5 minutes timeout for model loading
