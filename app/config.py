"""Configuration settings for the LLM service."""

# Model configuration
MODEL_NAME = "gemini-2.5-flash-lite"
MAX_TOKENS = 2048  # Increased for better response quality
MAX_QUERIES_PER_BATCH = 15  # Maximum number of queries to process in a single API call

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1200  # Maximum size of each text chunk
CHUNK_OVERLAP = 200  # Overlap between consecutive chunks
TOP_K_CHUNKS = 5  # Number of top chunks to retrieve per query

# Processing configuration
MAX_WORKERS = 4  # Number of worker threads for embedding generation
CACHE_TIMEOUT = 3600  # Cache timeout in seconds (1 hour)
