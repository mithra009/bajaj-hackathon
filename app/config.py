"""Configuration settings for the LLM service."""

# Model configuration
MODEL_NAME = "gemini-2.5-flash-lite"
MAX_TOKENS = 64000  # Increased from 8192 to handle larger documents
MAX_QUERIES_PER_BATCH = 50  # Maximum number of queries to process in a single API call
