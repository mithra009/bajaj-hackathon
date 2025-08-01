"""Configuration settings for the LLM service."""

# Model configuration
MODEL_NAME = "gemini-2.5-pro"
MAX_TOKENS = 8192  # Increased from 8192 to handle larger documents
MAX_QUERIES_PER_BATCH = 10  # Maximum number of queries to process in a single API call
