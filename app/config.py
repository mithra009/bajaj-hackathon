"""Configuration settings for the LLM service."""

# Model configuration
MODEL_NAME = "gemini-2.5-flash"
MAX_TOKENS = 5512  # Increased from 8192 to handle larger documents
MAX_QUERIES_PER_BATCH = 100  # Maximum number of queries to process in a single API call
