"""Configuration settings for the LLM service."""

# Model configuration
MODEL_NAME = "gpt-4.1-nano"
MAX_TOKENS = 8192  # Increased from 8192 to handle larger documents
MAX_QUERIES_PER_BATCH = 50  # Maximum number of queries to process in a single API call