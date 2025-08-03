#!/usr/bin/env python3
"""
Startup script to preload models and ensure system readiness.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.embedding_service import embedding_service
from app.config import EMBEDDING_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def preload_models():
    """Preload the embedding model to ensure it's ready."""
    try:
        logger.info("Starting model preloading...")
        
        # Test the embedding service
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        
        # Test with a simple text
        test_text = "This is a test document for model loading."
        chunks, embeddings = await embedding_service.generate_document_embeddings(test_text)
        
        logger.info(f"✅ Embedding model loaded successfully!")
        logger.info(f"   - Model: {EMBEDDING_MODEL}")
        logger.info(f"   - Test chunks: {len(chunks)}")
        logger.info(f"   - Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to preload models: {str(e)}")
        return False

def main():
    """Main startup function."""
    logger.info("🚀 Starting LLM Query API with Embedding Retrieval...")
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.info(f"📁 Logs directory: {logs_dir.absolute()}")
    
    # Preload models
    success = asyncio.run(preload_models())
    
    if success:
        logger.info("✅ System ready! Starting API server...")
        return 0
    else:
        logger.error("❌ System initialization failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 