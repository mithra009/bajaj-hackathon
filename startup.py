#!/usr/bin/env python3
"""
Startup script to preload models and ensure system readiness.
"""

import asyncio
import logging
import sys
import os
import signal
from pathlib import Path
from typing import Optional
import traceback

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global flag to track if we should continue running
should_exit = False

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global should_exit
    logger.info("🛑 Received shutdown signal. Cleaning up...")
    should_exit = True

async def preload_models():
    """Preload the embedding model to ensure it's ready."""
    from app.services.embedding_service import embedding_service
    from app.config import EMBEDDING_MODEL, MODEL_LOAD_TIMEOUT
    
    try:
        logger.info(f"🔍 Starting model preloading for: {EMBEDDING_MODEL}")
        
        # Set up a timeout for model loading
        try:
            # Test with a simple text
            test_text = "This is a test document for model loading."
            
            # Run the embedding generation with a timeout
            chunks, embeddings = await asyncio.wait_for(
                embedding_service.generate_document_embeddings(test_text),
                timeout=MODEL_LOAD_TIMEOUT
            )
            
            if len(chunks) > 0 and embeddings.size > 0:
                logger.info("✅ Embedding model loaded successfully!")
                logger.info(f"   - Model: {EMBEDDING_MODEL}")
                logger.info(f"   - Test chunks: {len(chunks)}")
                logger.info(f"   - Embedding shape: {embeddings.shape}")
                return True
            else:
                logger.error("❌ Model loaded but returned empty results")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"⏰ Model loading timed out after {MODEL_LOAD_TIMEOUT} seconds")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"❌ Failed to preload models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_environment() -> bool:
    """Check if all required environment variables are set."""
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    return True

def main() -> int:
    """Main startup function."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("🚀 Starting LLM Query API with Embedding Retrieval...")
    
    # Check environment first
    if not check_environment():
        return 1
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True, mode=0o755)
    logger.info(f"📁 Logs directory: {logs_dir.absolute()}")
    
    # Preload models with a timeout
    try:
        success = asyncio.run(preload_models())
    except Exception as e:
        logger.error(f"❌ Unhandled exception during model loading: {str(e)}")
        logger.error(traceback.format_exc())
        success = False
    
    if success:
        logger.info("✅ System ready! Starting API server...")
        return 0
    else:
        logger.error("❌ System initialization failed!")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)