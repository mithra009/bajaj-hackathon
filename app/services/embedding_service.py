import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import re
from app.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS, MAX_WORKERS

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        """
        Initialize the embedding service.
        
        Args:
            model_name: The sentence transformer model to use
            chunk_size: Maximum size of each text chunk
            overlap: Overlap between consecutive chunks
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the first chunk, try to break at a sentence boundary
            if start > 0:
                # Look for sentence endings in the overlap region
                overlap_start = max(start - self.overlap, 0)
                overlap_text = text[overlap_start:end]
                
                # Find the last sentence boundary in the overlap
                sentence_endings = ['.', '!', '?', '\n\n']
                last_sentence_end = -1
                
                for ending in sentence_endings:
                    pos = overlap_text.rfind(ending)
                    if pos != -1:
                        last_sentence_end = max(last_sentence_end, pos)
                
                if last_sentence_end != -1:
                    end = overlap_start + last_sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts asynchronously.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
            
        try:
            # Run embedding generation in a thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_document_embeddings(self, document_text: str) -> Tuple[List[str], np.ndarray]:
        """
        Generate embeddings for document chunks.
        
        Args:
            document_text: The document text to process
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        try:
            # Chunk the document
            chunks = self._chunk_text(document_text)
            if not chunks:
                logger.warning("No chunks created from document text")
                return [], np.array([])
            
            # Generate embeddings for all chunks
            embeddings = await self._generate_embeddings_batch(chunks)
            
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks, embeddings
            
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            raise
    
    async def generate_query_embeddings(self, queries: List[str]) -> np.ndarray:
        """
        Generate embeddings for queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            numpy array of query embeddings
        """
        try:
            embeddings = await self._generate_embeddings_batch(queries)
            logger.info(f"Generated embeddings for {len(queries)} queries")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating query embeddings: {str(e)}")
            raise
    
    def find_top_chunks(self, query_embeddings: np.ndarray, chunk_embeddings: np.ndarray, 
                       chunks: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Find the top-k most similar chunks for each query.
        
        Args:
            query_embeddings: Embeddings of queries
            chunk_embeddings: Embeddings of document chunks
            chunks: List of chunk texts
            top_k: Number of top chunks to retrieve per query
            
        Returns:
            List of lists, where each inner list contains (chunk_text, similarity_score) tuples
        """
        if len(chunk_embeddings) == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        try:
            # Calculate cosine similarity between queries and chunks
            similarities = cosine_similarity(query_embeddings, chunk_embeddings)
            
            top_chunks = []
            for i, query_similarities in enumerate(similarities):
                # Get indices of top-k most similar chunks
                top_indices = np.argsort(query_similarities)[::-1][:top_k]
                
                # Create list of (chunk_text, similarity_score) tuples
                query_top_chunks = []
                for idx in top_indices:
                    if query_similarities[idx] > 0:  # Only include positive similarities
                        query_top_chunks.append((chunks[idx], float(query_similarities[idx])))
                
                top_chunks.append(query_top_chunks)
            
            logger.info(f"Found top {top_k} chunks for {len(query_embeddings)} queries")
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error finding top chunks: {str(e)}")
            raise

# Singleton instance
embedding_service = EmbeddingService() 