#!/usr/bin/env python3
"""
Test script for the embedding-based retrieval system.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service

async def test_embedding_retrieval():
    """Test the embedding-based retrieval system."""
    
    # Sample document text
    document_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
    These machines can perform tasks that typically require human intelligence, such as visual perception,
    speech recognition, decision-making, and language translation. Machine learning is a subset of AI that
    focuses on algorithms and statistical models that enable computers to improve their performance on a
    specific task through experience. Deep learning, a subset of machine learning, uses neural networks
    with multiple layers to model and understand complex patterns in data.
    
    Natural Language Processing (NLP) is another important area of AI that deals with the interaction
    between computers and human language. It enables machines to understand, interpret, and generate
    human language in a meaningful way. Applications of NLP include chatbots, language translation,
    sentiment analysis, and text summarization.
    
    Computer Vision is a field of AI that trains computers to interpret and understand visual information
    from the world. It involves developing algorithms that can identify and process images and videos,
    enabling applications like facial recognition, object detection, and autonomous vehicles.
    """
    
    # Sample queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What is natural language processing?",
        "What are the applications of computer vision?",
        "How does deep learning differ from traditional machine learning?"
    ]
    
    print("Testing embedding-based retrieval system...")
    print(f"Document length: {len(document_text)} characters")
    print(f"Number of queries: {len(queries)}")
    print("\n" + "="*50)
    
    try:
        # Test document embedding generation
        print("1. Generating document embeddings...")
        chunks, chunk_embeddings = await embedding_service.generate_document_embeddings(document_text)
        print(f"   Created {len(chunks)} chunks")
        print(f"   Embedding shape: {chunk_embeddings.shape}")
        
        # Test query embedding generation
        print("\n2. Generating query embeddings...")
        query_embeddings = await embedding_service.generate_query_embeddings(queries)
        print(f"   Query embedding shape: {query_embeddings.shape}")
        
        # Test similarity search
        print("\n3. Finding top chunks for each query...")
        top_chunks_per_query = embedding_service.find_top_chunks(
            query_embeddings, chunk_embeddings, chunks, top_k=3
        )
        
        for i, (query, top_chunks) in enumerate(zip(queries, top_chunks_per_query)):
            print(f"\n   Query {i+1}: {query}")
            print(f"   Top chunks found: {len(top_chunks)}")
            for j, (chunk, score) in enumerate(top_chunks):
                print(f"     Chunk {j+1} (similarity: {score:.3f}): {chunk[:100]}...")
        
        # Test the full LLM service integration
        print("\n4. Testing LLM service with embedding retrieval...")
        # Create a mock document link for testing
        document_link = "https://example.com/test-document.pdf"
        
        # Get relevant context using the LLM service
        relevant_contexts = await llm_service._get_relevant_context(queries, document_text, document_link)
        
        print(f"   Generated relevant contexts for {len(relevant_contexts)} queries")
        for i, context in enumerate(relevant_contexts):
            print(f"   Context {i+1} length: {len(context)} characters")
            print(f"   Context {i+1} preview: {context[:100]}...")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_embedding_retrieval()) 