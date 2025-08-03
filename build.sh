#!/bin/bash

# Build script for LLM Query API
# This script builds the Docker image step by step

set -e

echo "🔨 Building LLM Query API Docker image..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "📦 Building with basic Dockerfile..."
docker build -f Dockerfile.basic -t llm-query-api:basic .

if [ $? -eq 0 ]; then
    echo "✅ Basic build successful!"
    
    echo "🧪 Testing the container..."
    docker run --rm -d --name test-container -p 8000:8000 llm-query-api:basic
    
    echo "⏳ Waiting for container to start..."
    sleep 15
    
    # Test if the container is running
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Container is running successfully!"
        echo "🌐 API is available at: http://localhost:8000"
        echo "📚 API documentation at: http://localhost:8000/docs"
        
        # Stop the test container
        docker stop test-container
        docker rm test-container
        
        echo ""
        echo "🎉 Build and test completed successfully!"
        echo ""
        echo "🚀 To run with docker-compose:"
        echo "docker-compose up -d"
        echo ""
        echo "🛑 To stop:"
        echo "docker-compose down"
    else
        echo "❌ Container failed to start properly."
        echo "📋 Checking logs..."
        docker logs test-container
        docker stop test-container
        docker rm test-container
        exit 1
    fi
else
    echo "❌ Build failed!"
    exit 1
fi 