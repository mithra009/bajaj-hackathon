#!/bin/bash

# LLM Query API Deployment Script
# This script builds and runs the application with embedding-based retrieval

set -e

echo "🚀 Deploying LLM Query API with Embedding Retrieval..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "📦 Building Docker image..."
docker-compose build

echo "🔧 Starting services..."
docker-compose up -d

echo "⏳ Waiting for service to be ready..."
sleep 10

# Check if service is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Service is running successfully!"
    echo "🌐 API is available at: http://localhost:8000"
    echo "📚 API documentation at: http://localhost:8000/docs"
    echo ""
    echo "📋 Quick test:"
    echo "curl -X GET http://localhost:8000/"
    echo ""
    echo "🔍 To view logs:"
    echo "docker-compose logs -f"
    echo ""
    echo "🛑 To stop the service:"
    echo "docker-compose down"
else
    echo "❌ Service failed to start properly."
    echo "📋 Checking logs..."
    docker-compose logs
    exit 1
fi 