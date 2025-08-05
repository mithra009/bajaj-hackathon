# Document Query API

A high-performance API for querying documents using Google's Gemini LLM with minimal latency. This service processes PDF documents, extracts text, generates embeddings, and answers questions using OpenAI's embeddings and Gemini Flash Lite model.

## Features

- **Fast Document Processing**: Efficient PDF text extraction and chunking
- **Low Latency**: Parallel query processing with multiple API keys
- **Semantic Search**: Uses OpenAI's text-embedding-3-small for accurate results
- **Intelligent Chunking**: Smart document segmentation with overlap
- **Scalable**: Built with FastAPI and async support
- **Secure**: API key authentication and CORS protection
- **Production Ready**: Containerized with Docker and optimized for deployment

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key (for embeddings)
- Gemini API key(s) (for text generation)

### 1. Clone the repository

```bash
git clone <repository-url>
cd llm-query-retrieval-system/testing
```

### 2. Set up environment variables

Copy the example environment file and update with your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys and configuration.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the service

#### Option 1: Using Docker (Recommended)

```bash
docker-compose up --build
```

#### Option 2: Local development

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the service is running, you can access:

- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

## Usage Example

### Query a Document

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your_api_key' \
  -H 'Content-Type: application/json' \
  -d '{
    "document_url": "https://example.com/insurance-policy.pdf",
    "queries": [
      "What is the policy coverage for emergency hospitalization?",
      "What is the claim submission process?"
    ],
    "timeout": 30
  }'
```

### Response

```json
{
  "query_1": "Answer to first question...",
  "query_2": "Answer to second question..."
}
```

## Configuration

See `.env.example` for all available configuration options. Key settings include:

- `OPENAI_API_KEY`: Your OpenAI API key (required for embeddings)
- `GEMINI_API_KEYS`: Comma-separated list of Gemini API keys
- `MAX_CONCURRENT_QUERIES`: Maximum parallel queries (default: 10)
- `MAX_CONCURRENT_EMBEDDINGS`: Maximum parallel embedding generations (default: 5)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 86400)

## How It Works

1. **Document Processing**:
   - Downloads and parses PDF documents
   - Chunks text into 1000-character segments with 200-character overlap
   - Generates embeddings using OpenAI's text-embedding-3-small

2. **Query Processing**:
   - Generates embeddings for each query
   - Finds most relevant document chunks using cosine similarity
   - Uses Gemini Flash Lite for generating answers
   - Processes multiple queries in parallel with API key rotation

3. **Optimizations**:
   - Asynchronous processing for maximum throughput
   - Connection pooling for API requests
   - Intelligent retry mechanism for failed requests
   - Response caching to minimize redundant processing

## API Usage

### Authentication
Include your API key in the Authorization header:
```
Authorization: Bearer your-api-key-here
```

### Query Endpoint
```
POST /hackrx/run
Content-Type: application/json

{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "Question 1?",
        "Question 2?"
    ]
}
```

## Development

### Local Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## License
MIT
