# LLM Query API

A FastAPI-based service for querying documents using Google's Gemini LLM with semantic search capabilities.

## Features
- Document querying via URL
- Multiple question support in a single request
- **Semantic search using all-MiniLM-L6-v2 embeddings**
- **Intelligent document chunking and retrieval**
- **Parallel processing with asyncio**
- **Local caching with automatic cleanup**
- Secure API key authentication
- Docker containerization

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mithra009/llm-query-api.git
   cd llm-query-api
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Google API key
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Build and run with Docker:
   ```bash
   docker build -t llm-query-api .
   docker run -d --name llm-query-api -p 8000:8000 --env-file .env llm-query-api
   ```

## How It Works

The system uses a sophisticated embedding-based retrieval approach:

1. **Document Processing**: Documents are automatically chunked into 1200-character segments with 200-character overlap
2. **Embedding Generation**: Both document chunks and queries are converted to embeddings using all-MiniLM-L6-v2
3. **Semantic Search**: For each query, the top 5 most relevant chunks are retrieved using cosine similarity
4. **Context-Aware Responses**: Each query receives only the most relevant document context
5. **Parallel Processing**: Multiple queries are processed simultaneously using asyncio
6. **Local Caching**: Embeddings are cached in memory and automatically cleaned up after processing

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
