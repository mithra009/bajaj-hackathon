# LLM Query API

A FastAPI-based service for querying documents using Google's Gemini LLM.

## Features
- Document querying via URL
- Multiple question support in a single request
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

3. Build and run with Docker:
   ```bash
   docker build -t llm-query-api .
   docker run -d --name llm-query-api -p 8000:8000 --env-file .env llm-query-api
   ```

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
