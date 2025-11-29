# Agentic RAG Knowledge Search

A production-ready AI Microservice that uses an Agentic Router to intelligently decide between retrieving answers from internal private documents (RAG) or searching the live web.

Built with FastAPI, LangChain, Docker, and Google Gemini.

## Features

- Agentic Routing: Uses an LLM to decide where to get information.

- Hybrid Tools:

    - lookup_internal_policy: Searches local PDFs using FAISS + Embeddings.
    
    - search_web: Searches DuckDuckGo for real-time info.

- REST API: Fully documented API using FastAPI.

- Containerized: Docker support for easy deployment.

## Project Structure

- src/rag_engine.py: Handles PDF ingestion and Vector Database (FAISS).

- src/agent.py: Defines the Agent, Tools, and LangChain logic.

- src/main.py: The FastAPI server entry point.

- data/: Place your PDF documents here.

## Setup & Installation

1. Prerequisites

- Get a Free Google Gemini API Key.

- Create a .env file in this directory:

```GOOGLE_API_KEY=AIzaSyD...your_key_here...```

2. Add Data

Place your PDF file (e.g., policy documents, course materials) into the data/ folder and rename it to policy.pdf (or update src/agent.py to match your filename).

3. Run with Docker (Recommended)

```docker build -t agentic-rag .```

```docker run -p 8000:8000 --env-file .env agentic-rag```

4. Run Locally

```pip install -r requirements.txt```

```python src/main.py``` 

## API Usage

```Endpoint: POST /chat

{
  "query": "What are the rules for termination in the policy?"
}



Response:

{
  "response": "According to the policy, termination requires...",
  "steps": [
    {
      "tool": "lookup_internal_policy",
      "tool_input": "termination rules"
    }
  ]
}

