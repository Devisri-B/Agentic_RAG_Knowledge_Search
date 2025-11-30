# Agentic RAG Knowledge Search

An Autonomous AI Microservice with Hybrid Retrieval & Self-Evaluation

## Overview

This project is a Production-Grade AI Microservice designed to solve the "Knowledge Silo" problem. Unlike traditional RAG systems that only look at internal documents, this Agentic System intelligently decides where to find the answer.

It uses a LangGraph Router to autonomously switch between:

1. Internal Knowledge Base: A vector database (FAISS) for proprietary policy documents.

1. External Web Search: DuckDuckGo for real-time, public information.

The system includes a custom LLM-as-a-Judge Evaluation Pipeline to continuously benchmark answer accuracy and hallucination rates.

## System Architecture

The system follows a Hybrid RAG architecture. The Agent acts as the central brain, routing user queries to the appropriate tool.
![Architecture](assets/Architecture.png)

## Key Features

- Agentic Reasoning: Replaces rigid if/else logic with a semantic router that understands intent.

- Hybrid Retrieval: Combines the security of local embeddings (HuggingFace) with the vast knowledge of the web.

- Zero-Cost Architecture: optimized to run entirely on Free Tier APIs (Gemini Flash) and CPU-based embeddings.

- Automated Evaluation: Includes a custom "Judge" pipeline that uses one LLM to grade the accuracy of another, producing detailed CSV reports.

- Containerized: Fully Dockerized for consistent deployment across any environment.

## Demo & Outputs

1. Interactive API (Swagger UI)

The service exposes a REST API documentation interface for easy testing.
![Swagger UI Interface](assets/swagger_screenshot.png)

2. Autonomous Tool Routing

The Agent correctly identifies when to look inside the PDF versus when to search the web.
![Agent PDF Routing Logic](assets/PDF_Search.png)
![Agent Web Routing Logic](assets/Web_Search.png)

3. Automated Evaluation Report

A generated CSV report scoring the agent's performance against ground truth data.
![Evaluation Score](assets/evaluation_score.png)

## Setup & Installation

### Prerequisites

- Python 3.10+ (Tested on 3.13)

- Google Gemini API Key

- Docker Desktop (Optional, for containerization)

1. Clone the Repository

    git clone [https://github.com/Devisri-B/Agentic_RAG_Knowledge_Search.git](https://github.com/Devisri-B/Agentic_RAG_Knowledge_Search.git)


2. Configure Environment

    Create a .env file in the root directory:
    
    ```GOOGLE_API_KEY=your_actual_api_key_here```


3. Option A: Run Locally (Python)

    Install Dependencies:
    
    ```pip install -r requirements.txt```
    
    
    Run the Application:
    
    ```python -m src.main```
    
    
    The API will be available at http://localhost:8000/docs.

4. Option B: Run with Docker 

    Build the Image:
    
    ```docker build -t agentic-rag-app .```
    
    
    Run the Container:
    
    ```docker run -p 8000:8000 --env-file .env agentic-rag-app```
    
    
    This isolates the application and ensures it runs consistently on any machine.

## Running Evaluations

This project prioritizes reliability. You can run the evaluation suite to test the agent against a "Golden Dataset" of questions and ground truths.

```python -m tests.evaluate```


This will:

1. Spin up the Agent.

2. Ask it a series of test questions.

3. Use a separate "Judge" LLM to grade the answers (1-10).

4. Generate an evaluation_report.csv file.

## API Reference

Endpoint: ```POST /chat``` click on Try it out button. In the Request body enter your query and click on Execute.

Request:

```{```
 ``` "query": "What are the termination conditions in the policy?"```
```}```


Response:

```{```
  ```"response": "The termination conditions vary depending on the type of treaty or function.\n\nFor provisional application of a multilateral treaty, termination can occur:\n*   By reasonable notice from the newly independent State, party, or contracting State, followed by the expiration of the notice.\n*   For treaties mentioned in Article 17, paragraph 3, by reasonable notice from the newly independent State or all parties/contracting States, followed by the expiration of the notice.\n\nFor provisional application of a bilateral treaty, termination can occur by reasonable notice of termination.\n\nThe functions of a head of delegation or other diplomatic staff can end upon notification of their termination by the sending State to the Organization or conference.\n\nA treaty can become void and terminate under Article 64 if it conflicts with a peremptory norm of general international law. In such cases, parties are released from further obligations, but rights, obligations, or legal situations created prior to termination may be maintained if they don't conflict with the new peremptory norm.\n\nRegarding the general termination of or withdrawal from a treaty:\n*   It can happen in conformity with the treaty's provisions.\n*   It can happen at any time by consent of all parties after consultation with other contracting States.\n\nA multilateral treaty does not terminate solely because the number of parties falls below the number necessary for its entry into force, unless the treaty specifies otherwise."```
```}```
