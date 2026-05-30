
# Agentic RAG Knowledge Search

An Autonomous AI Microservice with Hybrid Retrieval & Self-Evaluation. Built with FastAPI, LangChain, Docker, and Google Gemini.

> **Deploying on HuggingFace Spaces:** create a **Docker** Space and push this repo. No API-key
> secret is required — the app uses **Bring Your Own Key (BYOK)**: each visitor enters their own
> Google Gemini key in the UI, so the public demo never spends your quota. The container runs the
> FastAPI backend (internal, port 8000) and the Gradio UI (public, port 7860) together via
> `start.sh`. Models are baked into the image at build time, so cold starts are fast.

## Overview

This project is a Production-Grade AI Microservice designed to solve the "Knowledge Silo" problem. Unlike traditional RAG systems that only look at internal documents, this Agentic System intelligently decides where to find the answer.

It uses a LangGraph Router to autonomously switch between:

1. Internal Knowledge Base: A vector database (FAISS) for proprietary policy documents.

2. External Web Search: DuckDuckGo for real-time, public information.

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
  
- REST API: Fully documented API using FastAPI.

- User Document Upload: Users can upload their own files (PDF, DOCX, TXT, MD, CSV) at runtime. Uploaded documents are indexed instantly and searched first; if nothing is uploaded, the agent falls back to the built-in legal/policy document.

- Live, No-Cost Evaluation Metrics: Every answer is scored in real time using local models only (no extra API calls) — see [Live Evaluation Metrics](#live-evaluation-metrics) below.

- Single-Image Deployment: The FastAPI backend and Gradio UI run together from one Docker image (`start.sh`), ready for HuggingFace Spaces.

## User Document Upload

The Gradio UI includes an upload panel. Users can drag in one or more files and click **Process & Index Files**.

- Supported formats: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`
- Multiple files can be uploaded; new uploads are merged into the existing index.
- **Clear Uploaded Documents** resets the index back to the built-in default.
- If no files are uploaded, the agent uses the bundled `data/policy.pdf` as the default knowledge base.

## Live Evaluation Metrics

After every response, three metrics are computed **locally — no extra LLM/API calls** — and shown in the UI. This keeps hallucination/quality monitoring free and fast, even on CPU-only HuggingFace Spaces.

| Metric | Always shown? | How it works | What it catches |
|---|---|---|---|
| **Faithfulness** | Yes | NLI entailment (`cross-encoder/nli-deberta-v3-small`): each answer sentence is checked for *entailment* against the best-matching source passage it actually used (documents **or** web results). | Hallucinations and contradictions — not just topic drift. A claim that contradicts the source scores near 0. |
| **Answer Relevance** | Yes | Cosine similarity between the question and the answer (`all-MiniLM-L6-v2`). Needs no reference. | Off-topic or evasive answers. |
| **Accuracy** | Only with a reference | ROUGE-L F1 between the answer and a user-supplied reference answer. | Drift from a known-correct answer. |

> **Why NLI instead of plain similarity?** Cosine similarity measures *topical* overlap, so "the treaty can be terminated" and "the treaty cannot be terminated" score nearly identically despite opposite meaning. The NLI model checks logical *entailment*, so it correctly flags contradictions as unfaithful.

The offline `tests/evaluate.py` pipeline additionally uses an LLM-as-a-Judge for a second opinion against a golden dataset (see [Running Evaluations](#running-evaluations)).

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

### Project Structure

- src/rag_engine.py: Handles PDF ingestion and Vector Database (FAISS).
- src/agent.py: Defines the Agent, Tools, and LangChain logic.
- src/main.py: The FastAPI server entry point.
- src/file_processor.py: Indexes user-uploaded files (PDF/DOCX/TXT/MD/CSV) at runtime.
- src/embeddings.py: Shared, single-load embedding model reused by RAG and the evaluator.
- src/evaluator.py: Local evaluation metrics (NLI faithfulness, relevance, accuracy).
- src/prefetch_models.py: Downloads models at image-build time for fast cold starts.
- app.py: The Gradio user interface (chat, file upload, live metrics).
- start.sh: Launches the FastAPI backend and Gradio UI together (used by Docker).
- data/: Place your default PDF documents here. (used when user didn't upload any files)

### Prerequisites

- Python 3.10+ (Tested on 3.13)
- Google Gemini API Key
- Docker Desktop (Optional, for containerization)
  
1. Clone the Repository

    git clone [https://github.com/Devisri-B/Agentic_RAG_Knowledge_Search.git](https://github.com/Devisri-B/Agentic_RAG_Knowledge_Search.git)


2. Configure Environment

    The app uses **BYOK** — you enter your Gemini key directly in the UI, so no `.env` is needed to chat.

    A `.env` file is only required to run the **offline evaluation** suite (`tests/evaluate.py`):

    ```GOOGLE_API_KEY=your_actual_api_key_here```

3. Add Data

- Place your PDF file (e.g., policy documents, course materials) into the data/ folder and rename it to policy.pdf (or update src/agent.py to match your filename).

4. Option A: Run Locally (Python)

    Install Dependencies:
    
    ```pip install -r requirements.txt```
    
    
    Run the Application:
    
    ```python -m src.main```
    
    
    The API will be available at http://localhost:8000/docs.

5. Option B: Run with Docker 

    Build the Image:
    
    ```docker build -t agentic-rag-app .```
    
    
    Run the Container:
    
    ```docker run -p 8000:8000 --env-file .env agentic-rag-app```
    
    
    This isolates the application and ensures it runs consistently on any machine.

6. Option C: Run the full app (UI + API together)

    To run the Gradio UI and FastAPI backend together exactly as they run in the container:

    ```bash start.sh```

    Then open the UI at http://localhost:7860 (the API stays internal on port 8000).
    To run them separately during development, use two terminals: `python -m src.main` and `python app.py`.

## Deploying on HuggingFace Spaces

This repo is configured as a **Docker** Space (see the YAML frontmatter at the top of this file).

1. Create a new Space → choose **Docker** as the SDK.
2. Push this repository to the Space.
3. HuggingFace builds the image and serves the Gradio UI at the Space URL.

No API-key secret is needed: the app uses **Bring Your Own Key (BYOK)** — each visitor enters their own Google Gemini key in the UI (see below).

The container runs the FastAPI backend (internal, port 8000) and the Gradio UI (public, port 7860) together via `start.sh`. The embedding and NLI models are baked into the image at build time, so cold starts are fast and require no network access for models.

## API Key Strategy for Public Deployment

The Gemini free tier allows roughly **1,500 requests/day**, and each user question costs **2 calls** (one to route, one to answer) — about **750 questions/day total, shared across everyone**. If you publish a Space using your personal key, public traffic will exhaust it quickly and run on *your* quota.

**This app uses Bring Your Own Key (BYOK).** Each visitor enters their own Google Gemini API key in the UI:

- The key is sent only with that visitor's requests and is **never stored** on the server.
- A new free key takes seconds to create at [Google AI Studio](https://aistudio.google.com/apikey).
- The public Space therefore costs **you nothing** and can never exhaust your personal quota.
- A question cannot be submitted until a key is provided; an invalid key returns a clear error.

This is the standard pattern for public LLM demos. (Alternatives, if you ever want them: keep your own key as a Space secret with rate limiting, make the Space private, or upgrade to a paid Gemini plan.)

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
