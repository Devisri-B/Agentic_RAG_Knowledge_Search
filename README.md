#  Retrieval-Augmented Generation (RAG) Knowledge Search Engine

This project implements a robust Retrieval-Augmented Generation (RAG) pipeline designed to enable reliable, context-aware question answering over custom, private documents (PDFs, DOCX, CSV, TXT). It minimizes Large Language Model (LLM) "hallucinations" by grounding all answers strictly within the retrieved source material, providing full traceability and high domain-specific accuracy.

## Features

  * **Multi-Document Support:** Easily ingest and index documents of various formats (`.pdf`, `.txt`, `.docx`, `.csv`).
  * **Context Grounding:** Utilizes a custom prompt that forces the LLM to use **only** the provided context, ensuring factual and traceable answers  .
  * **Vectorization & Retrieval:** Leverages **Google Generative AI Embeddings** and the high-performance **FAISS** vector store for sub-second similarity search  .
  * **End-to-End Pipeline:** Wires the components into a `RetrievalQA` chain for seamless question processing  .
  * **Experiment Tracking:** Integrates **MLflow** to track key parameters (`chunk_size`, `chunk_overlap`, `session_id`) and log question/answer pairs along with source documents for model evaluation and future feedback  .

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | `LangChain` | Wires together the document processing, embedding, and LLM query steps  . |
| **LLM/Embeddings** | `Gemini 1.5 Pro` / `GoogleGenerativeAIEmbeddings` | Generative AI model and state-of-the-art embedding generation. |
| **Vector Database** | `FAISS (CPU)` | Efficient, in-memory index for fast similarity search during retrieval  . |
| **Experimentation** | `MLflow` | Tracks document processing parameters and conversation history/metrics  . |
| **Document Loaders**| `PyPDFLoader`, `UnstructuredWordDocumentLoader`, etc. | Handles ingestion of various file types  . |

##  RAG Pipeline Overview

The system utilizes a standard RAG pattern to answer questions:

1.  **Ingestion & Chunking:** Documents are loaded and split into small, overlapping chunks (default 1,000 tokens, 200 overlap) using `RecursiveCharacterTextSplitter`.
2.  **Embedding:** Each chunk is converted into a high-dimensional vector using **Google Generative AI Embeddings**.
3.  **Vector Store:** The vectors are indexed in the **FAISS** library for efficient storage and retrieval.
4.  **Retrieval:** At query time, the user's question is embedded and used to query FAISS for the Top-K (default $k=4$) most relevant document chunks.
5.  **Generation:** The retrieved chunks are passed to the **Gemini 1.5 Pro** LLM along with a strict prompt instructing it to answer **only** based on the provided context, producing the final, grounded response.


### Prerequisites

1.  **Python 3.8+**
2.  A **Google API Key** for the Gemini models (set as an environment variable or passed to the `DocumentQASystem` class).

### Usage

The core functionality is encapsulated in the `DocumentQASystem` class, which manages the entire RAG workflow.

1.  **Initialize the System:**
    ```python
    qa_system = DocumentQASystem(api_key='YOUR_GEMINI_API_KEY')
    ```
2.  **Process Documents:** Place your documents (`.pdf`, `.txt`, `.docx`, `.csv`) in a directory and provide the file paths.
    ```python
    sample_files = ["./data/manual.pdf", "./data/report.docx"]
    chunk_count = qa_system.process_documents(
        file_paths=sample_files,
        chunk_size=1000, 
        chunk_overlap=200
    )
    print(f"Processed {chunk_count} document chunks.")
    ```
3.  **Ask Questions:**
    ```python
    result = qa_system.ask_question("Under what conditions can a treaty be terminated?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    ```
### Contributing:
Contributions are welcome! Please open issues or submit pull requests to suggest improvements or new features.
