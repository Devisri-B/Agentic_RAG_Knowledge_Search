"""
File Processor for handling user-uploaded documents.
Processes PDFs, creates FAISS indices for semantic search.
"""

import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Handles processing of user-uploaded PDF files and creates searchable indices.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize FileProcessor with embeddings.

        Args:
            embedding_model: HuggingFace embedding model name
        """
        logger.info(f"Initializing FileProcessor with model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.status = "No files processed"

    def process_files(self, files: list) -> str:
        """
        Process uploaded PDF files and create a FAISS index.

        Args:
            files: List of file objects from Gradio

        Returns:
            Status message
        """
        if not files:
            self.status = "No files uploaded"
            return "No files selected. Using internal docs + web search."

        try:
            logger.info(f"Processing {len(files)} uploaded file(s)...")
            all_docs = []

            # Load all PDF files
            for file_obj in files:
                file_path = file_obj if isinstance(file_obj, str) else file_obj.name

                if file_path.lower().endswith(".pdf"):
                    logger.info(f"Loading PDF: {file_path}")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                else:
                    logger.warning(f"Skipping non-PDF file: {file_path}")

            if not all_docs:
                self.status = "No PDFs found in uploads"
                return "No valid PDF files uploaded. Using internal docs + web search."

            # Split documents into chunks
            logger.info(f"Splitting {len(all_docs)} documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(all_docs)

            # Create FAISS index
            logger.info(f"Creating FAISS index from {len(chunks)} chunks...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)

            self.status = (
                f"Indexed {len(chunks)} chunks from {len(all_docs)} pages"
            )
            logger.info(self.status)
            return f"✓ SUCCESS: {self.status} - Your uploaded documents will be searched first!"

        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            logger.error(error_msg)
            self.status = "Error loading files"
            return f"ERROR: {error_msg} - Falling back to internal docs + web search."

    def retrieve(self, query: str, k: int = 4) -> str:
        """
        Retrieve relevant chunks from uploaded files.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            Retrieved content or empty string if no index
        """
        if not self.vector_store:
            return ""

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            if not docs:
                return ""
            return "\n\n".join(
                [f"[Uploaded Document] {d.page_content}" for d in docs]
            )
        except Exception as e:
            logger.error(f"Error retrieving from uploaded files: {e}")
            return ""

    def has_documents(self) -> bool:
        """
        Check if vector store has documents.

        Returns:
            True if documents are loaded, False otherwise
        """
        return self.vector_store is not None

    def get_status(self) -> str:
        """
        Get current processing status.

        Returns:
            Status message
        """
        return self.status

    def reset(self) -> None:
        """Reset the file processor and clear loaded documents."""
        self.vector_store = None
        self.status = "No files processed"
        logger.info("File processor reset")
