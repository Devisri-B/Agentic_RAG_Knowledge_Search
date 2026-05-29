import logging
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}


def _loader_for(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(file_path)
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    return None


class FileProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.status = "No files processed"
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def process_files(self, file_paths: list[str]) -> str:
        """Index a list of file paths into the FAISS vector store."""
        if not file_paths:
            return "No files provided."

        all_docs = []
        skipped = []

        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append(os.path.basename(path))
                continue
            try:
                loader = _loader_for(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source_file"] = os.path.basename(path)
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} page(s) from {os.path.basename(path)}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                skipped.append(os.path.basename(path))

        if not all_docs:
            self.status = "No supported files could be loaded"
            return (
                f"Could not load any files. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. "
                + (f"Skipped: {', '.join(skipped)}" if skipped else "")
            )

        chunks = self._splitter.split_documents(all_docs)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            # Merge into existing index so previous uploads are retained
            new_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.merge_from(new_store)

        file_count = len(file_paths) - len(skipped)
        self.status = f"Indexed {len(chunks)} chunks from {file_count} file(s)"
        note = f" (skipped: {', '.join(skipped)})" if skipped else ""
        logger.info(self.status)
        return f"Done: {self.status}{note}"

    def retrieve(self, query: str, k: int = 4) -> str:
        if not self.vector_store:
            return ""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            if not docs:
                return ""
            return "\n\n".join(
                f"[File: {d.metadata.get('source_file', '?')}] {d.page_content}"
                for d in docs
            )
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return ""

    def has_documents(self) -> bool:
        return self.vector_store is not None

    def get_status(self) -> str:
        return self.status

    def reset(self) -> None:
        self.vector_store = None
        self.status = "No files processed"
        logger.info("FileProcessor reset")
