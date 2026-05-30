import logging
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}


def _loader_for(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(path)
    if ext == ".docx":
        return Docx2txtLoader(path)
    return None


class FileProcessor:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store = None
        self.status = "No files processed"
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_files(self, file_paths: list[str]) -> str:
        if not file_paths:
            return "No files provided."

        all_docs, skipped = [], []
        for path in file_paths:
            if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
                skipped.append(os.path.basename(path))
                continue
            try:
                docs = _loader_for(path).load()
                for d in docs:
                    d.metadata["source_file"] = os.path.basename(path)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                skipped.append(os.path.basename(path))

        if not all_docs:
            self.status = "No supported files could be loaded"
            return f"Could not load any files. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

        chunks = self._splitter.split_documents(all_docs)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.merge_from(FAISS.from_documents(chunks, self.embeddings))

        file_count = len(file_paths) - len(skipped)
        self.status = f"Indexed {len(chunks)} chunks from {file_count} file(s)"
        note = f" (skipped: {', '.join(skipped)})" if skipped else ""
        return f"Done: {self.status}{note}"

    def retrieve(self, query: str, k: int = 4) -> str:
        if not self.vector_store:
            return ""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join(
                f"[File: {d.metadata.get('source_file', '?')}] {d.page_content}" for d in docs
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
