import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vector_store = None
        self.index_path = "faiss_index"
        self.embeddings = get_embeddings()

    def load_and_index(self):
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                logger.info("Loaded FAISS index from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load cached index: {e}. Re-indexing...")

        if not os.path.exists(self.pdf_path):
            logger.warning(f"PDF not found: {self.pdf_path}")
            return

        docs = PyPDFLoader(self.pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.index_path)
        logger.info(f"Indexed {len(chunks)} chunks and saved to disk.")

    def retrieve(self, query: str, k: int = 4) -> str:
        if not self.vector_store:
            return "No internal documents have been indexed."
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join(f"[Source: Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs)
