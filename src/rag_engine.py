import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class KnowledgeBase:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vector_store = None
        # Initialize Local Embeddings
        print("Initializing Local Embeddings (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_path = "faiss_index"
        
    def load_and_index(self):
        # 1. Try to load existing index
        if os.path.exists(self.index_path):
            print("Found cached index on disk. Loading...")
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("Loaded index from disk.")
                return
            except Exception as e:
                print(f"Could not load cache: {e}. Re-indexing...")

        # 2. Check PDF
        if not os.path.exists(self.pdf_path):
            print(f"Warning: File not found at {self.pdf_path}")
            return

        # 3. Load and Split
        print(f"Loading PDF from: {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        # 4. Create Index (Local)
        print(f"Indexing {len(chunks)} chunks locally...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # 5. Save
        self.vector_store.save_local(self.index_path)
        print("Indexing Complete and Saved.")

    def retrieve(self, query: str, k: int = 4) -> str:
        if not self.vector_store:
            return "No internal documents have been indexed."
            
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([f"[Source: Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs])