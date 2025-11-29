import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KnowledgeBase:
    """
    Handles document loading, chunking, and vector storage.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vector_store = None
        # Using Gemini Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def load_and_index(self):
        """
        Loads the PDF, splits it into chunks, and builds the FAISS index.
        """
        if not os.path.exists(self.pdf_path):
            print(f"⚠️ Warning: File not found at {self.pdf_path}. Internal search will be empty.")
            return

        print(f"📄 Loading PDF from: {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        print("✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        print(f"💾 Indexing {len(chunks)} chunks into Vector Store...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("✅ Indexing Complete.")

    def retrieve(self, query: str, k: int = 4) -> str:
        """
        Retrieves the top-k most relevant text chunks for a query.
        Returns a single string of context.
        """
        if not self.vector_store:
            return "No internal documents have been indexed."
            
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Combine the content of the retrieved docs
        context = "\n\n".join([f"[Source: Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs])
        return context