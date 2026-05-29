import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from src.rag_engine import KnowledgeBase
from src.file_processor import FileProcessor

load_dotenv()

# --- Shared state ---
# file_processor is imported and mutated by main.py's /upload endpoint
file_processor = FileProcessor()

# Optional fallback KB (original policy.pdf); silently skipped if missing
_PDF_PATH = os.path.join("data", "policy.pdf")
_fallback_kb = KnowledgeBase(pdf_path=_PDF_PATH)
try:
    _fallback_kb.load_and_index()
except Exception as e:
    print(f"Fallback KB skipped: {e}")

# --- Tools ---

@tool
def lookup_documents(query: str) -> str:
    """Search the user-uploaded documents for relevant information.
    Use this for questions about content in any uploaded files."""
    if file_processor.has_documents():
        result = file_processor.retrieve(query)
        if result:
            return result
    # Fall back to original KB when no uploads exist
    return _fallback_kb.retrieve(query)

search_tool = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """Search the web for current events, news, or general knowledge not in uploaded documents."""
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"Search failed: {e}"

# --- Agent factory ---

def get_agent_executor():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    print("Initializing Gemini Agent (Model: gemini-2.5-flash-lite)...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    return create_react_agent(llm, [lookup_documents, search_web])
