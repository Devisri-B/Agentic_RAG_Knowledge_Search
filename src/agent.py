import os
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from src.rag_engine import KnowledgeBase
from src.file_processor import FileProcessor

load_dotenv()

file_processor = FileProcessor()

_fallback_kb = KnowledgeBase(pdf_path=os.path.join("data", "policy.pdf"))
try:
    _fallback_kb.load_and_index()
except Exception as e:
    print(f"Fallback KB skipped: {e}")

_search_tool = DuckDuckGoSearchRun()


@tool
def lookup_documents(query: str) -> str:
    """Search the user-uploaded documents for relevant information.
    Use this for questions about content in any uploaded files."""
    if file_processor.has_documents():
        result = file_processor.retrieve(query)
        if result:
            return result
    return _fallback_kb.retrieve(query)


@tool
def search_web(query: str) -> str:
    """Search the web for current events, news, or general knowledge not in uploaded documents."""
    try:
        return _search_tool.run(query)
    except Exception as e:
        return f"Search failed: {e}"


@lru_cache(maxsize=32)
def get_agent_executor(api_key: str):
    """Build (and cache) a LangGraph agent for the given Gemini API key.

    Each visitor supplies their own key (BYOK), so the LLM is created per key.
    The shared tools, embeddings, and RAG index are module-level and reused.
    Cached by key so repeat requests from the same user don't rebuild the graph."""
    if not api_key or not api_key.strip():
        raise ValueError("A Google Gemini API key is required.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=api_key.strip(),
    )
    return create_react_agent(llm, [lookup_documents, search_web])
