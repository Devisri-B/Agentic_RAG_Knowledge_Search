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

MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = (
    "You are a precise research assistant with two tools:\n"
    "- `lookup_documents`: searches the user's uploaded files and the internal knowledge base. "
    "Prefer this for questions about policies, documents, or any uploaded content.\n"
    "- `search_web`: searches the live web. Use this for current events, news, or general "
    "knowledge that is unlikely to be in the documents.\n\n"
    "Guidelines:\n"
    "1. Choose the tool that best fits the question; use both if needed.\n"
    "2. Ground your answer in the retrieved content — do not invent facts. If the documents "
    "do not contain the answer, say so and try the web.\n"
    "3. Use the conversation summary and recent turns to resolve follow-up questions "
    "(e.g. pronouns like 'it' or 'that').\n"
    "4. Be concise, accurate, and cite the source of your information when relevant."
)


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
def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    """Cached Gemini client for a given key — reused by the agent and the summarizer."""
    if not api_key or not api_key.strip():
        raise ValueError("A Google Gemini API key is required.")
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        google_api_key=api_key.strip(),
    )


@lru_cache(maxsize=32)
def get_agent_executor(api_key: str):
    """Build (and cache) a LangGraph ReAct agent for the given Gemini API key.

    Each visitor supplies their own key (BYOK). The shared tools, embeddings, and
    RAG index are module-level and reused; only the LLM is per-key. Cached by key
    so repeat requests from the same user don't rebuild the graph."""
    return create_react_agent(
        get_llm(api_key),
        [lookup_documents, search_web],
        prompt=SYSTEM_PROMPT,
    )
