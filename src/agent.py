import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from src.rag_engine import KnowledgeBase

load_dotenv()

# --- Configuration ---
PDF_PATH = os.path.join("data", "policy.pdf")

kb = KnowledgeBase(pdf_path=PDF_PATH)
try:
    kb.load_and_index()
except Exception as e:
    print(f"PDF Load skipped: {e}")

# --- Define Tools ---

@tool
def lookup_internal_policy(query: str) -> str:
    """Useful for answering questions about specific internal policies, documents, laws, or the PDF file."""
    return kb.retrieve(query)

# Initialize the tool ONCE (Global scope)
search_tool = DuckDuckGoSearchRun()

@tool
def search_web(query: str) -> str:
    """Useful for finding current events, news, or general knowledge not in the internal docs."""
    # Use the pre-initialized tool
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"Search failed: {e}"

# --- Initialize Agent ---

def get_agent_executor():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    print("Initializing Gemini Agent (Model: gemini-2.5-flash)...")
    
    # Using the model explicitly found in your check_model script
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    
    tools = [lookup_internal_policy, search_web]
    
    # Create the Agent (LangGraph)
    agent = create_react_agent(llm, tools)
    
    return agent