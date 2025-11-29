import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from dotenv import load_dotenv
from src.rag_engine import KnowledgeBase

load_dotenv()

# --- Configuration ---
PDF_PATH = os.path.join("data", "policy.pdf")

# Initialize the RAG engine
kb = KnowledgeBase(pdf_path=PDF_PATH)
# Index documents on startup
kb.load_and_index()

# --- Define Tools ---

@tool
def lookup_internal_policy(query: str) -> str:
    """
    Useful for answering questions about specific internal policies, 
    documents, laws, or the contents of the uploaded PDF file. 
    ALWAYS use this tool first if the question implies looking up specific rules or documents.
    """
    return kb.retrieve(query)

@tool
def search_web(query: str) -> str:
    """
    Useful for finding current events, news, general knowledge, 
    or information that is NOT contained in the internal policy documents.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)

# --- Initialize Agent ---

def get_agent_executor():
    """
    Constructs the Agent with the Gemini LLM and the tools.
    """
    # 1. Setup the Gemini LLM
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Or gemini-1.5-pro
        temperature=0,
        convert_system_message_to_human=True
    )
    
    # 2. Define the Toolkit
    tools = [lookup_internal_policy, search_web]
    
    # 3. Pull a standard prompt for Tool Calling from LangChain Hub
    # This prompt tells the LLM: "You have these tools. Use them to answer."
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # 4. Construct the Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 5. Create the Executor (Runtime)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, # Logs thoughts to console
        return_intermediate_steps=True # Return the thought process
    )
    return agent_executor