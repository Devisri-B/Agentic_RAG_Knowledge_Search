from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import get_agent_executor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG Service",
    description="An AI Microservice that routes between Internal Docs and Web Search using LangGraph.",
    version="2.0"
)

# Initialize Agent
try:
    agent_executor = get_agent_executor()
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent_executor = None

# --- API Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# --- Routes ---

@app.get("/")
async def root():
    return {
        "status": "active", 
        "service": "Agentic Knowledge Search (LangGraph)", 
        "docs_url": "/docs"
    }

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized (Check API Keys)")
    
    try:
        logger.info(f"Received query: {request.query}")
        
        # LangGraph Input Format
        # pass a dictionary with "messages"
        inputs = {"messages": [("user", request.query)]}
        
        # Invoke the agent
        result = agent_executor.invoke(inputs)
        
        # The result contains the entire conversation state. 
        # The last message is the AI's answer.
        last_message = result["messages"][-1]
        
        return QueryResponse(
            response=last_message.content
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)