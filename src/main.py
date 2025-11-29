from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent import get_agent_executor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG Service",
    description="An AI Microservice that routes between Internal Docs and Web Search.",
    version="1.0"
)

# Initialize Agent Logic
try:
    agent_executor = get_agent_executor()
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent_executor = None

# --- API Models ---
class QueryRequest(BaseModel):
    query: str

class StepInfo(BaseModel):
    tool: str
    tool_input: str

class QueryResponse(BaseModel):
    response: str
    steps: list[StepInfo] = []

# --- Routes ---

@app.get("/")
async def root():
    return {
        "status": "active", 
        "service": "Agentic Knowledge Search", 
        "docs_url": "/docs"
    }

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized (Check API Keys)")
    
    try:
        logger.info(f"Received query: {request.query}")
        
        # Invoke the agent
        result = agent_executor.invoke({"input": request.query})
        
        # Extract the 'Thought Process' (Intermediate Steps) to show the recruiter
        # 'intermediate_steps' is a list of tuples: (AgentAction, Observation)
        steps_data = []
        if "intermediate_steps" in result:
            for action, observation in result["intermediate_steps"]:
                steps_data.append(StepInfo(
                    tool=action.tool,
                    tool_input=str(action.tool_input)
                ))

        return QueryResponse(
            response=result["output"],
            steps=steps_data
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)