import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from src.agent import get_agent_executor, file_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG Service",
    description="AI microservice that routes between uploaded documents and web search.",
    version="3.0",
)

try:
    agent_executor = get_agent_executor()
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent_executor = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {
        "status": "active",
        "service": "Agentic Knowledge Search",
        "docs_url": "/docs",
        "uploaded_docs": file_processor.get_status(),
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accept uploaded files, index them for RAG, return a status message."""
    tmp_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(dest)
            logger.info(f"Saved upload: {upload.filename}")

        status = file_processor.process_files(saved_paths)
        logger.info(f"Processing result: {status}")
        return JSONResponse({"status": status})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/reset")
async def reset_documents():
    """Clear all uploaded documents from the index."""
    file_processor.reset()
    return {"status": "Uploaded documents cleared."}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized (check API key)")

    try:
        logger.info(f"Query: {request.query}")
        result = agent_executor.invoke({"messages": [("user", request.query)]})
        last_message = result["messages"][-1]

        content = last_message.content
        if isinstance(content, list):
            content = " ".join(
                block["text"] if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type") == "text"
            )

        return QueryResponse(response=str(content))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
