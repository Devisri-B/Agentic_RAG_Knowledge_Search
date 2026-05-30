import os
import re
import time
import shutil
import tempfile
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.agent import get_agent_executor, file_processor
from src.evaluator import faithfulness_score, answer_relevance_score, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG Service", version="3.0")


class QueryRequest(BaseModel):
    query: str
    api_key: Optional[str] = None        # BYOK: each visitor supplies their own key
    reference: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    source: str                          # "rag" | "web" | "rag+web"
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    accuracy: Optional[float] = None


def _retry_delay(error_str: str) -> float:
    match = re.search(r"retryDelay.*?(\d+\.?\d*)s", error_str)
    return float(match.group(1)) if match else 0.0


def _extract_content(message) -> str:
    content = message.content
    if isinstance(content, list):
        content = " ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content)


def _parse_tool_results(messages: list) -> tuple[str, str]:
    """Return (source_type, combined_tool_output) from the agent message chain.

    source_type is 'rag', 'web', or 'rag+web'.
    combined_tool_output is the actual text the agent received from its tools,
    which is what faithfulness should be measured against.
    """
    rag_parts, web_parts = [], []
    for msg in messages:
        # ToolMessage objects carry a .name attribute
        name = getattr(msg, "name", None)
        content = getattr(msg, "content", "") or ""
        if name == "lookup_documents":
            rag_parts.append(content)
        elif name == "search_web":
            web_parts.append(content)

    if rag_parts and web_parts:
        return "rag+web", " ".join(rag_parts + web_parts)
    if rag_parts:
        return "rag", " ".join(rag_parts)
    if web_parts:
        return "web", " ".join(web_parts)
    return "unknown", ""


@app.get("/")
async def root():
    return {"status": "active", "service": "Agentic Knowledge Search",
            "docs_url": "/docs", "uploaded_docs": file_processor.get_status()}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    tmp_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(dest)
        status = file_processor.process_files(saved_paths)
        return JSONResponse({"status": status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/reset")
async def reset_documents():
    file_processor.reset()
    return {"status": "Uploaded documents cleared."}


def _is_invalid_key(error_str: str) -> bool:
    markers = ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "API key expired")
    return any(m in error_str for m in markers)


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    # BYOK: a key must be supplied with every request
    if not request.api_key or not request.api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter your Google Gemini API key to ask a question.",
        )

    try:
        agent = get_agent_executor(request.api_key.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    for attempt in range(3):
        try:
            logger.info(f"Query (attempt {attempt + 1}): {request.query}")
            result = agent.invoke({"messages": [("user", request.query)]})
            answer = _extract_content(result["messages"][-1])

            source, tool_output = _parse_tool_results(result["messages"])

            # Faithfulness: answer vs the actual source content the agent used
            faith = faithfulness_score(answer, tool_output) if tool_output else None

            # Answer relevance: always computed, no reference needed
            relevance = answer_relevance_score(request.query, answer)

            # Accuracy: only when user provides a reference
            acc = accuracy_score(answer, request.reference) if request.reference else None

            return QueryResponse(
                response=answer,
                source=source,
                faithfulness=faith,
                answer_relevance=relevance,
                accuracy=acc,
            )

        except Exception as e:
            error_str = str(e)
            if _is_invalid_key(error_str):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or unauthorized API key. Please check your Google Gemini API key.",
                )
            if "RESOURCE_EXHAUSTED" not in error_str:
                logger.error(f"Chat error: {error_str}")
                raise HTTPException(status_code=500, detail=error_str)
            delay = _retry_delay(error_str)
            if delay and delay <= 120 and attempt < 2:
                logger.warning(f"Rate limited — retrying in {delay:.0f}s...")
                time.sleep(delay + 1)
                continue
            raise HTTPException(
                status_code=429,
                detail="Daily API quota exhausted. Please wait until tomorrow or upgrade your Gemini API plan.",
            )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
