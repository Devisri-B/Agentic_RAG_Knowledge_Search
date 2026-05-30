import os
import time
import shutil
import tempfile
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.agent import get_agent_executor, get_llm, file_processor
from src.evaluator import faithfulness_score, answer_relevance_score, accuracy_score
from src.memory import SessionStore
from src.utils import (
    extract_content,
    parse_tool_results,
    extract_citations,
    is_invalid_key,
    retry_delay,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic RAG Service", version="4.0")

sessions = SessionStore(max_turns=6, keep_recent=4)


class QueryRequest(BaseModel):
    query: str
    api_key: Optional[str] = None        # BYOK: each visitor supplies their own key
    session_id: Optional[str] = None     # conversation memory key
    reference: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    source: str                          # "rag" | "web" | "rag+web" | "unknown"
    citations: List[str] = []
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    accuracy: Optional[float] = None


def _summarizer(api_key: str):
    """Return a str->str summarizer backed by the user's own LLM."""
    llm = get_llm(api_key)

    def summarize(text: str) -> str:
        prompt = (
            "Condense the following conversation into a concise summary that preserves "
            "key facts, entities, decisions, and the user's goals. Keep it under 150 words.\n\n"
            f"{text}"
        )
        return extract_content(llm.invoke(prompt))

    return summarize


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


@app.post("/clear_memory")
async def clear_memory(session_id: str):
    sessions.clear(session_id)
    return {"status": "Conversation memory cleared."}


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    # BYOK: a key must be supplied with every request
    if not request.api_key or not request.api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter your Google Gemini API key to ask a question.",
        )

    api_key = request.api_key.strip()
    try:
        agent = get_agent_executor(api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    memory = sessions.get(request.session_id or "default")
    messages = memory.build_messages(request.query)

    for attempt in range(3):
        try:
            logger.info(f"Query (attempt {attempt + 1}, session={request.session_id}): {request.query}")
            result = agent.invoke({"messages": messages})
            answer = extract_content(result["messages"][-1])

            source, tool_output = parse_tool_results(result["messages"])
            citations = extract_citations(tool_output)

            faith = faithfulness_score(answer, tool_output) if tool_output else None
            relevance = answer_relevance_score(request.query, answer)
            acc = accuracy_score(answer, request.reference) if request.reference else None

            # Update conversation memory, summarizing older turns when it grows
            memory.add_turn(request.query, answer)
            try:
                memory.summarize_if_needed(_summarizer(api_key))
            except Exception as e:
                logger.warning(f"Summarization skipped: {e}")

            return QueryResponse(
                response=answer,
                source=source,
                citations=citations,
                faithfulness=faith,
                answer_relevance=relevance,
                accuracy=acc,
            )

        except Exception as e:
            error_str = str(e)
            if is_invalid_key(error_str):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or unauthorized API key. Please check your Google Gemini API key.",
                )
            if "RESOURCE_EXHAUSTED" not in error_str:
                logger.error(f"Chat error: {error_str}")
                raise HTTPException(status_code=500, detail=error_str)
            delay = retry_delay(error_str)
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
