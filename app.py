import os
import uuid
import gradio as gr
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"
UPLOAD_ENDPOINT = f"{FASTAPI_URL}/upload"
RESET_ENDPOINT = f"{FASTAPI_URL}/reset"
CLEAR_MEMORY_ENDPOINT = f"{FASTAPI_URL}/clear_memory"

SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".md", ".csv"]

CSS = """
#submit-btn { margin-top: auto; margin-bottom: auto; height: 80px; }
#input-row  { align-items: center; }
"""

SOURCE_LABELS = {
    "rag":     "Uploaded Documents (RAG)",
    "web":     "Web Search",
    "rag+web": "Documents + Web Search",
    "unknown": "Unknown",
}


def upload_files(files) -> str:
    if not files:
        return "No files selected."
    try:
        multipart = []
        for f in files:
            path = f if isinstance(f, str) else f.name
            with open(path, "rb") as fp:
                multipart.append(("files", (os.path.basename(path), fp.read(), "application/octet-stream")))
        resp = requests.post(UPLOAD_ENDPOINT, files=multipart, timeout=120)
        resp.raise_for_status()
        return resp.json().get("status", "Files processed.")
    except requests.exceptions.RequestException as e:
        return f"Upload failed: {e}"


def reset_documents() -> str:
    try:
        resp = requests.post(RESET_ENDPOINT, timeout=10)
        resp.raise_for_status()
        return resp.json().get("status", "Documents cleared.")
    except requests.exceptions.RequestException as e:
        return f"Reset failed: {e}"


def _bar(value: float, width: int = 12) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _fmt(label: str, value: float, hint: str) -> str:
    return f"**{label}**  \n`{_bar(value)}  {value:.2f} / 1.0`  \n_{hint}_\n"


def _faith_hint(score: float, source: str) -> str:
    what = "web results" if source == "web" else "retrieved documents"
    if score >= 0.75:
        return f"Answer is well-grounded in the {what}"
    if score >= 0.50:
        return f"Answer is mostly grounded in the {what}, minor unsupported details possible"
    return f"Low grounding — answer may contain content not present in the {what}"


def _relevance_hint(score: float) -> str:
    if score >= 0.70:
        return "Answer directly addresses the question"
    if score >= 0.45:
        return "Answer is related but may not fully address the question"
    return "Answer may be off-topic or incomplete"


def _accuracy_hint(score: float) -> str:
    if score >= 0.75:
        return "Strong match with your reference answer"
    if score >= 0.40:
        return "Partial match — some key points differ from your reference"
    return "Low overlap with your reference answer"


def _format_metrics(source: str, faithfulness, answer_relevance, accuracy) -> str:
    if faithfulness is None and answer_relevance is None:
        return ""

    src_label = SOURCE_LABELS.get(source, source)
    lines = [f"**Answer source: {src_label}**", "---"]

    if faithfulness is not None:
        faith_label = "Faithfulness — grounded in documents" if source != "web" else "Faithfulness — grounded in web results"
        lines.append(_fmt(faith_label, faithfulness, _faith_hint(faithfulness, source)))

    if answer_relevance is not None:
        lines.append(_fmt("Answer Relevance — addresses the question", answer_relevance, _relevance_hint(answer_relevance)))

    if accuracy is not None:
        lines.append(_fmt("Accuracy — matches your reference", accuracy, _accuracy_hint(accuracy)))
    else:
        lines.append("_Accuracy: provide a reference answer above to see this score._")

    return "\n".join(lines)


def process_query(message: str, api_key: str, reference: str, session_id: str, chat_history: list) -> tuple[list, str, str]:
    if not api_key.strip():
        return chat_history, "Enter your Google Gemini API key above to start.", ""
    if not message.strip():
        return chat_history, "Please enter a question.", ""
    try:
        payload = {"query": message, "api_key": api_key.strip(), "session_id": session_id}
        if reference.strip():
            payload["reference"] = reference.strip()

        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=180)

        if resp.status_code in (400, 401, 429):
            detail = resp.json().get("detail", "Request could not be completed.")
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": f"Sorry, I can't respond right now:\n\n{detail}"})
            return chat_history, detail, ""

        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "")
        source = data.get("source", "unknown")
        citations = data.get("citations", [])

        src_label = SOURCE_LABELS.get(source, source)
        footer = f"\n\n--- Source: {src_label} ---"
        if citations:
            footer += "\n📎 Sources used: " + "; ".join(citations)

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"{response_text}{footer}"})

        metrics_md = _format_metrics(
            source,
            data.get("faithfulness"),
            data.get("answer_relevance"),
            data.get("accuracy"),
        )
        return chat_history, f"Done — answered via {src_label}", metrics_md

    except requests.exceptions.RequestException as e:
        error = f"Request failed: {e}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"ERROR: {error}"})
        return chat_history, error, ""


def clear_chat(session_id: str) -> tuple[list, str, str]:
    """Clear the UI and the server-side conversation memory for this session."""
    try:
        requests.post(CLEAR_MEMORY_ENDPOINT, params={"session_id": session_id}, timeout=10)
    except requests.exceptions.RequestException:
        pass
    return [], "", ""


with gr.Blocks(title="Agentic RAG Knowledge Search") as demo:
    # Per-browser-session id for server-side conversation memory
    session_id = gr.State()

    gr.Markdown("# Agentic RAG Knowledge Search")
    gr.Markdown(
        "Upload your own documents and ask questions. "
        "The agent searches your files via RAG first, then falls back to web search if needed. "
        "It remembers the conversation, so follow-up questions work. "
        "**No files uploaded?** The agent uses a built-in legal/policy document as the default knowledge base."
    )

    with gr.Group():
        gr.Markdown(
            "### Your Gemini API Key (required)\n"
            "This app uses **your own** Google Gemini key — it is sent only with your requests and never stored. "
            "Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)."
        )
        api_key_input = gr.Textbox(
            label="Google Gemini API Key",
            placeholder="Paste your API key here (starts with AIza...)",
            type="password",
            lines=1,
        )

    with gr.Group():
        gr.Markdown(
            f"### Upload Documents\n"
            f"Supported: **{', '.join(SUPPORTED_TYPES)}** — multiple files allowed, new uploads add to the index."
        )
        file_input = gr.File(label="Select Files", file_count="multiple", file_types=SUPPORTED_TYPES)
        with gr.Row():
            upload_btn = gr.Button("Process & Index Files", variant="primary")
            reset_btn = gr.Button("Clear Uploaded Documents", variant="secondary")
        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=2,
                                   placeholder="Upload status will appear here...")

    gr.Markdown("---")

    with gr.Group():
        gr.Markdown(
            "### Ask a Question\n"
            "> Press **Enter** or click **Submit**. The agent decides whether to search your documents, the web, or both."
        )
        chatbot = gr.Chatbot(label="Conversation", height=420)

        with gr.Row(elem_id="input-row"):
            user_input = gr.Textbox(
                placeholder="Ask anything about your documents or the web...",
                lines=2, scale=5, show_label=False, container=False,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1, elem_id="submit-btn")

        reference_input = gr.Textbox(
            label="Reference Answer (optional)",
            placeholder="Paste an expected correct answer here to also see an Accuracy score...",
            lines=2,
        )

        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

        status_output = gr.Textbox(label="Status", interactive=False, lines=1,
                                   placeholder="Status will appear here...")

    gr.Markdown("---")

    with gr.Group():
        gr.Markdown(
            "### Evaluation Metrics\n"
            "Computed automatically after every response — **no extra API calls, no reference needed** for the first two.\n\n"
            "| Metric | Always shown? | What it measures |\n"
            "|---|---|---|\n"
            "| **Faithfulness** | Yes | Is the answer grounded in the source it used? (docs or web results) |\n"
            "| **Answer Relevance** | Yes | Does the answer actually address your question? |\n"
            "| **Accuracy** | Only with reference | Does the answer match an expected correct answer? |"
        )
        metrics_output = gr.Markdown()

    # Assign a fresh session id when the page loads
    demo.load(fn=lambda: str(uuid.uuid4()), inputs=[], outputs=[session_id])

    upload_btn.click(fn=upload_files, inputs=[file_input], outputs=[upload_status])
    reset_btn.click(fn=reset_documents, inputs=[], outputs=[upload_status])

    submit_btn.click(
        fn=process_query,
        inputs=[user_input, api_key_input, reference_input, session_id, chatbot],
        outputs=[chatbot, status_output, metrics_output],
    ).then(fn=lambda: "", inputs=[], outputs=[user_input])

    user_input.submit(
        fn=process_query,
        inputs=[user_input, api_key_input, reference_input, session_id, chatbot],
        outputs=[chatbot, status_output, metrics_output],
    ).then(fn=lambda: "", inputs=[], outputs=[user_input])

    clear_btn.click(fn=clear_chat, inputs=[session_id], outputs=[chatbot, status_output, metrics_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=CSS)
