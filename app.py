import os
import logging
import gradio as gr
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"
UPLOAD_ENDPOINT = f"{FASTAPI_URL}/upload"
RESET_ENDPOINT = f"{FASTAPI_URL}/reset"

SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".md", ".csv"]


def upload_files(files) -> str:
    if not files:
        return "No files selected."
    try:
        multipart = []
        for f in files:
            path = f if isinstance(f, str) else f.name
            filename = os.path.basename(path)
            with open(path, "rb") as fp:
                multipart.append(("files", (filename, fp.read(), "application/octet-stream")))

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


def determine_source(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["search", "web", "duckduckgo", "internet", "online"]):
        return "Web Search"
    if any(k in lower for k in ["uploaded", "file:", "page", "document", "policy", "pdf"]):
        return "Uploaded Documents (RAG)"
    return "RAG + Web Search"


def process_query(message: str, chat_history: list) -> tuple[list, str]:
    if not message.strip():
        return chat_history, "Please enter a question."
    try:
        resp = requests.post(CHAT_ENDPOINT, json={"query": message}, timeout=120)
        resp.raise_for_status()
        response_text = resp.json().get("response", "")

        source = determine_source(response_text)
        full_response = f"{response_text}\n\n--- Source: {source} ---"

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": full_response})
        return chat_history, f"Done — answered via {source}"
    except requests.exceptions.RequestException as e:
        error = f"Request failed: {e}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"ERROR: {error}"})
        return chat_history, error


def clear_chat() -> tuple[list, str]:
    return [], ""


# --- UI ---
with gr.Blocks(title="Agentic RAG Knowledge Search") as demo:
    gr.Markdown("# Agentic RAG Knowledge Search")
    gr.Markdown(
        "Upload your documents, then ask questions. "
        "The agent searches your files first, then the web if needed."
    )

    # File upload panel
    with gr.Group():
        gr.Markdown("### Upload Documents")
        gr.Markdown(f"Supported: {', '.join(SUPPORTED_TYPES)}")
        with gr.Row():
            file_input = gr.File(
                label="Select Files",
                file_count="multiple",
                file_types=SUPPORTED_TYPES,
            )
        with gr.Row():
            upload_btn = gr.Button("Process Files", variant="primary")
            reset_btn = gr.Button("Clear Uploaded Documents", variant="secondary")
        upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=2)

    gr.Markdown("---")

    # Chat panel
    with gr.Group():
        gr.Markdown("### Ask a Question")
        chatbot = gr.Chatbot(
            label="Conversation",
            height=420,
        )
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Ask anything about your documents or the web...",
                label="Your Question",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Submit", variant="primary", scale=1)
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        status_output = gr.Textbox(label="Status", interactive=False, lines=1)

    # Wiring
    upload_btn.click(fn=upload_files, inputs=[file_input], outputs=[upload_status])
    reset_btn.click(fn=reset_documents, inputs=[], outputs=[upload_status])

    submit_btn.click(
        fn=process_query,
        inputs=[user_input, chatbot],
        outputs=[chatbot, status_output],
    ).then(fn=lambda: "", inputs=[], outputs=[user_input])

    user_input.submit(
        fn=process_query,
        inputs=[user_input, chatbot],
        outputs=[chatbot, status_output],
    ).then(fn=lambda: "", inputs=[], outputs=[user_input])

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, status_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
