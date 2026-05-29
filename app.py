import os
import logging
import gradio as gr
import requests

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI endpoint configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"


def determine_source(response_text) -> str:
    """
    Attempt to determine if the response came from RAG or Web Search.
    Looks for keywords or source indicators in the response.
    """
    # Convert list to string if needed
    if isinstance(response_text, list):
        response_text = " ".join(str(item) for item in response_text)
    
    response_lower = str(response_text).lower()
    
    # Check for web search indicators
    if any(keyword in response_lower for keyword in ["search", "web", "duckduckgo", "internet", "online"]):
        return "Source: Web Search"
    
    # Check for RAG/document indicators
    if "source:" in response_lower and "page" in response_lower:
        return "Source: Internal Documents (RAG)"
    
    if any(keyword in response_lower for keyword in ["document", "policy", "pdf", "internal", "knowledge base"]):
        return "Source: Internal Documents (RAG)"
    
    # Default fallback - cannot determine
    return "Source: Mixed (RAG + Web Search)"

def process_query(message: str, chat_history: list) -> tuple[list, str]:
    """
    Process user query by calling the FastAPI /chat endpoint.
    
    Args:
        message: User query
        chat_history: Current chat history in Gradio format
        
    Returns:
        Tuple of (updated_chat_history, status_message)
    """
    try:
        logger.info(f"Sending query to FastAPI: {message}")
        response = requests.post(CHAT_ENDPOINT, json={"query": message}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        response_text = payload.get("response", "")
        
        if isinstance(response_text, list):
            response_text = "\n".join(str(item) for item in response_text)
        else:
            response_text = str(response_text)
        
        source = determine_source(response_text)
        full_response = f"{response_text}\n\n--- {source} ---"
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": full_response})
        
        logger.info("Response received from FastAPI")
        return chat_history, f"Query processed. {source}"
    except requests.exceptions.RequestException as e:
        error_msg = f"FastAPI request failed: {str(e)}"
        logger.error(error_msg)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"ERROR: {error_msg}"})
        return chat_history, error_msg
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"ERROR: {error_msg}"})
        return chat_history, error_msg

def clear_chat() -> tuple[list, str]:
    """Clear chat history."""
    return [], ""

# Build Gradio Interface
with gr.Blocks(title="Agentic RAG Knowledge Search") as demo:
    gr.Markdown("# 🔍 Agentic RAG Knowledge Search")
    gr.Markdown("Ask questions answered from internal documents (RAG) or the live web — the agent decides.")
    with gr.Group():
        chatbot = gr.Chatbot(
            label="Conversation History",
            show_label=True,
            height=400
        )
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask a question about policy documents or anything on the web...",
            label="Your Question",
            lines=2
        )
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")
    
    status_output = gr.Textbox(
        label="Status",
        interactive=False,
        lines=1
    )
    
    # Wire up chat interactions
    submit_btn.click(
        fn=process_query,
        inputs=[user_input, chatbot],
        outputs=[chatbot, status_output]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[user_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, status_output]
    )
    
    # Allow Enter key to submit
    user_input.submit(
        fn=process_query,
        inputs=[user_input, chatbot],
        outputs=[chatbot, status_output]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[user_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
