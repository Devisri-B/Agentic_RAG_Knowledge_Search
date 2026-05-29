import os
import gradio as gr
from src.agent import get_agent_executor
import logging

# Suppress HuggingFace tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Agent
try:
    agent_executor = get_agent_executor()
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent_executor = None

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
    Process user query through the agent and return updated chat history and status.
    
    Args:
        message: User query
        chat_history: Current chat history in Gradio format
        
    Returns:
        Tuple of (updated_chat_history, status_message)
    """
    if not agent_executor:
        error_msg = "Agent not initialized. Please check GOOGLE_API_KEY."
        chat_history.append([message, f"ERROR: {error_msg}"])
        return chat_history, error_msg
    
    try:
        logger.info(f"Processing query: {message}")
        
        # Format input for LangGraph agent
        inputs = {"messages": [("user", message)]}
        
        # Invoke agent
        result = agent_executor.invoke(inputs)
        
        # Extract response from agent output
        last_message = result["messages"][-1]
        response_text = last_message.content
        
        # Convert to string if it's a list
        if isinstance(response_text, list):
            response_text = "\n".join(str(item) for item in response_text)
        else:
            response_text = str(response_text)
        
        # Determine source
        source = determine_source(response_text)
        
        # Format response with source indicator
        full_response = f"{response_text}\n\n--- {source} ---"
        
        # Append to chat history as list [user, assistant]
        chat_history.append([message, full_response])
        
        logger.info(f"Response generated successfully")
        return chat_history, f"Query processed. {source}"
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        chat_history.append([message, f"ERROR: {error_msg}"])
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
    
    # Wire up interactions
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
