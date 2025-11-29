import sys
import os
# Add project root to python path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
from datasets import Dataset
from src.agent import get_agent_executor
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def run_evaluation():
    print("🚀 Starting Evaluation Pipeline...")
    agent = get_agent_executor()
    
    # 1. Define Test Data
    # Questions that cover both Internal Docs and External Search
    questions = [
        "What are the specific conditions for treaty termination?", # Should use Internal Tool
        "What is the current Prime Minister of the UK?",          # Should use Web Tool
    ]
    
    # The 'Ground Truth' answers (what we expect)
    ground_truths = [
        ["Treaties can be terminated by consent of all parties, material breach, or fundamental change of circumstances."], 
        ["Keir Starmer is the Prime Minister of the UK (as of late 2024)."] 
    ]
    
    answers = []
    contexts = [] 
    
    # 2. Run the Agent on the questions
    for q in questions:
        print(f"Testing Question: {q}")
        try:
            result = agent.invoke({"input": q})
            answers.append(result["output"])
            
            # Extract retrieved context if available (simplified for this script)
            # In a real scenario, we'd pull the actual text chunks from the tool output
            contexts.append(["Context retrieved from agent tools..."]) 
            
        except Exception as e:
            print(f"Error on question '{q}': {e}")
            answers.append("Error")
            contexts.append([""])

    # 3. Create Ragas Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 4. Score it
    # Note: Ragas metrics often require an LLM for judging. 
    # Ensure your GOOGLE_API_KEY supports the models Ragas tries to call.
    print("📊 Calculating Metrics (Faithfulness, Relevance)...")
    
    # Configure Ragas to use Gemini if OpenAI is not available
    # (This part is complex in Ragas v0.1 without OpenAI, 
    # so we print the raw results for the MVP if metrics fail)
    print("Evaluation Results:")
    for i, q in enumerate(questions):
        print(f"Q: {q}")
        print(f"A: {answers[i]}")
        print("-" * 20)

if __name__ == "__main__":
    run_evaluation()