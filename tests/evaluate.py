import sys
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Add 'src' to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.agent import get_agent_executor
except ImportError:
    print("Could not import agent. Make sure you are running this from the root folder!")
    sys.exit(1)

def run_evaluation():
    print("Starting Custom Evaluation Pipeline (LLM-as-a-Judge)...")
    
    # 1. Setup the Judge (Gemini)
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 2. Define the Grading Logic
    grading_prompt = PromptTemplate.from_template(
        """
        You are a strict teacher grading a student's answer.
        
        Question: {question}
        Ground Truth: {ground_truth}
        Student Answer: {answer}
        
        On a scale of 1-10, how accurate is the student's answer compared to the ground truth?
        
        IMPORTANT: 
        - If the Student Answer matches the meaning of the Ground Truth, give a high score (8-10).
        - If the Student Answer adds extra correct details from the document, that is GOOD.
        - Only penalize if the information is factually wrong or completely unrelated.
        
        Format:
        Score: [1-10]
        Reason: [Text]
        """
    )

    # 3. Test Data 
    questions = [
        "What are the reporting requirements for State Parties?",
        "What happens if a State Party denounces the Protocol?",
    ]
    
    ground_truths = [
        "State Parties must submit a comprehensive report initially, followed by further information included in reports to the Committee on the Rights of the Child. Other State Parties need to submit reports every five years.", 
        "Denunciation does not affect acts or situations occurring before the denunciation becomes effective. It also does not prejudice the continued consideration of matters already under consideration."
    ]
    
    results = []

    # 4. Run the Agent & Grade it
    print("Generating and Grading answers...")
    try:
        agent = get_agent_executor()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    for i, q in enumerate(questions):
        print(f"\nTest Case {i+1}: {q}")
        try:
            # A. Get Agent Answer
            inputs = {"messages": [("user", q)]}
            result = agent.invoke(inputs)
            agent_answer = result["messages"][-1].content
            print(f"   Agent Answer: {agent_answer[:100]}...")

            # B. Grade the Answer
            grade_request = grading_prompt.format(
                question=q,
                ground_truth=ground_truths[i],
                answer=agent_answer
            )
            grading_result = eval_llm.invoke(grade_request).content
            
            # C. Store Result
            results.append({
                "Question": q,
                "Agent Answer": agent_answer,
                "Ground Truth": ground_truths[i],
                "Grading": grading_result
            })
            print(f"   Judge: {grading_result.splitlines()[0]}")

        except Exception as e:
            print(f"Failed: {e}")

    # 5. Save Report
    if results:
        df = pd.DataFrame(results)
        df.to_csv("evaluation_report.csv", index=False)
        print("\nEvaluation Complete! Report saved to 'evaluation_report.csv'")
        print(df[["Question", "Grading"]])

if __name__ == "__main__":
    run_evaluation()