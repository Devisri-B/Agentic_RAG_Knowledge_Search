"""Offline LLM-as-a-judge evaluation. Grades the agent on two metrics per question:
faithfulness (grounded in the retrieved context?) and accuracy (matches ground truth?).
Run from the project root: python -m tests.evaluate"""

import sys
import os
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

try:
    from src.agent import get_agent_executor, file_processor, _fallback_kb
except ImportError:
    print("Run this from the project root: python -m tests.evaluate")
    sys.exit(1)


FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI answer is grounded in the provided source context.

Question: {question}
Retrieved Context: {context}
AI Answer: {answer}

Does the answer contain claims NOT supported by the retrieved context?
A faithful answer only uses information present in the context.
A hallucinated answer invents facts or adds information not in the context.

Score 1-10 where:
  9-10 = fully grounded, no unsupported claims
  6-8  = mostly grounded, minor additions
  3-5  = several unsupported claims
  1-2  = mostly fabricated

Format:
Score: [1-10]
Reason: [one sentence]"""

ACCURACY_PROMPT = """\
You are a strict teacher grading a student's answer.

Question: {question}
Ground Truth: {ground_truth}
Student Answer: {answer}

On a scale of 1-10, how accurate and complete is the student's answer compared to the ground truth?
- 9-10: matches ground truth meaning, may add correct extra details
- 6-8:  partially correct, missing some key points
- 3-5:  relevant but significantly incomplete or partially wrong
- 1-2:  incorrect or completely off-topic

Format:
Score: [1-10]
Reason: [one sentence]"""


def extract_content(message) -> str:
    content = message.content
    if isinstance(content, list):
        content = " ".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        )
    return str(content)


def parse_score(text: str) -> int:
    match = re.search(r"Score:\s*(\d+)", text)
    return int(match.group(1)) if match else 0


def get_context(question: str) -> str:
    """Retrieve the same context the agent would use for RAG questions."""
    if file_processor.has_documents():
        ctx = file_processor.retrieve(question)
        if ctx:
            return ctx
    return _fallback_kb.retrieve(question)


# Add your own Q&A pairs here. Leave ground_truth as None for web questions
# (accuracy is then skipped, since there is nothing to compare against).
TEST_CASES = [
    {
        "question": "What are the reporting requirements for State Parties?",
        "ground_truth": (
            "State Parties must submit a comprehensive report initially, followed by further "
            "information included in reports to the Committee on the Rights of the Child. "
            "Other State Parties need to submit reports every five years."
        ),
        "source": "rag",
    },
    {
        "question": "What happens if a State Party denounces the Protocol?",
        "ground_truth": (
            "Denunciation does not affect acts or situations occurring before the denunciation "
            "becomes effective. It also does not prejudice the continued consideration of matters "
            "already under consideration."
        ),
        "source": "rag",
    },
]


def run_evaluation(test_cases: list = None):
    cases = test_cases or TEST_CASES
    print(f"Starting evaluation ({len(cases)} test case(s))\n")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY in your .env to run the offline evaluation.")
        return

    judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=api_key)

    try:
        agent = get_agent_executor(api_key)
    except Exception as e:
        print(f"Could not initialize agent: {e}")
        return

    results = []

    for i, case in enumerate(cases, 1):
        question = case["question"]
        ground_truth = case.get("ground_truth")
        source = case.get("source", "rag")

        print(f"[{i}/{len(cases)}] {question}")

        try:
            result = agent.invoke({"messages": [("user", question)]})
            answer = extract_content(result["messages"][-1])
        except Exception as e:
            print(f"  Agent error: {e}\n")
            results.append({"Question": question, "Answer": f"ERROR: {e}",
                            "Ground Truth": ground_truth,
                            "Faithfulness Score": "-", "Faithfulness Reason": str(e),
                            "Accuracy Score": "-", "Accuracy Reason": str(e)})
            continue

        print(f"  Answer: {answer[:120]}...")

        # Faithfulness only applies to document-grounded answers
        faithfulness_score, faithfulness_reason = "-", "N/A (web search question)"
        if source == "rag":
            try:
                context = get_context(question)
                response = judge.invoke(
                    FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
                )
                faith_text = extract_content(response)
                faithfulness_score = parse_score(faith_text)
                faithfulness_reason = faith_text.split("Reason:")[-1].strip()
                print(f"  Faithfulness: {faithfulness_score}/10")
            except Exception as e:
                faithfulness_reason = str(e)
                print(f"  Faithfulness check failed: {e}")

        # Accuracy only when a ground truth is provided
        accuracy_score, accuracy_reason = "-", "N/A (no ground truth)"
        if ground_truth:
            try:
                response = judge.invoke(
                    ACCURACY_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer)
                )
                acc_text = extract_content(response)
                accuracy_score = parse_score(acc_text)
                accuracy_reason = acc_text.split("Reason:")[-1].strip()
                print(f"  Accuracy:     {accuracy_score}/10")
            except Exception as e:
                accuracy_reason = str(e)
                print(f"  Accuracy check failed: {e}")

        results.append({
            "Question": question,
            "Answer": answer,
            "Ground Truth": ground_truth or "",
            "Faithfulness Score": faithfulness_score,
            "Faithfulness Reason": faithfulness_reason,
            "Accuracy Score": accuracy_score,
            "Accuracy Reason": accuracy_reason,
        })
        print()

    # Save report
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)
    df.to_csv("evaluation_report.csv", index=False)

    # Print summary
    print("=" * 50)
    numeric_faith = [r["Faithfulness Score"] for r in results if isinstance(r["Faithfulness Score"], int)]
    numeric_acc   = [r["Accuracy Score"] for r in results if isinstance(r["Accuracy Score"], int)]
    if numeric_faith:
        print(f"Avg Faithfulness (hallucination): {sum(numeric_faith)/len(numeric_faith):.1f}/10")
    if numeric_acc:
        print(f"Avg Accuracy:                     {sum(numeric_acc)/len(numeric_acc):.1f}/10")
    print("\nReport saved to evaluation_report.csv")


if __name__ == "__main__":
    run_evaluation()
