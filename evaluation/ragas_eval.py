"""
evaluation/ragas_eval.py

RAGAS evaluation — measures RAG pipeline quality with real metrics.

Three metrics we care about:
- Faithfulness     : Is the answer grounded in retrieved context?
- Answer Relevancy : Does the answer actually address the question?
- Context Recall   : Did we retrieve the right chunks?

Run with: python -m evaluation.ragas_eval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from graph.rag_graph import run_pipeline
from dotenv import load_dotenv

load_dotenv()


def run_ragas_evaluation(test_cases: list) -> dict:
    """
    Run RAGAS evaluation on a list of test cases.

    Each test case:
    {
        "question": "What is gradient descent?",
        "ground_truth": "Gradient descent is an optimization algorithm..."
    }

    Returns dict with metric scores.
    """
    print(f"\n[RAGAS] Running evaluation on {len(test_cases)} test cases...")
    print("=" * 60)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, case in enumerate(test_cases):
        print(f"\n[RAGAS] Test {i+1}/{len(test_cases)}: {case['question'][:50]}...")

        # Run pipeline
        result = run_pipeline(case["question"])

        # Collect outputs
        questions.append(case["question"])
        answers.append(result.get("final_answer", ""))
        contexts.append([
            chunk["text"]
            for chunk in result.get("retrieved_chunks", [])[:3]
        ])
        ground_truths.append(case["ground_truth"])

        print(f"  Critic Score    : {result.get('critic_score', 0):.2f}")
        print(f"  Healing Loops   : {result.get('healing_loop_count', 0)}")
        print(f"  Verification    : {result.get('verification_status', 'N/A')}")

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run evaluation
    print("\n[RAGAS] Computing metrics...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Faithfulness     : {results['faithfulness']:.4f}")
    print(f"Answer Relevancy : {results['answer_relevancy']:.4f}")
    print(f"Context Recall   : {results['context_recall']:.4f}")
    print("=" * 60)

    return results


# ── Sample test cases — replace with your own ─────────────────────────────────
SAMPLE_TEST_CASES = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "The document covers the main subject area of the uploaded PDF."
    },
    {
        "question": "Summarize the key points discussed.",
        "ground_truth": "The key points include the major themes present in the document."
    }
]


if __name__ == "__main__":
    print("Agentic Self-Healing RAG — RAGAS Evaluation")
    print("Make sure you have ingested documents before running.")
    print()

    results = run_ragas_evaluation(SAMPLE_TEST_CASES)
    print("\nDone. Use these scores in your README and interviews.")