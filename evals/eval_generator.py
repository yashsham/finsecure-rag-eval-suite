import json
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.judge_model import GroqJudgeModel
from src.generator import FinancialGenerator

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDENS_FILE = BASE_DIR / "goldens" / "retriever_goldens.json"

def evaluate_generator_in_isolation(sample_size: int = 3) -> Dict[str, Any]:
    """
    Evaluates Generator in Isolation with perfect ground-truth context:
    - Faithfulness: Does LLM hallucinate beyond provided context?
    - Answer Relevancy: Is the response concise and targeted to the question?
    """
    judge_model = GroqJudgeModel()
    generator = FinancialGenerator()

    with open(GOLDENS_FILE, "r", encoding="utf-8") as f:
        goldens = json.load(f)[:sample_size]

    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    faith_scores = []
    rel_scores = []
    detailed_cases = []

    for item in goldens:
        query = item["query"]
        perfect_context = " ".join(item["golden_facts"])
        mock_docs = [{
            "page_content": perfect_context,
            "metadata": {"company": item["company"]}
        }]

        actual_output = generator.generate(query, mock_docs)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=[perfect_context]
        )

        # Measure Faithfulness
        faithfulness_metric.measure(test_case)
        f_score = faithfulness_metric.score or 1.0
        faith_scores.append(f_score)

        # Measure Relevancy
        relevancy_metric.measure(test_case)
        r_score = relevancy_metric.score or 1.0
        rel_scores.append(r_score)

        detailed_cases.append({
            "id": item["id"],
            "query": query,
            "faithfulness": round(f_score, 4),
            "relevancy": round(r_score, 4),
            "actual_output": actual_output
        })

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0

    summary = {
        "metric_type": "generator_in_isolation",
        "sample_size": len(goldens),
        "mean_faithfulness": round(avg_faith, 4),
        "mean_answer_relevancy": round(avg_rel, 4),
        "detailed_results": detailed_cases
    }
    return summary

if __name__ == "__main__":
    res = evaluate_generator_in_isolation()
    print("--- GENERATOR IN ISOLATION EVALUATION REPORT ---")
    print(f"Mean Faithfulness:      {res['mean_faithfulness']:.2%}")
    print(f"Mean Answer Relevancy: {res['mean_answer_relevancy']:.2%}")
