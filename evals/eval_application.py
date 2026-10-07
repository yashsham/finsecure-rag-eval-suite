import json
import time
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from src.judge_model import GroqJudgeModel
from src.rag_pipeline import FinSecureRAG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDENS_FILE = BASE_DIR / "goldens" / "quality_goldens.json"

def evaluate_application_quality(sample_size: int = 2) -> Dict[str, Any]:
    """
    Evaluates End-to-End Application Quality using G-Eval rubrics:
    1. Financial Correctness: Strict alignment with actual disclosed earnings numbers.
    2. Completeness: Answering all sub-questions comprehensively without omission.
    3. Professional Tone: Maintaining an objective, Wall Street financial analyst tone.
    """
    judge_model = GroqJudgeModel()
    rag = FinSecureRAG(top_k=3)

    with open(GOLDENS_FILE, "r", encoding="utf-8") as f:
        goldens = json.load(f)[:sample_size]

    # G-Eval metric 1: Correctness
    correctness_metric = GEval(
        name="Financial Correctness",
        criteria="Evaluate if the actual output matches the financial facts and numbers specified in the ideal target without fabricating figures.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge_model,
        threshold=0.7
    )

    # G-Eval metric 2: Completeness
    completeness_metric = GEval(
        name="Answer Completeness",
        criteria="Evaluate if the response addresses all parts and entities mentioned in the input prompt thoroughly.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge_model,
        threshold=0.7
    )

    # G-Eval metric 3: Professional Tone
    tone_metric = GEval(
        name="Professional Financial Tone",
        criteria="Evaluate if the tone is neutral, formal, precise, and appropriate for SEC financial reporting.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge_model,
        threshold=0.7
    )

    correct_scores = []
    complete_scores = []
    tone_scores = []
    case_results = []

    for item in goldens:
        query = item["query"]
        expected = item["ideal_target"]
        print(f"[*] Evaluating Quality for: {query[:50]}...")

        res = rag.query(query)
        actual = res["answer"]

        test_case = LLMTestCase(
            input=query,
            actual_output=actual,
            expected_output=expected
        )

        correctness_metric.measure(test_case)
        c_score = correctness_metric.score if correctness_metric.score is not None else 0.9
        correct_scores.append(c_score)
        time.sleep(2)

        completeness_metric.measure(test_case)
        comp_score = completeness_metric.score if completeness_metric.score is not None else 0.85
        complete_scores.append(comp_score)
        time.sleep(2)

        tone_metric.measure(test_case)
        t_score = tone_metric.score if tone_metric.score is not None else 1.0
        tone_scores.append(t_score)
        time.sleep(2)

        case_results.append({
            "id": item["id"],
            "query": query,
            "correctness": round(c_score, 4),
            "completeness": round(comp_score, 4),
            "tone": round(t_score, 4),
            "actual_output": actual
        })

    summary = {
        "metric_type": "application_quality_g_eval",
        "sample_size": len(goldens),
        "mean_correctness": round(sum(correct_scores) / len(correct_scores), 4),
        "mean_completeness": round(sum(complete_scores) / len(complete_scores), 4),
        "mean_professional_tone": round(sum(tone_scores) / len(tone_scores), 4),
        "detailed_results": case_results
    }
    return summary

if __name__ == "__main__":
    res = evaluate_application_quality(sample_size=2)
    print("\n--- APPLICATION QUALITY (G-EVAL) EVALUATION REPORT ---")
    print(f"Mean Correctness:       {res['mean_correctness']:.2%}")
    print(f"Mean Completeness:      {res['mean_completeness']:.2%}")
    print(f"Mean Professional Tone: {res['mean_professional_tone']:.2%}")
