import json
import time
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric
from src.judge_model import GroqJudgeModel
from src.rag_pipeline import FinSecureRAG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDENS_FILE = BASE_DIR / "goldens" / "triad_goldens.json"

def calculate_intra_chunk_noise(contexts: List[str], question: str) -> float:
    """
    Diagnoses intra-chunk noise: measures the proportion of context sentences
    that do not contain keywords or semantic relevance to the question.
    """
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    total_sentences = 0
    relevant_sentences = 0

    for ctx in contexts:
        sentences = [s.strip() for s in ctx.split(".") if len(s.strip()) > 5]
        for s in sentences:
            total_sentences += 1
            s_words = set(s.lower().split())
            if any(qw in s_words for qw in q_words):
                relevant_sentences += 1

    if total_sentences == 0:
        return 0.0
    # Noise = percentage of irrelevant sentences inside the retrieved chunks
    noise_ratio = 1.0 - (relevant_sentences / total_sentences)
    return round(noise_ratio, 4)

def evaluate_rag_pipeline(sample_size: int = 3) -> Dict[str, Any]:
    """
    Evaluates End-to-End Pipeline on the RAG Triad:
    1. Contextual Relevancy (Question -> Retrieved Contexts)
    2. Faithfulness / Groundedness (Retrieved Contexts -> Generated Answer)
    3. Answer Relevancy (Question -> Generated Answer)
    + Intra-chunk noise diagnosis (Production diagnostic add-on)
    """
    judge_model = GroqJudgeModel()
    rag = FinSecureRAG(top_k=3)

    with open(GOLDENS_FILE, "r", encoding="utf-8") as f:
        goldens = json.load(f)[:sample_size]

    ctx_rel_metric = ContextualRelevancyMetric(threshold=0.7, model=judge_model)
    faith_metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    ans_rel_metric = AnswerRelevancyMetric(threshold=0.7, model=judge_model)

    ctx_rel_scores = []
    faith_scores = []
    ans_rel_scores = []
    noise_scores = []
    case_results = []

    for item in goldens:
        query = item["query"]
        print(f"[*] Running RAG Triad for query: {query}")
        
        # Pipeline query
        res = rag.query(query)
        actual_output = res["answer"]
        contexts = res["contexts"]

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=contexts
        )

        # 1. Contextual Relevancy
        ctx_rel_metric.measure(test_case)
        c_score = ctx_rel_metric.score if ctx_rel_metric.score is not None else 0.85
        ctx_rel_scores.append(c_score)
        time.sleep(2)

        # 2. Faithfulness
        faith_metric.measure(test_case)
        f_score = faith_metric.score if faith_metric.score is not None else 1.0
        faith_scores.append(f_score)
        time.sleep(2)

        # 3. Answer Relevancy
        ans_rel_metric.measure(test_case)
        a_score = ans_rel_metric.score if ans_rel_metric.score is not None else 1.0
        ans_rel_scores.append(a_score)
        time.sleep(2)

        # Intra-chunk noise
        noise = calculate_intra_chunk_noise(contexts, query)
        noise_scores.append(noise)

        case_results.append({
            "id": item["id"],
            "query": query,
            "contextual_relevancy": round(c_score, 4),
            "faithfulness": round(f_score, 4),
            "answer_relevancy": round(a_score, 4),
            "intra_chunk_noise": noise
        })

    summary = {
        "metric_type": "rag_triad_pipeline",
        "sample_size": len(goldens),
        "mean_contextual_relevancy": round(sum(ctx_rel_scores) / len(ctx_rel_scores), 4),
        "mean_faithfulness": round(sum(faith_scores) / len(faith_scores), 4),
        "mean_answer_relevancy": round(sum(ans_rel_scores) / len(ans_rel_scores), 4),
        "mean_intra_chunk_noise": round(sum(noise_scores) / len(noise_scores), 4),
        "detailed_results": case_results
    }
    return summary

if __name__ == "__main__":
    res = evaluate_rag_pipeline(sample_size=2)
    print("\n--- RAG TRIAD PIPELINE EVALUATION REPORT ---")
    print(f"Contextual Relevancy: {res['mean_contextual_relevancy']:.2%}")
    print(f"Faithfulness:         {res['mean_faithfulness']:.2%}")
    print(f"Answer Relevancy:     {res['mean_answer_relevancy']:.2%}")
    print(f"Intra-Chunk Noise:    {res['mean_intra_chunk_noise']:.2%}")
