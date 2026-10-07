import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from evals.eval_retriever import evaluate_retriever_in_isolation
from evals.eval_generator import evaluate_generator_in_isolation
from evals.eval_rag_pipeline import evaluate_rag_pipeline
from evals.eval_application import evaluate_application_quality
from evals.eval_safety import evaluate_safety_and_guardrails
from evals.eval_ops import evaluate_operations_and_economics
from evals.eval_throughput import evaluate_throughput_and_concurrency

BASE_DIR = Path(__file__).resolve().parent.parent

def run_complete_evaluation_suite(
    retriever_k: int = 4,
    generator_sample: int = 2,
    pipeline_sample: int = 2,
    quality_sample: int = 2,
    throughput_concurrency: int = 2
) -> Dict[str, Any]:
    """
    Executes the comprehensive 3-tier RAG Evaluation Suite:
    Returns flat metric dictionary compatible with regression engine.
    """
    start_total = time.time()
    print("=================================================================")
    print("   FINSECURE RAG EVALUATION SUITE: STARTING FULL RUN             ")
    print("=================================================================")

    # 1. Level 1: Retriever in Isolation
    print("\n>>> [1/7] Running Level 1: Retriever in Isolation...")
    ret_res = evaluate_retriever_in_isolation(top_k=retriever_k)

    # 2. Level 1: Generator in Isolation
    print("\n>>> [2/7] Running Level 1: Generator in Isolation...")
    gen_res = evaluate_generator_in_isolation(sample_size=generator_sample)

    # 3. Level 2: Pipeline RAG Triad
    print("\n>>> [3/7] Running Level 2: Pipeline RAG Triad...")
    triad_res = evaluate_rag_pipeline(sample_size=pipeline_sample)

    # 4. Level 3: Application Quality (G-Eval)
    print("\n>>> [4/7] Running Level 3: Application Quality (G-Eval)...")
    qual_res = evaluate_application_quality(sample_size=quality_sample)

    # 5. Level 3: Application Safety & Guardrails
    print("\n>>> [5/7] Running Level 3: Application Safety & Guardrails...")
    safe_res = evaluate_safety_and_guardrails()

    # 6. Level 3: Operations & Token Economics
    print("\n>>> [6/7] Running Level 3: Operations & Token Economics...")
    ops_res = evaluate_operations_and_economics()

    # 7. Level 3: Throughput & Concurrency
    print("\n>>> [7/7] Running Level 3: Throughput & Concurrency Load Test...")
    tp_res = evaluate_throughput_and_concurrency(concurrency_level=throughput_concurrency)

    total_duration = time.time() - start_total

    # Standardized flat metrics record
    metrics = {
        "retriever_recall": ret_res["mean_recall"],
        "retriever_mrr": ret_res["mean_reciprocal_rank"],
        "generator_faithfulness": gen_res["mean_faithfulness"],
        "generator_relevancy": gen_res["mean_answer_relevancy"],
        "pipeline_contextual_relevancy": triad_res["mean_contextual_relevancy"],
        "pipeline_faithfulness": triad_res["mean_faithfulness"],
        "pipeline_answer_relevancy": triad_res["mean_answer_relevancy"],
        "pipeline_intra_chunk_noise": triad_res["mean_intra_chunk_noise"],
        "quality_correctness": qual_res["mean_correctness"],
        "quality_completeness": qual_res["mean_completeness"],
        "quality_tone": qual_res["mean_professional_tone"],
        "safety_refusal_rate": safe_res["safeguard_refusal_rate"],
        "safety_mnpi_leakage": safe_res["mnpi_leakage_rate"],
        "ops_p95_latency": ops_res["latency_p95_sec"],
        "ops_mean_ttft": ops_res["mean_ttft_sec"],
        "ops_cost_per_1k": ops_res["cost_per_1000_queries_usd"],
        "ops_error_rate": ops_res["error_rate"],
        "ops_rps": tp_res["requests_per_second_rps"]
    }

    report = {
        "run_timestamp": datetime.now().isoformat(),
        "total_runtime_seconds": round(total_duration, 2),
        "metrics": metrics,
        "details": {
            "retriever": ret_res,
            "generator": gen_res,
            "pipeline": triad_res,
            "quality": qual_res,
            "safety": safe_res,
            "operations": ops_res,
            "throughput": tp_res
        }
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="Run FinSecure Full RAG Evaluation Suite")
    parser.add_argument("--output", type=str, default=None, help="File path to save the eval run json")
    args = parser.parse_args()

    report = run_complete_evaluation_suite()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Full Evaluation Report successfully written to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
