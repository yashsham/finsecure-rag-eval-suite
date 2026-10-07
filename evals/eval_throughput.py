import time
import json
import statistics
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.rag_pipeline import FinSecureRAG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = BASE_DIR / "goldens" / "ops_benchmark.json"

def execute_single_worker(query_item: Dict[str, Any], rag: FinSecureRAG) -> Dict[str, Any]:
    start = time.time()
    query = query_item["query"]
    status = "SUCCESS"
    err_msg = ""
    try:
        res = rag.query(query)
        ans = res["answer"]
    except Exception as e:
        status = "FAILED"
        err_msg = str(e)

    total_time = time.time() - start
    return {
        "id": query_item["id"],
        "latency_sec": round(total_time, 3),
        "status": status,
        "error": err_msg
    }

def evaluate_throughput_and_concurrency(concurrency_level: int = 3) -> Dict[str, Any]:
    """
    Evaluates Throughput (Requests Per Second - RPS) under concurrent user traffic.
    Simulates production workload spikes.
    """
    rag = FinSecureRAG(top_k=2)
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)[:concurrency_level]

    start_batch = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(execute_single_worker, item, rag) for item in benchmarks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_duration = time.time() - start_batch
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    rps = success_count / total_duration if total_duration > 0 else 0.0

    summary = {
        "metric_type": "throughput_and_concurrency",
        "concurrency_level": concurrency_level,
        "total_requests": len(benchmarks),
        "successful_requests": success_count,
        "total_duration_sec": round(total_duration, 3),
        "requests_per_second_rps": round(rps, 2),
        "worker_results": results
    }
    return summary

if __name__ == "__main__":
    res = evaluate_throughput_and_concurrency(concurrency_level=2)
    print("\n--- CONCURRENCY & THROUGHPUT STRESS REPORT ---")
    print(f"Concurrency Level:  {res['concurrency_level']}")
    print(f"Completed Duration: {res['total_duration_sec']}s")
    print(f"Throughput:         {res['requests_per_second_rps']} RPS")
    print(f"Success Count:      {res['successful_requests']}/{res['total_requests']}")
