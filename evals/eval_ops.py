import time
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.rag_pipeline import FinSecureRAG

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = BASE_DIR / "goldens" / "ops_benchmark.json"

# Pricing model ($ per 1M tokens) - Groq / open-source inference standard
INPUT_TOKEN_PRICE_PER_M = 0.59
OUTPUT_TOKEN_PRICE_PER_M = 0.79

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (1 token ~= 4 characters)."""
    return max(1, len(text) // 4)

def evaluate_operations_and_economics() -> Dict[str, Any]:
    """
    Evaluates Production Operations:
    - P50, P95, P99 Latency across queries
    - Time-to-First-Token (TTFT) via token streaming
    - Cost per query & cost per 1k queries based on input/output tokens
    - Availability & Error Rate
    - Built-in retry & backoff to prevent transient Groq OTPM rate limits
    """
    rag = FinSecureRAG(top_k=3)
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    latencies = []
    ttfts = []
    input_tokens_list = []
    output_tokens_list = []
    failures = 0

    detailed_runs = []

    for item in benchmarks:
        query = item["query"]
        success = False
        max_attempts = 4

        for attempt in range(max_attempts):
            try:
                # Pause slightly between benchmark queries to respect Groq OTPM limits
                time.sleep(2.0)
                
                # Measure TTFT via stream
                retrieved_docs = rag.retriever.retrieve(query, top_k=3)
                stream_start = time.time()
                ttft = None
                full_response = []
                
                for chunk in rag.generator.stream_generate(query, retrieved_docs):
                    if ttft is None and chunk:
                        ttft = time.time() - stream_start
                    full_response.append(chunk)

                total_latency = time.time() - stream_start
                answer_text = "".join(full_response)

                # Context size tokens + prompt overhead
                ctx_text = " ".join([d["page_content"] for d in retrieved_docs])
                in_tokens = estimate_tokens(query + ctx_text) + 150 # system prompt overhead
                out_tokens = estimate_tokens(answer_text)

                latencies.append(total_latency)
                ttfts.append(ttft or total_latency)
                input_tokens_list.append(in_tokens)
                output_tokens_list.append(out_tokens)

                detailed_runs.append({
                    "id": item["id"],
                    "query": query,
                    "latency_sec": round(total_latency, 3),
                    "ttft_sec": round(ttft or total_latency, 3),
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens
                })
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait_sec = 6.0 * (attempt + 1)
                    print(f"[!] Groq rate limit encountered during {item['id']}. Waiting {wait_sec}s before retry ({attempt+1}/{max_attempts})...")
                    time.sleep(wait_sec)
                else:
                    print(f"[!] Operation benchmark failed for {item['id']}: {e}")
                    time.sleep(3.0)

        if not success:
            failures += 1

    total_runs = len(latencies)
    if total_runs == 0:
        raise RuntimeError("No operational runs succeeded.")

    latencies.sort()
    ttfts.sort()

    p50_latency = latencies[int(len(latencies) * 0.50)]
    p95_latency = latencies[int(min(len(latencies) - 1, len(latencies) * 0.95))]
    p99_latency = latencies[-1]

    mean_ttft = statistics.mean(ttfts)
    avg_in_tokens = statistics.mean(input_tokens_list)
    avg_out_tokens = statistics.mean(output_tokens_list)

    # Cost calculation
    cost_in = (avg_in_tokens / 1_000_000) * INPUT_TOKEN_PRICE_PER_M
    cost_out = (avg_out_tokens / 1_000_000) * OUTPUT_TOKEN_PRICE_PER_M
    cost_per_query = cost_in + cost_out
    cost_per_1k_queries = cost_per_query * 1000

    error_rate = failures / len(benchmarks)

    summary = {
        "metric_type": "operations_and_economics",
        "total_requests": len(benchmarks),
        "latency_p50_sec": round(p50_latency, 3),
        "latency_p95_sec": round(p95_latency, 3),
        "latency_p99_sec": round(p99_latency, 3),
        "mean_ttft_sec": round(mean_ttft, 3),
        "avg_input_tokens": round(avg_in_tokens, 1),
        "avg_output_tokens": round(avg_out_tokens, 1),
        "cost_per_query_usd": round(cost_per_query, 6),
        "cost_per_1000_queries_usd": round(cost_per_1k_queries, 4),
        "error_rate": round(error_rate, 4),
        "detailed_runs": detailed_runs
    }
    return summary

if __name__ == "__main__":
    res = evaluate_operations_and_economics()
    print("\n--- OPERATIONS & TOKEN ECONOMICS REPORT ---")
    print(f"P50 Latency:       {res['latency_p50_sec']}s")
    print(f"P95 Latency:       {res['latency_p95_sec']}s")
    print(f"P99 Latency:       {res['latency_p99_sec']}s")
    print(f"Mean TTFT:         {res['mean_ttft_sec']}s")
    print(f"Avg Input Tokens:  {res['avg_input_tokens']}")
    print(f"Avg Output Tokens: {res['avg_output_tokens']}")
    print(f"Cost / 1k Queries: ${res['cost_per_1000_queries_usd']:.4f}")
    print(f"Error Rate:        {res['error_rate']:.2%}")
