import json
from pathlib import Path
from typing import Dict, Any, List

from src.retriever import HybridFinancialRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDENS_FILE = BASE_DIR / "goldens" / "retriever_goldens.json"

def evaluate_retriever_in_isolation(top_k: int = 4) -> Dict[str, Any]:
    """
    Evaluates the Retriever in Isolation:
    - Context Recall: Percentage of expected golden facts/keywords retrieved.
    - Rank-Aware Precision (MRR & NDCG-like hit rate at top-1, top-3).
    - Latency per retrieval query.
    """
    retriever = HybridFinancialRetriever(top_k=top_k)
    with open(GOLDENS_FILE, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    total_queries = len(goldens)
    recalls = []
    hit_at_1 = 0
    hit_at_3 = 0
    mrr_list = []

    results = []

    for item in goldens:
        query = item["query"]
        expected_keywords = item["expected_keywords"]
        company = item["company"]

        docs = retriever.retrieve(query, top_k=top_k)
        retrieved_texts = [d["page_content"] for d in docs]
        combined_text = " ".join(retrieved_texts)

        # 1. Recall calculation: proportion of golden keywords present in retrieved chunks
        matched_kw = [kw for kw in expected_keywords if kw.lower() in combined_text.lower()]
        recall_score = len(matched_kw) / len(expected_keywords) if expected_keywords else 0.0
        recalls.append(recall_score)

        # 2. Rank-aware metrics (MRR: reciprocal rank of first relevant doc matching company & keyword)
        first_hit_rank = None
        for rank, d in enumerate(docs, 1):
            doc_company = d.get("metadata", {}).get("company", "")
            # check if doc contains at least one primary keyword
            has_kw = any(kw.lower() in d["page_content"].lower() for kw in expected_keywords)
            if doc_company == company and has_kw:
                first_hit_rank = rank
                break

        if first_hit_rank == 1:
            hit_at_1 += 1
            hit_at_3 += 1
            mrr_list.append(1.0)
        elif first_hit_rank and first_hit_rank <= 3:
            hit_at_3 += 1
            mrr_list.append(1.0 / first_hit_rank)
        elif first_hit_rank:
            mrr_list.append(1.0 / first_hit_rank)
        else:
            mrr_list.append(0.0)

        results.append({
            "id": item["id"],
            "query": query,
            "recall": recall_score,
            "first_hit_rank": first_hit_rank,
            "matched_keywords": matched_kw
        })

    avg_recall = sum(recalls) / total_queries if total_queries else 0.0
    avg_mrr = sum(mrr_list) / total_queries if total_queries else 0.0
    precision_at_1 = hit_at_1 / total_queries if total_queries else 0.0
    precision_at_3 = hit_at_3 / total_queries if total_queries else 0.0

    summary = {
        "metric_type": "retriever_in_isolation",
        "total_queries": total_queries,
        "mean_recall": round(avg_recall, 4),
        "precision_at_1": round(precision_at_1, 4),
        "precision_at_3": round(precision_at_3, 4),
        "mean_reciprocal_rank": round(avg_mrr, 4),
        "detailed_results": results
    }
    return summary

if __name__ == "__main__":
    res = evaluate_retriever_in_isolation()
    print("--- RETRIEVER EVALUATION REPORT ---")
    print(f"Mean Recall: {res['mean_recall']:.2%}")
    print(f"Precision@1: {res['precision_at_1']:.2%}")
    print(f"Precision@3: {res['precision_at_3']:.2%}")
    print(f"MRR:         {res['mean_reciprocal_rank']:.4f}")
