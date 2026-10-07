import time
from typing import Dict, Any, List
from dotenv import load_dotenv

from langsmith import traceable
from src.retriever import HybridFinancialRetriever
from src.generator import FinancialGenerator

load_dotenv()

class FinSecureRAG:
    """
    End-to-end Enterprise RAG Pipeline for SEC 10-K & Earnings Disclosures.
    Instrumented with LangSmith nested tracing across retriever, generator, and chain.
    """
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.retriever = HybridFinancialRetriever(top_k=top_k)
        self.generator = FinancialGenerator()

    @traceable(name="hybrid_retriever", run_type="retriever")
    def retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(question, top_k=top_k)

    @traceable(name="financial_generator", run_type="llm")
    def generate(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        return self.generator.generate(question, retrieved_docs)

    @traceable(name="finsecure_rag_pipeline", run_type="chain")
    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Retrieve with nested trace
        retrieval_start = time.time()
        retrieved_docs = self.retrieve(question, self.top_k)
        retrieval_latency = time.time() - retrieval_start

        # Extract contexts
        contexts = [d["page_content"] for d in retrieved_docs]

        # 2. Generate with nested trace
        gen_start = time.time()
        answer = self.generate(question, retrieved_docs)
        gen_latency = time.time() - gen_start

        total_latency = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "retrieved_docs": retrieved_docs,
            "metadata": {
                "total_latency_sec": total_latency,
                "retrieval_latency_sec": retrieval_latency,
                "generation_latency_sec": gen_latency,
                "top_k": self.top_k,
                "num_contexts": len(contexts)
            }
        }

if __name__ == "__main__":
    rag = FinSecureRAG(top_k=2)
    res = rag.query("What was Apple's total revenue in Q4 FY2024 and how much did Services contribute?")
    print(f"Question: {res['question']}")
    print(f"Latency: {res['metadata']['total_latency_sec']:.2f}s")
    print(f"Answer:\n{res['answer']}")
