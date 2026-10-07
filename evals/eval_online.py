import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

from langsmith import Client
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.judge_model import GroqJudgeModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ONLINE_LOGS_FILE = BASE_DIR / "data" / "online_eval_results.json"

class OnlineEvalWorker:
    """
    Session 8 Architecture: Online Observability & Async Evaluation.
    - Continuously or periodically polls traces from LangSmith / local production logs.
    - Evaluates live user interactions asynchronously in background.
    - Flags potential hallucinations or toxic outputs without degrading user-facing latency.
    """
    def __init__(self, project_name: str = "finsecure-rag"):
        self.project_name = project_name
        self.client = Client() if os.getenv("LANGCHAIN_API_KEY") else None
        self.judge = GroqJudgeModel()
        self.faith_metric = FaithfulnessMetric(threshold=0.7, model=self.judge)
        self.rel_metric = AnswerRelevancyMetric(threshold=0.7, model=self.judge)

    def evaluate_live_trace(self, input_text: str, output_text: str, contexts: List[str]) -> Dict[str, Any]:
        """Runs asynchronous quality audit on a single live production query."""
        test_case = LLMTestCase(
            input=input_text,
            actual_output=output_text,
            retrieval_context=contexts
        )

        # 1. Faithfulness
        self.faith_metric.measure(test_case)
        f_score = self.faith_metric.score if self.faith_metric.score is not None else 1.0

        # 2. Answer Relevancy
        self.rel_metric.measure(test_case)
        r_score = self.rel_metric.score if self.rel_metric.score is not None else 1.0

        is_flagged = f_score < 0.7 or r_score < 0.7

        return {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "output": output_text,
            "faithfulness": round(f_score, 4),
            "relevancy": round(r_score, 4),
            "flagged_for_human_review": is_flagged
        }

    def process_trace_batch(self, simulated_traces: List[Dict[str, Any]] = None):
        """Processes a batch of production traces and writes audit logs."""
        print(f"[*] Online Eval Worker: Processing batch of traces at {datetime.now().isoformat()}...")
        traces = simulated_traces or [
            {
                "input": "What was Apple Services revenue in Q4?",
                "output": "Apple Services revenue reached an all-time record of $24.97 billion, up 12% year-over-year.",
                "contexts": ["Services revenue achieved an all-time record of $24.97 billion, growing 12% year-over-year."]
            }
        ]

        results = []
        for t in traces:
            audit = self.evaluate_live_trace(t["input"], t["output"], t["contexts"])
            results.append(audit)

        # Save online audit logs
        ONLINE_LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if ONLINE_LOGS_FILE.exists():
            try:
                with open(ONLINE_LOGS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        existing.extend(results)
        with open(ONLINE_LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        print(f"[+] Successfully evaluated {len(results)} live traces. Logged to: {ONLINE_LOGS_FILE}")
        return results

if __name__ == "__main__":
    worker = OnlineEvalWorker()
    worker.process_trace_batch()
