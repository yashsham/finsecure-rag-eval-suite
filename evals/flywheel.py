import json
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
ONLINE_LOGS_FILE = BASE_DIR / "data" / "online_eval_results.json"
GOLDENS_DIR = BASE_DIR / "goldens"
SYNTHESIZED_FILE = GOLDENS_DIR / "curated_flywheel_cases.json"

def run_data_flywheel(faithfulness_threshold: float = 0.7) -> int:
    """
    Session 8 Data Flywheel:
    - Mines online production traces that failed evaluation or were flagged for review.
    - Curates and transforms edge-case production failures into high-value golden dataset entries.
    - Retrains and expands offline test suites to prevent future regression loops.
    """
    print("[*] Running Autonomous Data Flywheel...")
    if not ONLINE_LOGS_FILE.exists():
        print(f"[!] No online evaluation logs found at {ONLINE_LOGS_FILE}")
        return 0

    with open(ONLINE_LOGS_FILE, "r", encoding="utf-8") as f:
        traces = json.load(f)

    curated_cases = []
    for item in traces:
        # Extract edge cases or high-value audits
        if item.get("flagged_for_human_review", False) or item.get("faithfulness", 1.0) < faithfulness_threshold:
            curated_cases.append({
                "source": "production_online_trace",
                "timestamp": item.get("timestamp"),
                "query": item.get("input"),
                "problematic_output": item.get("output"),
                "faithfulness_score": item.get("faithfulness"),
                "review_status": "PENDING_SME_ANNOTATION"
            })
        else:
            # Also sample high-confidence queries for expanding golden benchmarks
            curated_cases.append({
                "source": "production_online_trace_high_quality",
                "timestamp": item.get("timestamp"),
                "query": item.get("input"),
                "verified_output": item.get("output"),
                "faithfulness_score": item.get("faithfulness"),
                "review_status": "AUTO_ACCEPTED"
            })

    SYNTHESIZED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTHESIZED_FILE, "w", encoding="utf-8") as f:
        json.dump(curated_cases, f, indent=2)

    print(f"[+] Flywheel complete! Synthesized {len(curated_cases)} cases into: {SYNTHESIZED_FILE}")
    return len(curated_cases)

if __name__ == "__main__":
    run_data_flywheel()
