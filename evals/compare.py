import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

from evals.metric_registry import METRIC_REGISTRY, MetricDefinition

def compare_eval_runs(baseline_path: Path, candidate_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Compares candidate run against baseline using Metric Registry:
    - Applies directional comparisons (higher vs lower is better)
    - Enforces noise margins (tolerates minor random fluctuations within 2*sigma)
    - Checks critical metric violations
    - Returns (passed: bool, diff_report: Dict)
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(candidate_path, "r", encoding="utf-8") as f:
        cand_data = json.load(f)

    base_metrics = base_data.get("metrics", {})
    cand_metrics = cand_data.get("metrics", {})

    diffs = []
    has_critical_regression = False
    has_any_regression = False

    for key, defn in METRIC_REGISTRY.items():
        if key not in base_metrics or key not in cand_metrics:
            continue

        b_val = base_metrics[key]
        c_val = cand_metrics[key]
        delta = c_val - b_val
        noise_margin = defn.noise_margin

        regressed = False
        status = "PASSED"

        if defn.direction == "higher_is_better":
            # Regressed if candidate dropped by more than noise margin
            if delta < -noise_margin:
                regressed = True
        else: # lower_is_better
            # Regressed if candidate rose by more than noise margin
            if delta > noise_margin:
                regressed = True

        if regressed:
            has_any_regression = True
            if defn.critical:
                has_critical_regression = True
                status = "FAILED_CRITICAL"
            else:
                status = "WARNING_NON_CRITICAL"

        diffs.append({
            "key": key,
            "name": defn.name,
            "tier": defn.tier,
            "baseline": round(b_val, 4),
            "candidate": round(c_val, 4),
            "delta": round(delta, 4),
            "direction": defn.direction,
            "critical": defn.critical,
            "status": status
        })

    overall_passed = not has_critical_regression

    report = {
        "overall_passed": overall_passed,
        "has_critical_regression": has_critical_regression,
        "has_any_regression": has_any_regression,
        "baseline_file": str(baseline_path),
        "candidate_file": str(candidate_path),
        "comparisons": diffs
    }
    return overall_passed, report

def format_markdown_diff_table(report: Dict[str, Any]) -> str:
    lines = [
        "### [REGRESSION ENGINE] FinSecure RAG Evaluation Results",
        f"**Decision**: {'[APPROVED] APPROVED FOR DEPLOYMENT' if report['overall_passed'] else '[BLOCKED] CRITICAL REGRESSION DETECTED'}\n",
        "| Metric | Tier | Baseline | Candidate | Delta | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for d in report["comparisons"]:
        lines.append(
            f"| {d['name']} | `{d['tier']}` | {d['baseline']} | {d['candidate']} | {d['delta']:+} | {d['status']} |"
        )
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="FinSecure RAG Regression Engine")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline eval json")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate eval json")
    parser.add_argument("--report", type=str, default=None, help="Optional output markdown report path")
    args = parser.parse_args()

    passed, report = compare_eval_runs(Path(args.baseline), Path(args.candidate))
    md_table = format_markdown_diff_table(report)
    print("\n" + md_table + "\n")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(md_table)

    if not passed:
        print("[!] CI/CD Check FAILED due to critical regression.")
        sys.exit(1)
    else:
        print("[+] CI/CD Check PASSED. Ready for promotion.")
        sys.exit(0)

if __name__ == "__main__":
    main()
