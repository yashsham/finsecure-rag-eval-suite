import sys
import shutil
import argparse
from pathlib import Path

def promote_candidate_to_baseline(candidate_path: Path, baseline_path: Path):
    """
    Safely promotes an approved candidate eval JSON to become the new production baseline.
    """
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate file does not exist: {candidate_path}")

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, baseline_path)
    print(f"[+] Successfully promoted candidate ({candidate_path.name}) to baseline ({baseline_path.name})")

def main():
    parser = argparse.ArgumentParser(description="Promote Candidate Eval to Production Baseline")
    parser.add_argument("--candidate", type=str, required=True, help="Path to verified candidate JSON")
    parser.add_argument("--baseline", type=str, default="baselines/baseline.json", help="Path to production baseline JSON")
    args = parser.parse_args()

    promote_candidate_to_baseline(Path(args.candidate), Path(args.baseline))

if __name__ == "__main__":
    main()
