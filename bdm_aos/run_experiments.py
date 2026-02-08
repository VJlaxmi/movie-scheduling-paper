"""
Main entry point: Run all experiments and generate analysis.

Usage:
    python -m bdm_aos.run_experiments
    python -m bdm_aos.run_experiments --quick    # Reduced settings for testing
"""

import sys
import os
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdm_aos.experiment_runner import ExperimentRunner
from bdm_aos.analysis import ResultAnalyzer


def main():
    parser = argparse.ArgumentParser(description="BDM-AOS Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (fewer seeds, smaller instances)")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of seeds per algorithm")
    parser.add_argument("--configs", nargs="+",
                        default=["S10", "S15", "S20", "S25", "S30", "S50"])
    parser.add_argument("--algorithms", nargs="+",
                        default=["BDM_AOS", "BDM_NoAOS", "GA", "PSO", "MILP"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.quick:
        args.seeds = 5
        args.configs = ["S10", "S15", "S20"]
        print("=== QUICK TEST MODE ===")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output or os.path.join(base_dir, "experiments")

    print("="*60)
    print("BDM-AOS: Bi-Level Decomposition Matheuristic")
    print("with Adaptive Operator Selection for MSSP")
    print("="*60)
    print(f"Configs: {args.configs}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Run experiments
    runner = ExperimentRunner(output_dir, num_seeds=args.seeds)
    df = runner.run_all(configs=args.configs, algorithms=args.algorithms)

    # Analyze results
    figures_dir = os.path.join(base_dir, "latex", "figures")
    analyzer = ResultAnalyzer(output_dir, figures_dir=figures_dir)
    analyzer.generate_all()

    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
