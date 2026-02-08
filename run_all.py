"""Run all experiments and generate analysis."""
import sys
import os

# Ensure correct path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bdm_aos.experiment_runner import ExperimentRunner
from bdm_aos.analysis import ResultAnalyzer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "experiments")
    figures_dir = os.path.join(base_dir, "latex", "figures")

    print("=" * 60)
    print("BDM-AOS Experiment Suite")
    print("=" * 60)

    configs = ["S10", "S15", "S20", "S25"]
    algorithms = ["GA", "PSO", "MILP", "BDM_NoAOS", "BDM_AOS"]
    num_seeds = 3

    print(f"Configs: {configs}")
    print(f"Algorithms: {algorithms}")
    print(f"Seeds: {num_seeds}")
    print("=" * 60)

    runner = ExperimentRunner(output_dir, num_seeds=num_seeds)
    df = runner.run_all(configs=configs, algorithms=algorithms)

    print("\n\nRunning analysis...")
    analyzer = ResultAnalyzer(output_dir, figures_dir=figures_dir)
    analyzer.generate_all()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
