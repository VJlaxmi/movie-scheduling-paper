"""
Statistical Analysis and Visualization

Produces:
- Summary tables with mean/std/min/max
- Wilcoxon rank-sum tests
- Friedman test
- Convergence plots
- Pareto front visualization
- Box plots
- AOS operator usage analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple
from itertools import combinations


class ResultAnalyzer:
    """Analyze experimental results and generate figures/tables."""

    def __init__(self, results_dir: str, figures_dir: str = None):
        self.results_dir = results_dir
        self.figures_dir = figures_dir or os.path.join(results_dir, "..", "latex", "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.df = pd.read_csv(os.path.join(results_dir, "results.csv"))

    def summary_table(self) -> pd.DataFrame:
        """Generate summary statistics per algorithm per instance."""
        summary = self.df[self.df["status"] == "success"].groupby(
            ["config", "algorithm"]
        ).agg(
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            min_cost=("cost", "min"),
            max_cost=("cost", "max"),
            mean_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
            mean_runtime=("runtime", "mean"),
            num_runs=("cost", "count"),
        ).round(2)

        summary.to_csv(os.path.join(self.results_dir, "summary.csv"))
        print("\n=== SUMMARY TABLE ===")
        print(summary.to_string())
        return summary

    def wilcoxon_tests(self) -> pd.DataFrame:
        """Pairwise Wilcoxon rank-sum tests between algorithms."""
        results = []
        configs = self.df["config"].unique()
        algorithms = [a for a in self.df["algorithm"].unique() if a != "MILP"]

        for config in configs:
            config_data = self.df[(self.df["config"] == config) &
                                  (self.df["status"] == "success")]
            for a1, a2 in combinations(algorithms, 2):
                costs_a1 = config_data[config_data["algorithm"] == a1]["cost"].values
                costs_a2 = config_data[config_data["algorithm"] == a2]["cost"].values

                if len(costs_a1) < 3 or len(costs_a2) < 3:
                    continue

                try:
                    stat, p_val = stats.mannwhitneyu(
                        costs_a1, costs_a2, alternative='two-sided'
                    )
                    # Effect size (rank-biserial correlation)
                    n1, n2 = len(costs_a1), len(costs_a2)
                    effect_size = 1 - 2 * stat / (n1 * n2)

                    results.append({
                        "config": config,
                        "algo1": a1,
                        "algo2": a2,
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(p_val), 6),
                        "significant": bool(p_val < 0.05),
                        "effect_size": round(float(effect_size), 4),
                        "mean_diff": round(float(np.mean(costs_a1) - np.mean(costs_a2)), 2),
                        "better": a1 if np.mean(costs_a1) < np.mean(costs_a2) else a2,
                    })
                except Exception as e:
                    pass

        df_tests = pd.DataFrame(results)
        df_tests.to_csv(os.path.join(self.results_dir, "wilcoxon_tests.csv"), index=False)
        print("\n=== WILCOXON RANK-SUM TESTS ===")
        print(df_tests.to_string())
        return df_tests

    def friedman_test(self) -> Dict:
        """Friedman test across all algorithms and instances."""
        algorithms = [a for a in self.df["algorithm"].unique() if a != "MILP"]
        configs = self.df["config"].unique()

        # Build rank matrix
        ranks_per_config = []
        for config in configs:
            config_data = self.df[(self.df["config"] == config) &
                                  (self.df["status"] == "success")]
            means = {}
            for algo in algorithms:
                algo_data = config_data[config_data["algorithm"] == algo]["cost"]
                if len(algo_data) > 0:
                    means[algo] = algo_data.mean()

            if len(means) >= 2:
                sorted_algos = sorted(means.keys(), key=lambda a: means[a])
                ranks = {algo: rank + 1 for rank, algo in enumerate(sorted_algos)}
                ranks_per_config.append(ranks)

        if not ranks_per_config:
            return {"error": "Insufficient data for Friedman test"}

        # Build matrix
        algo_list = list(set().union(*[r.keys() for r in ranks_per_config]))
        matrix = []
        for ranks in ranks_per_config:
            row = [ranks.get(a, len(algo_list)) for a in algo_list]
            matrix.append(row)

        matrix = np.array(matrix)

        if matrix.shape[0] >= 3 and matrix.shape[1] >= 2:
            try:
                stat, p_val = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
                result = {
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p_val), 6),
                    "significant": bool(p_val < 0.05),
                    "avg_ranks": {algo: round(float(np.mean(matrix[:, i])), 2)
                                  for i, algo in enumerate(algo_list)},
                }
            except Exception:
                result = {"error": "Friedman test failed", "avg_ranks": {}}
        else:
            result = {"error": "Not enough data for Friedman test"}

        with open(os.path.join(self.results_dir, "friedman_test.json"), 'w') as f:
            json.dump(result, f, indent=2)

        print("\n=== FRIEDMAN TEST ===")
        print(json.dumps(result, indent=2))
        return result

    def plot_convergence(self):
        """Plot convergence curves for each instance."""
        configs = self.df["config"].unique()

        for config in configs:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Load individual run files for convergence history
            for algo in ["BDM_AOS", "BDM_NoAOS", "GA", "PSO"]:
                run_dir = os.path.join(self.results_dir, "runs", config, algo)
                if not os.path.exists(run_dir):
                    continue

                all_histories = []
                for fname in sorted(os.listdir(run_dir))[:5]:  # First 5 seeds
                    fpath = os.path.join(run_dir, fname)
                    try:
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                        hist = data.get("cost_history", [])
                        if hist:
                            all_histories.append(hist)
                    except Exception:
                        pass

                if all_histories:
                    # Pad to same length
                    max_len = max(len(h) for h in all_histories)
                    padded = []
                    for h in all_histories:
                        padded.append(h + [h[-1]] * (max_len - len(h)))
                    mean_curve = np.mean(padded, axis=0)
                    ax.plot(mean_curve, label=algo, linewidth=1.5)

            ax.set_xlabel("Generation / Iteration")
            ax.set_ylabel("Best Cost")
            ax.set_title(f"Convergence — {config}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f"convergence_{config}.pdf"),
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, f"convergence_{config}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def plot_boxplots(self):
        """Box plots comparing algorithms across instances."""
        configs = sorted(self.df["config"].unique())
        algorithms = [a for a in ["BDM_AOS", "BDM_NoAOS", "GA", "PSO"]
                       if a in self.df["algorithm"].values]

        fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 5),
                                 sharey=False)
        if len(configs) == 1:
            axes = [axes]

        for idx, config in enumerate(configs):
            ax = axes[idx]
            data_to_plot = []
            labels = []
            for algo in algorithms:
                vals = self.df[(self.df["config"] == config) &
                               (self.df["algorithm"] == algo) &
                               (self.df["status"] == "success")]["cost"].values
                if len(vals) > 0:
                    data_to_plot.append(vals)
                    labels.append(algo.replace("BDM_", "BDM-\n"))

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
                for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            ax.set_title(config)
            ax.set_ylabel("Cost" if idx == 0 else "")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Cost Distribution by Algorithm and Instance Size", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "boxplots.pdf"),
                    dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "boxplots.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_runtime_scaling(self):
        """Plot runtime vs instance size."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for algo in ["BDM_AOS", "GA", "PSO", "MILP"]:
            algo_data = self.df[(self.df["algorithm"] == algo) &
                                 (self.df["status"] == "success")]
            if algo_data.empty:
                continue
            grouped = algo_data.groupby("num_scenes")["runtime"].mean()
            ax.plot(grouped.index, grouped.values, 'o-',
                    label=algo, linewidth=1.5, markersize=6)

        ax.set_xlabel("Number of Scenes")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Runtime Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "runtime_scaling.pdf"),
                    dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "runtime_scaling.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_pareto_fronts(self):
        """Plot Pareto fronts for BDM-AOS."""
        configs = self.df["config"].unique()

        for config in configs:
            run_dir = os.path.join(self.results_dir, "runs", config, "BDM_AOS")
            if not os.path.exists(run_dir):
                continue

            fig, ax = plt.subplots(figsize=(7, 5))

            for fname in sorted(os.listdir(run_dir))[:3]:
                fpath = os.path.join(run_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    pf = data.get("pareto_front", [])
                    if pf:
                        costs = [p[0] for p in pf]
                        makespans = [p[1] for p in pf]
                        seed = data.get("seed", "?")
                        ax.scatter(costs, makespans, alpha=0.7,
                                   label=f"Seed {seed}", s=30)
                except Exception:
                    pass

            ax.set_xlabel("Total Cost")
            ax.set_ylabel("Makespan (days)")
            ax.set_title(f"Pareto Front — {config}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f"pareto_{config}.pdf"),
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, f"pareto_{config}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def generate_all(self):
        """Run all analyses and generate all figures."""
        print("="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        summary = self.summary_table()
        wilcoxon = self.wilcoxon_tests()
        friedman = self.friedman_test()

        print("\n" + "="*60)
        print("GENERATING FIGURES")
        print("="*60)

        self.plot_convergence()
        print("  ✓ Convergence plots")

        self.plot_boxplots()
        print("  ✓ Box plots")

        self.plot_runtime_scaling()
        print("  ✓ Runtime scaling")

        self.plot_pareto_fronts()
        print("  ✓ Pareto fronts")

        print(f"\nAll figures saved to {self.figures_dir}")
        return summary, wilcoxon, friedman
