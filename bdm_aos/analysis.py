"""
Statistical Analysis and Visualization

Produces:
- Summary tables with mean/std/min/max
- Wilcoxon rank-sum tests
- Friedman test with Nemenyi post-hoc
- Convergence plots
- Pareto front visualization
- Box plots
- AOS operator usage analysis
- Hypervolume (HV) and IGD metrics
- Sensitivity analysis plots
- Operator frequency heatmaps
- Runtime scalability breakdown
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    # ================================================================
    #  Core statistics
    # ================================================================

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
                except Exception:
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

    # ================================================================
    #  Hypervolume (HV) and IGD metrics
    # ================================================================

    def _compute_hv_2d(self, front: List[Tuple[float, float]],
                       ref: Tuple[float, float]) -> float:
        """Compute 2-D hypervolume indicator."""
        if not front:
            return 0.0
        pts = sorted(set(front), key=lambda p: p[0])
        # Filter dominated
        nd = []
        for p in pts:
            if p[0] < ref[0] and p[1] < ref[1]:
                while nd and nd[-1][1] >= p[1]:
                    nd.pop()
                nd.append(p)
        hv = 0.0
        prev_x = 0.0
        for i, (cx, cy) in enumerate(nd):
            next_x = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
            hv += (next_x - cx) * (ref[1] - cy)
        return hv

    def _compute_igd(self, front: List[Tuple[float, float]],
                     ref_front: List[Tuple[float, float]]) -> float:
        """Compute Inverted Generational Distance."""
        if not front or not ref_front:
            return float('inf')
        front_arr = np.array(front)
        ref_arr = np.array(ref_front)
        total = 0.0
        for r in ref_arr:
            dists = np.sqrt(np.sum((front_arr - r) ** 2, axis=1))
            total += np.min(dists)
        return total / len(ref_arr)

    def compute_hv_igd_table(self) -> pd.DataFrame:
        """Compute HV and IGD for all BDM variants across configs."""
        bdm_algos = [a for a in self.df["algorithm"].unique()
                     if a.startswith("BDM")]
        configs = sorted(self.df["config"].unique())
        rows = []

        for config in configs:
            # Collect ALL Pareto fronts for this config to build a reference front
            all_points = []
            fronts_by_algo = {}

            for algo in bdm_algos:
                run_dir = os.path.join(self.results_dir, "runs", config, algo)
                if not os.path.exists(run_dir):
                    continue
                algo_fronts = []
                for fname in sorted(os.listdir(run_dir)):
                    fpath = os.path.join(run_dir, fname)
                    try:
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                        pf = data.get("pareto_front", [])
                        if pf:
                            pts = [(p[0], p[1]) for p in pf
                                   if p[0] < float('inf')]
                            algo_fronts.append(pts)
                            all_points.extend(pts)
                    except Exception:
                        pass
                fronts_by_algo[algo] = algo_fronts

            if not all_points:
                continue

            # Reference point for HV
            max_cost = max(p[0] for p in all_points) * 1.1 + 1
            max_ms = max(p[1] for p in all_points) * 1.1 + 1
            ref_point = (max_cost, max_ms)

            # Build combined reference front (non-dominated from ALL runs)
            unique_pts = list(set(all_points))
            ref_front = self._non_dominated(unique_pts)

            for algo, fronts_list in fronts_by_algo.items():
                hvs = []
                igds = []
                for front in fronts_list:
                    if front:
                        hv = self._compute_hv_2d(front, ref_point)
                        igd = self._compute_igd(front, ref_front)
                        hvs.append(hv)
                        igds.append(igd)
                if hvs:
                    rows.append({
                        "config": config,
                        "algorithm": algo,
                        "mean_hv": round(np.mean(hvs), 2),
                        "std_hv": round(np.std(hvs), 2),
                        "mean_igd": round(np.mean(igds), 4),
                        "std_igd": round(np.std(igds), 4),
                        "num_runs": len(hvs),
                    })

        df_hv = pd.DataFrame(rows)
        if not df_hv.empty:
            df_hv.to_csv(os.path.join(self.results_dir, "hv_igd_metrics.csv"),
                         index=False)
            print("\n=== HV/IGD METRICS ===")
            print(df_hv.to_string())
        return df_hv

    def _non_dominated(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Extract non-dominated front from a set of points."""
        nd = []
        for p in points:
            dominated = False
            for q in points:
                if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                    dominated = True
                    break
            if not dominated:
                nd.append(p)
        return sorted(nd, key=lambda x: x[0])

    # ================================================================
    #  Figures
    # ================================================================

    def plot_convergence(self):
        """Plot convergence curves for each instance."""
        configs = sorted(self.df["config"].unique())
        main_algos = ["BDM_AOS", "BDM_NoAOS", "GA", "PSO"]

        for config in configs:
            fig, ax = plt.subplots(figsize=(8, 5))

            for algo in main_algos:
                run_dir = os.path.join(self.results_dir, "runs", config, algo)
                if not os.path.exists(run_dir):
                    continue

                all_histories = []
                for fname in sorted(os.listdir(run_dir))[:10]:
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
                    max_len = max(len(h) for h in all_histories)
                    padded = []
                    for h in all_histories:
                        padded.append(h + [h[-1]] * (max_len - len(h)))
                    mean_curve = np.mean(padded, axis=0)
                    std_curve = np.std(padded, axis=0)
                    gens = np.arange(len(mean_curve))
                    ax.plot(gens, mean_curve, label=algo, linewidth=1.5)
                    ax.fill_between(gens, mean_curve - std_curve,
                                    mean_curve + std_curve, alpha=0.15)

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

        n_configs = len(configs)
        fig, axes = plt.subplots(1, n_configs, figsize=(3.5 * n_configs, 5),
                                 sharey=False)
        if n_configs == 1:
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
            ax.set_title(config, fontsize=10)
            ax.set_ylabel("Cost" if idx == 0 else "")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', labelsize=7)

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
            grouped = algo_data.groupby("num_scenes")["runtime"].agg(["mean", "std"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                        fmt='o-', label=algo, linewidth=1.5, markersize=6,
                        capsize=3)

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
        configs = sorted(self.df["config"].unique())

        for config in configs:
            run_dir = os.path.join(self.results_dir, "runs", config, "BDM_AOS")
            if not os.path.exists(run_dir):
                continue

            fig, ax = plt.subplots(figsize=(7, 5))

            for fname in sorted(os.listdir(run_dir))[:5]:
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

    def plot_ablation_study(self):
        """Plot ablation study: BDM-AOS vs all BDM variants."""
        ablation_algos = [a for a in ["BDM_AOS", "BDM_NoAOS", "BDM_NoSIG",
                                      "BDM_NoSpecOps", "BDM_Flat"]
                          if a in self.df["algorithm"].values]
        if len(ablation_algos) < 2:
            return

        configs = sorted(self.df["config"].unique())
        # Grouped bar chart: mean cost per config per ablation variant
        fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.5), 6))
        x = np.arange(len(configs))
        width = 0.8 / len(ablation_algos)
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

        for i, algo in enumerate(ablation_algos):
            means = []
            stds = []
            for config in configs:
                vals = self.df[(self.df["config"] == config) &
                               (self.df["algorithm"] == algo) &
                               (self.df["status"] == "success")]["cost"]
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)
            label = algo.replace("BDM_", "BDM-")
            ax.bar(x + i * width - 0.4 + width / 2, means, width,
                   yerr=stds, label=label, color=colors[i % len(colors)],
                   alpha=0.8, capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_xlabel("Instance Size")
        ax.set_ylabel("Mean Cost")
        ax.set_title("Ablation Study: Component Contribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "ablation_study.pdf"),
                    dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "ablation_study.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_operator_frequency_heatmap(self):
        """Heatmap of operator selection frequency across generations."""
        configs = sorted(self.df["config"].unique())

        for config in configs:
            run_dir = os.path.join(self.results_dir, "runs", config, "BDM_AOS")
            if not os.path.exists(run_dir):
                continue

            # Aggregate operator frequency from all runs
            op_counts = {}  # (cx, mut) -> count
            gen_op_counts = {}  # gen -> {(cx,mut): count}

            for fname in sorted(os.listdir(run_dir)):
                fpath = os.path.join(run_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    aos_stats = data.get("aos_stats", {})
                    freq_log = aos_stats.get("operator_frequency_log", [])
                    for entry in freq_log:
                        cx = entry.get("cx_op", "?")
                        mut = entry.get("mut_op", "?")
                        gen = entry.get("gen", 0)
                        key = f"{cx}+{mut}"
                        op_counts[key] = op_counts.get(key, 0) + 1
                        if gen not in gen_op_counts:
                            gen_op_counts[gen] = {}
                        gen_op_counts[gen][key] = gen_op_counts[gen].get(key, 0) + 1
                except Exception:
                    pass

            if not op_counts:
                continue

            # Overall frequency bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ops = sorted(op_counts.keys())
            counts = [op_counts[o] for o in ops]
            total = sum(counts)
            pcts = [c / total * 100 for c in counts]
            ax.barh(ops, pcts, color='#3498db', alpha=0.8)
            ax.set_xlabel("Selection Frequency (%)")
            ax.set_title(f"AOS Operator Selection — {config}")
            for i, (pct, cnt) in enumerate(zip(pcts, counts)):
                ax.text(pct + 0.5, i, f"{pct:.1f}% ({cnt})", va='center', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f"operator_freq_{config}.pdf"),
                        dpi=150, bbox_inches='tight')
            plt.savefig(os.path.join(self.figures_dir, f"operator_freq_{config}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # Generation-wise heatmap (if enough data)
            if gen_op_counts and len(gen_op_counts) > 2:
                all_ops = sorted(set().union(*[set(v.keys()) for v in gen_op_counts.values()]))
                gens = sorted(gen_op_counts.keys())
                matrix = np.zeros((len(all_ops), len(gens)))
                for j, gen in enumerate(gens):
                    gen_total = sum(gen_op_counts[gen].values())
                    for i, op in enumerate(all_ops):
                        matrix[i, j] = gen_op_counts[gen].get(op, 0) / max(1, gen_total)

                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
                ax.set_yticks(range(len(all_ops)))
                ax.set_yticklabels(all_ops, fontsize=8)
                ax.set_xlabel("Generation")
                ax.set_title(f"Operator Selection Over Generations — {config}")
                plt.colorbar(im, ax=ax, label="Selection Proportion")
                plt.tight_layout()
                plt.savefig(os.path.join(self.figures_dir, f"operator_heatmap_{config}.pdf"),
                            dpi=150, bbox_inches='tight')
                plt.savefig(os.path.join(self.figures_dir, f"operator_heatmap_{config}.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()

    def plot_scalability_breakdown(self):
        """Plot cost improvement ratio of BDM-AOS vs GA/PSO across scales."""
        configs = sorted(self.df["config"].unique())
        success = self.df[self.df["status"] == "success"]
        main_algos = ["GA", "PSO"]

        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = []
        for config in configs:
            bdm_data = success[(success["config"] == config) &
                               (success["algorithm"] == "BDM_AOS")]
            if bdm_data.empty:
                continue
            bdm_mean = bdm_data["cost"].mean()
            row_sizes = bdm_data["num_scenes"].values
            if len(row_sizes) > 0:
                ns = int(row_sizes[0])
            else:
                continue
            sizes.append(ns)

            for algo in main_algos:
                algo_data = success[(success["config"] == config) &
                                    (success["algorithm"] == algo)]
                if algo_data.empty:
                    continue
                algo_mean = algo_data["cost"].mean()
                pct_diff = (algo_mean - bdm_mean) / algo_mean * 100
                marker = 'o' if algo == "GA" else 's'
                color = '#e74c3c' if algo == "GA" else '#f39c12'
                ax.scatter(ns, pct_diff, marker=marker, color=color, s=80, zorder=5)

        # Connect with lines
        for algo in main_algos:
            xs, ys = [], []
            for config in configs:
                bdm_data = success[(success["config"] == config) &
                                   (success["algorithm"] == "BDM_AOS")]
                algo_data = success[(success["config"] == config) &
                                    (success["algorithm"] == algo)]
                if bdm_data.empty or algo_data.empty:
                    continue
                ns_vals = bdm_data["num_scenes"].values
                if len(ns_vals) == 0:
                    continue
                ns = int(ns_vals[0])
                pct = (algo_data["cost"].mean() - bdm_data["cost"].mean()) / algo_data["cost"].mean() * 100
                xs.append(ns)
                ys.append(pct)
            if xs:
                ax.plot(xs, ys, '-', label=f"vs {algo}", linewidth=1.5,
                        color='#e74c3c' if algo == "GA" else '#f39c12')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Number of Scenes")
        ax.set_ylabel("BDM-AOS Improvement over Baseline (%)")
        ax.set_title("Scalability: BDM-AOS Cost Improvement")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "scalability_improvement.pdf"),
                    dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "scalability_improvement.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def plot_hv_comparison(self):
        """Bar chart comparing HV across BDM variants."""
        hv_csv = os.path.join(self.results_dir, "hv_igd_metrics.csv")
        if not os.path.exists(hv_csv):
            self.compute_hv_igd_table()
        if not os.path.exists(hv_csv):
            return

        df_hv = pd.read_csv(hv_csv)
        if df_hv.empty:
            return

        configs = sorted(df_hv["config"].unique())
        algos = sorted(df_hv["algorithm"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # HV
        ax = axes[0]
        x = np.arange(len(configs))
        width = 0.8 / max(1, len(algos))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        for i, algo in enumerate(algos):
            means = []
            stds = []
            for config in configs:
                row = df_hv[(df_hv["config"] == config) & (df_hv["algorithm"] == algo)]
                means.append(row["mean_hv"].values[0] if len(row) > 0 else 0)
                stds.append(row["std_hv"].values[0] if len(row) > 0 else 0)
            label = algo.replace("BDM_", "BDM-")
            ax.bar(x + i * width - 0.4 + width / 2, means, width,
                   yerr=stds, label=label, color=colors[i % len(colors)],
                   alpha=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylabel("Hypervolume (↑ better)")
        ax.set_title("Hypervolume Comparison")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        # IGD
        ax = axes[1]
        for i, algo in enumerate(algos):
            means = []
            stds = []
            for config in configs:
                row = df_hv[(df_hv["config"] == config) & (df_hv["algorithm"] == algo)]
                means.append(row["mean_igd"].values[0] if len(row) > 0 else 0)
                stds.append(row["std_igd"].values[0] if len(row) > 0 else 0)
            label = algo.replace("BDM_", "BDM-")
            ax.bar(x + i * width - 0.4 + width / 2, means, width,
                   yerr=stds, label=label, color=colors[i % len(colors)],
                   alpha=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylabel("IGD (↓ better)")
        ax.set_title("Inverted Generational Distance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "hv_igd_comparison.pdf"),
                    dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, "hv_igd_comparison.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ================================================================
    #  Generate all
    # ================================================================

    def generate_all(self):
        """Run all analyses and generate all figures."""
        print("=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)

        summary = self.summary_table()
        wilcoxon = self.wilcoxon_tests()
        friedman = self.friedman_test()

        print("\n" + "=" * 60)
        print("COMPUTING HV/IGD METRICS")
        print("=" * 60)

        hv_igd = self.compute_hv_igd_table()

        print("\n" + "=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)

        self.plot_convergence()
        print("  ✓ Convergence plots")

        self.plot_boxplots()
        print("  ✓ Box plots")

        self.plot_runtime_scaling()
        print("  ✓ Runtime scaling")

        self.plot_pareto_fronts()
        print("  ✓ Pareto fronts")

        self.plot_ablation_study()
        print("  ✓ Ablation study")

        self.plot_operator_frequency_heatmap()
        print("  ✓ Operator frequency heatmaps")

        self.plot_scalability_breakdown()
        print("  ✓ Scalability improvement")

        self.plot_hv_comparison()
        print("  ✓ HV/IGD comparison")

        print(f"\nAll figures saved to {self.figures_dir}")
        return summary, wilcoxon, friedman
