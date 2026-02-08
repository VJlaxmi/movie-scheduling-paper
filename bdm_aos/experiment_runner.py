"""
Experiment Runner

Runs all algorithms on all instances with multiple seeds,
collects results, and saves to JSON/CSV for analysis.
"""

import os
import json
import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List
from copy import deepcopy

from .data_generator import ScaledDataGenerator, MSSPInstance
from .matheuristic import BDM_AOS
from .baselines import StandaloneGA, StandalonePSO, FullMILP


class ExperimentRunner:
    """Run comprehensive experiments across instances and seeds."""

    def __init__(self, output_dir: str, num_seeds: int = 30):
        self.output_dir = output_dir
        self.num_seeds = num_seeds
        os.makedirs(output_dir, exist_ok=True)

    def run_all(self, configs: List[str] = None,
                algorithms: List[str] = None) -> pd.DataFrame:
        """
        Run all experiments.

        Args:
            configs: Instance configs to run (default: all)
            algorithms: Algorithms to run (default: all)
        """
        if configs is None:
            configs = ["S10", "S15", "S20", "S25", "S30", "S50", "S75", "S100"]
        if algorithms is None:
            algorithms = ["BDM_AOS", "BDM_NoAOS", "BDM_NoSIG", "BDM_NoSpecOps",
                          "BDM_Flat", "GA", "PSO", "MILP"]

        # Generate instances
        gen = ScaledDataGenerator(seed=42)
        instances_dir = os.path.join(self.output_dir, "instances")
        gen.generate_all(instances_dir, num_variants=1)

        all_results = []

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Instance: {config}")
            print(f"{'='*60}")

            # Load instance
            inst_path = os.path.join(instances_dir, f"{config}_v0.json")
            with open(inst_path, 'r') as f:
                inst_data = json.load(f)
            inst = MSSPInstance.from_dict(inst_data)

            for algo_name in algorithms:
                # Skip MILP for large instances
                if algo_name == "MILP" and inst.num_scenes > 15:
                    print(f"  Skipping {algo_name} for {config} (too large)")
                    all_results.append({
                        "config": config, "algorithm": algo_name,
                        "seed": 0, "cost": float('inf'),
                        "makespan": float('inf'), "runtime": 0,
                        "status": "skipped"
                    })
                    continue

                seeds = [1] if algo_name == "MILP" else list(range(1, self.num_seeds + 1))

                for seed in seeds:
                    # Skip if already completed
                    run_dir = os.path.join(self.output_dir, "runs",
                                           config, algo_name)
                    run_file = os.path.join(run_dir, f"seed_{seed}.json")
                    if os.path.exists(run_file):
                        # Load cached result
                        with open(run_file, 'r') as rf:
                            cached = json.load(rf)
                        row = {
                            "config": config, "algorithm": algo_name,
                            "seed": seed,
                            "cost": cached.get("best_cost", float('inf')),
                            "makespan": cached.get("best_makespan", float('inf')),
                            "runtime": cached.get("runtime", 0),
                            "status": "success",
                            "num_scenes": inst.num_scenes,
                            "num_actors": inst.num_actors,
                        }
                        if "num_blocks" in cached:
                            row["num_blocks"] = cached["num_blocks"]
                        if "aos_stats" in cached and cached["aos_stats"]:
                            row["best_operator"] = cached["aos_stats"].get("best_operator", "")
                        if "pareto_front" in cached:
                            row["pareto_size"] = len(cached["pareto_front"])
                        all_results.append(row)
                        print(f"  {algo_name} seed={seed} on {config}: CACHED (cost={row['cost']:.1f})")
                        continue

                    print(f"\n  {algo_name} seed={seed} on {config}...")

                    try:
                        result = self._run_single(inst, algo_name, seed)

                        row = {
                            "config": config,
                            "algorithm": algo_name,
                            "seed": seed,
                            "cost": result.get("best_cost", float('inf')),
                            "makespan": result.get("best_makespan", float('inf')),
                            "runtime": result.get("runtime", 0),
                            "status": "success",
                            "num_scenes": inst.num_scenes,
                            "num_actors": inst.num_actors,
                        }

                        # Add BDM-specific stats
                        if "num_blocks" in result:
                            row["num_blocks"] = result["num_blocks"]
                        if "aos_stats" in result and result["aos_stats"]:
                            row["best_operator"] = result["aos_stats"].get("best_operator", "")
                        if "pareto_front" in result:
                            row["pareto_size"] = len(result["pareto_front"])

                        all_results.append(row)

                        # Save individual run
                        run_dir = os.path.join(self.output_dir, "runs",
                                               config, algo_name)
                        os.makedirs(run_dir, exist_ok=True)
                        run_file = os.path.join(run_dir, f"seed_{seed}.json")

                        # Convert non-serializable types
                        save_data = {}
                        for k, v in result.items():
                            if isinstance(v, (int, float, str, list, dict, bool)):
                                save_data[k] = v
                            elif hasattr(v, 'objectives'):
                                save_data[k] = {"cost": v.cost, "makespan": v.makespan}
                        with open(run_file, 'w') as f:
                            json.dump(save_data, f, indent=2, default=str)

                    except Exception as e:
                        print(f"    ERROR: {e}")
                        traceback.print_exc()
                        all_results.append({
                            "config": config, "algorithm": algo_name,
                            "seed": seed, "cost": float('inf'),
                            "makespan": float('inf'), "runtime": 0,
                            "status": f"error: {str(e)}"
                        })

        # Save results
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        return df

    def _run_single(self, inst: MSSPInstance,
                    algo_name: str, seed: int) -> Dict:
        """Run a single algorithm on a single instance with given seed."""

        if algo_name == "BDM_AOS":
            algo = BDM_AOS(
                population_size=20,
                generations=40,
                crossover_rate=0.9,
                mutation_rate=0.3,
                use_aos=True,
                use_sig=True,
                use_spec_ops=True,
                flat_mode=False,
                seed=seed,
                milp_time_limit=10,
            )
            return algo.run(inst)

        elif algo_name == "BDM_NoAOS":
            algo = BDM_AOS(
                population_size=20,
                generations=40,
                crossover_rate=0.9,
                mutation_rate=0.3,
                use_aos=False,
                use_sig=True,
                use_spec_ops=True,
                flat_mode=False,
                seed=seed,
                milp_time_limit=10,
            )
            return algo.run(inst)

        elif algo_name == "BDM_NoSIG":
            algo = BDM_AOS(
                population_size=20,
                generations=40,
                crossover_rate=0.9,
                mutation_rate=0.3,
                use_aos=True,
                use_sig=False,
                use_spec_ops=True,
                flat_mode=False,
                seed=seed,
                milp_time_limit=10,
            )
            return algo.run(inst)

        elif algo_name == "BDM_NoSpecOps":
            algo = BDM_AOS(
                population_size=20,
                generations=40,
                crossover_rate=0.9,
                mutation_rate=0.3,
                use_aos=True,
                use_sig=True,
                use_spec_ops=False,
                flat_mode=False,
                seed=seed,
                milp_time_limit=10,
            )
            return algo.run(inst)

        elif algo_name == "BDM_Flat":
            algo = BDM_AOS(
                population_size=20,
                generations=40,
                crossover_rate=0.9,
                mutation_rate=0.3,
                use_aos=True,
                use_sig=True,
                use_spec_ops=True,
                flat_mode=True,
                seed=seed,
                milp_time_limit=10,
            )
            return algo.run(inst)

        elif algo_name == "GA":
            algo = StandaloneGA(
                population_size=50,
                generations=100,
                crossover_rate=0.85,
                mutation_rate=0.15,
                seed=seed,
            )
            return algo.run(inst)

        elif algo_name == "PSO":
            algo = StandalonePSO(
                swarm_size=30,
                iterations=100,
                seed=seed,
            )
            return algo.run(inst)

        elif algo_name == "MILP":
            algo = FullMILP(time_limit=60, seed=seed)
            return algo.run(inst)

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
