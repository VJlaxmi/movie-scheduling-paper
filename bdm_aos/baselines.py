"""
Baseline Algorithms for Comparison

1. Standalone GA (single-objective)
2. Standalone PSO (single-objective)
3. Full MILP (exact solver, for small instances)

Each returns results in a consistent format for fair comparison.
"""

import random
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from copy import deepcopy

from .data_generator import MSSPInstance
from .milp_solver import MILPSubProblemSolver


# ===================================================================
#  Cost Evaluation (shared by GA and PSO)
# ===================================================================

def evaluate_permutation(perm: List[int], inst: MSSPInstance) -> Tuple[float, float, Dict]:
    """
    Evaluate a scene permutation using greedy constructive decoding.
    Returns (cost, makespan, schedule).
    """
    schedule = {}
    day_hours = {}        # day -> hours used
    actor_days = {}       # actor -> set of days
    current_day = 1
    total_cost = 0.0

    for scene in perm:
        dur = inst.scene_duration.get(scene, 2.0)
        actors = inst.scene_actors.get(scene, [])
        locs = inst.scene_locations.get(scene, [])

        placed = False
        for d in range(current_day, inst.num_days + 1):
            used_h = day_hours.get(d, 0)
            if used_h + dur > 8.0:
                continue

            # Check actor availability
            all_avail = all(
                d in inst.actor_availability.get(a, inst.days)
                for a in actors
            )
            if not all_avail:
                continue

            # Check time window
            tw = inst.time_windows.get(scene)
            if tw and not (tw[0] <= d <= tw[1]):
                continue

            # Choose best location (cheapest transfer from last)
            chosen_loc = None
            min_tc = float('inf')
            last_loc = None
            if schedule:
                last_s = max(schedule, key=lambda s: schedule[s]["day"])
                last_loc = schedule[last_s].get("location")

            for loc in locs:
                if d not in inst.location_availability.get(loc, inst.days):
                    continue
                tc = 0
                if last_loc:
                    tc = inst.transfer_cost.get((last_loc, loc), 0)
                if tc < min_tc:
                    min_tc = tc
                    chosen_loc = loc

            if chosen_loc is None and locs:
                chosen_loc = locs[0]

            schedule[scene] = {
                "day": d,
                "location": chosen_loc,
                "actors": actors,
                "duration": dur
            }
            day_hours[d] = used_h + dur

            # Costs
            for a in actors:
                if a not in actor_days:
                    actor_days[a] = set()
                if d not in actor_days[a]:
                    actor_days[a].add(d)
                    total_cost += inst.actor_wage.get(a, 0)

            if last_loc and chosen_loc:
                total_cost += inst.transfer_cost.get((last_loc, chosen_loc), 0)

            # Deadline penalty
            if scene in inst.scene_deadline:
                dl = inst.scene_deadline[scene]
                if d > dl:
                    total_cost += inst.deadline_penalty.get(scene, 100) * (d - dl)

            placed = True
            break

        if not placed:
            # Force-place
            d = inst.num_days
            loc = locs[0] if locs else None
            schedule[scene] = {
                "day": d, "location": loc,
                "actors": actors, "duration": dur
            }
            total_cost += 10000  # Large penalty for infeasible

    if schedule:
        days_used = [v["day"] for v in schedule.values()]
        makespan = max(days_used) - min(days_used) + 1
    else:
        makespan = 1

    return total_cost, float(makespan), schedule


# ===================================================================
#  Standalone GA
# ===================================================================

class StandaloneGA:
    """Standard GA for MSSP (single-objective: minimize cost)."""

    def __init__(self, population_size: int = 100, generations: int = 200,
                 crossover_rate: float = 0.85, mutation_rate: float = 0.15,
                 elite_size: int = 5, seed: int = 42):
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.elite_size = elite_size
        self.seed = seed

    def run(self, inst: MSSPInstance) -> Dict:
        """Run standalone GA."""
        rng = random.Random(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        start = time.time()

        scenes = list(inst.scenes)
        n = len(scenes)

        # Initialize
        population = []
        for _ in range(self.pop_size):
            perm = scenes[:]
            rng.shuffle(perm)
            population.append(perm)

        fitness = [evaluate_permutation(p, inst)[0] for p in population]
        best_cost_history = []
        best_ever = float('inf')
        best_schedule = None
        best_perm = None

        for gen in range(self.generations):
            current_best_idx = int(np.argmin(fitness))
            if fitness[current_best_idx] < best_ever:
                best_ever = fitness[current_best_idx]
                best_perm = population[current_best_idx][:]
                _, _, best_schedule = evaluate_permutation(best_perm, inst)
            best_cost_history.append(best_ever)

            if gen % 20 == 0:
                print(f"    GA Gen {gen:3d}: Best={best_ever:.1f}")

            # Elitism
            sorted_idx = np.argsort(fitness)
            new_pop = [population[i][:] for i in sorted_idx[:self.elite_size]]

            # Generate offspring
            while len(new_pop) < self.pop_size:
                # Tournament selection
                t1 = rng.sample(range(len(population)), 3)
                p1 = population[min(t1, key=lambda i: fitness[i])][:]
                t2 = rng.sample(range(len(population)), 3)
                p2 = population[min(t2, key=lambda i: fitness[i])][:]

                # OX crossover
                if rng.random() < self.cx_rate:
                    start_pt, end_pt = sorted(rng.sample(range(n), 2))
                    c1 = [None] * n
                    c1[start_pt:end_pt] = p1[start_pt:end_pt]
                    used = set(c1[start_pt:end_pt])
                    idx = end_pt
                    for i in range(n):
                        pos = (end_pt + i) % n
                        if p2[pos] not in used:
                            c1[idx % n] = p2[pos]
                            used.add(p2[pos])
                            idx += 1
                else:
                    c1 = p1[:]

                # Swap mutation
                if rng.random() < self.mut_rate:
                    i, j = rng.sample(range(n), 2)
                    c1[i], c1[j] = c1[j], c1[i]

                if sorted(c1) == sorted(scenes):
                    new_pop.append(c1)

            population = new_pop[:self.pop_size]
            fitness = [evaluate_permutation(p, inst)[0] for p in population]

        runtime = time.time() - start
        best_cost, best_makespan, best_schedule = evaluate_permutation(best_perm, inst)

        return {
            "best_cost": best_cost,
            "best_makespan": best_makespan,
            "best_schedule": best_schedule,
            "cost_history": best_cost_history,
            "runtime": runtime,
            "algorithm": "StandaloneGA",
            "seed": self.seed,
        }


# ===================================================================
#  Standalone PSO
# ===================================================================

class StandalonePSO:
    """Discrete PSO for MSSP (following Tekin 2023 approach)."""

    def __init__(self, swarm_size: int = 50, iterations: int = 200,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 seed: int = 42):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def run(self, inst: MSSPInstance) -> Dict:
        """Run discrete PSO."""
        rng = random.Random(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        start = time.time()

        scenes = list(inst.scenes)
        n = len(scenes)

        # Initialize swarm
        particles = []
        velocities = []
        personal_best = []
        personal_best_cost = []

        for _ in range(self.swarm_size):
            perm = scenes[:]
            rng.shuffle(perm)
            particles.append(perm)
            velocities.append([rng.uniform(-1, 1) for _ in range(n)])
            personal_best.append(perm[:])
            cost, _, _ = evaluate_permutation(perm, inst)
            personal_best_cost.append(cost)

        global_best_idx = int(np.argmin(personal_best_cost))
        global_best = personal_best[global_best_idx][:]
        global_best_cost = personal_best_cost[global_best_idx]

        cost_history = []

        for it in range(self.iterations):
            for i in range(self.swarm_size):
                # Discrete PSO: apply swap sequence based on velocity
                new_perm = particles[i][:]

                # Personal best influence
                if rng.random() < self.c1 / 3:
                    # Apply swap to move towards personal best
                    for j in range(n):
                        if new_perm[j] != personal_best[i][j]:
                            if rng.random() < 0.3:
                                target = personal_best[i][j]
                                idx = new_perm.index(target)
                                new_perm[j], new_perm[idx] = new_perm[idx], new_perm[j]

                # Global best influence
                if rng.random() < self.c2 / 3:
                    for j in range(n):
                        if new_perm[j] != global_best[j]:
                            if rng.random() < 0.2:
                                target = global_best[j]
                                idx = new_perm.index(target)
                                new_perm[j], new_perm[idx] = new_perm[idx], new_perm[j]

                # Random perturbation (inertia)
                if rng.random() < self.w:
                    i1, i2 = rng.sample(range(n), 2)
                    new_perm[i1], new_perm[i2] = new_perm[i2], new_perm[i1]

                # Evaluate
                cost, _, _ = evaluate_permutation(new_perm, inst)
                particles[i] = new_perm

                if cost < personal_best_cost[i]:
                    personal_best[i] = new_perm[:]
                    personal_best_cost[i] = cost

                    if cost < global_best_cost:
                        global_best = new_perm[:]
                        global_best_cost = cost

            cost_history.append(global_best_cost)

            if it % 20 == 0:
                print(f"    PSO Iter {it:3d}: Best={global_best_cost:.1f}")

        runtime = time.time() - start
        best_cost, best_makespan, best_schedule = evaluate_permutation(global_best, inst)

        return {
            "best_cost": best_cost,
            "best_makespan": best_makespan,
            "best_schedule": best_schedule,
            "cost_history": cost_history,
            "runtime": runtime,
            "algorithm": "StandalonePSO",
            "seed": self.seed,
        }


# ===================================================================
#  Full MILP Baseline
# ===================================================================

class FullMILP:
    """Solve the complete MILP (for small instances)."""

    def __init__(self, time_limit: int = 300, seed: int = 42):
        self.time_limit = time_limit
        self.seed = seed
        self.solver = MILPSubProblemSolver(time_limit=time_limit)

    def run(self, inst: MSSPInstance) -> Dict:
        """Run full MILP solver on all scenes at once."""
        start = time.time()

        result = self.solver.solve_block(inst, inst.scenes)

        runtime = time.time() - start

        if result:
            # Compute makespan
            if result["schedule"]:
                days = [v["day"] for v in result["schedule"].values()]
                makespan = max(days) - min(days) + 1
            else:
                makespan = 0

            return {
                "best_cost": result["cost"],
                "best_makespan": float(makespan),
                "best_schedule": result["schedule"],
                "cost_history": [result["cost"]],
                "runtime": runtime,
                "algorithm": "FullMILP",
                "solver_status": result["solver_status"],
                "seed": self.seed,
            }
        else:
            return {
                "best_cost": float('inf'),
                "best_makespan": float('inf'),
                "best_schedule": {},
                "cost_history": [],
                "runtime": runtime,
                "algorithm": "FullMILP",
                "solver_status": "infeasible",
                "seed": self.seed,
            }
