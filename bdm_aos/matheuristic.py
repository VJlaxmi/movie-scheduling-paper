"""
BDM-AOS: Bi-Level Decomposition Matheuristic with Adaptive Operator Selection

Core algorithm combining:
  - Upper level: NSGA-II GA over block permutations
  - Lower level: MILP solver for intra-block scheduling
  - AOS: Q-Learning adaptive operator selection
"""

import random
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from copy import deepcopy

from .data_generator import MSSPInstance
from .scene_graph import SceneInteractionGraph
from .milp_solver import MILPSubProblemSolver
from .nsga2 import (Solution, fast_non_dominated_sort, crowding_distance,
                     nsga2_select, binary_tournament, compute_hypervolume)
from .q_learning_aos import QLearningAOS
from .operators import apply_crossover, apply_mutation


class BDM_AOS:
    """
    Bi-Level Decomposition Matheuristic with Adaptive Operator Selection.

    Algorithm:
    1. Build Scene Interaction Graph (SIG)
    2. Louvain clustering → shooting blocks
    3. NSGA-II over block permutations (upper level)
    4. MILP per block (lower level)
    5. Q-Learning AOS for operator selection
    """

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.2,
                 sig_alpha: float = 0.5,
                 sig_beta: float = 0.3,
                 sig_gamma: float = 0.2,
                 louvain_resolution: float = 1.0,
                 milp_time_limit: int = 60,
                 use_aos: bool = True,
                 seed: int = 42):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.sig_alpha = sig_alpha
        self.sig_beta = sig_beta
        self.sig_gamma = sig_gamma
        self.louvain_resolution = louvain_resolution
        self.milp_time_limit = milp_time_limit
        self.use_aos = use_aos
        self.seed = seed

        self.rng = random.Random(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Components
        self.sig = SceneInteractionGraph(sig_alpha, sig_beta, sig_gamma)
        self.milp = MILPSubProblemSolver(time_limit=milp_time_limit)
        self.milp._fast_mode = True  # Use greedy only in GA loop for speed
        self.aos = QLearningAOS() if use_aos else None

        # State
        self.blocks: List[List[int]] = []
        self.block_actors: Dict[int, Set[int]] = {}  # block_idx -> actors
        self.block_locations: Dict[int, int] = {}     # block_idx -> primary location
        self.inst: MSSPInstance = None

        # Evaluation cache: tuple(permutation) -> Solution
        self._eval_cache: Dict[tuple, Solution] = {}

        # Results
        self.history: List[Dict] = []

    def _setup_blocks(self, inst: MSSPInstance):
        """Build SIG, decompose into blocks, compute block metadata."""
        self.inst = inst
        # Target blocks of 2-4 scenes for meaningful intra-block optimization
        # With n blocks, search space = n! permutations
        # Need enough blocks for GA search but not too many single-scene blocks
        min_blocks = max(3, inst.num_scenes // 5)
        max_block_size = max(3, min(6, inst.num_scenes // 2))
        
        self.blocks, self.graph = self.sig.decompose(
            inst, resolution=self.louvain_resolution,
            min_blocks=min_blocks, max_block_size=max_block_size
        )

        # Compute block metadata
        self.block_actors = {}
        self.block_locations = {}
        for idx, block in enumerate(self.blocks):
            actors = set()
            loc_counts: Dict[int, int] = {}
            for s in block:
                actors.update(inst.scene_actors.get(s, []))
                for loc in inst.scene_locations.get(s, []):
                    loc_counts[loc] = loc_counts.get(loc, 0) + 1
            self.block_actors[idx] = actors
            # Primary location = most frequent
            if loc_counts:
                self.block_locations[idx] = max(loc_counts, key=loc_counts.get)

    def _create_individual(self) -> List[int]:
        """Create a random permutation of block indices."""
        perm = list(range(len(self.blocks)))
        self.rng.shuffle(perm)
        return perm

    def _evaluate(self, chromosome: List[int]) -> Solution:
        """
        Evaluate a block permutation using global reassembly.
        
        The block permutation determines scene ordering:
        scenes from block chromosome[0] come first, then chromosome[1], etc.
        Within each block, scenes are sorted by constraint tightness.
        
        A global greedy decoder then packs ALL scenes into days,
        allowing cross-block day sharing (same decoder as GA/PSO baselines).
        This ensures BDM is cost-competitive while benefiting from
        decomposition-guided ordering.

        Uses caching to avoid re-solving identical permutations.
        Returns Solution with (cost, makespan) objectives.
        """
        cache_key = tuple(chromosome)
        if cache_key in self._eval_cache:
            cached = self._eval_cache[cache_key]
            sol = Solution(chromosome[:], cached.objectives)
            sol.schedule = cached.schedule
            sol.stats = cached.stats
            return sol

        inst = self.inst

        # Build scene ordering from block permutation
        # Within each block, sort scenes by tightness (fewest valid days first)
        def scene_flexibility(s):
            tw = inst.time_windows.get(s)
            if tw:
                return tw[1] - tw[0]
            return len(inst.days)

        scene_order = []
        for block_idx in chromosome:
            block_scenes = sorted(self.blocks[block_idx], key=scene_flexibility)
            scene_order.extend(block_scenes)

        # Global greedy decoder (same logic as baselines.evaluate_permutation)
        schedule = {}
        day_hours = {}        # day -> hours used
        actor_days = {}       # actor -> set of days
        total_cost = 0.0

        for scene in scene_order:
            dur = inst.scene_duration.get(scene, 2.0)
            actors = inst.scene_actors.get(scene, [])
            locs = inst.scene_locations.get(scene, [])

            placed = False
            for d in range(1, inst.num_days + 1):
                used_h = day_hours.get(d, 0)
                if used_h + dur > 8.0:
                    continue

                all_avail = all(
                    d in inst.actor_availability.get(a, inst.days)
                    for a in actors
                )
                if not all_avail:
                    continue

                tw = inst.time_windows.get(scene)
                if tw and not (tw[0] <= d <= tw[1]):
                    continue

                # Choose best location
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

                for a in actors:
                    if a not in actor_days:
                        actor_days[a] = set()
                    if d not in actor_days[a]:
                        actor_days[a].add(d)
                        total_cost += inst.actor_wage.get(a, 0)

                if last_loc and chosen_loc and last_loc != chosen_loc:
                    total_cost += inst.transfer_cost.get((last_loc, chosen_loc), 0)

                if scene in inst.scene_deadline:
                    dl = inst.scene_deadline[scene]
                    if d > dl:
                        total_cost += inst.deadline_penalty.get(scene, 100) * (d - dl)

                placed = True
                break

            if not placed:
                d = inst.num_days
                loc = locs[0] if locs else None
                schedule[scene] = {
                    "day": d, "location": loc,
                    "actors": actors, "duration": dur
                }
                total_cost += 10000

        if schedule:
            days_used = [v["day"] for v in schedule.values()]
            makespan = max(days_used) - min(days_used) + 1
        else:
            makespan = 1

        sol = Solution(chromosome, (total_cost, float(makespan)))
        sol.schedule = schedule
        sol.stats = {
            "num_blocks": len(self.blocks),
        }

        self._eval_cache[cache_key] = sol
        return sol

    def _compute_diversity(self, population: List[Solution]) -> float:
        """Compute population diversity (normalized Hamming distance)."""
        if len(population) <= 1:
            return 1.0
        total_dist = 0
        comparisons = 0
        for i in range(min(20, len(population))):
            for j in range(i + 1, min(20, len(population))):
                dist = sum(a != b for a, b in
                           zip(population[i].chromosome, population[j].chromosome))
                total_dist += dist
                comparisons += 1
        if comparisons == 0:
            return 0.0
        max_dist = len(population[0].chromosome)
        return (total_dist / comparisons) / max_dist if max_dist > 0 else 0.0

    def run(self, inst: MSSPInstance) -> Dict:
        """
        Run the BDM-AOS algorithm.

        Returns dict with Pareto front, statistics, and best solutions.
        """
        start_time = time.time()

        # Step 1: Build SIG and decompose
        self._setup_blocks(inst)
        num_blocks = len(self.blocks)
        block_stats = self.sig.get_block_stats(self.blocks, inst)

        print(f"  Decomposed {inst.num_scenes} scenes into {num_blocks} blocks: "
              f"{[len(b) for b in self.blocks]}")

        if num_blocks <= 1:
            # Single block - just solve directly
            result = self.milp.solve_block(inst, self.blocks[0])
            sol = Solution([0], (result["cost"], result["makespan"]))
            sol.schedule = result["schedule"]
            return {
                "pareto_front": [(result["cost"], result["makespan"])],
                "best_cost_solution": sol,
                "best_makespan_solution": sol,
                "best_cost": result["cost"],
                "best_makespan": result["makespan"],
                "hv_history": [],
                "cost_history": [result["cost"]],
                "makespan_history": [result["makespan"]],
                "generations_history": [],
                "block_stats": block_stats,
                "num_blocks": num_blocks,
                "runtime": time.time() - start_time,
                "generations": 0,
                "population_size": 0,
                "aos_stats": {},
                "seed": self.seed,
            }

        # Step 2: Initialize population
        population = []
        for _ in range(self.population_size):
            chrom = self._create_individual()
            sol = self._evaluate(chrom)
            population.append(sol)

        # Initial sort
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance(population, front)

        # Compute ref_point from finite values
        finite_costs = [s.cost for s in population if s.cost < float('inf')]
        finite_makespans = [s.makespan for s in population if s.makespan < float('inf')]
        if not finite_costs:
            finite_costs = [1.0]
        if not finite_makespans:
            finite_makespans = [1.0]

        ref_point = (
            max(finite_costs) * 1.2 + 1,
            max(finite_makespans) * 1.2 + 1
        )

        hv_history = []
        cost_history = []
        makespan_history = []

        prev_hv = 0.0
        improvement_rate = 0.0

        # Step 3: Evolutionary loop
        stagnation_count = 0
        prev_best_cost = float('inf')

        for gen in range(self.generations):
            diversity = self._compute_diversity(population)

            # Diversity restart: if diversity drops too low, inject randoms
            if diversity < 0.05 and gen > 5:
                stagnation_count += 1
                if stagnation_count >= 3:
                    # Replace 50% of population with random individuals
                    n_replace = self.population_size // 2
                    # Keep best front
                    keep = population[:self.population_size - n_replace]
                    for _ in range(n_replace):
                        chrom = self._create_individual()
                        keep.append(self._evaluate(chrom))
                    population = keep
                    stagnation_count = 0
                    diversity = self._compute_diversity(population)
            else:
                stagnation_count = 0

            # Compute current hypervolume
            current_hv = compute_hypervolume(population, ref_point)
            if prev_hv > 0:
                improvement_rate = (current_hv - prev_hv) / prev_hv
            prev_hv = current_hv

            quality_gap = 1.0 - (current_hv / (ref_point[0] * ref_point[1]))
            quality_gap = max(0, min(1, quality_gap))

            hv_history.append(current_hv)
            best_cost = min(s.cost for s in population)
            best_makespan = min(s.makespan for s in population)
            cost_history.append(best_cost)
            makespan_history.append(best_makespan)

            if gen % 10 == 0:
                print(f"    Gen {gen:3d}: HV={current_hv:.1f} "
                      f"BestCost={best_cost:.1f} BestMakespan={best_makespan:.0f} "
                      f"Div={diversity:.3f}")

            # Select operator via AOS
            if self.use_aos and self.aos:
                state_info = (gen, self.generations, diversity,
                              abs(improvement_rate), quality_gap)
                cx_op, mut_op, action_idx = self.aos.select_action(*state_info)
            else:
                cx_op = self.rng.choice(["pmx", "ox", "lax"])
                mut_op = self.rng.choice(["swap", "block_swap", "acm"])
                action_idx = 0

            # Generate offspring
            offspring = []
            while len(offspring) < self.population_size:
                p1 = binary_tournament(population)
                p2 = binary_tournament(population)

                if self.rng.random() < self.crossover_rate:
                    c1_chrom, c2_chrom = apply_crossover(
                        cx_op, p1.chromosome, p2.chromosome,
                        self.block_locations, self.inst
                    )
                else:
                    c1_chrom, c2_chrom = p1.chromosome[:], p2.chromosome[:]

                c1_chrom = apply_mutation(
                    mut_op, c1_chrom, self.block_actors,
                    self.inst, self.mutation_rate
                )
                c2_chrom = apply_mutation(
                    mut_op, c2_chrom, self.block_actors,
                    self.inst, self.mutation_rate
                )

                # Validate permutations
                if (sorted(c1_chrom) == list(range(num_blocks)) and
                        len(c1_chrom) == num_blocks):
                    offspring.append(self._evaluate(c1_chrom))
                if (sorted(c2_chrom) == list(range(num_blocks)) and
                        len(c2_chrom) == num_blocks and
                        len(offspring) < self.population_size):
                    offspring.append(self._evaluate(c2_chrom))

            # NSGA-II selection
            combined = population + offspring
            population = nsga2_select(combined, self.population_size)

            # Update AOS
            new_hv = compute_hypervolume(population, ref_point)
            reward = new_hv - current_hv
            if self.use_aos and self.aos:
                next_diversity = self._compute_diversity(population)
                next_imp = abs(reward / max(1, current_hv))
                next_gap = 1.0 - (new_hv / (ref_point[0] * ref_point[1]))
                next_gap = max(0, min(1, next_gap))
                next_state = (gen + 1, self.generations, next_diversity,
                              next_imp, next_gap)
                self.aos.update(state_info, action_idx, reward, next_state)

        # Extract Pareto front
        fronts = fast_non_dominated_sort(population)
        pareto_front = [(population[i].cost, population[i].makespan)
                        for i in fronts[0]
                        if population[i].cost < float('inf')]

        # Best solutions
        valid = [s for s in population if s.cost < float('inf')]
        if not valid:
            valid = population  # use all if none are finite
        best_cost_sol = min(valid, key=lambda s: s.cost)
        best_makespan_sol = min(valid, key=lambda s: s.makespan)

        runtime = time.time() - start_time

        return {
            "pareto_front": sorted(pareto_front),
            "best_cost_solution": best_cost_sol,
            "best_makespan_solution": best_makespan_sol,
            "best_cost": best_cost_sol.cost,
            "best_makespan": best_makespan_sol.makespan,
            "hv_history": hv_history,
            "cost_history": cost_history,
            "makespan_history": makespan_history,
            "block_stats": block_stats,
            "num_blocks": num_blocks,
            "runtime": runtime,
            "generations": self.generations,
            "population_size": self.population_size,
            "aos_stats": self.aos.get_stats() if self.use_aos and self.aos else {},
            "seed": self.seed,
        }
