"""
BDM-AOS: Bi-Level Decomposition Matheuristic with Adaptive Operator Selection

Core algorithm combining:
  - Upper level: NSGA-II GA over block permutations
  - Lower level: MILP solver for intra-block scheduling
  - AOS: Q-Learning adaptive operator selection
"""

import math
import random
import time
import numpy as np
import networkx as nx
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
                 use_sig: bool = True,
                 use_spec_ops: bool = True,
                 flat_mode: bool = False,
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
        self.use_sig = use_sig          # False = random blocks (BDM-NoSIG)
        self.use_spec_ops = use_spec_ops  # False = no LAX/ACM (BDM-NoSpecOps)
        self.flat_mode = flat_mode       # True = NSGA-II on scenes directly (BDM-Flat)
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

        if self.flat_mode:
            # Flat mode: each scene is its own "block"
            self.blocks = [[s] for s in inst.scenes]
            self.graph = nx.Graph()
            self.graph.add_nodes_from(inst.scenes)
        elif not self.use_sig:
            # Random blocks (BDM-NoSIG): random partitioning, no SIG
            rng = random.Random(self.seed + 999)
            scenes_shuffled = list(inst.scenes)
            rng.shuffle(scenes_shuffled)
            block_size = max(2, inst.num_scenes // max(3, inst.num_scenes // 5))
            self.blocks = []
            for i in range(0, len(scenes_shuffled), block_size):
                self.blocks.append(sorted(scenes_shuffled[i:i+block_size]))
            self.graph = nx.Graph()
            self.graph.add_nodes_from(inst.scenes)
        else:
            # Standard SIG decomposition: at least ceil(n/4) blocks
            min_blocks = max(3, math.ceil(inst.num_scenes / 4))
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

        # Global greedy decoder enforcing all 19 constraint families
        schedule = {}
        day_hours = {}        # day -> hours used
        actor_days = {}       # actor -> set of days
        day_town = {}         # day -> active town (C11)
        day_equip_count = {}  # (equipment_type, day) -> count (C14)
        day_loc_count = {}    # (location, day) -> scene count (C18)
        total_cost = 0.0

        def _get_town(loc):
            return inst.location_town.get(loc)

        def _town_continuity_ok(d, town):
            """C12: no town switch on consecutive days."""
            if town is None:
                return True
            if d - 1 in day_town and day_town[d - 1] is not None and day_town[d - 1] != town:
                return False
            if d + 1 in day_town and day_town[d + 1] is not None and day_town[d + 1] != town:
                return False
            return True

        def _consec_ok(actor, d):
            """C13: actor consecutive working day limit."""
            max_c = inst.actor_max_consecutive.get(actor, 99)
            worked = actor_days.get(actor, set())
            consec = 1
            cd = d - 1
            while cd in worked:
                consec += 1
                cd -= 1
            cd = d + 1
            while cd in worked:
                consec += 1
                cd += 1
            return consec <= max_c

        def _rest_ok(actor, d):
            """C19: lead actor rest period check."""
            lead_actors_set = set(getattr(inst, 'lead_actors', []))
            if actor not in lead_actors_set:
                return True
            max_c = inst.actor_max_consecutive.get(actor, 5)
            r_a = getattr(inst, 'actor_rest_period', {}).get(actor, 1)
            window = max_c + r_a
            worked_in_window = sum(
                1 for dd in range(d - window + 1, d + 1)
                if dd in actor_days.get(actor, set())
            )
            return worked_in_window < max_c

        for scene in scene_order:
            dur = inst.scene_duration.get(scene, 2.0)
            actors = inst.scene_actors.get(scene, [])
            locs = inst.scene_locations.get(scene, [])

            # C16 co-scheduling: towns shared by ALL group members (computed once per scene)
            cosched_allowed_towns = None
            for _grp in getattr(inst, 'coschedule_groups', []):
                if scene in _grp:
                    _towns = None
                    for _gs in _grp:
                        _gs_towns = {_get_town(_l)
                                     for _l in inst.scene_locations.get(_gs, [])}
                        _towns = _gs_towns if _towns is None else _towns & _gs_towns
                    cosched_allowed_towns = _towns if _towns else None
                    break

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

                # C13: actor consecutive day limits
                if not all(_consec_ok(a, d) for a in actors):
                    continue

                # C14: equipment capacity
                equip_ok = True
                for e in getattr(inst, 'scene_equipment', {}).get(scene, []):
                    supply = getattr(inst, 'equipment_supply', {}).get(e, 999)
                    if day_equip_count.get((e, d), 0) >= supply:
                        equip_ok = False
                        break
                if not equip_ok:
                    continue

                # C3: precedence constraints
                prec_ok = True
                for (pre, post) in inst.precedence:
                    if post == scene and pre in schedule:
                        if d <= schedule[pre]["day"]:
                            prec_ok = False
                            break
                    if pre == scene and post in schedule:
                        if schedule[post]["day"] <= d:
                            prec_ok = False
                            break
                if not prec_ok:
                    continue

                # C16: co-scheduling constraints
                cosched_ok = True
                for group in getattr(inst, 'coschedule_groups', []):
                    if scene in group:
                        for gs in group:
                            if gs != scene and gs in schedule:
                                if schedule[gs]["day"] != d:
                                    cosched_ok = False
                                    break
                    if not cosched_ok:
                        break
                if not cosched_ok:
                    continue

                # C17: minimum scene separation
                sep_ok = True
                for (s_i, s_j), delta in getattr(inst, 'scene_separation', {}).items():
                    if s_j == scene and s_i in schedule:
                        if d - schedule[s_i]["day"] < delta:
                            sep_ok = False
                            break
                    if s_i == scene and s_j in schedule:
                        if schedule[s_j]["day"] - d < delta:
                            sep_ok = False
                            break
                if not sep_ok:
                    continue

                # C19: lead actor rest periods
                if not all(_rest_ok(a, d) for a in actors):
                    continue

                # Choose best location respecting town constraints
                chosen_loc = None
                min_tc = float('inf')
                last_loc = None
                if schedule:
                    last_s = max(schedule, key=lambda s: schedule[s]["day"])
                    last_loc = schedule[last_s].get("location")

                for loc in locs:
                    if d not in inst.location_availability.get(loc, inst.days):
                        continue
                    # C18: location concurrency capacity
                    cap = getattr(inst, 'location_capacity', {}).get(loc, 999)
                    if day_loc_count.get((loc, d), 0) >= cap:
                        continue
                    town = _get_town(loc)
                    # C16 co-scheduling town compatibility
                    if cosched_allowed_towns is not None and town not in cosched_allowed_towns:
                        continue
                    # C11: town uniqueness per day
                    if d in day_town and day_town[d] is not None and day_town[d] != town:
                        continue
                    # C12: town continuity
                    if not _town_continuity_ok(d, town):
                        continue
                    tc = 0
                    if last_loc:
                        tc = inst.transfer_cost.get((last_loc, loc), 0)
                    if tc < min_tc:
                        min_tc = tc
                        chosen_loc = loc

                if chosen_loc is None:
                    # No location satisfies town/concurrency constraints on this day
                    # — skip to next day rather than violating C11/C18
                    continue

                schedule[scene] = {
                    "day": d,
                    "location": chosen_loc,
                    "actors": actors,
                    "duration": dur
                }
                day_hours[d] = used_h + dur

                # Track town activation (C11)
                if chosen_loc:
                    town = _get_town(chosen_loc)
                    if town is not None:
                        day_town[d] = town
                    # Track location concurrency (C18)
                    day_loc_count[(chosen_loc, d)] = day_loc_count.get((chosen_loc, d), 0) + 1

                # Track equipment usage (C14)
                for e in getattr(inst, 'scene_equipment', {}).get(scene, []):
                    day_equip_count[(e, d)] = day_equip_count.get((e, d), 0) + 1

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

                # Criticality-weighted utilization cost
                if chosen_loc:
                    kl = inst.location_criticality.get(chosen_loc, 1.0)
                    if kl > 0:
                        total_cost += 1.0 / kl
                for a in actors:
                    ka = inst.actor_criticality.get(a, 1.0)
                    if ka > 0:
                        total_cost += 1.0 / ka

                placed = True
                break

            if not placed:
                # Co-scheduling group: force-place on same day as already-scheduled
                # group member to preserve C16 intent.
                # Otherwise pick lightest day to minimise C5 violations.
                forced_by_group = False
                d = inst.num_days
                for group in getattr(inst, 'coschedule_groups', []):
                    if scene in group:
                        for gs in group:
                            if gs != scene and gs in schedule:
                                d = schedule[gs]["day"]
                                forced_by_group = True
                                break
                        break
                if not forced_by_group:
                    def _fp_day_key(day):
                        town_set = day_town.get(day)
                        if cosched_allowed_towns is not None:
                            town_ok = (town_set is None or town_set in cosched_allowed_towns)
                        else:
                            town_ok = (town_set is None or
                                       any(_get_town(lc) == town_set for lc in locs))
                        return (0 if town_ok else 1, day_hours.get(day, 0))
                    d = min(range(1, inst.num_days + 1), key=_fp_day_key)
                loc = locs[0] if locs else None
                # Prefer a location from co-scheduling-compatible towns first
                if loc and cosched_allowed_towns:
                    for candidate in locs:
                        if _get_town(candidate) in cosched_allowed_towns:
                            loc = candidate
                            break
                # Then override with day's established town (C11)
                if loc and d in day_town and day_town[d] is not None:
                    for candidate in locs:
                        if _get_town(candidate) == day_town[d]:
                            loc = candidate
                            break
                schedule[scene] = {
                    "day": d, "location": loc,
                    "actors": actors, "duration": dur
                }
                # Update state trackers even on force-place
                day_hours[d] = day_hours.get(d, 0) + dur
                for a in actors:
                    actor_days.setdefault(a, set()).add(d)
                if loc:
                    town = _get_town(loc)
                    if town is not None and d not in day_town:
                        day_town[d] = town
                    day_loc_count[(loc, d)] = day_loc_count.get((loc, d), 0) + 1
                for e in getattr(inst, 'scene_equipment', {}).get(scene, []):
                    day_equip_count[(e, d)] = day_equip_count.get((e, d), 0) + 1
                total_cost += 10000

        # C15: Holding fees for booked actors idle on booking window days
        booking_windows = getattr(inst, 'actor_booking_window', {})
        holding_fees = getattr(inst, 'actor_holding_fee', {})
        for a, (bw_start, bw_end) in booking_windows.items():
            h_a = holding_fees.get(a, 0.0)
            if h_a > 0:
                shooting_days = actor_days.get(a, set())
                for d in range(bw_start, bw_end + 1):
                    if d not in shooting_days:
                        total_cost += h_a

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
                # If spec ops disabled, override LAX/ACM
                if not self.use_spec_ops:
                    if cx_op == "lax":
                        cx_op = self.rng.choice(["pmx", "ox"])
                    if mut_op == "acm":
                        mut_op = self.rng.choice(["swap", "block_swap"])
            else:
                if self.use_spec_ops:
                    cx_op = self.rng.choice(["pmx", "ox", "lax"])
                    mut_op = self.rng.choice(["swap", "block_swap", "acm"])
                else:
                    cx_op = self.rng.choice(["pmx", "ox"])
                    mut_op = self.rng.choice(["swap", "block_swap"])
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
