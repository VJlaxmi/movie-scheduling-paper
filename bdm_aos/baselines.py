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
    Enforces all 19 constraint families including town uniqueness (C11),
    town continuity (C12), actor consecutive day limits (C13), equipment
    capacity (C14), booking/holding fees (C15), co-scheduling (C16),
    scene separation (C17), location concurrency (C18), and lead-actor
    rest periods (C19).
    Returns (cost, makespan, schedule).
    """
    schedule = {}
    day_hours = {}        # day -> hours used
    actor_days = {}       # actor -> set of days
    day_town = {}         # day -> active town (C11)
    day_equip_count = {}  # (equipment_type, day) -> count (C14)
    day_loc_count = {}    # (location, day) -> count (C18)
    current_day = 1
    total_cost = 0.0

    def _get_town(loc):
        return inst.location_town.get(loc) if hasattr(inst, 'location_town') else None

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
        max_c = inst.actor_max_consecutive.get(actor, 99) if hasattr(inst, 'actor_max_consecutive') else 99
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

    for scene in perm:
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
        for d in range(current_day, inst.num_days + 1):
            used_h = day_hours.get(d, 0)
            if used_h + dur > 8.0:
                continue

            # C6: Check actor availability
            all_avail = all(
                d in inst.actor_availability.get(a, inst.days)
                for a in actors
            )
            if not all_avail:
                continue

            # C4: Check time window
            tw = inst.time_windows.get(scene)
            if tw and not (tw[0] <= d <= tw[1]):
                continue

            # C13: Check actor consecutive working day limits
            if not all(_consec_ok(a, d) for a in actors):
                continue

            # C14: Check equipment availability
            equip_ok = True
            for e in getattr(inst, 'scene_equipment', {}).get(scene, []):
                supply = getattr(inst, 'equipment_supply', {}).get(e, 999)
                if day_equip_count.get((e, d), 0) >= supply:
                    equip_ok = False
                    break
            if not equip_ok:
                continue

            # C3: Precedence constraints
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

            # C16: Co-scheduling constraints (group members must share a day)
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

            # C17: Check minimum scene separation
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

            # C19: Lead actor rest period check
            rest_ok = True
            lead_actors = getattr(inst, 'lead_actors', [])
            rest_periods = getattr(inst, 'actor_rest_period', {})
            for a in actors:
                if a in lead_actors:
                    max_c = inst.actor_max_consecutive.get(a, 5)
                    r_a = rest_periods.get(a, 1)
                    window = max_c + r_a
                    worked_in_window = sum(
                        1 for dd in range(d - window + 1, d + 1)
                        if dd in actor_days.get(a, set())
                    )
                    if worked_in_window >= max_c:
                        rest_ok = False
                        break
            if not rest_ok:
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

            # Costs: actor wages
            for a in actors:
                if a not in actor_days:
                    actor_days[a] = set()
                if d not in actor_days[a]:
                    actor_days[a].add(d)
                    total_cost += inst.actor_wage.get(a, 0)

            if last_loc and chosen_loc and last_loc != chosen_loc:
                total_cost += inst.transfer_cost.get((last_loc, chosen_loc), 0)

            # Deadline penalty
            if scene in inst.scene_deadline:
                dl = inst.scene_deadline[scene]
                if d > dl:
                    total_cost += inst.deadline_penalty.get(scene, 100) * (d - dl)

            # Criticality-weighted utilization
            if chosen_loc:
                kl = getattr(inst, 'location_criticality', {}).get(chosen_loc, 1.0)
                if kl > 0:
                    total_cost += 1.0 / kl
            for a in actors:
                ka = getattr(inst, 'actor_criticality', {}).get(a, 1.0)
                if ka > 0:
                    total_cost += 1.0 / ka

            placed = True
            break

        if not placed:
            # Force-place: update state trackers to maintain consistency
            # If scene is in a co-scheduling group, force-place on the same day
            # as any already-scheduled group member (preserves C16 intent).
            # Otherwise, pick the lightest day to minimise C5 violations.
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
            total_cost += 10000  # Large penalty for infeasible

    # C15: Add holding fees for booked actors idle on booking window days
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

    return total_cost, float(makespan), schedule


# ===================================================================
#  Standalone GA
# ===================================================================

class StandaloneGA:
    """Standard GA for MSSP (single-objective: minimize cost)."""

    def __init__(self, population_size: int = 50, generations: int = 100,
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

    def __init__(self, swarm_size: int = 30, iterations: int = 100,
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
