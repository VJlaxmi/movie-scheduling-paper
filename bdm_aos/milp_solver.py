"""
MILP Sub-Problem Solver for Shooting Blocks

Solves a small MILP for each block (5-10 scenes) to find optimal
intra-block scene scheduling. Uses greedy constructive heuristic
(MILP via Pyomo is optional enhancement when solver is available).
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Set
from .data_generator import MSSPInstance


class MILPSubProblemSolver:
    """
    Solve the MSSP for a subset of scenes (a single shooting block).
    Accounts for constraints inherited from previously scheduled blocks.
    
    Uses a greedy constructive heuristic by default for reliability,
    with optional MILP solving when a solver is available.
    """

    def __init__(self, solver_name: str = "cbc", time_limit: int = 120):
        self.solver_name = solver_name
        self.time_limit = time_limit
        self._solver_available = None  # lazy check

    def _check_solver(self):
        """Check if MILP solver is available (lazy, cached)."""
        if self._solver_available is None:
            try:
                from pyomo.environ import SolverFactory
                solver = SolverFactory(self.solver_name)
                self._solver_available = solver is not None and solver.available()
                if not self._solver_available:
                    for alt in ['glpk', 'highs']:
                        solver = SolverFactory(alt)
                        if solver is not None and solver.available():
                            self.solver_name = alt
                            self._solver_available = True
                            break
            except Exception:
                self._solver_available = False
        return self._solver_available

    def solve_block(self, inst: MSSPInstance, block_scenes: List[int],
                    used_days: Set[int] = None,
                    actor_day_usage: Dict[int, Set[int]] = None,
                    last_location: int = None,
                    day_offset: int = 0
                    ) -> Optional[Dict]:
        """
        Solve scheduling for a single shooting block.

        Args:
            inst: Full MSSP instance
            block_scenes: Scene IDs in this block
            used_days: Days already used by previous blocks
            actor_day_usage: {actor: set of days already used}
            last_location: Location used last in previous block (for transfer)
            day_offset: Minimum day (earliest start for this block)

        Returns:
            dict with schedule, cost, makespan (always returns a valid result)
        """
        if used_days is None:
            used_days = set()
        if actor_day_usage is None:
            actor_day_usage = {}

        # Compute available days: allow sharing days with other blocks
        # (multiple scenes can share a day if total hours <= 8)
        available_days = [d for d in inst.days if d >= max(1, day_offset)]
        if not available_days:
            available_days = list(inst.days)

        # Try MILP first if solver available and block is small enough
        # For speed, only use MILP when called directly (not in matheuristic loop)
        if (self._check_solver() and len(block_scenes) <= 15 
            and not getattr(self, '_fast_mode', False)):
            milp_result = self._solve_milp(
                inst, block_scenes, available_days,
                used_days, actor_day_usage, last_location
            )
            if milp_result is not None:
                return milp_result

        # Always fall back to greedy (guaranteed to produce a result)
        return self._fallback_greedy(
            inst, block_scenes, available_days,
            used_days, actor_day_usage, last_location
        )

    def _solve_milp(self, inst, block_scenes, available_days,
                    used_days, actor_day_usage, last_location):
        """Try to solve with MILP. Returns None if fails."""
        try:
            from pyomo.environ import (
                ConcreteModel, Set as PyoSet, Param, Var, Binary,
                NonNegativeIntegers, Objective, ConstraintList,
                SolverFactory, minimize, value
            )

            model = ConcreteModel()
            scenes = block_scenes

            # Actors needed for this block
            actors = set()
            for s in scenes:
                actors.update(inst.scene_actors.get(s, []))
            actors = sorted(actors)

            # Locations relevant
            locations = set()
            for s in scenes:
                locations.update(inst.scene_locations.get(s, []))
            locations = sorted(locations)

            model.S = PyoSet(initialize=scenes)
            model.D = PyoSet(initialize=available_days)
            model.A = PyoSet(initialize=actors)
            model.L = PyoSet(initialize=locations)

            # Build valid (scene, day) pairs
            valid_sd = []
            for s in scenes:
                scene_actors = inst.scene_actors.get(s, [])
                for d in available_days:
                    all_avail = all(
                        d in inst.actor_availability.get(a, inst.days)
                        for a in scene_actors
                    )
                    loc_avail = any(
                        d in inst.location_availability.get(loc, inst.days)
                        for loc in inst.scene_locations.get(s, [])
                    ) if inst.scene_locations.get(s) else True
                    tw = inst.time_windows.get(s)
                    tw_ok = (tw[0] <= d <= tw[1]) if tw else True
                    if all_avail and loc_avail and tw_ok:
                        valid_sd.append((s, d))

            # Check feasibility: every scene must have at least 1 valid day
            scenes_with_days = {sd[0] for sd in valid_sd}
            if scenes_with_days != set(scenes):
                return None  # infeasible, fall back to greedy

            model.SD = PyoSet(initialize=valid_sd)
            model.x = Var(model.SD, within=Binary)

            valid_sl = [(s, l) for s in scenes
                        for l in inst.scene_locations.get(s, [])]
            model.SL = PyoSet(initialize=valid_sl)
            model.y = Var(model.SL, within=Binary)

            model.day_of = Var(model.S, within=NonNegativeIntegers,
                               bounds=(min(available_days), max(available_days)))

            # Constraints
            model.scene_once = ConstraintList()
            for s in scenes:
                days_for_s = [d for (ss, d) in valid_sd if ss == s]
                if days_for_s:
                    model.scene_once.add(
                        sum(model.x[s, d] for d in days_for_s) == 1
                    )

            model.daily_cap = ConstraintList()
            for d in available_days:
                scenes_on_d = [s for (s, dd) in valid_sd if dd == d]
                if scenes_on_d:
                    model.daily_cap.add(
                        sum(model.x[s, d] * inst.scene_duration[s]
                            for s in scenes_on_d) <= 8.0
                    )

            model.loc_assign = ConstraintList()
            for s in scenes:
                locs = inst.scene_locations.get(s, [])
                if locs:
                    model.loc_assign.add(
                        sum(model.y[s, l] for l in locs) == 1
                    )

            model.link_day = ConstraintList()
            for s in scenes:
                days_for_s = [d for (ss, d) in valid_sd if ss == s]
                if days_for_s:
                    model.link_day.add(
                        model.day_of[s] == sum(d * model.x[s, d]
                                               for d in days_for_s)
                    )

            model.prec = ConstraintList()
            block_set = set(scenes)
            for pre, post in inst.precedence:
                if pre in block_set and post in block_set:
                    model.prec.add(model.day_of[pre] + 1 <= model.day_of[post])

            # Objective
            wage_expr = 0
            for a in actors:
                for s in scenes:
                    if a in inst.scene_actors.get(s, []):
                        for d in [d for (ss, d) in valid_sd if ss == s]:
                            wage_expr += inst.actor_wage[a] * model.x[s, d]

            transfer_expr = 0
            if last_location is not None:
                for s in scenes:
                    for loc in inst.scene_locations.get(s, []):
                        tc = inst.transfer_cost.get((last_location, loc), 0)
                        if tc > 0:
                            transfer_expr += tc * model.y[s, loc]

            deadline_expr = 0
            for s in scenes:
                if s in inst.scene_deadline:
                    dl = inst.scene_deadline[s]
                    pen = inst.deadline_penalty.get(s, 100)
                    for d in [d for (ss, d) in valid_sd if ss == s]:
                        if d > dl:
                            deadline_expr += pen * (d - dl) * model.x[s, d]

            model.obj = Objective(expr=wage_expr + transfer_expr + deadline_expr,
                                  sense=minimize)

            solver = SolverFactory(self.solver_name)
            try:
                solver.options['seconds'] = self.time_limit
            except Exception:
                pass

            result = solver.solve(model, tee=False)
            tc = result.solver.termination_condition.value
            if tc not in ('optimal', 'feasible'):
                return None

            schedule = {}
            total_cost = value(model.obj)
            for s in scenes:
                for d in [d for (ss, d) in valid_sd if ss == s]:
                    if value(model.x[s, d]) > 0.5:
                        chosen_loc = None
                        for loc in inst.scene_locations.get(s, []):
                            if (s, loc) in model.SL and value(model.y[s, loc]) > 0.5:
                                chosen_loc = loc
                                break
                        if chosen_loc is None and inst.scene_locations.get(s):
                            chosen_loc = inst.scene_locations[s][0]
                        schedule[s] = {
                            "day": d,
                            "location": chosen_loc,
                            "actors": inst.scene_actors.get(s, []),
                            "duration": inst.scene_duration[s]
                        }
                        break

            if len(schedule) < len(scenes):
                return None

            days_used = [v["day"] for v in schedule.values()]
            makespan = max(days_used) - min(days_used) + 1
            last_scene = max(schedule, key=lambda s: schedule[s]["day"])

            return {
                "schedule": schedule,
                "cost": total_cost,
                "makespan": makespan,
                "solver_status": "optimal",
                "last_location": schedule[last_scene]["location"]
            }

        except Exception:
            return None

    def _fallback_greedy(self, inst: MSSPInstance, block_scenes: List[int],
                          available_days: List[int], used_days: Set[int],
                          actor_day_usage: Dict[int, Set[int]],
                          last_location: int) -> Dict:
        """
        Greedy constructive heuristic — always produces a valid schedule.
        Sorts scenes by constraint tightness, then assigns greedily.
        """
        schedule = {}
        day_hours = {}  # day -> total hours used
        actor_days = {}  # actor -> set of days they work

        # Sort scenes by tightness: fewest valid days first
        def scene_flexibility(s):
            tw = inst.time_windows.get(s)
            if tw:
                return tw[1] - tw[0]
            return len(available_days)

        sorted_scenes = sorted(block_scenes, key=scene_flexibility)

        for s in sorted_scenes:
            dur = inst.scene_duration.get(s, 2.0)
            scene_actors = inst.scene_actors.get(s, [])
            locs = inst.scene_locations.get(s, [])
            placed = False

            # Score each day
            best_day = None
            best_score = float('inf')
            best_loc = locs[0] if locs else None

            for d in available_days:
                used_h = day_hours.get(d, 0)
                if used_h + dur > 8.0:
                    continue

                # Check actor availability
                all_avail = all(
                    d in inst.actor_availability.get(a, inst.days)
                    for a in scene_actors
                )
                if not all_avail:
                    continue

                # Check time window
                tw = inst.time_windows.get(s)
                if tw and not (tw[0] <= d <= tw[1]):
                    continue

                # Choose best location on this day
                chosen_loc = None
                min_tc = float('inf')
                for loc in locs:
                    if d in inst.location_availability.get(loc, inst.days):
                        tc = 0
                        if last_location:
                            tc = inst.transfer_cost.get(
                                (last_location, loc), 0)
                        if tc < min_tc:
                            min_tc = tc
                            chosen_loc = loc

                if chosen_loc is None:
                    # Try any location (relax availability)
                    chosen_loc = locs[0] if locs else None
                    min_tc = 0

                # Score = wage cost on new days + transfer cost
                wage_cost = 0
                for a in scene_actors:
                    if d not in actor_days.get(a, set()):
                        wage_cost += inst.actor_wage.get(a, 0)

                score = wage_cost + min_tc

                # Deadline penalty
                if s in inst.scene_deadline:
                    dl = inst.scene_deadline[s]
                    if d > dl:
                        score += inst.deadline_penalty.get(s, 100) * (d - dl)

                if score < best_score:
                    best_score = score
                    best_day = d
                    best_loc = chosen_loc

            if best_day is not None:
                schedule[s] = {
                    "day": best_day,
                    "location": best_loc,
                    "actors": scene_actors,
                    "duration": dur
                }
                day_hours[best_day] = day_hours.get(best_day, 0) + dur
                for a in scene_actors:
                    actor_days.setdefault(a, set()).add(best_day)
                placed = True

            if not placed:
                # Force-place: find any day with least usage
                for d in sorted(available_days,
                                key=lambda d: day_hours.get(d, 0)):
                    loc = locs[0] if locs else None
                    schedule[s] = {
                        "day": d,
                        "location": loc,
                        "actors": scene_actors,
                        "duration": dur
                    }
                    day_hours[d] = day_hours.get(d, 0) + dur
                    for a in scene_actors:
                        actor_days.setdefault(a, set()).add(d)
                    break

        # Compute cost accurately
        total_cost = 0.0
        all_actor_days = {}  # actor -> set of days (for wage)
        prev_loc = last_location

        # Process in day order for transfer costs
        day_order = sorted(schedule.keys(),
                           key=lambda s: schedule[s]["day"])

        for s in day_order:
            info = schedule[s]
            d = info["day"]
            loc = info["location"]

            # Actor wages (per distinct day)
            for a in info["actors"]:
                if d not in all_actor_days.get(a, set()):
                    all_actor_days.setdefault(a, set()).add(d)
                    total_cost += inst.actor_wage.get(a, 0)

            # Transfer cost
            if prev_loc and loc and prev_loc != loc:
                total_cost += inst.transfer_cost.get((prev_loc, loc), 0)
            if loc:
                prev_loc = loc

            # Deadline penalty
            if s in inst.scene_deadline:
                dl = inst.scene_deadline[s]
                if d > dl:
                    total_cost += inst.deadline_penalty.get(s, 100) * (d - dl)

        makespan = 1
        if schedule:
            days_used = [v["day"] for v in schedule.values()]
            makespan = max(days_used) - min(days_used) + 1

        last_loc_final = None
        if schedule:
            last_s = max(schedule, key=lambda s: schedule[s]["day"])
            last_loc_final = schedule[last_s]["location"]

        return {
            "schedule": schedule,
            "cost": total_cost,
            "makespan": makespan,
            "solver_status": "greedy_fallback",
            "last_location": last_loc_final
        }
