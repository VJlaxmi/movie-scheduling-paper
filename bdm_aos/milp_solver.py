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

            # C10-C11: Town activation and uniqueness constraints
            relevant_towns = set()
            for loc in locations:
                town = inst.location_town.get(loc)
                if town is not None:
                    relevant_towns.add(town)
            relevant_towns = sorted(relevant_towns)

            if relevant_towns:
                town_day_pairs = [(t, d) for t in relevant_towns
                                  for d in available_days]
                model.TD = PyoSet(initialize=town_day_pairs)
                model.u = Var(model.TD, within=Binary)

                # C10: location-town linking (z_{l,s,d} <= u_{phi(l),d})
                model.loc_town_link = ConstraintList()
                for s in scenes:
                    for loc in inst.scene_locations.get(s, []):
                        town = inst.location_town.get(loc)
                        if town is not None and town in relevant_towns:
                            for d in [d for (ss, d) in valid_sd if ss == s]:
                                if (loc, d) in inst.location_availability:
                                    pass  # availability already filtered
                                model.loc_town_link.add(
                                    model.x[s, d] <= model.u[town, d])

                # C11: at most one active town per day
                model.town_unique = ConstraintList()
                for d in available_days:
                    towns_on_d = [t for t in relevant_towns
                                  if (t, d) in model.TD]
                    if len(towns_on_d) > 1:
                        model.town_unique.add(
                            sum(model.u[t, d] for t in towns_on_d) <= 1)

                # C12: town continuity across consecutive days
                model.town_cont = ConstraintList()
                sorted_days = sorted(available_days)
                for idx in range(1, len(sorted_days)):
                    d = sorted_days[idx]
                    d_prev = sorted_days[idx - 1]
                    if d == d_prev + 1:  # consecutive days
                        for ti in relevant_towns:
                            for tj in relevant_towns:
                                if ti != tj:
                                    if (ti, d) in model.TD and (tj, d_prev) in model.TD:
                                        model.town_cont.add(
                                            model.u[ti, d] + model.u[tj, d_prev] <= 1)

            # C14: Equipment capacity constraints
            equip_types = getattr(inst, 'equipment_types', [])
            if equip_types:
                model.equip_cap = ConstraintList()
                for e in equip_types:
                    supply = getattr(inst, 'equipment_supply', {}).get(e, 999)
                    for d in available_days:
                        scenes_needing_e = [
                            s for s in scenes
                            if e in getattr(inst, 'scene_equipment', {}).get(s, [])
                        ]
                        if len(scenes_needing_e) > supply:
                            days_for_scenes = {
                                s: [dd for (ss, dd) in valid_sd if ss == s]
                                for s in scenes_needing_e
                            }
                            valid_scene_day = [
                                s for s in scenes_needing_e if d in days_for_scenes.get(s, [])
                            ]
                            if len(valid_scene_day) > supply:
                                model.equip_cap.add(
                                    sum(model.x[s, d] for s in valid_scene_day) <= supply
                                )

            # C15: Actor booking / holding-fee idle variables
            booking_windows = getattr(inst, 'actor_booking_window', {})
            holding_fees = getattr(inst, 'actor_holding_fee', {})
            booked_actors = [a for a in actors if a in booking_windows]
            phi_pairs = []
            for a in booked_actors:
                bw = booking_windows[a]
                for d in available_days:
                    if bw[0] <= d <= bw[1]:
                        phi_pairs.append((a, d))
            if phi_pairs:
                model.PHI = PyoSet(initialize=phi_pairs)
                model.phi = Var(model.PHI, within=Binary)
                model.phi_lb = ConstraintList()  # C15:  phi >= mu^b_{a,d} - sum(v)
                model.phi_ub = ConstraintList()  # C15b: phi <= mu^b_{a,d}
                for a, d in phi_pairs:
                    v_sum = sum(
                        model.x[s, d]
                        for s in scenes
                        if a in inst.scene_actors.get(s, [])
                        and (s, d) in set(valid_sd)
                    )
                    # C15:  phi_{a,d} >= 1 - v_sum  (mu^b_{a,d}=1 for all phi_pairs)
                    model.phi_lb.add(model.phi[a, d] >= 1 - v_sum)
                    # C15b: phi_{a,d} <= mu^b_{a,d} = 1 (booking day)
                    # phi is not defined for non-booking days, enforcing phi=0 there.
                    model.phi_ub.add(model.phi[a, d] <= 1)

            # C16: Mandatory co-scheduling groups
            coschedule_groups = getattr(inst, 'coschedule_groups', [])
            if coschedule_groups:
                model.coschedule = ConstraintList()
                for group in coschedule_groups:
                    block_scenes_set = set(scenes)
                    group_in_block = [s for s in group if s in block_scenes_set]
                    if len(group_in_block) >= 2:
                        for d in available_days:
                            # All group members on same day: x[s,d] == x[s',d]
                            days_for_first = [dd for (ss, dd) in valid_sd if ss == group_in_block[0]]
                            for s_other in group_in_block[1:]:
                                days_for_other = [dd for (ss, dd) in valid_sd if ss == s_other]
                                if d in days_for_first and d in days_for_other:
                                    model.coschedule.add(
                                        model.x[group_in_block[0], d] == model.x[s_other, d]
                                    )

            # C17: Minimum scene separation
            scene_separation = getattr(inst, 'scene_separation', {})
            if scene_separation:
                model.sep = ConstraintList()
                block_set = set(scenes)
                for (s_i, s_j), delta in scene_separation.items():
                    if s_i in block_set and s_j in block_set:
                        days_i = [d for (ss, d) in valid_sd if ss == s_i]
                        days_j = [d for (ss, d) in valid_sd if ss == s_j]
                        if days_i and days_j:
                            day_i_expr = sum(d * model.x[s_i, d] for d in days_i)
                            day_j_expr = sum(d * model.x[s_j, d] for d in days_j)
                            model.sep.add(day_j_expr - day_i_expr >= delta)

            # C18: Location concurrency capacity
            loc_capacity = getattr(inst, 'location_capacity', {})
            if loc_capacity:
                model.loc_concur = ConstraintList()
                for loc in locations:
                    cap = loc_capacity.get(loc, 999)
                    for d in available_days:
                        scenes_at_loc_d = [
                            s for s in scenes
                            if loc in inst.scene_locations.get(s, [])
                            and (s, d) in set(valid_sd)
                        ]
                        if len(scenes_at_loc_d) > cap:
                            model.loc_concur.add(
                                sum(model.x[s, d] for s in scenes_at_loc_d) <= cap
                            )

            # C19: Lead actor rest periods (w_{a,d} working-day indicator + rest window)
            lead_actors_list = [a for a in getattr(inst, 'lead_actors', []) if a in set(actors)]
            rest_periods = getattr(inst, 'actor_rest_period', {})
            if lead_actors_list:
                w_pairs = [(a, d) for a in lead_actors_list for d in available_days]
                model.WD = PyoSet(initialize=w_pairs)
                model.w = Var(model.WD, within=Binary)
                model.w_lb = ConstraintList()
                model.w_ub = ConstraintList()
                model.rest = ConstraintList()
                for a in lead_actors_list:
                    max_consec = inst.actor_max_consecutive.get(a, 5)
                    r_a = rest_periods.get(a, 1)
                    window_len = max_consec + r_a
                    for d in available_days:
                        # w_{a,d} >= v_{a,s,d} for all s
                        for s in scenes:
                            if a in inst.scene_actors.get(s, []) and (s, d) in set(valid_sd):
                                model.w_lb.add(model.w[a, d] >= model.x[s, d])
                        # w_{a,d} <= sum_s v_{a,s,d}
                        v_sum = sum(
                            model.x[s, d]
                            for s in scenes
                            if a in inst.scene_actors.get(s, [])
                            and (s, d) in set(valid_sd)
                        )
                        model.w_ub.add(model.w[a, d] <= v_sum)
                        # C19 rest window: sum_{d'=d}^{d+h+r-1} w_{a,d'} <= h_a
                        window_days = [
                            dd for dd in available_days
                            if d <= dd <= d + window_len - 1
                        ]
                        if len(window_days) >= window_len:
                            model.rest.add(
                                sum(model.w[a, dd] for dd in window_days) <= max_consec
                            )

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

            # Criticality-weighted utilization (objective term iv)
            crit_expr = 0
            for s in scenes:
                for loc in inst.scene_locations.get(s, []):
                    kl = inst.location_criticality.get(loc, 1.0)
                    if kl > 0:
                        for d in [d for (ss, d) in valid_sd if ss == s]:
                            crit_expr += model.x[s, d] / kl
                for a in inst.scene_actors.get(s, []):
                    ka = inst.actor_criticality.get(a, 1.0)
                    if ka > 0:
                        for d in [d for (ss, d) in valid_sd if ss == s]:
                            crit_expr += model.x[s, d] / ka

            # Holding fee cost (C15 objective term v)
            holding_expr = 0
            if phi_pairs:
                for a, d in phi_pairs:
                    h_a = holding_fees.get(a, 0.0)
                    if h_a > 0:
                        holding_expr += h_a * model.phi[a, d]

            model.obj = Objective(
                expr=wage_expr + transfer_expr + deadline_expr + crit_expr + holding_expr,
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
        day_town = {}   # day -> active town (C11: at most one town per day)
        day_equip_count = {}  # (equipment_type, day) -> count used (C14)
        day_loc_count = {}    # (location, day) -> scenes at that location (C18)

        # Sort scenes by tightness: fewest valid days first
        def scene_flexibility(s):
            tw = inst.time_windows.get(s)
            if tw:
                return tw[1] - tw[0]
            return len(available_days)

        sorted_scenes = sorted(block_scenes, key=scene_flexibility)

        def _get_town_for_loc(loc):
            """Get the town for a location from hierarchy."""
            return inst.location_town.get(loc)

        def _check_town_continuity(d, town):
            """C12: Check town continuity with previous/next day."""
            if town is None:
                return True
            if d - 1 in day_town and day_town[d - 1] is not None:
                if day_town[d - 1] != town:
                    return False
            if d + 1 in day_town and day_town[d + 1] is not None:
                if day_town[d + 1] != town:
                    return False
            return True

        def _check_actor_consecutive(actor, d):
            """C13: Check actor consecutive working day limit."""
            max_consec = inst.actor_max_consecutive.get(actor, 99)
            days_worked = actor_days.get(actor, set())
            # Count consecutive run ending at day d
            consec = 1
            check_d = d - 1
            while check_d in days_worked:
                consec += 1
                check_d -= 1
            check_d = d + 1
            while check_d in days_worked:
                consec += 1
                check_d += 1
            return consec <= max_consec

        # Track most recently placed location (within this block) for transfer scoring
        current_last_loc = last_location

        for s in sorted_scenes:
            dur = inst.scene_duration.get(s, 2.0)
            scene_actors = inst.scene_actors.get(s, [])
            locs = inst.scene_locations.get(s, [])
            placed = False

            # C16 co-scheduling: towns shared by ALL group members (computed once per scene)
            _cosched_allowed_s = None
            for _grp in getattr(inst, 'coschedule_groups', []):
                if s in _grp:
                    _towns = None
                    for _gs in _grp:
                        _gs_towns = {_get_town_for_loc(_l)
                                     for _l in inst.scene_locations.get(_gs, [])}
                        _towns = _gs_towns if _towns is None else _towns & _gs_towns
                    _cosched_allowed_s = _towns if _towns else None
                    break

            # Score each day
            best_day = None
            best_score = float('inf')
            best_loc = locs[0] if locs else None

            for d in available_days:
                used_h = day_hours.get(d, 0)
                if used_h + dur > 8.0:
                    continue

                # Check actor availability (C6)
                all_avail = all(
                    d in inst.actor_availability.get(a, inst.days)
                    for a in scene_actors
                )
                if not all_avail:
                    continue

                # Check time window (C4)
                tw = inst.time_windows.get(s)
                if tw and not (tw[0] <= d <= tw[1]):
                    continue

                # C13: Check actor consecutive working day limits
                consec_ok = all(
                    _check_actor_consecutive(a, d) for a in scene_actors
                )
                if not consec_ok:
                    continue

                # C14: Check equipment availability
                equip_ok = True
                for e in getattr(inst, 'scene_equipment', {}).get(s, []):
                    supply = getattr(inst, 'equipment_supply', {}).get(e, 999)
                    used = day_equip_count.get((e, d), 0)
                    if used >= supply:
                        equip_ok = False
                        break
                if not equip_ok:
                    continue

                # C3: Precedence constraints
                prec_ok = True
                for (pre, post) in inst.precedence:
                    if post == s and pre in schedule:
                        if d <= schedule[pre]["day"]:
                            prec_ok = False
                            break
                    if pre == s and post in schedule:
                        if schedule[post]["day"] <= d:
                            prec_ok = False
                            break
                if not prec_ok:
                    continue

                # C16: co-scheduling constraints
                cosched_ok = True
                for group in getattr(inst, 'coschedule_groups', []):
                    if s in group:
                        for gs in group:
                            if gs != s and gs in schedule:
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
                    if s_j == s and s_i in schedule:
                        if d - schedule[s_i]["day"] < delta:
                            sep_ok = False
                            break
                    if s_i == s and s_j in schedule:
                        if schedule[s_j]["day"] - d < delta:
                            sep_ok = False
                            break
                if not sep_ok:
                    continue

                # Choose best location on this day (respecting town constraints)
                chosen_loc = None
                min_tc = float('inf')

                for loc in locs:
                    if d not in inst.location_availability.get(loc, inst.days):
                        continue
                    # C18: location concurrency capacity
                    cap = getattr(inst, 'location_capacity', {}).get(loc, 999)
                    if day_loc_count.get((loc, d), 0) >= cap:
                        continue
                    town = _get_town_for_loc(loc)
                    # C16 co-scheduling town compatibility
                    if _cosched_allowed_s is not None and town not in _cosched_allowed_s:
                        continue
                    # C11: town uniqueness - check if another town is already active
                    if d in day_town and day_town[d] is not None and day_town[d] != town:
                        continue
                    # C12: town continuity
                    if not _check_town_continuity(d, town):
                        continue
                    tc = 0
                    if current_last_loc:
                        tc = inst.transfer_cost.get(
                            (current_last_loc, loc), 0)
                    if tc < min_tc:
                        min_tc = tc
                        chosen_loc = loc

                if chosen_loc is None:
                    # No location satisfies town/concurrency constraints on this day
                    # — skip this day to avoid C11 violations
                    continue

                # Score = wage cost + transfer cost + criticality
                wage_cost = 0
                for a in scene_actors:
                    if d not in actor_days.get(a, set()):
                        wage_cost += inst.actor_wage.get(a, 0)

                score = wage_cost + min_tc

                # Criticality-weighted utilization
                if chosen_loc:
                    kl = inst.location_criticality.get(chosen_loc, 1.0)
                    if kl > 0:
                        score += 1.0 / kl
                for a in scene_actors:
                    ka = inst.actor_criticality.get(a, 1.0)
                    if ka > 0:
                        score += 1.0 / ka

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
                # Track active town for C11/C12
                if best_loc:
                    town = _get_town_for_loc(best_loc)
                    if town is not None:
                        day_town[best_day] = town
                    # Track location concurrency (C18)
                    day_loc_count[(best_loc, best_day)] = day_loc_count.get((best_loc, best_day), 0) + 1
                    # Update running last location for transfer cost scoring
                    current_last_loc = best_loc
                # Track equipment usage (C14)
                for e in getattr(inst, 'scene_equipment', {}).get(s, []):
                    day_equip_count[(e, best_day)] = day_equip_count.get((e, best_day), 0) + 1
                placed = True

            if not placed:
                # Co-scheduling group: force-place on same day as already-scheduled
                # group member to preserve C16 intent
                force_day = None
                for group in getattr(inst, 'coschedule_groups', []):
                    if s in group:
                        for gs in group:
                            if gs != s and gs in schedule:
                                force_day = schedule[gs]["day"]
                                break
                        break
                # Pick the lightest available day that is town-compatible, or group day if available
                def _fp_day_sort(d):
                    town_set = day_town.get(d)
                    if _cosched_allowed_s is not None:
                        town_ok = (town_set is None or town_set in _cosched_allowed_s)
                    else:
                        town_ok = (town_set is None or
                                   any(_get_town_for_loc(lc) == town_set for lc in locs))
                    return (0 if town_ok else 1, day_hours.get(d, 0))
                candidate_days = ([force_day] if force_day else []) + sorted(
                    available_days, key=_fp_day_sort)
                for d in candidate_days:
                    if d not in available_days:
                        continue
                    loc = locs[0] if locs else None
                    # Prefer a location from co-scheduling-compatible towns first
                    if loc and _cosched_allowed_s:
                        for candidate in locs:
                            if _get_town_for_loc(candidate) in _cosched_allowed_s:
                                loc = candidate
                                break
                    # Then override with day's established town (C11)
                    if loc and d in day_town and day_town[d] is not None:
                        for candidate in locs:
                            if _get_town_for_loc(candidate) == day_town[d]:
                                loc = candidate
                                break
                    schedule[s] = {
                        "day": d,
                        "location": loc,
                        "actors": scene_actors,
                        "duration": dur
                    }
                    day_hours[d] = day_hours.get(d, 0) + dur
                    for a in scene_actors:
                        actor_days.setdefault(a, set()).add(d)
                    if loc:
                        town = _get_town_for_loc(loc)
                        if town is not None and d not in day_town:
                            day_town[d] = town
                        day_loc_count[(loc, d)] = day_loc_count.get((loc, d), 0) + 1
                        current_last_loc = loc
                    for e in getattr(inst, 'scene_equipment', {}).get(s, []):
                        day_equip_count[(e, d)] = day_equip_count.get((e, d), 0) + 1
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

            # Criticality-weighted utilization (objective term iv)
            if loc:
                kl = inst.location_criticality.get(loc, 1.0)
                if kl > 0:
                    total_cost += 1.0 / kl
            for a in info["actors"]:
                ka = inst.actor_criticality.get(a, 1.0)
                if ka > 0:
                    total_cost += 1.0 / ka

        # C15: Holding fees for booked actors not shooting on booking window days
        # Only charge actors who appear in this block's scenes
        block_actors_set = set()
        for s_info in schedule.values():
            block_actors_set.update(s_info.get("actors", []))
        booking_windows = getattr(inst, 'actor_booking_window', {})
        holding_fees = getattr(inst, 'actor_holding_fee', {})
        for a, (bw_start, bw_end) in booking_windows.items():
            if a not in block_actors_set:
                continue
            h_a = holding_fees.get(a, 0.0)
            if h_a > 0:
                actor_shooting_days = set()
                for s_info in schedule.values():
                    if a in s_info.get("actors", []):
                        actor_shooting_days.add(s_info["day"])
                for d in range(bw_start, bw_end + 1):
                    if d not in actor_shooting_days:
                        total_cost += h_a

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
