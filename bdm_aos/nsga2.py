"""
NSGA-II Multi-Objective Optimization Components

Implements non-dominated sorting and crowding distance for
bi-objective optimization (minimize cost, minimize makespan).
"""

import random
import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy


class Solution:
    """Represents a solution in objective space."""

    def __init__(self, chromosome: List[int], objectives: Tuple[float, float] = None):
        self.chromosome = chromosome  # block permutation
        self.objectives = objectives  # (cost, makespan)
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.schedule: Dict = None    # full decoded schedule
        self.stats: Dict = {}

    @property
    def cost(self) -> float:
        return self.objectives[0] if self.objectives else float('inf')

    @property
    def makespan(self) -> float:
        return self.objectives[1] if self.objectives else float('inf')

    def dominates(self, other: "Solution") -> bool:
        """True if self dominates other (all objectives <=, at least one <)."""
        if self.objectives is None or other.objectives is None:
            return False
        all_leq = all(a <= b for a, b in zip(self.objectives, other.objectives))
        any_lt = any(a < b for a, b in zip(self.objectives, other.objectives))
        return all_leq and any_lt


def fast_non_dominated_sort(population: List[Solution]) -> List[List[int]]:
    """
    NSGA-II fast non-dominated sorting.
    Returns list of fronts (each front is list of indices into population).
    """
    n = len(population)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if population[i].dominates(population[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif population[j].dominates(population[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = k + 1
                    next_front.append(j)
        k += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return [f for f in fronts if f]


def crowding_distance(population: List[Solution],
                      front: List[int]) -> None:
    """
    Compute crowding distance for solutions in a front.
    Updates crowding_distance attribute in-place.
    """
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding_distance = float('inf')
        return

    for idx in front:
        population[idx].crowding_distance = 0.0

    num_objectives = 2  # cost, makespan
    for m in range(num_objectives):
        # Sort front by objective m
        sorted_front = sorted(front, key=lambda i: population[i].objectives[m])

        # Boundary points get infinite distance
        population[sorted_front[0]].crowding_distance = float('inf')
        population[sorted_front[-1]].crowding_distance = float('inf')

        # Range of this objective
        obj_range = (population[sorted_front[-1]].objectives[m] -
                     population[sorted_front[0]].objectives[m])
        if obj_range == 0:
            continue

        for k in range(1, len(sorted_front) - 1):
            population[sorted_front[k]].crowding_distance += (
                (population[sorted_front[k + 1]].objectives[m] -
                 population[sorted_front[k - 1]].objectives[m]) / obj_range
            )


def crowded_comparison(sol_a: Solution, sol_b: Solution) -> int:
    """
    NSGA-II crowded comparison operator.
    Returns -1 if a is preferred, 1 if b preferred, 0 if tie.
    """
    if sol_a.rank < sol_b.rank:
        return -1
    elif sol_a.rank > sol_b.rank:
        return 1
    elif sol_a.crowding_distance > sol_b.crowding_distance:
        return -1
    elif sol_a.crowding_distance < sol_b.crowding_distance:
        return 1
    return 0


def nsga2_select(population: List[Solution], num_select: int) -> List[Solution]:
    """
    Select individuals using NSGA-II environmental selection.
    """
    fronts = fast_non_dominated_sort(population)

    selected = []
    for front in fronts:
        crowding_distance(population, front)
        if len(selected) + len(front) <= num_select:
            selected.extend(front)
        else:
            # Sort by crowding distance descending
            remaining = num_select - len(selected)
            front_sorted = sorted(front,
                                  key=lambda i: population[i].crowding_distance,
                                  reverse=True)
            selected.extend(front_sorted[:remaining])
            break

    return [deepcopy(population[i]) for i in selected]


def binary_tournament(population: List[Solution]) -> Solution:
    """Binary tournament selection based on crowded comparison."""
    i, j = random.sample(range(len(population)), 2)
    if crowded_comparison(population[i], population[j]) <= 0:
        return deepcopy(population[i])
    return deepcopy(population[j])


def compute_hypervolume(population: List[Solution],
                        ref_point: Tuple[float, float]) -> float:
    """
    Compute 2D hypervolume indicator.
    ref_point should dominate all solutions (worst case values).
    """
    # Get non-dominated solutions
    non_dom = []
    for sol in population:
        if sol.objectives is None:
            continue
        if sol.objectives[0] < ref_point[0] and sol.objectives[1] < ref_point[1]:
            is_dominated = False
            for other in population:
                if other.objectives is None:
                    continue
                if other is not sol and other.dominates(sol):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dom.append(sol)

    if not non_dom:
        return 0.0

    # Sort by first objective ascending
    non_dom.sort(key=lambda s: s.objectives[0])

    hv = 0.0
    prev_y = ref_point[1]
    for k, sol in enumerate(non_dom):
        if sol.objectives[1] < prev_y:
            # Width: from current x to next point's x (or ref_point[0] for last)
            next_x = non_dom[k + 1].objectives[0] if k + 1 < len(non_dom) else ref_point[0]
            width = next_x - sol.objectives[0]
            height = prev_y - sol.objectives[1]
            hv += width * height
            prev_y = sol.objectives[1]

    return hv


def compute_igd(population: List[Solution],
                reference_front: List[Tuple[float, float]]) -> float:
    """
    Compute Inverted Generational Distance (IGD).
    Measures how well the found front approximates the reference front.
    """
    if not reference_front or not population:
        return float('inf')

    # Get non-dominated solutions
    obj_points = []
    for sol in population:
        if sol.objectives:
            obj_points.append(sol.objectives)

    if not obj_points:
        return float('inf')

    obj_array = np.array(obj_points)
    ref_array = np.array(reference_front)

    # For each reference point, find minimum distance to obtained front
    distances = []
    for ref in ref_array:
        dists = np.sqrt(np.sum((obj_array - ref) ** 2, axis=1))
        distances.append(np.min(dists))

    return np.mean(distances)
