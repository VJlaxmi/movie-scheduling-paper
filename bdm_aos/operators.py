"""
Novel Problem-Specific Genetic Operators for MSSP

Includes standard operators (PMX, OX, swap) and two novel operators:
1. Location-Aware Crossover (LAX)
2. Actor-Continuity Mutation (ACM)
"""

import random
from typing import List, Tuple, Dict, Set
from .data_generator import MSSPInstance


# ===================================================================
#  Standard Operators (for baseline GA and ablation)
# ===================================================================

def pmx_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Partially Mapped Crossover."""
    size = len(p1)
    if size < 3:
        return p1[:], p2[:]
    start, end = sorted(random.sample(range(size), 2))
    c1 = [-1] * size
    c2 = [-1] * size
    c1[start:end] = p1[start:end]
    c2[start:end] = p2[start:end]
    mapping1 = dict(zip(p1[start:end], p2[start:end]))
    mapping2 = dict(zip(p2[start:end], p1[start:end]))
    for i in range(size):
        if c1[i] == -1:
            val = p2[i]
            while val in mapping1:
                val = mapping1[val]
            c1[i] = val
        if c2[i] == -1:
            val = p1[i]
            while val in mapping2:
                val = mapping2[val]
            c2[i] = val
    return c1, c2


def ox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover."""
    size = len(p1)
    if size < 3:
        return p1[:], p2[:]
    start, end = sorted(random.sample(range(size), 2))
    c1 = [None] * size
    c2 = [None] * size
    c1[start:end] = p1[start:end]
    c2[start:end] = p2[start:end]
    used1 = set(c1[start:end])
    used2 = set(c2[start:end])
    idx1, idx2 = end, end
    for i in range(size):
        pos = (end + i) % size
        if p2[pos] not in used1:
            c1[idx1 % size] = p2[pos]
            used1.add(p2[pos])
            idx1 += 1
        if p1[pos] not in used2:
            c2[idx2 % size] = p1[pos]
            used2.add(p1[pos])
            idx2 += 1
    return c1, c2


def swap_mutation(individual: List[int], prob: float = 0.15) -> List[int]:
    """Swap two random positions."""
    result = individual[:]
    if random.random() < prob and len(result) >= 2:
        i, j = random.sample(range(len(result)), 2)
        result[i], result[j] = result[j], result[i]
    return result


def block_swap_mutation(individual: List[int], prob: float = 0.15) -> List[int]:
    """Swap two contiguous sub-blocks."""
    result = individual[:]
    if random.random() < prob and len(result) >= 4:
        size = len(result)
        # Pick block1
        b1_start = random.randint(0, size - 2)
        b1_len = random.randint(1, min(3, size - b1_start))
        # Pick block2 (non-overlapping)
        remaining = [i for i in range(size) if i < b1_start or i >= b1_start + b1_len]
        if remaining:
            b2_start = random.choice(remaining)
            b2_len = min(b1_len, size - b2_start)
            # Extract blocks
            block1 = result[b1_start:b1_start + b1_len]
            block2 = result[b2_start:b2_start + b2_len]
            # Swap (simple: just swap the two positions for now)
            if b1_start < b2_start:
                new = (result[:b1_start] + block2 +
                       result[b1_start + b1_len:b2_start] + block1 +
                       result[b2_start + b2_len:])
            else:
                new = (result[:b2_start] + block1 +
                       result[b2_start + b2_len:b1_start] + block2 +
                       result[b1_start + b1_len:])
            # Validate it's still a valid permutation
            if sorted(new) == sorted(result) and len(new) == len(result):
                result = new
    return result


# ===================================================================
#  Novel Operators
# ===================================================================

def location_aware_crossover(p1: List[int], p2: List[int],
                              block_locations: Dict[int, int],
                              inst: MSSPInstance) -> Tuple[List[int], List[int]]:
    """
    Location-Aware Crossover (LAX) — Novel Operator.

    Preserves sub-sequences where consecutive blocks share the same
    primary location (minimizing transfer costs).

    Strategy:
    1. Identify "location runs" in parent1 (consecutive blocks at same location)
    2. Preserve the longest location run from parent1
    3. Fill remaining positions from parent2 in order
    """
    size = len(p1)
    if size < 3:
        return p1[:], p2[:]

    # Find location runs in parent1
    def get_location_runs(perm: List[int]) -> List[Tuple[int, int]]:
        """Return (start, end) of consecutive same-location blocks."""
        runs = []
        start = 0
        for i in range(1, len(perm)):
            loc_prev = block_locations.get(perm[i - 1], -1)
            loc_curr = block_locations.get(perm[i], -2)
            if loc_prev != loc_curr:
                if i - start >= 2:
                    runs.append((start, i))
                start = i
        if len(perm) - start >= 2:
            runs.append((start, len(perm)))
        return runs

    runs1 = get_location_runs(p1)

    if not runs1:
        # Fall back to OX
        return ox_crossover(p1, p2)

    # Pick the longest run to preserve
    best_run = max(runs1, key=lambda r: r[1] - r[0])
    preserved = p1[best_run[0]:best_run[1]]
    preserved_set = set(preserved)

    # Fill from parent2
    remaining = [b for b in p2 if b not in preserved_set]
    c1 = remaining[:best_run[0]] + preserved + remaining[best_run[0]:]

    # Ensure valid permutation
    if sorted(c1) != sorted(p1) or len(c1) != size:
        return ox_crossover(p1, p2)

    # Second child: swap parents
    runs2 = get_location_runs(p2)
    if runs2:
        best_run2 = max(runs2, key=lambda r: r[1] - r[0])
        preserved2 = p2[best_run2[0]:best_run2[1]]
        preserved_set2 = set(preserved2)
        remaining2 = [b for b in p1 if b not in preserved_set2]
        c2 = remaining2[:best_run2[0]] + preserved2 + remaining2[best_run2[0]:]
        if sorted(c2) != sorted(p2) or len(c2) != size:
            c2 = p2[:]
    else:
        c2 = p2[:]

    return c1, c2


def actor_continuity_mutation(individual: List[int],
                               block_actors: Dict[int, Set[int]],
                               inst: MSSPInstance,
                               prob: float = 0.15) -> List[int]:
    """
    Actor-Continuity Mutation (ACM) — Novel Operator.

    Identifies blocks where the same actor has large idle gaps
    between appearances. Moves blocks closer together to minimize
    actor idle days (reducing wage costs).

    Strategy:
    1. For each actor, find blocks they appear in
    2. Compute "idle gap" = distance between consecutive appearances
    3. Find the block pair with largest gap
    4. Move the later block right after the earlier one
    """
    result = individual[:]
    if random.random() >= prob or len(result) < 3:
        return result

    # Build actor -> block positions mapping
    actor_positions: Dict[int, List[int]] = {}
    for pos, block_id in enumerate(result):
        for actor in block_actors.get(block_id, set()):
            actor_positions.setdefault(actor, []).append(pos)

    # Find largest idle gap
    max_gap = 0
    move_from = -1
    move_to = -1
    for actor, positions in actor_positions.items():
        positions.sort()
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            if gap > max_gap:
                max_gap = gap
                move_from = positions[i + 1]
                move_to = positions[i] + 1  # right after first occurrence

    if max_gap <= 1 or move_from < 0:
        return result

    # Perform the move
    if move_from != move_to and 0 <= move_from < len(result):
        block = result.pop(move_from)
        insert_pos = min(move_to, len(result))
        result.insert(insert_pos, block)

    return result


# ===================================================================
#  Operator dispatcher
# ===================================================================

def apply_crossover(method: str, p1: List[int], p2: List[int],
                    block_locations: Dict[int, int] = None,
                    inst: MSSPInstance = None) -> Tuple[List[int], List[int]]:
    """Dispatch to the appropriate crossover operator."""
    if method == "pmx":
        return pmx_crossover(p1, p2)
    elif method == "ox":
        return ox_crossover(p1, p2)
    elif method == "lax":
        if block_locations and inst:
            return location_aware_crossover(p1, p2, block_locations, inst)
        return ox_crossover(p1, p2)
    else:
        return ox_crossover(p1, p2)


def apply_mutation(method: str, individual: List[int],
                   block_actors: Dict[int, Set[int]] = None,
                   inst: MSSPInstance = None,
                   prob: float = 0.15) -> List[int]:
    """Dispatch to the appropriate mutation operator."""
    if method == "swap":
        return swap_mutation(individual, prob)
    elif method == "block_swap":
        return block_swap_mutation(individual, prob)
    elif method == "acm":
        if block_actors and inst:
            return actor_continuity_mutation(individual, block_actors, inst, prob)
        return swap_mutation(individual, prob)
    else:
        return swap_mutation(individual, prob)
