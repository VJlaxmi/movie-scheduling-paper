"""
Scene Interaction Graph (SIG) and Louvain Decomposition

Novel contribution: Graph-theoretic representation of scene coupling
based on shared actors, shared locations, and transfer costs.
Louvain community detection partitions scenes into shooting blocks.
"""

import networkx as nx
import community as community_louvain
import numpy as np
from typing import Dict, List, Tuple
from .data_generator import MSSPInstance


class SceneInteractionGraph:
    """
    Build a weighted undirected graph where:
      - Nodes = scenes
      - Edge weight = coupling strength w(si, sj)

    w(si, sj) = alpha * Jaccard_actor(si,sj)
              + beta  * Jaccard_location(si,sj)
              + gamma * 1/(1 + transfer_cost(si,sj))
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _jaccard(self, set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient."""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _avg_transfer_cost(self, s1: int, s2: int, inst: MSSPInstance) -> float:
        """Average transfer cost between locations of s1 and s2."""
        locs1 = inst.scene_locations.get(s1, [])
        locs2 = inst.scene_locations.get(s2, [])
        if not locs1 or not locs2:
            return 0.0
        costs = []
        for l1 in locs1:
            for l2 in locs2:
                c = inst.transfer_cost.get((l1, l2), 0.0)
                costs.append(c)
        return np.mean(costs) if costs else 0.0

    def build_graph(self, inst: MSSPInstance) -> nx.Graph:
        """Build the Scene Interaction Graph."""
        G = nx.Graph()
        G.add_nodes_from(inst.scenes)

        # Build co-scheduling group membership lookup for O(1) checks
        cosched_pairs: set = set()
        for group in getattr(inst, 'coschedule_groups', []):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    cosched_pairs.add((min(group[i], group[j]),
                                       max(group[i], group[j])))

        for i, s1 in enumerate(inst.scenes):
            for s2 in inst.scenes[i + 1:]:
                # C16: co-scheduling groups get weight=1 (max) to force same block
                pair = (min(s1, s2), max(s1, s2))
                if pair in cosched_pairs:
                    G.add_edge(s1, s2, weight=1.0,
                               actor_coupling=1.0,
                               location_coupling=1.0,
                               transfer_proximity=1.0)
                    continue

                actors1 = set(inst.scene_actors.get(s1, []))
                actors2 = set(inst.scene_actors.get(s2, []))
                locs1 = set(inst.scene_locations.get(s1, []))
                locs2 = set(inst.scene_locations.get(s2, []))

                j_actor = self._jaccard(actors1, actors2)
                j_loc = self._jaccard(locs1, locs2)
                tc = self._avg_transfer_cost(s1, s2, inst)
                proximity = 1.0 / (1.0 + tc)

                weight = (self.alpha * j_actor +
                          self.beta * j_loc +
                          self.gamma * proximity)

                if weight > 0.01:  # Only add meaningful edges
                    G.add_edge(s1, s2, weight=weight,
                               actor_coupling=j_actor,
                               location_coupling=j_loc,
                               transfer_proximity=proximity)

        return G

    def decompose(self, inst: MSSPInstance,
                  resolution: float = 1.0,
                  min_blocks: int = 4,
                  max_block_size: int = 8) -> Tuple[List[List[int]], nx.Graph]:
        """
        Decompose scenes into shooting blocks using Louvain clustering.
        
        Ensures at least min_blocks blocks and no block exceeds max_block_size.
        If Louvain produces too few blocks, increases resolution iteratively.
        Large blocks are split based on internal graph structure.

        Returns:
            blocks: List of scene groups (shooting blocks)
            graph: The SIG
        """
        G = self.build_graph(inst)

        # Try increasing resolution until we get enough blocks
        best_blocks = None
        for res_mult in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            res = resolution * res_mult
            partition = community_louvain.best_partition(
                G, weight='weight', resolution=res, random_state=42
            )
            communities: Dict[int, List[int]] = {}
            for scene, comm_id in partition.items():
                communities.setdefault(comm_id, []).append(scene)

            blocks = [sorted(scenes) for scenes in communities.values()]
            if len(blocks) >= min_blocks:
                best_blocks = blocks
                break
            best_blocks = blocks

        blocks = best_blocks

        # Split any blocks that are too large
        final_blocks = []
        for block in blocks:
            if len(block) <= max_block_size:
                final_blocks.append(block)
            else:
                # Split large block into sub-blocks of max_block_size
                subG = G.subgraph(block)
                # Try Louvain on subgraph
                sub_part = community_louvain.best_partition(
                    subG, weight='weight', resolution=2.0, random_state=42
                )
                sub_comm: Dict[int, List[int]] = {}
                for s, c in sub_part.items():
                    sub_comm.setdefault(c, []).append(s)
                sub_blocks = [sorted(sc) for sc in sub_comm.values()]
                if len(sub_blocks) > 1:
                    final_blocks.extend(sub_blocks)
                else:
                    # Force split by chunks
                    for i in range(0, len(block), max_block_size):
                        final_blocks.append(sorted(block[i:i+max_block_size]))

        # Ensure minimum number of blocks
        while len(final_blocks) < min_blocks and any(
                len(b) > 2 for b in final_blocks):
            # Find largest block and split it
            largest_idx = max(range(len(final_blocks)),
                              key=lambda i: len(final_blocks[i]))
            block = final_blocks.pop(largest_idx)
            mid = len(block) // 2
            final_blocks.append(sorted(block[:mid]))
            final_blocks.append(sorted(block[mid:]))

        # Sort blocks by min scene ID
        final_blocks.sort(key=lambda b: min(b))

        # Merge tiny blocks (size 1) with their most similar neighbor
        merged = True
        while merged:
            merged = False
            for i in range(len(final_blocks)):
                if len(final_blocks[i]) < 2 and len(final_blocks) > 2:
                    # Find most similar neighbor to merge with
                    best_j = -1
                    best_weight = -1
                    for j in range(len(final_blocks)):
                        if i == j:
                            continue
                        # Combined size must be <= max_block_size
                        if len(final_blocks[i]) + len(final_blocks[j]) > max_block_size:
                            continue
                        # Compute coupling between blocks
                        w = 0
                        for s1 in final_blocks[i]:
                            for s2 in final_blocks[j]:
                                if G.has_edge(s1, s2):
                                    w += G[s1][s2]['weight']
                        if w > best_weight:
                            best_weight = w
                            best_j = j
                    if best_j >= 0:
                        final_blocks[best_j].extend(final_blocks[i])
                        final_blocks[best_j].sort()
                        final_blocks.pop(i)
                        merged = True
                        break

        # Re-sort by min scene ID
        final_blocks.sort(key=lambda b: min(b))

        return final_blocks, G

    def get_block_stats(self, blocks: List[List[int]],
                        inst: MSSPInstance) -> List[Dict]:
        """Compute statistics for each shooting block."""
        stats = []
        for i, block in enumerate(blocks):
            # All actors in this block
            actors = set()
            for s in block:
                actors.update(inst.scene_actors.get(s, []))

            # All locations
            locs = set()
            for s in block:
                locs.update(inst.scene_locations.get(s, []))

            # Total duration
            total_dur = sum(inst.scene_duration.get(s, 0) for s in block)

            # Internal precedence constraints
            block_set = set(block)
            internal_prec = [(p, q) for p, q in inst.precedence
                             if p in block_set and q in block_set]

            # Cross-block precedence (scenes in this block that have
            # precedence to/from scenes in other blocks)
            external_prec = [(p, q) for p, q in inst.precedence
                             if (p in block_set) != (q in block_set)
                             and (p in block_set or q in block_set)]

            stats.append({
                "block_id": i,
                "scenes": block,
                "num_scenes": len(block),
                "actors": sorted(actors),
                "num_actors": len(actors),
                "locations": sorted(locs),
                "num_locations": len(locs),
                "total_duration_hours": round(total_dur, 1),
                "internal_precedence": internal_prec,
                "external_precedence": external_prec,
            })
        return stats
