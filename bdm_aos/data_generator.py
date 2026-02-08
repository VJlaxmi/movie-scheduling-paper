"""
Scaled Data Generator for Movie Scene Scheduling Problem (MSSP)

Generates realistic test instances at multiple scales:
  10, 15, 20, 25, 30, 50 scenes
following the methodology of Liu et al. (2019), Long & Zhao (2020),
and Baran Tekin (2023).

Includes: actor-scene mapping, location compatibility, transfer costs,
precedence constraints, time windows, actor blocking periods, deadlines.
"""

import random
import json
import os
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations


class MSSPInstance:
    """A complete MSSP problem instance."""

    def __init__(self, name: str, num_scenes: int, num_actors: int,
                 num_locations: int, num_days: int, seed: int):
        self.name = name
        self.num_scenes = num_scenes
        self.num_actors = num_actors
        self.num_locations = num_locations
        self.num_days = num_days
        self.seed = seed

        self.scenes: List[int] = []            # scene IDs
        self.actors: List[int] = []            # actor IDs
        self.locations: List[int] = []         # location IDs
        self.days: List[int] = []              # day IDs

        self.scene_duration: Dict[int, float] = {}      # scene -> hours
        self.actor_wage: Dict[int, float] = {}           # actor -> $/day
        self.actor_scenes: Dict[int, List[int]] = {}     # actor -> [scenes]
        self.scene_actors: Dict[int, List[int]] = {}     # scene -> [actors]
        self.scene_locations: Dict[int, List[int]] = {}  # scene -> [locations]
        self.location_scenes: Dict[int, List[int]] = {}  # location -> [scenes]

        self.transfer_cost: Dict[Tuple[int, int], float] = {}  # (loc, loc) -> cost
        self.location_city: Dict[int, int] = {}                 # location -> city

        self.precedence: List[Tuple[int, int]] = []      # (pre, post)
        self.time_windows: Dict[int, Tuple[int, int]] = {}  # scene -> (earliest, latest)
        self.actor_availability: Dict[int, List[int]] = {}   # actor -> available days
        self.location_availability: Dict[int, List[int]] = {}  # loc -> available days
        self.scene_deadline: Dict[int, int] = {}                # scene -> deadline day
        self.deadline_penalty: Dict[int, float] = {}            # scene -> penalty/day

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "metadata": {
                "name": self.name,
                "num_scenes": self.num_scenes,
                "num_actors": self.num_actors,
                "num_locations": self.num_locations,
                "num_days": self.num_days,
                "seed": self.seed
            },
            "scenes": self.scenes,
            "actors": self.actors,
            "locations": self.locations,
            "days": self.days,
            "scene_duration": {str(k): v for k, v in self.scene_duration.items()},
            "actor_wage": {str(k): v for k, v in self.actor_wage.items()},
            "actor_scenes": {str(k): v for k, v in self.actor_scenes.items()},
            "scene_actors": {str(k): v for k, v in self.scene_actors.items()},
            "scene_locations": {str(k): v for k, v in self.scene_locations.items()},
            "location_scenes": {str(k): v for k, v in self.location_scenes.items()},
            "transfer_cost": {f"{k[0]}_{k[1]}": v for k, v in self.transfer_cost.items()},
            "location_city": {str(k): v for k, v in self.location_city.items()},
            "precedence": self.precedence,
            "time_windows": {str(k): list(v) for k, v in self.time_windows.items()},
            "actor_availability": {str(k): v for k, v in self.actor_availability.items()},
            "location_availability": {str(k): v for k, v in self.location_availability.items()},
            "scene_deadline": {str(k): v for k, v in self.scene_deadline.items()},
            "deadline_penalty": {str(k): v for k, v in self.deadline_penalty.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MSSPInstance":
        """Deserialize from dict."""
        m = d["metadata"]
        inst = cls(m["name"], m["num_scenes"], m["num_actors"],
                   m["num_locations"], m["num_days"], m["seed"])
        inst.scenes = d["scenes"]
        inst.actors = d["actors"]
        inst.locations = d["locations"]
        inst.days = d["days"]
        inst.scene_duration = {int(k): v for k, v in d["scene_duration"].items()}
        inst.actor_wage = {int(k): v for k, v in d["actor_wage"].items()}
        inst.actor_scenes = {int(k): v for k, v in d["actor_scenes"].items()}
        inst.scene_actors = {int(k): v for k, v in d["scene_actors"].items()}
        inst.scene_locations = {int(k): v for k, v in d["scene_locations"].items()}
        inst.location_scenes = {int(k): v for k, v in d["location_scenes"].items()}
        inst.transfer_cost = {
            (int(k.split("_")[0]), int(k.split("_")[1])): v
            for k, v in d["transfer_cost"].items()
        }
        inst.location_city = {int(k): v for k, v in d["location_city"].items()}
        inst.precedence = [tuple(p) for p in d["precedence"]]
        inst.time_windows = {int(k): tuple(v) for k, v in d["time_windows"].items()}
        inst.actor_availability = {int(k): v for k, v in d["actor_availability"].items()}
        inst.location_availability = {int(k): v for k, v in d["location_availability"].items()}
        inst.scene_deadline = {int(k): v for k, v in d.get("scene_deadline", {}).items()}
        inst.deadline_penalty = {int(k): v for k, v in d.get("deadline_penalty", {}).items()}
        return inst


class ScaledDataGenerator:
    """
    Generate MSSP instances following literature methodology:
    - Liu et al. (2019): actor wages $80-$1000, transfer costs $0-$10000
    - Baran Tekin (2023): precedence graphs, time windows, blocking periods
    """

    # Instance configurations: (scenes, actors, locations, days)
    CONFIGS = {
        "S10":  (10, 5,  3,  15),
        "S15":  (15, 8,  4,  25),
        "S20":  (20, 10, 5,  35),
        "S25":  (25, 12, 6,  45),
        "S30":  (30, 15, 7,  55),
        "S50":  (50, 20, 10, 90),
    }

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def generate_instance(self, config_name: str, variant: int = 0) -> MSSPInstance:
        """Generate a single instance for given config and variant."""
        if config_name not in self.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}")

        ns, na, nl, nd = self.CONFIGS[config_name]
        seed = self.base_seed + variant * 1000 + hash(config_name) % 100
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        inst = MSSPInstance(f"{config_name}_v{variant}", ns, na, nl, nd, seed)
        inst.scenes = list(range(1, ns + 1))
        inst.actors = list(range(1, na + 1))
        inst.locations = list(range(1, nl + 1))
        inst.days = list(range(1, nd + 1))

        # --- Scene durations: 1-4 hours (following Liu 2019) ---
        for s in inst.scenes:
            inst.scene_duration[s] = round(rng.uniform(1.0, 4.0), 1)

        # --- Actor wages: $80-$1000/day (following Liu 2019) ---
        for a in inst.actors:
            inst.actor_wage[a] = round(rng.uniform(80, 1000), 2)

        # --- Actor-Scene mapping: each scene needs 1-4 actors ---
        for a in inst.actors:
            inst.actor_scenes[a] = []
        for s in inst.scenes:
            n_actors = rng.randint(1, min(4, na))
            assigned = rng.sample(inst.actors, n_actors)
            inst.scene_actors[s] = assigned
            for a in assigned:
                inst.actor_scenes[a].append(s)
        # Ensure every actor appears in at least 1 scene
        for a in inst.actors:
            if not inst.actor_scenes[a]:
                s = rng.choice(inst.scenes)
                inst.actor_scenes[a].append(s)
                inst.scene_actors[s].append(a)

        # --- Locations with city hierarchy ---
        num_cities = max(2, nl // 3)
        cities = list(range(1, num_cities + 1))
        for loc in inst.locations:
            inst.location_city[loc] = cities[(loc - 1) % num_cities]

        # --- Scene-Location compatibility: each scene 1-2 locations ---
        for loc in inst.locations:
            inst.location_scenes[loc] = []
        for s in inst.scenes:
            n_loc = rng.randint(1, min(2, nl))
            assigned = rng.sample(inst.locations, n_loc)
            inst.scene_locations[s] = assigned
            for loc in assigned:
                inst.location_scenes[loc].append(s)

        # --- Transfer costs based on city distance (following Liu 2019) ---
        for l1 in inst.locations:
            for l2 in inst.locations:
                if l1 == l2:
                    inst.transfer_cost[(l1, l2)] = 0.0
                elif inst.location_city[l1] == inst.location_city[l2]:
                    # Same city: low cost
                    inst.transfer_cost[(l1, l2)] = round(rng.uniform(100, 500), 2)
                else:
                    # Different cities: high cost (following Liu 2019 range)
                    inst.transfer_cost[(l1, l2)] = round(rng.uniform(1000, 8000), 2)
                # Ensure symmetry
                inst.transfer_cost[(l2, l1)] = inst.transfer_cost[(l1, l2)]

        # --- Precedence constraints (DAG, ~20% of scene pairs) ---
        # Create a topological ordering and add forward edges
        topo_order = list(inst.scenes)
        rng.shuffle(topo_order)
        topo_rank = {s: i for i, s in enumerate(topo_order)}
        for i, s1 in enumerate(topo_order):
            for s2 in topo_order[i + 1:]:
                if rng.random() < 0.08:  # sparse DAG
                    inst.precedence.append((s1, s2))
        # Limit total precedence to ~20% of scenes
        max_prec = max(2, ns // 5)
        if len(inst.precedence) > max_prec:
            inst.precedence = rng.sample(inst.precedence, max_prec)

        # --- Time windows (for ~40% of scenes) ---
        for s in inst.scenes:
            if rng.random() < 0.4:
                earliest = rng.randint(1, max(1, nd // 3))
                latest = rng.randint(earliest + max(1, nd // 4), nd)
                inst.time_windows[s] = (earliest, latest)

        # --- Actor availability: available ~80% of days ---
        for a in inst.actors:
            available = [d for d in inst.days if rng.random() < 0.8]
            if len(available) < nd // 3:  # ensure enough days
                available = rng.sample(inst.days, nd // 2)
            inst.actor_availability[a] = sorted(available)

        # --- Location availability: available ~85% of days ---
        for loc in inst.locations:
            available = [d for d in inst.days if rng.random() < 0.85]
            if len(available) < nd // 3:
                available = rng.sample(inst.days, nd // 2)
            inst.location_availability[loc] = sorted(available)

        # --- Deadlines for ~30% of scenes ---
        for s in inst.scenes:
            if rng.random() < 0.3:
                inst.scene_deadline[s] = rng.randint(nd // 2, nd)
                inst.deadline_penalty[s] = round(rng.uniform(50, 500), 2)

        return inst

    def generate_all(self, output_dir: str, num_variants: int = 1):
        """Generate all instances and save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        manifest = {}

        for config_name in self.CONFIGS:
            for v in range(num_variants):
                inst = self.generate_instance(config_name, v)
                fname = f"{inst.name}.json"
                fpath = os.path.join(output_dir, fname)
                data = inst.to_dict()
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=2)
                # Compute checksum
                with open(fpath, 'rb') as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                manifest[fname] = {
                    "config": config_name,
                    "variant": v,
                    "scenes": inst.num_scenes,
                    "actors": inst.num_actors,
                    "sha256": checksum
                }

        with open(os.path.join(output_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Generated {len(manifest)} instances in {output_dir}")
        return manifest


if __name__ == "__main__":
    gen = ScaledDataGenerator(seed=42)
    gen.generate_all(
        os.path.join(os.path.dirname(__file__), "instances"),
        num_variants=1
    )
