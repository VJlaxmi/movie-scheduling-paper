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
    """A complete MSSP problem instance with hierarchical location model."""

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

        # --- Hierarchical location model: City -> Town -> Location ---
        self.cities: List[int] = []                      # city IDs
        self.towns: List[int] = []                       # town IDs
        self.location_town: Dict[int, int] = {}          # location -> town
        self.town_city: Dict[int, int] = {}              # town -> city
        self.town_locations: Dict[int, List[int]] = {}   # town -> [locations]
        self.city_towns: Dict[int, List[int]] = {}       # city -> [towns]

        # --- Criticality weights ---
        self.actor_criticality: Dict[int, float] = {}    # actor -> criticality weight
        self.location_criticality: Dict[int, float] = {} # location -> criticality weight

        # --- Actor labor constraints ---
        self.actor_max_consecutive: Dict[int, int] = {}  # actor -> max consecutive working days

        self.precedence: List[Tuple[int, int]] = []      # (pre, post)
        self.time_windows: Dict[int, Tuple[int, int]] = {}  # scene -> (earliest, latest)
        self.actor_availability: Dict[int, List[int]] = {}   # actor -> available days
        self.location_availability: Dict[int, List[int]] = {}  # loc -> available days
        self.scene_deadline: Dict[int, int] = {}                # scene -> deadline day
        self.deadline_penalty: Dict[int, float] = {}            # scene -> penalty/day

        # --- C14: Equipment constraints ---
        self.equipment_types: List[int] = []                      # equipment type IDs
        self.scene_equipment: Dict[int, List[int]] = {}           # scene -> [equipment types]
        self.equipment_supply: Dict[int, int] = {}                # equipment type -> available units (xi_e)

        # --- C15: Actor booking / holding fee (pay-or-play contracts) ---
        self.actor_booking_window: Dict[int, Tuple[int, int]] = {}  # actor -> (beta_start, beta_end)
        self.actor_holding_fee: Dict[int, float] = {}               # actor -> holding fee h_a per idle day

        # --- C16: Mandatory co-scheduling groups ---
        self.coschedule_groups: List[List[int]] = []  # list of [scene_ids] that must be same day

        # --- C17: Minimum scene separation ---
        self.scene_separation: Dict[Tuple[int, int], int] = {}  # (s_i, s_j) -> min days gap delta_ij

        # --- C18: Location concurrency capacity ---
        self.location_capacity: Dict[int, int] = {}  # location -> mu_l (max concurrent scenes/day)

        # --- C18b/C18c + C19: Lead actor rest periods ---
        self.lead_actors: List[int] = []             # subset A^P of principal actors
        self.actor_rest_period: Dict[int, int] = {}  # lead actor -> r_a min rest days after full block

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
            "cities": self.cities,
            "towns": self.towns,
            "scene_duration": {str(k): v for k, v in self.scene_duration.items()},
            "actor_wage": {str(k): v for k, v in self.actor_wage.items()},
            "actor_scenes": {str(k): v for k, v in self.actor_scenes.items()},
            "scene_actors": {str(k): v for k, v in self.scene_actors.items()},
            "scene_locations": {str(k): v for k, v in self.scene_locations.items()},
            "location_scenes": {str(k): v for k, v in self.location_scenes.items()},
            "transfer_cost": {f"{k[0]}_{k[1]}": v for k, v in self.transfer_cost.items()},
            "location_city": {str(k): v for k, v in self.location_city.items()},
            "location_town": {str(k): v for k, v in self.location_town.items()},
            "town_city": {str(k): v for k, v in self.town_city.items()},
            "town_locations": {str(k): v for k, v in self.town_locations.items()},
            "city_towns": {str(k): v for k, v in self.city_towns.items()},
            "actor_criticality": {str(k): v for k, v in self.actor_criticality.items()},
            "location_criticality": {str(k): v for k, v in self.location_criticality.items()},
            "actor_max_consecutive": {str(k): v for k, v in self.actor_max_consecutive.items()},
            "precedence": self.precedence,
            "time_windows": {str(k): list(v) for k, v in self.time_windows.items()},
            "actor_availability": {str(k): v for k, v in self.actor_availability.items()},
            "location_availability": {str(k): v for k, v in self.location_availability.items()},
            "scene_deadline": {str(k): v for k, v in self.scene_deadline.items()},
            "deadline_penalty": {str(k): v for k, v in self.deadline_penalty.items()},
            # New constraint parameters (C14-C19)
            "equipment_types": self.equipment_types,
            "scene_equipment": {str(k): v for k, v in self.scene_equipment.items()},
            "equipment_supply": {str(k): v for k, v in self.equipment_supply.items()},
            "actor_booking_window": {str(k): list(v) for k, v in self.actor_booking_window.items()},
            "actor_holding_fee": {str(k): v for k, v in self.actor_holding_fee.items()},
            "coschedule_groups": self.coschedule_groups,
            "scene_separation": {f"{k[0]}_{k[1]}": v for k, v in self.scene_separation.items()},
            "location_capacity": {str(k): v for k, v in self.location_capacity.items()},
            "lead_actors": self.lead_actors,
            "actor_rest_period": {str(k): v for k, v in self.actor_rest_period.items()},
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
        inst.cities = d.get("cities", [])
        inst.towns = d.get("towns", [])
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
        inst.location_town = {int(k): v for k, v in d.get("location_town", {}).items()}
        inst.town_city = {int(k): v for k, v in d.get("town_city", {}).items()}
        inst.town_locations = {int(k): v for k, v in d.get("town_locations", {}).items()}
        inst.city_towns = {int(k): v for k, v in d.get("city_towns", {}).items()}
        inst.actor_criticality = {int(k): v for k, v in d.get("actor_criticality", {}).items()}
        inst.location_criticality = {int(k): v for k, v in d.get("location_criticality", {}).items()}
        inst.actor_max_consecutive = {int(k): v for k, v in d.get("actor_max_consecutive", {}).items()}
        inst.precedence = [tuple(p) for p in d["precedence"]]
        inst.time_windows = {int(k): tuple(v) for k, v in d["time_windows"].items()}
        inst.actor_availability = {int(k): v for k, v in d["actor_availability"].items()}
        inst.location_availability = {int(k): v for k, v in d["location_availability"].items()}
        inst.scene_deadline = {int(k): v for k, v in d.get("scene_deadline", {}).items()}
        inst.deadline_penalty = {int(k): v for k, v in d.get("deadline_penalty", {}).items()}
        # New constraint parameters (C14-C19)
        inst.equipment_types = d.get("equipment_types", [])
        inst.scene_equipment = {int(k): v for k, v in d.get("scene_equipment", {}).items()}
        inst.equipment_supply = {int(k): v for k, v in d.get("equipment_supply", {}).items()}
        inst.actor_booking_window = {int(k): tuple(v) for k, v in d.get("actor_booking_window", {}).items()}
        inst.actor_holding_fee = {int(k): v for k, v in d.get("actor_holding_fee", {}).items()}
        inst.coschedule_groups = [list(g) for g in d.get("coschedule_groups", [])]
        inst.scene_separation = {
            (int(k.split("_")[0]), int(k.split("_")[1])): v
            for k, v in d.get("scene_separation", {}).items()
        }
        inst.location_capacity = {int(k): v for k, v in d.get("location_capacity", {}).items()}
        inst.lead_actors = d.get("lead_actors", [])
        inst.actor_rest_period = {int(k): v for k, v in d.get("actor_rest_period", {}).items()}
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
        "S75":  (75, 35, 12, 120),
        "S100": (100, 43, 15, 150),
    }

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def generate_instance(self, config_name: str, variant: int = 0) -> MSSPInstance:
        """Generate a single instance for given config and variant."""
        if config_name not in self.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}")

        ns, na, nl, nd = self.CONFIGS[config_name]
        # Use sum of ASCII values for deterministic hash (avoids Python hash randomization)
        _cfg_hash = sum(ord(c) for c in config_name) % 100
        seed = self.base_seed + variant * 1000 + _cfg_hash
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

        # --- Three-tier location hierarchy: City -> Town -> Location ---
        num_cities = max(1, nl // 3)
        num_towns = max(2, 2 * nl // 3)
        inst.cities = list(range(1, num_cities + 1))
        inst.towns = list(range(1, num_towns + 1))

        # Assign towns to cities
        for t in inst.towns:
            inst.town_city[t] = inst.cities[(t - 1) % num_cities]
        for c in inst.cities:
            inst.city_towns[c] = [t for t in inst.towns if inst.town_city[t] == c]

        # Assign locations to towns
        for loc in inst.locations:
            inst.location_town[loc] = inst.towns[(loc - 1) % num_towns]
            inst.location_city[loc] = inst.town_city[inst.location_town[loc]]
        for t in inst.towns:
            inst.town_locations[t] = [l for l in inst.locations if inst.location_town[l] == t]

        # --- Scene-Location compatibility: each scene 1-2 locations ---
        for loc in inst.locations:
            inst.location_scenes[loc] = []
        for s in inst.scenes:
            n_loc = rng.randint(1, min(2, nl))
            assigned = rng.sample(inst.locations, n_loc)
            inst.scene_locations[s] = assigned
            for loc in assigned:
                inst.location_scenes[loc].append(s)

        # --- Hierarchical transfer costs (Eq. 1 in paper) ---
        for l1 in inst.locations:
            for l2 in inst.locations:
                if l1 == l2:
                    inst.transfer_cost[(l1, l2)] = 0.0
                elif inst.location_town[l1] == inst.location_town[l2]:
                    # Same town: intra-town cost only
                    inst.transfer_cost[(l1, l2)] = round(rng.uniform(100, 500), 2)
                elif inst.location_city[l1] == inst.location_city[l2]:
                    # Same city, different towns: town + loc costs
                    inst.transfer_cost[(l1, l2)] = round(
                        rng.uniform(500, 2000) + rng.uniform(100, 500), 2)
                else:
                    # Different cities: city + town + loc costs
                    inst.transfer_cost[(l1, l2)] = round(
                        rng.uniform(2000, 8000) + rng.uniform(500, 2000) + rng.uniform(100, 500), 2)
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

        # --- Actor criticality weights (C8 linking, objective term iv) ---
        for a in inst.actors:
            inst.actor_criticality[a] = round(rng.uniform(1.0, 10.0), 2)

        # --- Actor max consecutive working days (C13 labor constraint) ---
        for a in inst.actors:
            inst.actor_max_consecutive[a] = rng.choice([3, 4, 5, 6])

        # --- Location availability: available ~85% of days ---
        for loc in inst.locations:
            available = [d for d in inst.days if rng.random() < 0.85]
            if len(available) < nd // 3:
                available = rng.sample(inst.days, nd // 2)
            inst.location_availability[loc] = sorted(available)

        # --- Location criticality weights (C9 linking, objective term iv) ---
        for loc in inst.locations:
            inst.location_criticality[loc] = round(rng.uniform(1.0, 10.0), 2)

        # --- Deadlines for ~30% of scenes ---
        for s in inst.scenes:
            if rng.random() < 0.3:
                inst.scene_deadline[s] = rng.randint(nd // 2, nd)
                inst.deadline_penalty[s] = round(rng.uniform(50, 500), 2)

        # --- C14: Equipment types and supply ---
        num_equip = max(2, nl // 2)  # number of specialized equipment types
        inst.equipment_types = list(range(1, num_equip + 1))
        for e in inst.equipment_types:
            inst.equipment_supply[e] = rng.choice([1, 2])  # 1 or 2 units available
        for s in inst.scenes:
            # Each scene requires 0-2 equipment types (50% of scenes need special equipment)
            if rng.random() < 0.5:
                n_equip = rng.randint(1, min(2, num_equip))
                inst.scene_equipment[s] = rng.sample(inst.equipment_types, n_equip)
            else:
                inst.scene_equipment[s] = []

        # --- C15: Actor booking windows and holding fees (~60% of actors) ---
        for a in inst.actors:
            if rng.random() < 0.6:
                # Booking window: a contiguous window within the planning horizon
                window_start = rng.randint(1, max(1, nd // 2))
                window_length = rng.randint(max(1, nd // 4), max(2, nd * 3 // 4))
                window_end = min(nd, window_start + window_length - 1)
                inst.actor_booking_window[a] = (window_start, window_end)
                # Holding fee: 20-60% of daily wage
                inst.actor_holding_fee[a] = round(inst.actor_wage[a] * rng.uniform(0.2, 0.6), 2)

        # --- C16: Mandatory co-scheduling groups ---
        # ~10-15% of scenes grouped in pairs/triples of 2-3
        # Groups must share at least one common town to be C11-feasible.
        num_groups = max(1, ns // 8)
        ungrouped_scenes = list(inst.scenes)
        rng.shuffle(ungrouped_scenes)

        def _scene_towns(s):
            return {inst.location_town.get(l)
                    for l in inst.scene_locations.get(s, [])
                    if l in inst.location_town}

        for _ in range(num_groups):
            group_size = rng.randint(2, min(3, len(ungrouped_scenes)))
            if len(ungrouped_scenes) < group_size:
                break
            group = ungrouped_scenes[:group_size]
            # Verify the group shares at least one common town (C11 feasibility)
            shared_towns = _scene_towns(group[0])
            for gs in group[1:]:
                shared_towns &= _scene_towns(gs)
            if not shared_towns:
                # No common town — shrink to a pair that does share a town
                valid_group = [group[0]]
                t0 = _scene_towns(group[0])
                for gs in group[1:]:
                    if t0 & _scene_towns(gs):
                        valid_group.append(gs)
                        t0 &= _scene_towns(gs)
                        if len(valid_group) >= 2:
                            break
                if len(valid_group) < 2:
                    # Cannot form any valid pair; consume scenes but skip this group
                    ungrouped_scenes = ungrouped_scenes[group_size:]
                    continue
                group = valid_group
            ungrouped_scenes = [s for s in ungrouped_scenes if s not in group]
            inst.coschedule_groups.append(group)

        # --- C17: Minimum scene separation (10% of scene pairs) ---
        max_sep_pairs = max(1, ns // 10)
        scene_pairs = [(s1, s2) for s1 in inst.scenes for s2 in inst.scenes if s1 < s2]
        sep_pairs = rng.sample(scene_pairs, min(max_sep_pairs, len(scene_pairs)))
        for s1, s2 in sep_pairs:
            inst.scene_separation[(s1, s2)] = rng.randint(2, max(2, nd // 10))

        # --- C18: Location concurrency capacity ---
        for loc in inst.locations:
            inst.location_capacity[loc] = rng.choice([1, 2, 3])

        # --- C19: Lead actors and rest periods ---
        # ~20% of actors are lead actors requiring rest periods
        num_leads = max(1, na // 5)
        inst.lead_actors = rng.sample(inst.actors, num_leads)
        for a in inst.lead_actors:
            # Rest period: 1-2 days after completing max consecutive block
            inst.actor_rest_period[a] = rng.randint(1, 2)

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
