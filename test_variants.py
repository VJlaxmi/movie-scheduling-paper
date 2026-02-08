"""Quick sanity test for all ablation variants and new configs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bdm_aos.data_generator import ScaledDataGenerator
from bdm_aos.matheuristic import BDM_AOS
from bdm_aos.baselines import StandaloneGA, StandalonePSO

gen = ScaledDataGenerator(seed=42)

# Test all ablation variants on S10
inst = gen.generate_instance("S10", 0)
print(f"S10: {inst.num_scenes} scenes, {inst.num_actors} actors")

variants = [
    ("BDM_AOS",       dict(use_aos=True,  use_sig=True,  use_spec_ops=True,  flat_mode=False)),
    ("BDM_NoAOS",     dict(use_aos=False, use_sig=True,  use_spec_ops=True,  flat_mode=False)),
    ("BDM_NoSIG",     dict(use_aos=True,  use_sig=False, use_spec_ops=True,  flat_mode=False)),
    ("BDM_NoSpecOps", dict(use_aos=True,  use_sig=True,  use_spec_ops=False, flat_mode=False)),
    ("BDM_Flat",      dict(use_aos=True,  use_sig=True,  use_spec_ops=True,  flat_mode=True)),
]

for name, kwargs in variants:
    algo = BDM_AOS(population_size=10, generations=5, seed=42, milp_time_limit=5, **kwargs)
    result = algo.run(inst)
    cost = result["best_cost"]
    ms = result["best_makespan"]
    nb = result.get("num_blocks", "?")
    print(f"  {name:20s}: cost={cost:10.1f}  makespan={ms:3.0f}  blocks={nb}")

# Test S75 and S100
for cfg in ["S75", "S100"]:
    inst_big = gen.generate_instance(cfg, 0)
    print(f"\n{cfg}: {inst_big.num_scenes} scenes, {inst_big.num_actors} actors, {inst_big.num_days} days")
    algo = BDM_AOS(population_size=10, generations=3, seed=42, milp_time_limit=5)
    result = algo.run(inst_big)
    cost = result["best_cost"]
    ms = result["best_makespan"]
    nb = result["num_blocks"]
    aos_stats = result.get("aos_stats", {})
    freq_log = aos_stats.get("operator_frequency_log", [])
    print(f"  BDM_AOS: cost={cost:.1f}, makespan={ms:.0f}, blocks={nb}")
    print(f"  Operator log entries: {len(freq_log)}")

# Test GA and PSO on S10
ga = StandaloneGA(population_size=20, generations=20, seed=42)
r = ga.run(inst)
print(f"\nGA on S10: cost={r['best_cost']:.1f}")

pso = StandalonePSO(swarm_size=10, iterations=20, seed=42)
r = pso.run(inst)
print(f"PSO on S10: cost={r['best_cost']:.1f}")

print("\n=== ALL VARIANTS OK ===")
