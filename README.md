# BDM-AOS: Bi-Level Decomposition Matheuristic with Adaptive Operator Selection for Movie Scene Scheduling

A novel approach to the **Movie Scene Scheduling Problem (MSSP)** that combines graph-based decomposition, multi-objective evolutionary optimization, and reinforcement learning-based adaptive operator selection.

## Overview

Film production scheduling involves sequencing scenes across shooting days to minimize total production cost (actor wages + location transfer costs) subject to actor availability, precedence, time-window, and capacity constraints. The MSSP is NP-hard.

**BDM-AOS** introduces:
1. **Scene Interaction Graph (SIG)** — captures actor-sharing, location-sharing, and transfer-cost coupling between scenes; Louvain community detection partitions scenes into tightly coupled *shooting blocks*
2. **Bi-level Matheuristic** — NSGA-II evolves block permutations (upper level); a global greedy decoder schedules scenes in block-guided order (lower level), optimizing cost and makespan simultaneously
3. **Q-Learning Adaptive Operator Selection** — dynamically selects among 9 crossover–mutation combinations including two novel operators:
   - *Location-Aware Crossover (LAX)* — preserves spatial locality
   - *Actor-Continuity Mutation (ACM)* — reduces actor idle gaps

## Project Structure

```
├── bdm_aos/                # Core Python package
│   ├── __init__.py
│   ├── data_generator.py   # Synthetic MSSP instance generator
│   ├── scene_graph.py      # Scene Interaction Graph + Louvain decomposition
│   ├── matheuristic.py     # BDM-AOS algorithm (NSGA-II + global greedy decoder)
│   ├── nsga2.py            # NSGA-II components (non-dominated sorting, crowding)
│   ├── operators.py        # Crossover/mutation operators (PMX, OX, LAX, ACM, etc.)
│   ├── q_learning_aos.py   # Q-learning adaptive operator selection
│   ├── milp_solver.py      # MILP baseline solver (Pyomo + CBC)
│   ├── baselines.py        # Standalone GA, PSO, MILP baselines
│   ├── experiment_runner.py # Experiment orchestration
│   ├── analysis.py         # Statistical analysis + figure generation
│   └── run_experiments.py  # CLI entry point
├── run_all.py              # Run full experiment pipeline
├── experiments/            # Experiment outputs
│   ├── results.csv         # Main results table
│   ├── summary.csv         # Aggregated summary
│   ├── friedman_test.json  # Friedman statistical test
│   └── wilcoxon_tests.csv  # Pairwise Wilcoxon tests
├── latex/                  # NeurIPS 2025 paper
│   ├── main.tex
│   ├── main.bib
│   ├── neurips_2025.sty
│   ├── Sections/           # Paper sections (0–7)
│   └── figures/            # Generated figures (PDF + PNG)
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.10
- networkx, python-louvain (community detection)
- numpy, scipy, pandas, matplotlib
- pyomo (MILP baseline only)

## Usage

### Run Full Experiment Pipeline

```bash
python run_all.py
```

This runs all 5 algorithms (GA, PSO, MILP, BDM-NoAOS, BDM-AOS) across 4 instance sizes (S10, S15, S20, S25) with 3 seeds each, generates statistical analysis and figures.

### Run Specific Configurations

```bash
python -m bdm_aos.run_experiments --seeds 5 --configs S10 S25 --algorithms BDM_AOS GA PSO
```

## Results

| Config | GA | PSO | BDM-AOS | BDM-NoAOS |
|--------|----------|----------|----------|-----------|
| S10 | **11,780** | 11,959 | 11,965 | 11,965 |
| S15 | 20,567 | **18,721** | 28,102 | 28,102 |
| S20 | 29,478 | **28,655** | 37,338 | 37,338 |
| S25 | **34,394** | 36,808 | 36,947 | 37,479 |

Friedman test: χ² = 9.9, p = 0.019 (statistically significant).

## Citation

If you use this code, please cite:

```bibtex
@article{lendale2025bdmaos,
  title={BDM-AOS: A Bi-Level Decomposition Matheuristic with Adaptive Operator Selection for Multi-Objective Movie Scene Scheduling},
  author={Lendale, Vijaylaxmi},
  year={2025}
}
```

## License

This project is for academic research purposes.
