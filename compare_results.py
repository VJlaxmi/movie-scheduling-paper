import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scorer import load_data, score_schedule

ROOT = "/Users/vjsandy/Documents/ResearchPapers/MovieScheduling"


def find_latest_baseline(root: str) -> str:
    art = os.path.join(root, "artifacts")
    candidates = sorted(glob.glob(os.path.join(art, "baseline_*")))
    if not candidates:
        raise FileNotFoundError("No baseline artifacts found")
    return candidates[-1]


def load_milp_objective(root: str) -> float:
    latest = find_latest_baseline(root)
    # Rescore MILP CSV with unified scorer for apples-to-apples
    csv_path = os.path.join(latest, "output_pyomo.csv")
    df = pd.read_csv(csv_path)
    res = score_schedule(df, load_data())
    return float(res.get("objective", 0.0)), latest


def load_ga_runs(root: str) -> pd.DataFrame:
    runs_dir = os.path.join(root, "runs")
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "ga_seed*.json")):
        with open(path, 'r') as f:
            r = json.load(f)
        rows.append({
            "seed": r.get("seed"),
            "algorithm": "GA",
            "objective": float(r.get("best_objective", 0.0)),
            "violations": bool(r.get("violations")),
            "path": path,
        })
    return pd.DataFrame(rows)


def load_pso_runs(root: str) -> pd.DataFrame:
    runs_dir = os.path.join(root, "runs")
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "pso_seed*.json")):
        with open(path, 'r') as f:
            r = json.load(f)
        rows.append({
            "seed": r.get("seed"),
            "algorithm": "PSO",
            "objective": float(r.get("best_objective", 0.0)),
            "violations": bool(r.get("violations")),
            "path": path,
        })
    return pd.DataFrame(rows)


def main():
    milp_obj, baseline_dir = load_milp_objective(ROOT)
    ga_df = load_ga_runs(ROOT)
    pso_df = load_pso_runs(ROOT)
    if ga_df.empty and pso_df.empty:
        raise RuntimeError("No GA/PSO run files found in runs/")
    # Compute gap (%) vs MILP
    for df in (ga_df, pso_df):
        if not df.empty:
            df["milp_objective"] = milp_obj
            df["gap_pct"] = 100.0 * (df["objective"] - milp_obj) / milp_obj if milp_obj != 0 else float('nan')

    # Build MILP row
    milp_row = pd.DataFrame([{ "seed": 0, "algorithm": "MILP", "objective": milp_obj, "violations": False, "path": baseline_dir, "milp_objective": milp_obj, "gap_pct": 0.0 }])
    frames = [milp_row]
    if not ga_df.empty:
        frames.append(ga_df)
    if not pso_df.empty:
        frames.append(pso_df)
    comp = pd.concat(frames, ignore_index=True)

    # Save CSV
    out_csv = os.path.join(ROOT, "runs", "comparison.csv")
    comp.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    # Sort for readability
    comp_sorted = comp.sort_values(by=["algorithm","seed"])
    labels = [f"{a}-{int(s)}" if a in ("GA","PSO") else "MILP" for a,s in zip(comp_sorted["algorithm"], comp_sorted["seed"])]
    colors = []
    for a in comp_sorted["algorithm"]:
        if a == "MILP": colors.append("#1f77b4")
        elif a == "GA": colors.append("#ff7f0e")
        else: colors.append("#2ca02c")
    ax.bar(labels, comp_sorted["objective"], color=colors)
    ax.set_ylabel("Objective")
    ax.set_title("Objective comparison: MILP vs GA (lower is better)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_png = os.path.join(ROOT, "runs", "comparison.png")
    plt.savefig(out_png, dpi=150)
    print(out_csv)
    print(out_png)


if __name__ == "__main__":
    main()


