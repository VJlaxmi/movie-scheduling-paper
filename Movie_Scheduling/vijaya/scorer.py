import os
import json
from typing import Dict, List, Tuple
import pandas as pd

ROOT = "/Users/vjsandy/Documents/ResearchPapers/MovieScheduling"
DATASET = os.path.join(ROOT, "datasets", "canonical_v1.xlsx")


def load_data(path: str = DATASET) -> Dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    return sheets


def score_schedule(assignments: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> Dict:
    """
    assignments columns required:
      - D_ID (int)
      - Scene_Num (int)
      - Location (str)
      - Actor (str) [optional per-row, we dedupe for scene-day accounting]
    """
    # Deduplicate by (day, scene) for duration and core checks
    uniq_scene_day = assignments.drop_duplicates(subset=["D_ID","Scene_Num"])  # (day, scene)

    # Load needed tables
    scene_time = data["SCENE_TIME"][ ["Scene_Number","Shoot_time"] ].rename(columns={"Scene_Number":"Scene_Num"})
    actor_av = data["ACTOR_AVAILABILITY"][ ["Actors","Availability"] ]
    actor_scn = data["ACTOR_SCENES"][ ["Actors","Scene_Number"] ].rename(columns={"Scene_Number":"Scene_Num","Actors":"Actor"})
    loc_av = data["LOCATION_AVAILABILITY"][ ["Shoot_Location","Availability"] ]
    loc_sum = data["LOCATION_SUMMARY"][ ["Shoot_Location","Criticality"] ]
    actor_sum = data["ACTOR_SUMMARY"][ ["Actors","Criticality"] ]
    cal = data["CALENDAR"][ ["D_ID","DATE"] ]

    # Optional sheets
    windows = data.get("WINDOWS", pd.DataFrame())
    prec = data.get("PRECEDENCE", pd.DataFrame())
    loc_cap = data.get("LOCATION_CAPACITY", pd.DataFrame())
    transfers = data.get("TRANSFERS", pd.DataFrame())
    actor_costs = data.get("ACTOR_COSTS", pd.DataFrame())
    scene_policy = data.get("SCENE_POLICY", pd.DataFrame())

    # Feasibility checks
    violations = {}

    # Each scene exactly once
    sc_counts = uniq_scene_day.groupby("Scene_Num")["D_ID"].nunique()
    v1 = sc_counts[sc_counts != 1]
    if not v1.empty:
        violations["scene_once"] = v1.to_dict()

    # Day capacity (≤8h) by summing durations per day
    with_time = uniq_scene_day.merge(scene_time, on="Scene_Num", how="left")
    day_hours = with_time.groupby("D_ID")["Shoot_time"].sum()
    v2 = day_hours[day_hours > 8]
    if not v2.empty:
        violations["day_capacity"] = v2.to_dict()

    # Actor availability and membership
    v3_missing_av, v3b_missing_map = [], []
    if "Actor" in assignments.columns and not assignments["Actor"].isna().all():
        actor_ok = assignments.merge(actor_av, left_on=["Actor","D_ID"], right_on=["Actors","Availability"], how="left")
        v3_missing_av = actor_ok[actor_ok["Actors"].isna()][["Actor","Scene_Num","D_ID"]].to_dict('records')
        # Only run membership check if the actor_scn table contains any rows
        if not actor_scn.empty:
            actor_scene_ok = assignments.merge(actor_scn, on=["Actor","Scene_Num"], how="left")
            v3b_missing_map = actor_scene_ok[actor_scene_ok["Actor"].isna()][["Actor","Scene_Num"]].to_dict('records')
    if v3_missing_av:
        violations["actor_availability"] = v3_missing_av
    if v3b_missing_map:
        violations["actor_scene_membership"] = v3b_missing_map

    # Location availability
    loc_ok = uniq_scene_day.merge(loc_av, left_on=["Location","D_ID"], right_on=["Shoot_Location","Availability"], how="left")
    v4 = loc_ok[loc_ok["Shoot_Location"].isna()][["Location","Scene_Num","D_ID"]].to_dict('records')
    if v4:
        violations["location_availability"] = v4

    # Windows
    if not windows.empty:
        wmap = windows.set_index("Scene_Number")[ ["Earliest_Day","Latest_Day"] ].to_dict('index')
        bad_w = []
        for _, r in uniq_scene_day.iterrows():
            w = wmap.get(int(r.Scene_Num))
            if w and not (int(r.D_ID) >= int(w["Earliest_Day"]) and int(r.D_ID) <= int(w["Latest_Day"])):
                bad_w.append({"Scene_Num": int(r.Scene_Num), "D_ID": int(r.D_ID)})
        if bad_w:
            violations["windows"] = bad_w

    # Precedence
    if not prec.empty:
        order = uniq_scene_day.set_index("Scene_Num")["D_ID"].to_dict()
        bad_p = []
        for _, r in prec.iterrows():
            pre, post = int(r.Pre), int(r.Post)
            if pre in order and post in order and not (order[post] >= order[pre]):
                bad_p.append({"Pre": pre, "Post": post})
        if bad_p:
            violations["precedence"] = bad_p

    # Location capacity per day
    if not loc_cap.empty:
        cap = loc_cap.set_index("Shoot_Location")["Daily_Capacity"].to_dict()
        loc_day_count = uniq_scene_day.groupby(["Location","D_ID"]).size()
        bad_c = []
        for (j, n), cnt in loc_day_count.items():
            if int(cnt) > int(cap.get(j, 1)):
                bad_c.append({"Location": j, "D_ID": int(n), "count": int(cnt), "cap": int(cap.get(j,1))})
        if bad_c:
            violations["location_capacity"] = bad_c

    # Objective components
    # Base priorities
    crit_loc = loc_sum.rename(columns={"Shoot_Location":"Location"})
    obj_loc = assignments.merge(crit_loc, on="Location", how="left")["Criticality"].apply(lambda x: 1/float(x) if pd.notna(x) and x!=0 else 0).sum()
    crit_act = assignments.merge(actor_sum, left_on="Actor", right_on="Actors", how="left")["Criticality"].apply(lambda x: 1/float(x) if pd.notna(x) and x!=0 else 0).sum() if "Actor" in assignments.columns else 0.0

    # Actor wage (count each actor-day once)
    if not actor_costs.empty and "Actor" in assignments.columns:
        wages = actor_costs.set_index("Actors")["Daily_Wage"].to_dict()
        actor_days = assignments.drop_duplicates(subset=["Actor","D_ID"])  # one wage per actor-day
        obj_wage = actor_days["Actor"].map(lambda a: float(wages.get(a, 0.0))).sum()
    else:
        obj_wage = 0.0

    # Travel cost proxy (per location-day used)
    loc_days_used = uniq_scene_day.drop_duplicates(subset=["Location","D_ID"]).shape[0]
    avg_tc = float(transfers["Travel_Cost"].mean()) if not transfers.empty else 0.0
    obj_travel = loc_days_used * avg_tc

    # Deadline penalties
    obj_deadline = 0.0
    if not scene_policy.empty:
        dl = scene_policy.set_index("Scene_Number")[ ["Deadline_Day","Late_Penalty_per_Day"] ].to_dict('index')
        day_of = uniq_scene_day.set_index("Scene_Num")["D_ID"].to_dict()
        for s, v in dl.items():
            dday = v.get("Deadline_Day")
            pene = v.get("Late_Penalty_per_Day")
            if pd.notna(dday) and pd.notna(pene) and s in day_of and int(day_of[s]) > int(dday):
                obj_deadline += float(pene) * (int(day_of[s]) - int(dday))

    objective = float(obj_loc + crit_act + obj_wage + obj_travel + obj_deadline)

    return {
        "violations": violations,
        "objective": objective,
        "components": {
            "priority_location": float(obj_loc),
            "priority_actor": float(crit_act),
            "wage": float(obj_wage),
            "travel": float(obj_travel),
            "deadline": float(obj_deadline),
        },
    }


if __name__ == "__main__":
    # Demo: score the MILP CSV if present
    csv_path = os.path.join(ROOT, "output_pyomo.csv")
    if os.path.exists(csv_path):
        data = load_data()
        df = pd.read_csv(csv_path)
        res = score_schedule(df, data)
        print(json.dumps(res, indent=2))
    else:
        print("No schedule CSV found to score.")


