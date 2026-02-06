
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import shutil
import sys
import os
import os.path
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )

from pyomo.environ import *
from pyomo.gdp import *

"""## Defining the model type"""

model = ConcreteModel()

"""## Import the data"""

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)

# Use the enriched Excel if present, else fallback
path_enriched = "/Users/vjsandy/Documents/ResearchPapers/MovieScheduling/testing_data_enriched.xlsx"
path_default = "/Users/vjsandy/Documents/ResearchPapers/MovieScheduling/testing_data.xlsx"
path = path_enriched if os.path.exists(path_enriched) else path_default

# Validate file existence and required sheets
if not os.path.exists(path):
    raise FileNotFoundError(f"Data file not found at {path}. Place 'testing_data.xlsx' in the project root.")

required_sheets = [
    "SCENE_SUMMARY",
    "ACTOR_SUMMARY",
    "LOCATION_SUMMARY",
    "CALENDAR",
    "LOCATION_AVAILABILITY",
    "SCENE_TIME",
    "ACTOR_AVAILABILITY",
    "ACTOR_SCENES",
    "LOCATION_SCENES",
]
optional_sheets = [
    "PRECEDENCE",
    "WINDOWS",
    "LOCATION_CAPACITY",
    "ACTOR_COSTS",
    "TRANSFERS",
    "SCENE_POLICY",
]

xl = pd.ExcelFile(path)
missing_sheets = [s for s in required_sheets if s not in xl.sheet_names]
if missing_sheets:
    raise ValueError(f"Missing required sheets in Excel file: {missing_sheets}. Available: {xl.sheet_names}")

SCENE_SUMMARY = pd.read_excel(path, sheet_name= "SCENE_SUMMARY")
ACTOR_SUMMARY = pd.read_excel(path, sheet_name= "ACTOR_SUMMARY")
LOCATION_SUMMARY = pd.read_excel(path, sheet_name= "LOCATION_SUMMARY")
CALENDAR = pd.read_excel(path, sheet_name= "CALENDAR")
LOCATION_AVAILABILITY = pd.read_excel(path, sheet_name= "LOCATION_AVAILABILITY")
SCENE_TIME = pd.read_excel(path, sheet_name= "SCENE_TIME")
ACTOR_AVAILABILITY = pd.read_excel(path, sheet_name= "ACTOR_AVAILABILITY")
ACTOR_SCENES = pd.read_excel(path, sheet_name= "ACTOR_SCENES")

ACTOR_SUMMARY.head()

ACTOR_INFO = pd.merge(ACTOR_AVAILABILITY,ACTOR_SCENES,on='Actors',how='inner')
ACTOR_INFO['scen_avai'] = ACTOR_INFO[['Scene_Number','Availability']].apply(tuple, axis=1)
ACTOR_INFO['actor_avai'] = ACTOR_INFO[['Actors','Availability']].apply(tuple, axis=1)
ACTOR_INFO['scene_actor'] = ACTOR_INFO[['Scene_Number','Actors']].apply(tuple, axis=1)
ACTOR_INFO['asn'] = ACTOR_INFO[['Actors','Scene_Number','Availability']].apply(tuple, axis=1)

SCENE_SUMMARY = pd.read_excel(path, sheet_name= "SCENE_SUMMARY")
ACTOR_SUMMARY = pd.read_excel(path, sheet_name= "ACTOR_SUMMARY")
LOCATION_SUMMARY = pd.read_excel(path, sheet_name= "LOCATION_SUMMARY")
CALENDAR = pd.read_excel(path, sheet_name= "CALENDAR")
LOCATION_AVAILABILITY = pd.read_excel(path, sheet_name= "LOCATION_AVAILABILITY")
SCENE_TIME = pd.read_excel(path, sheet_name= "SCENE_TIME")
ACTOR_AVAILABILITY = pd.read_excel(path, sheet_name= "ACTOR_AVAILABILITY")
ACTOR_SCENES = pd.read_excel(path, sheet_name= "ACTOR_SCENES")
ACTOR_INFO = pd.merge(ACTOR_AVAILABILITY,ACTOR_SCENES,on='Actors',how='inner')
ACTOR_INFO['scen_avai'] = ACTOR_INFO[['Scene_Number','Availability']].apply(tuple, axis=1)
ACTOR_INFO['actor_avai'] = ACTOR_INFO[['Actors','Availability']].apply(tuple, axis=1)
ACTOR_INFO['scene_actor'] = ACTOR_INFO[['Scene_Number','Actors']].apply(tuple, axis=1)
ACTOR_INFO['asn'] = ACTOR_INFO[['Actors','Scene_Number','Availability']].apply(tuple, axis=1)

LOCATION_SCENES = pd.read_excel(path, sheet_name= "LOCATION_SCENES")
LOCATION_INFO = pd.merge(LOCATION_AVAILABILITY,LOCATION_SCENES,on=['Shoot_Location','Shoot_Town'],how='inner')
LOCATION_INFO = pd.merge(LOCATION_INFO,LOCATION_SUMMARY[['Shoot_Location','Shoot_Town','Shoot_City']],on=['Shoot_Location','Shoot_Town'],how='left')
LOCATION_INFO['scen_avai'] = LOCATION_INFO[['Scene_Number','Availability']].apply(tuple, axis=1)
LOCATION_INFO['town_avai'] = LOCATION_INFO[['Shoot_Town','Availability']].apply(tuple, axis=1)
LOCATION_INFO['town_city_avai'] = LOCATION_INFO[['Shoot_Town','Shoot_City','Availability']].apply(tuple, axis=1)
LOCATION_INFO['loc_avai'] = LOCATION_INFO[['Shoot_Location','Availability']].apply(tuple, axis=1)
LOCATION_INFO['city_avai'] = LOCATION_INFO[['Shoot_City','Availability']].apply(tuple, axis=1)
LOCATION_INFO['lsn'] = LOCATION_INFO[['Shoot_Location','Scene_Number','Availability']].apply(tuple, axis=1)
LOCATION_INFO['clsn'] = LOCATION_INFO[['Shoot_City','Shoot_Location','Scene_Number','Availability']].apply(tuple, axis=1)
LOCATION_INFO['ctsn'] = LOCATION_INFO[['Shoot_City','Shoot_Town','Scene_Number','Availability']].apply(tuple, axis=1)
LOCATION_INFO['tsn'] = LOCATION_INFO[['Shoot_Town','Scene_Number','Availability']].apply(tuple, axis=1)
LOCATION_INFO['cij'] = LOCATION_INFO[['Shoot_City','Shoot_Town','Shoot_Location']].apply(tuple, axis=1)
LOCATION_INFO['ij'] = LOCATION_INFO[['Shoot_Town','Shoot_Location']].apply(tuple, axis=1)
LOCATION_INFO['sj'] = LOCATION_INFO[['Scene_Number','Shoot_Location']].apply(tuple, axis=1)
LOCATION_INFO['sc'] = LOCATION_INFO[['Scene_Number','Shoot_City']].apply(tuple, axis=1)
LOCATION_INFO['scj'] = LOCATION_INFO[['Scene_Number','Shoot_City','Shoot_Location']].apply(tuple, axis=1)
LOCATION_INFO['ijn'] = LOCATION_INFO[['Shoot_Town','Shoot_Location','Availability']].apply(tuple, axis=1)


from functools import reduce

scene_list = ACTOR_INFO.Scene_Number.unique().tolist()
scene_list.sort()

df1 = ACTOR_INFO.groupby(by=["Scene_Number"])

# Compute (Scene_Number, Availability) pairs where all required actors are available
Actor_avail = []
for i in scene_list:
    df2 = df1.get_group(i)
    actor_to_days = df2.groupby("Actors")["Availability"].apply(lambda s: set(s.tolist()))
    if len(actor_to_days) == 0:
        continue
    common_days = set.intersection(*actor_to_days.tolist()) if len(actor_to_days) > 0 else set()
    for day in sorted(common_days):
        Actor_avail.append((i, day))
location_avail = LOCATION_INFO.scen_avai.unique().tolist()
def Intersection(lst1, lst2):
    return set(lst2).intersection(lst1)

# Apply time windows if present (Earliest/Latest per scene)
windows_df = pd.read_excel(path, sheet_name="WINDOWS") if "WINDOWS" in xl.sheet_names else pd.DataFrame()
if not windows_df.empty:
    windows_map = windows_df.set_index("Scene_Number")[['Earliest_Day','Latest_Day']].to_dict('index')
    def in_window(pair):
        s, n = pair
        w = windows_map.get(s)
        if w is None:
            return True
        return (n >= int(w['Earliest_Day'])) and (n <= int(w['Latest_Day']))
    actor_loc_pairs = Intersection(Actor_avail,location_avail)
    fin_lis = [p for p in actor_loc_pairs if in_window(p)]
else:
    fin_lis = Intersection(Actor_avail,location_avail)
FINAL_ACTOR_INFO = ACTOR_INFO.copy()
FINAL_ACTOR_INFO = FINAL_ACTOR_INFO[FINAL_ACTOR_INFO['scen_avai'].isin(fin_lis)]
FINAL_LOCATION_INFO = LOCATION_INFO.copy()
FINAL_LOCATION_INFO = FINAL_LOCATION_INFO[FINAL_LOCATION_INFO['scen_avai'].isin(fin_lis)]

FINAL_LOCATION_INFO

"""## Defining Sets"""

model.i = Set(initialize=LOCATION_SUMMARY["Shoot_Town"].tolist())
model.j = Set(initialize=LOCATION_SUMMARY["Shoot_Location"].tolist())
model.s = Set(initialize=SCENE_SUMMARY["Scene_Number"].tolist())
model.a = Set(initialize=ACTOR_SUMMARY["Actors"].tolist())
model.n = Set(initialize=CALENDAR['D_ID'].tolist())
model.an = Set(initialize=FINAL_ACTOR_INFO["actor_avai"].tolist())
model.sa = Set(initialize=FINAL_ACTOR_INFO["scene_actor"].tolist())
model.asn = Set(initialize=FINAL_ACTOR_INFO["asn"].tolist())
model.jn = Set(initialize=FINAL_LOCATION_INFO["loc_avai"].tolist())
model.jsn = Set(initialize=FINAL_LOCATION_INFO["lsn"].tolist())
model.tn = Set(initialize=FINAL_LOCATION_INFO["town_avai"].tolist())
model.isn = Set(initialize=FINAL_LOCATION_INFO["tsn"].tolist())
model.ij = Set(initialize=FINAL_LOCATION_INFO["ij"].tolist())
model.sj = Set(initialize=FINAL_LOCATION_INFO["sj"].tolist())
# model.split = Set(initialize=SCENE_SUMMARY["Split"].tolist())
# shoot city
model.clsn = Set(initialize=FINAL_LOCATION_INFO["clsn"].tolist())
model.ctsn = Set(initialize=FINAL_LOCATION_INFO["ctsn"].tolist())
model.cij = Set(initialize=FINAL_LOCATION_INFO["cij"].tolist())
model.scj = Set(initialize=FINAL_LOCATION_INFO["scj"].tolist())
model.cn = Set(initialize=FINAL_LOCATION_INFO["city_avai"].tolist())
model.c = Set(initialize=FINAL_LOCATION_INFO["Shoot_City"].tolist())
model.ctn = Set(initialize=FINAL_LOCATION_INFO["town_city_avai"].tolist())
model.ijn = Set(initialize=FINAL_LOCATION_INFO["ijn"].tolist())

# Precedence set (pairs of scenes) if present
if "PRECEDENCE" in xl.sheet_names:
  PRECEDENCE = pd.read_excel(path, sheet_name="PRECEDENCE")
  if not PRECEDENCE.empty:
    model.prec = Set(dimen=2, initialize=[(int(r.Pre), int(r.Post)) for _, r in PRECEDENCE.iterrows()])
  else:
    model.prec = Set(dimen=2, initialize=[])
else:
  model.prec = Set(dimen=2, initialize=[])

"""## Define the parameters"""

set_ele1 = pd.Series(ACTOR_SUMMARY.Criticality.values,index=ACTOR_SUMMARY.Actors).to_dict()
model.CA = Param(model.a,initialize=set_ele1)
model.CA.pprint()

set_elem = pd.Series(SCENE_TIME.Shoot_time.values,index=SCENE_TIME.Scene_Number).to_dict()
model.TS = Param(model.s,initialize=set_elem)
model.TS.pprint()

set_elex = pd.Series(LOCATION_SUMMARY.Criticality.values,index=LOCATION_SUMMARY.Shoot_Location).to_dict()
model.CL = Param(model.j,initialize=set_elex)
model.CL.pprint()

model.Bsa = Param(model.sa, initialize=1)
model.Bsa.pprint()

model.Can = Param(model.an, initialize=1)
model.Can.pprint()

model.Asj = Param(model.sj, initialize=1)
model.Asj.pprint()

model.Djn = Param(model.jn, initialize=1)
model.Djn.pprint()

model.Eij = Param(model.ij, initialize=1)
model.Eij.pprint()

model.Ecij = Param(model.cij, initialize=1)
model.Ecij.pprint()

set_ele4 = pd.Series(SCENE_TIME.Shoot_time.values,index=SCENE_TIME.Scene_Number).to_dict()
model.Tsh = Param(model.s ,initialize=set_ele4)
model.Tsh.pprint()

# Location capacity per day (default 1)
if "LOCATION_CAPACITY" in xl.sheet_names:
  cap_df = pd.read_excel(path, sheet_name="LOCATION_CAPACITY")
  cap_map = pd.Series(cap_df.Daily_Capacity.values, index=cap_df.Shoot_Location).to_dict()
else:
  cap_map = {j:1 for j in LOCATION_SUMMARY["Shoot_Location"].tolist()}
model.CapJ = Param(model.j, initialize=cap_map, within=NonNegativeIntegers)
model.CapJ.pprint()

# Actor daily wage (optional)
if "ACTOR_COSTS" in xl.sheet_names:
  costs_df = pd.read_excel(path, sheet_name="ACTOR_COSTS")
  wage_map = pd.Series(costs_df.Daily_Wage.values, index=costs_df.Actors).to_dict()
else:
  wage_map = {a: 0.0 for a in ACTOR_SUMMARY["Actors"].tolist()}
model.Wage = Param(model.a, initialize=wage_map, within=NonNegativeReals)
model.Wage.pprint()

# Transfers (average time and cost for surrogate travel modeling)
if "TRANSFERS" in xl.sheet_names:
  tf_df = pd.read_excel(path, sheet_name="TRANSFERS")
  try:
    avg_tt = float(tf_df["Travel_Time"].mean()) if not tf_df.empty else 0.0
    avg_tc = float(tf_df["Travel_Cost"].mean()) if not tf_df.empty else 0.0
  except Exception:
    avg_tt, avg_tc = 0.0, 0.0
else:
  avg_tt, avg_tc = 0.0, 0.0
model.AvgTravelTime = Param(initialize=avg_tt)
model.AvgTravelCost = Param(initialize=avg_tc)
print("Avg travel time:", value(model.AvgTravelTime), "Avg travel cost:", value(model.AvgTravelCost))

# Scene deadlines and penalties
if "SCENE_POLICY" in xl.sheet_names:
  pol_df = pd.read_excel(path, sheet_name="SCENE_POLICY")
  dl_map = {}
  pen_map = {}
  for _, r in pol_df.iterrows():
    s = int(r["Scene_Number"]) if not pd.isna(r["Scene_Number"]) else None
    if s is None:
      continue
    dl = 0 if pd.isna(r.get("Deadline_Day")) else int(r.get("Deadline_Day"))
    pe = 0.0 if pd.isna(r.get("Late_Penalty_per_Day")) else float(r.get("Late_Penalty_per_Day"))
    dl_map[s] = dl
    pen_map[s] = pe
else:
  dl_map = {int(s): 0 for s in SCENE_SUMMARY["Scene_Number"].tolist()}
  pen_map = {int(s): 0.0 for s in SCENE_SUMMARY["Scene_Number"].tolist()}
model.Deadline = Param(model.s, initialize=dl_map, within=NonNegativeIntegers)
model.LatePenalty = Param(model.s, initialize=pen_map, within=NonNegativeReals)
model.Deadline.pprint()
model.LatePenalty.pprint()

"""## Define Variables"""

model.Vijn = Var(model.ijn, within=Binary)
model.Vijn.pprint()

model.Wjn = Var(model.jn, within=Binary)
model.Wjn.pprint()

model.Usn = Var(fin_lis,within=Binary)
model.Usn.pprint()

model.Vasn = Var(model.asn, within=Binary)
model.Vasn.pprint()

model.Wcin = Var( model.ctn ,within=Binary)
model.Wcin.pprint()

model.Wcn = Var( model.cn ,within=Binary)
model.Wcn.pprint()

# model.Wisn = Var( model.jsn ,within=Binary)
# model.Wisn.pprint()

model.Win = Var(model.tn,within=Binary)
model.Win.pprint()

model.Xjsn = Var( model.jsn ,within=Binary)
model.Xjsn.pprint()

model.Dursn = Var( model.s,model.n)
model.Dursn.pprint()

model.AVsn = Var( model.s,model.n)
model.AVsn.pprint()

model.SHsn = Var( model.s,model.n)
model.SHsn.pprint()

model.DELsn =Var(model.s,model.n, within=Binary)
model.DELsn.pprint()

# Scene day assignment variable (derived day as integer)
model.DayOfScene = Var(model.s, within=NonNegativeIntegers)

# Location usage indicator per day
model.ZjnUse = Var(model.jn, within=Binary)

"""## Define Constraints"""

# Initialize a dictionary to hold the count of days available for each scene
scene_day_count = {scene: 0 for scene in model.s}

# Iterate over the Usn set to count the available days for each scene
for scene_day_pair in model.Usn:
    scene = scene_day_pair[0]
    scene_day_count[scene] += 1

# Check for any scene with zero available days
scenes_with_no_days = [scene for scene, count in scene_day_count.items() if count == 0]

# Output the scenes with no available days
print("Scenes with no scheduling days:", scenes_with_no_days)

# Summation over n across Usn ==1
# def rule2(model, s):
#   return sum(model.Usn[s,iter] for iter in model.n if(s,iter) in model.Usn) == 1
# model.const1 = Constraint( model.s, rule=rule2 )
# model.const1.pprint()

from pyomo.environ import Constraint, ConstraintList

def rule2(model, s):
    # Your original constraint logic
    constraint_expr = sum(model.Usn[s,iter] for iter in model.n if (s,iter) in model.Usn) == 1
    if constraint_expr is False:
        return Constraint.Infeasible
    else:
        return constraint_expr

model.const1 = Constraint(model.s, rule=rule2)
model.const1.pprint()

# Link DayOfScene to chosen day: DayOfScene[s] == sum_n n * Usn[s,n]
def day_link_rule(model, s):
  return model.DayOfScene[s] == sum(n * model.Usn[s,n] for n in model.n if (s,n) in model.Usn)
model.const_day_link = Constraint(model.s, rule=day_link_rule)
model.const_day_link.pprint()

# Late indicator DELsn and lateness logic
def late_rule(model, s, n):
  if (s,n) not in model.DELsn:
    return Constraint.Skip
  # DELsn[s,n] == 1 if DayOfScene[s] >= n
  return model.DELsn[s,n] >= (model.DayOfScene[s] - n) / max(1, max(value(nn) for nn in model.n))
model.const_late = Constraint(model.s, model.n, rule=late_rule)
model.const_late.pprint()

# Summation over s Usn*TS(s) <= WT
sc_num = []
for iter1 in model.s:
  for iter2 in model.n:
    if (iter1,iter2) in model.Usn:
      sc_num.append(iter2)

def rule10(model,n):
        return sum(model.Usn[iter,n]*model.TS[iter] for iter in model.s if (iter,n) in model.Usn) <= 8
model.const2 = Constraint(sc_num, rule = rule10)
model.const2.pprint()

#Vasn == Usn*Bsa*Cna
model.const3 = ConstraintList()
for iter1 in model.n:
  for iter2 in model.s:
     for iter3 in model.a:
         if (iter3,iter2,iter1) in model.Vasn and (iter2,iter1) in model.Usn and (iter3,iter1) in model.Can and (iter2,iter3) in model.Bsa:
           model.const3.add(model.Vasn[iter3,iter2,iter1] == model.Usn[iter2,iter1] * model.Bsa[iter2,iter3] * model.Can[iter3,iter1])
         else:
           continue
model.const3.pprint()

#Xjsn == Usn * Asj *Djn
model.const4 = ConstraintList()
for iter1 in model.j:
    for iter2 in model.n:
      for iter3 in model.s:
        if (iter1,iter3,iter2) in model.Xjsn and (iter1,iter2) in model.Djn and (iter3,iter1) in model.Asj:
          model.const4.add(model.Xjsn[iter1,iter3,iter2] == model.Usn[iter3,iter2]*model.Asj[iter3,iter1]*model.Djn[iter1,iter2])
        else:
          continue

model.const4.pprint()

#Xjsn <= Win * Eij * Djn
model.const5 = ConstraintList()
for iter1 in model.j:
  for iter2 in model.s:
     for iter3 in model.n:
       for iter4 in model.i:
         if (iter1,iter2,iter3) in model.Xjsn and (iter4,iter3) in model.Win and (iter1,iter3) in model.Djn and (iter4,iter1) in model.Eij:
           model.const5.add(model.Xjsn[iter1,iter2,iter3] <= model.Win[iter4,iter3] * model.Eij[iter4,iter1]* model.Djn[iter1,iter3])
         else:
           continue

model.const5.pprint()

# Location capacity per day: sum_s Xjsn[j,s,n] <= CapJ[j]
model.const_capacity = ConstraintList()
for j in model.j:
  for n in model.n:
    terms = [model.Xjsn[j,s,n] for s in model.s if (j,s,n) in model.Xjsn]
    if len(terms) == 0:
      model.const_capacity.add(Constraint.Feasible)
    else:
      model.const_capacity.add(sum(terms) <= model.CapJ[j])
model.const_capacity.pprint()

# Summation over i across Win <=1
sc_num1 = []
for iter in model.tn:
  if (iter) in model.tn:
    sc_num1.append(iter[1])


def rule4(model, n):
  return sum(model.Win[iter1,n] for iter1 in model.i if(iter1,n) in model.tn) <= 1
model.const6 = Constraint(sc_num1, rule=rule4)
model.const6.pprint()

model.const7 = ConstraintList()
for iter1 in model.i:
  for iter3 in model.i:
    for iter2 in model.n:
      if iter2 != 1 and iter1!=iter3:
        if (iter1,iter2) in model.Win and (iter3,iter2-1) in model.Win :
          model.const7.add(model.Win[iter1,iter2]<= 1 - model.Win[iter3,iter2-1])
      else:
        continue

model.const7.pprint()

# Precedence constraints: DayOfScene[post] >= DayOfScene[pre]
model.const_prec = ConstraintList()
for (pre, post) in list(model.prec.value) if hasattr(model.prec, 'value') else []:
  if pre in model.s and post in model.s:
    model.const_prec.add(model.DayOfScene[post] >= model.DayOfScene[pre])
model.const_prec.pprint()

# Location usage activation: if any Xjsn[j,s,n]==1 then ZjnUse[j,n]==1
model.const_loc_use = ConstraintList()
for j in model.j:
  for n in model.n:
    terms = [model.Xjsn[j,s,n] for s in model.s if (j,s,n) in model.Xjsn]
    if len(terms) == 0:
      continue
    model.const_loc_use.add(sum(terms) <= len(terms) * model.ZjnUse[j,n])
model.const_loc_use.pprint()

"""## Objective Function"""

model.CL.pprint()

model.obj1 = Objective( expr = 
    # original priority-like terms
    sum(model.Xjsn[idx]*(1/model.CL[idx[0]]) for idx in model.jsn)
  + sum(model.Vasn[idx]*(1/model.CA[idx[0]]) for idx in model.asn)
  # actor wage cost (one day when present)
  + sum(model.Vasn[a,s,n] * model.Wage[a] for (a,s,n) in model.asn)
  # average travel cost for using any location/day
  + sum(model.ZjnUse[j,n] * model.AvgTravelCost for (j,n) in model.jn)
  # late penalties (approximate via cumulative DELsn up to deadline)
  + sum(model.LatePenalty[s] * sum(model.DELsn[s,n] for n in model.n if n > model.Deadline[s]) for s in model.s)
)
model.obj1.pprint()

"""## Solver"""

# Prefer CBC if available; otherwise try HiGHS (appsi) or highs
solver = None
preferred_solvers = ['cbc', 'appsi_highs', 'highs']
for name in preferred_solvers:
    try:
        sf = SolverFactory(name)
        if sf is not None and sf.available():
            solver = sf
            break
    except Exception:
        continue

if solver is None:
    raise RuntimeError("No suitable MILP solver found. Install CBC or HiGHS. For macOS: brew install cbc or pip install highspy")

resolve = solver.solve(model)

resolve.write()

print(model.obj1())

"""## Final Output"""

import numpy as np
day_num1 = []
loc = []
scene_nums1 = []
for x in model.jsn:
  if model.Xjsn[x]() == 1.0:
    #print(model.Xjsn[x])
    day_num1.append(x[2])
    loc.append(x[0])
    scene_nums1.append(x[1])
res_df2 =  pd.DataFrame(
    {'D_ID': day_num1,
     'Location': loc,
     'Scene_Num': scene_nums1
    })


day_num = []
actor = []
scene_nums = []
for x in model.asn:
  if model.Vasn[x]() == 1.0:
    day_num.append(x[2])
    actor.append(x[0])
    scene_nums.append(x[1])

res_df1 =  pd.DataFrame(
    {'D_ID': day_num,
     'Actor': actor,
     'Scene_Num': scene_nums
    })

numbers = []
town = []
for iter1 in model.tn:
  if model.Win[iter1]() == 1.0:
    numbers.append(iter1[1])
    town.append(iter1[0])

res_df3 =  pd.DataFrame(
    {'D_ID': numbers,
     'Shoot_Town': town
    })


res_df4 = pd.merge(res_df2,res_df3,on='D_ID',how='inner')
res_df5 = pd.merge(res_df1,res_df4,on=['D_ID','Scene_Num'],how='inner')
final_dff = pd.merge(res_df5,CALENDAR,on='D_ID',how='inner')
final_dff.drop(['Holiday'], axis=1, inplace=True)
final_dff.to_csv("output_pyomo.csv")

final_dff


"""## Verification Summary"""

# 1) Each scene scheduled exactly once
_scene_counts = final_dff.groupby("Scene_Num")["D_ID"].nunique()
_viol_1 = _scene_counts[_scene_counts != 1]

# 2) Day capacity ≤ 8 hours (deduplicate scene-day before summing)
_unique_scene_day = final_dff.drop_duplicates(subset=["D_ID","Scene_Num"])\
    .merge(SCENE_TIME, left_on="Scene_Num", right_on="Scene_Number", how="left")
_day_hours = _unique_scene_day.groupby("D_ID")["Shoot_time"].sum()
_viol_2 = _day_hours[_day_hours > 8]

# 3) Actor availability on assigned day
_actor_ok = final_dff.merge(ACTOR_AVAILABILITY, left_on=["Actor","D_ID"], right_on=["Actors","Availability"], how="left")
_viol_3 = _actor_ok[_actor_ok["Actors"].isna()]

# 3b) Actor assigned to that scene
_actor_scene_ok = final_dff.merge(ACTOR_SCENES, left_on=["Actor","Scene_Num"], right_on=["Actors","Scene_Number"], how="left")
_viol_3b = _actor_scene_ok[_actor_scene_ok["Actors"].isna()]

# 4) Location availability on assigned day
_loc_ok = final_dff.merge(LOCATION_AVAILABILITY, left_on=["Location","D_ID"], right_on=["Shoot_Location","Availability"], how="left")
_viol_4 = _loc_ok[_loc_ok["Shoot_Location"].isna()]

# 5) One town per day
_viol_5 = final_dff.groupby("D_ID")["Shoot_Town"].nunique()
_viol_5 = _viol_5[_viol_5 > 1]

# 6) Approx objective for sanity (sum 1/CL + 1/CA over rows)
_obj_part_loc = final_dff.merge(LOCATION_SUMMARY[["Shoot_Location","Criticality"]], left_on="Location", right_on="Shoot_Location", how="left")
_obj_loc = (1/_obj_part_loc["Criticality"]).sum()
_obj_part_act = final_dff.merge(ACTOR_SUMMARY[["Actors","Criticality"]], left_on="Actor", right_on="Actors", how="left")
_obj_act = (1/_obj_part_act["Criticality"]).sum()
_objective_estimate = float(_obj_loc + _obj_act)

print("\n=== Verification Summary ===")
print("- Each scene exactly once:", "OK" if _viol_1.empty else f"Bad -> {_viol_1.to_dict()}")
print("- Day ≤ 8h:", "OK" if _viol_2.empty else f"Bad -> {_viol_2.to_dict()}")
print("- Actor availability:", "OK" if _viol_3.empty else f"Bad -> {_viol_3[['Actor','Scene_Num','D_ID']].head().to_dict('records')}")
print("- Actor assigned to scene:", "OK" if _viol_3b.empty else f"Bad -> {_viol_3b[['Actor','Scene_Num']].head().to_dict('records')}")
print("- Location availability:", "OK" if _viol_4.empty else f"Bad -> {_viol_4[['Location','Scene_Num','D_ID']].head().to_dict('records')}")
print("- One town per day:", "OK" if _viol_5.empty else f"Bad -> {_viol_5.to_dict()}")
print(f"- Objective (approx): {_objective_estimate}")


"""## Freeze Baseline Artifacts"""
import json
from datetime import datetime

_art_root = os.path.join(os.path.dirname(path), "artifacts")
os.makedirs(_art_root, exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_run_dir = os.path.join(_art_root, f"baseline_{_ts}")
os.makedirs(_run_dir, exist_ok=True)

# Save CSV copy
final_dff.to_csv(os.path.join(_run_dir, "output_pyomo.csv"), index=False)

# Capture solver info if available
_solver_name = None
try:
  _solver_name = solver.name if hasattr(solver, "name") else str(solver)
except Exception:
  _solver_name = None

_termination = None
try:
  _termination = str(resolve.solver.termination_condition)
except Exception:
  _termination = None

_meta = {
  "timestamp": _ts,
  "data_path": path,
  "solver": _solver_name,
  "termination_condition": _termination,
  "objective_value": float(model.obj1()),
  "verification": {
    "each_scene_once": _viol_1.empty,
    "day_hours_leq_8": _viol_2.empty,
    "actor_availability": _viol_3.empty,
    "actor_scene_membership": _viol_3b.empty,
    "location_availability": _viol_4.empty,
    "one_town_per_day": _viol_5.empty,
    "objective_estimate": _objective_estimate,
  },
}

with open(os.path.join(_run_dir, "baseline_run.json"), "w") as f:
  json.dump(_meta, f, indent=2)

print(f"Saved baseline artifacts to: {_run_dir}")