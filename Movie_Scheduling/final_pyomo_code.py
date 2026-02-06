path = ""
final_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Pyomo/Backend_Script/Final_DF.csv")

def pyomo_function(path, final_df):
    
    # Importing Libraries
    %matplotlib inline
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd
    from functools import reduce
    
    
    import shutil
    import sys
    import os.path
    
    if not shutil.which("pyomo"):
        !pip install -q pyomo
        assert(shutil.which("pyomo"))
    
    if not (shutil.which("cbc") or os.path.isfile("cbc")):
        if "google.colab" in sys.modules:
            !apt-get install -y -qq coinor-cbc
        else:
            try:
                !conda install -c conda-forge coincbc 
            except:
                pass
    
    assert(shutil.which("cbc") or os.path.isfile("cbc"))
    
    from pyomo.environ import *
    from pyomo.gdp import *
    
    
    # Model type
    model = ConcreteModel()
    
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
    
    #main_df = pd.DataFrame(["scen_avail"])
    main_list = []
    for i in scene_list:
        df2 = df1.get_group(i)
        actors_list = df2.Actors.unique().tolist()
        #print("---------------")
        
        list = []
        df3 = df2.groupby("Actors")
        for j in actors_list:
            list.append(df3.get_group(j))
                
        final_df = reduce(lambda  left,right: pd.merge(left,right,on=['Availability'], how='inner'), list)
        #print(final_df.shape)
        
        main_list.append(final_df.iloc[:,4:5].values)
    
    list1 =[]
    for i in range(len(main_list)):
      #print(i)
      list1.append(main_list[i].tolist())
    
    
    flat_list = [item for sublist in list1 for item in sublist]
    flat_list1 = [item for sublist in flat_list for item in sublist]
    Actor_avail = flat_list1
    location_avail = LOCATION_INFO.scen_avai.unique().tolist()
    def Intersection(lst1, lst2):
        return set(lst2).intersection(lst1)
    
    fin_lis = Intersection(Actor_avail,location_avail)
    FINAL_ACTOR_INFO = ACTOR_INFO.copy()
    FINAL_ACTOR_INFO = FINAL_ACTOR_INFO[FINAL_ACTOR_INFO['scen_avai'].isin(fin_lis)]
    FINAL_LOCATION_INFO = LOCATION_INFO.copy()
    FINAL_LOCATION_INFO = FINAL_LOCATION_INFO[FINAL_LOCATION_INFO['scen_avai'].isin(fin_lis)]
    
    
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
    
    set_ele1 = pd.Series(ACTOR_SUMMARY.Criticality.values,index=ACTOR_SUMMARY.Actors).to_dict()
    model.CA = Param(model.a,initialize=set_ele1)
    set_elem = pd.Series(SCENE_TIME.Shoot_time.values,index=SCENE_TIME.Scene_Number).to_dict()
    model.TS = Param(model.s,initialize=set_elem)model.Can = Param(model.an, initialize=1)
    set_elex = pd.Series(LOCATION_SUMMARY.Criticality.values,index=LOCATION_SUMMARY.Shoot_Location).to_dict()
    model.CL = Param(model.j,initialize=set_elex)
    model.Bsa = Param(model.sa, initialize=1)
    model.Can = Param(model.an, initialize=1)
    model.Asj = Param(model.sj, initialize=1)
    model.Djn = Param(model.jn, initialize=1)
    model.Eij = Param(model.ij, initialize=1)
    model.Ecij = Param(model.cij, initialize=1)
    set_ele4 = pd.Series(SCENE_TIME.Shoot_time.values,index=SCENE_TIME.Scene_Number).to_dict()
    model.Tsh = Param(model.s ,initialize=set_ele4)


    model.Vijn = Var(model.ijn, within=Binary)
    model.Wjn = Var(model.jn, within=Binary)
    model.Wcin = Var( model.ctn ,within=Binary)
    model.Wcn = Var( model.cn ,within=Binary)
    model.Dursn = Var( model.s,model.n)
    model.AVsn = Var( model.s,model.n)
    model.SHsn = Var( model.s,model.n)
    model.DELsn =Var(model.s,model.n, within=Binary)
    model.Usn = Var(fin_lis,within=Binary)
    model.Vasn = Var(model.asn, within=Binary)
    model.Win = Var(model.tn,within=Binary)
    model.Xjsn = Var( model.jsn ,within=Binary)
    
    
    # Summation over n across Usn ==1
    def rule2(model, s):
      return sum(model.Usn[s,iter] for iter in model.n if(s,iter) in model.Usn) == 1
    model.const1 = Constraint( model.s, rule=rule2 )
    # Summation over s Usn*TS(s) <= WT
    sc_num = [] 
    for iter1 in model.s:
      for iter2 in model.n:
        if (iter1,iter2) in model.Usn:
          sc_num.append(iter2)
    
    def rule10(model,n):
            return sum(model.Usn[iter,n]*model.TS[iter] for iter in model.s if (iter,n) in model.Usn) <= 8
    model.const2 = Constraint(sc_num, rule = rule10)
    
    #Vasn == Usn*Bsa*Cna
    model.const3 = ConstraintList()
    for iter1 in model.n:
      for iter2 in model.s:
         for iter3 in model.a:
             if (iter3,iter2,iter1) in model.Vasn and (iter2,iter1) in model.Usn and (iter3,iter1) in model.Can and (iter2,iter3) in model.Bsa:
               model.const3.add(model.Vasn[iter3,iter2,iter1] == model.Usn[iter2,iter1] * model.Bsa[iter2,iter3] * model.Can[iter3,iter1])
             else:
               continue
               
    #Xjsn == Usn * Asj *Djn
    model.const4 = ConstraintList()
    for iter1 in model.j:
        for iter2 in model.n:
          for iter3 in model.s:
            if (iter1,iter3,iter2) in model.Xjsn and (iter1,iter2) in model.Djn and (iter3,iter1) in model.Asj:
              model.const4.add(model.Xjsn[iter1,iter3,iter2] == model.Usn[iter3,iter2]*model.Asj[iter3,iter1]*model.Djn[iter1,iter2])
            else:
              continue
              
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
    # Summation over i across Win <=1
    sc_num1 = []
    for iter in model.tn:
      if (iter) in model.tn:
        sc_num1.append(iter[1])
    
    
    def rule4(model, n):
      return sum(model.Win[iter1,n] for iter1 in model.i if(iter1,n) in model.tn) <= 1
    model.const6 = Constraint(sc_num1, rule=rule4)
    
    model.const7 = ConstraintList()
    for iter1 in model.i:
      for iter3 in model.i:
        for iter2 in model.n:
          if iter2 != 1 and iter1!=iter3:
            if (iter1,iter2) in model.Win and (iter3,iter2-1) in model.Win :
              model.const7.add(model.Win[iter1,iter2]<= 1 - model.Win[iter3,iter2-1])
          else:
            continue
            
    model.obj1 = Objective( expr = sum( model.Xjsn[iter1]*(1/model.CL[iter1[0]]) for iter1 in model.jsn) + sum(model.Vasn[iter2]* (1/model.CA[iter2[0]]) for iter2 in model.asn))
    from pyomo.environ import *
    solver = SolverFactory('cbc')
    resolve = solver.solve(model)
    resolve.write()
    
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
     
    final_df = final_df.sort_values("D_ID")
    final_df = final_df.reindex(['Shoot_City','Shoot_Town','Scene_Num','Location','Actor','DATE',"D_ID"], axis=1)
    final_df.groupby((final_df['Shoot_City'].shift() != final_df['Shoot_City']).cumsum())
    dfs = []
    for k, v in final_df.groupby((final_df['Shoot_City'].shift() != final_df['Shoot_City']).cumsum()):
        v["Schedule"] = k
        dfs.append(v)
    final_df1 = pd.concat(dfs)
    final_df1 = final_df1.reindex(["Schedule",'Shoot_City','Shoot_Town','Scene_Num','Location','Actor','DATE',"D_ID"], axis=1)
    final_df2 = final_df1.groupby(['Schedule'])
    final_df2 = final_df2.agg(Minimum_Date=('DATE', np.min), Maximum_Date=('DATE', np.max))
    final_df2['Schedule'] = final_df2.index
    final_df2["list_of_tuples"] = final_df2.apply(tuple, axis=1)
    list_of_tuples = final_df2['list_of_tuples'].to_list()
    From_date = []
    To_date = []
    for row in final_df1['Schedule']:
        if row ==1 :    From_date.append(list_of_tuples[0][1]),To_date.append(list_of_tuples[0][2])
        elif row==2:   From_date.append(list_of_tuples[1][1]),To_date.append(list_of_tuples[1][2])
        elif row==3:   From_date.append(list_of_tuples[2][1]),To_date.append(list_of_tuples[2][2])
        elif row==4:   From_date.append(list_of_tuples[3][1]),To_date.append(list_of_tuples[3][2])
        elif row==5:   From_date.append(list_of_tuples[4][1]),To_date.append(list_of_tuples[4][2])
        elif row==6:   From_date.append(list_of_tuples[5][1]),To_date.append(list_of_tuples[5][2])
        elif row==7:   From_date.append(list_of_tuples[6][1]),To_date.append(list_of_tuples[6][2])
        else: From_date.append('NA'),To_date.append('NA')
    
    final_df1['From_Date'] = From_date
    final_df1['To_Date'] = To_date
    final_df1.to_csv("output_pyomo.csv")   
    return final_df1

        
    
            
            
    	