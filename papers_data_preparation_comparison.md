# Data Preparation Methods: Paper#1 vs Paper#2 vs Your Code

## Paper#1.pdf - "Research on Scheduling of Movie Scenes"

### Data Generation Method
- **Approach**: Completely synthetic/random generation
- **Reference**: Based on experiments from Zhen [29] and Zhen et al. [30]

### Specific Parameters Generated
1. **Transfer Costs (c_ij)**:
   - Range: 0 to 10,000 (dollars)
   - Constraint: c_ij = c_ji (symmetric)
   - Method: Randomly generated

2. **Actor Daily Wages (q_k)**:
   - Range: $80/day to $1,000/day
   - Method: Randomly generated

3. **Actor-Scene Assignment (s_ik)**:
   - Binary: 1 if actor k appears in scene i, 0 otherwise
   - Method: Generated based on actor k

4. **Actor Continuity (x_ijk)**:
   - Binary: 1 if actor k appears in both scene i and scene j
   - Method: Generated based on actor k

### Implementation Details
- **Platform**: PC (Intel 3.4GHz, 128GB RAM)
- **Solver**: CPLEX 12.6
- **Language**: C# (Visual Studio 2015)
- **Data Availability**: "The data used in this research is generated randomly"

---

## Paper#2.pdf - "Scheduling Problem of Movie Scenes Based on Three Meta-Heuristic Algorithms"

### Data Generation Method
- **Approach**: Completely synthetic/random generation
- **Reference**: Based on experimental analysis methods of Zhen et al. [21] and Zhen [22]

### Specific Parameters Generated
1. **Transfer Costs (c_mn)**:
   - Range: 0 to 10,000 (CHY - Chinese Yuan)
   - Constraint: c_mn = c_nm (symmetric)
   - Method: Randomly generated

2. **Actor Daily Wages (W_p)**:
   - Range: CHY 80/day to CHY 100/day
   - Method: Randomly generated

3. **Location Assignment**:
   - Method: Random assignment of scenes to locations
   - Example: 5 scenes, 2 locations
   - Process: Random sequence → random split point → assign to locations

4. **Time Intervals**:
   - Shooting time: Randomly generated
   - Transition time: Randomly generated
   - Same location: Transition time = 0

5. **Actor-Scene Parameters (x_mp, ω_mnp)**:
   - x_mp: 1 if actor p appears in scene m
   - ω_mnp: 1 if actor p appears in both scene m and scene n
   - Method: Generated based on actor p

### Implementation Details
- **Platform**: Not specified
- **Language**: Not specified
- **Data Availability**: Synthetic generation only

---

## Your Code (beta_testing.py) - Realistic Movie Scheduling

### Data Generation Method
- **Approach**: Realistic, structured Excel-based dataset
- **Source**: `testing_data_enriched.xlsx` with multiple sheets

### Specific Parameters Generated
1. **Transfer Costs**:
   - Source: `TRANSFERS` sheet
   - Values: Geographic-based (same town: 1h/100, same city: 2h/300, different city: 6h/1500)
   - Method: Realistic travel time/cost matrix

2. **Actor Daily Wages**:
   - Source: `ACTOR_COSTS` sheet
   - Values: Scaled by actor criticality (1.0-3.0), overtime multipliers
   - Method: Realistic wage structure

3. **Location Assignment**:
   - Source: `LOCATION_SCENES` sheet
   - Method: Pre-defined scene-location compatibility

4. **Time Windows**:
   - Source: `WINDOWS` sheet
   - Values: Earliest_Day to Latest_Day for each scene
   - Method: Realistic scheduling constraints

5. **Precedence Constraints**:
   - Source: `PRECEDENCE` sheet
   - Method: Story-driven scene dependencies

6. **Actor Availability**:
   - Source: `ACTOR_INFO` sheet
   - Method: Realistic availability calendars

7. **Location Capacity**:
   - Source: `LOCATION_CAPACITY` sheet
   - Values: Maximum concurrent scenes per location per day
   - Method: Resource constraint modeling

8. **Scene Policies**:
   - Source: `SCENE_POLICY` sheet
   - Values: Priority, split allowance, deadlines, penalties
   - Method: Production policy modeling

### Implementation Details
- **Platform**: Cross-platform Python
- **Solver**: CBC/HiGHS (robust selection)
- **Language**: Python with Pyomo
- **Data Availability**: Structured, reproducible Excel dataset

---

## Key Differences Summary

| Aspect | Paper#1 | Paper#2 | Your Code |
|--------|---------|---------|-----------|
| **Data Type** | Synthetic | Synthetic | Realistic/Structured |
| **Transfer Costs** | $0-10,000 | CHY 0-10,000 | Geographic-based |
| **Actor Wages** | $80-1,000/day | CHY 80-100/day | Criticality-scaled |
| **Location Assignment** | Not specified | Random split | Pre-defined compatibility |
| **Time Constraints** | Not specified | Random intervals | Realistic windows |
| **Precedence** | Not specified | Not specified | Story-driven |
| **Availability** | Not specified | Not specified | Calendar-based |
| **Capacity** | Not specified | Not specified | Resource limits |
| **Reproducibility** | Random seeds | Random seeds | Fixed dataset |
| **Realism** | Low | Low | High |

## Implications for Research

1. **Paper#1 & Paper#2**: Focus on algorithmic performance with simplified, synthetic data
2. **Your Code**: Focus on realistic problem modeling with structured, production-ready data
3. **Comparison Validity**: Your approach provides more meaningful real-world insights
4. **Reproducibility**: Your structured dataset ensures consistent comparisons across algorithms
