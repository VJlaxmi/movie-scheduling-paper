# Synthetic Data Generator for Movie Scene Scheduling

This package generates synthetic datasets for the Movie Scene Scheduling Problem (MSSP) following the methodology from research papers. It creates realistic test instances that can be used to develop and test genetic algorithms and other optimization methods.

## Overview

The synthetic data generator creates datasets with all the parameters needed for movie scene scheduling optimization:

1. **Set of Scenes** - Individual movie scenes to be scheduled
2. **Set of Actors** - Available actors for the production
3. **Set of Locations** - Shooting locations (cities and venue types)
4. **Daily wage of each actor** - Cost per day for each actor
5. **Actor-Scene Mapping** - Which actors are required for each scene
6. **Scene-Location Mapping** - Which locations are compatible with each scene
7. **Location cost for each day** - Daily cost for using each location
8. **Cost of moving from Scene i to Scene j** - Transfer costs between scenes
9. **Number of days required to shoot Scene i** - Duration of each scene

## Files

- `synthetic_data_generator.py` - Main data generator class
- `validate_synthetic_data.py` - Data validation and consistency checking
- `test_synthetic_data.py` - Test suite with example usage
- `SYNTHETIC_DATA_README.md` - This documentation

## Quick Start

### Basic Usage

```python
from synthetic_data_generator import SyntheticDataGenerator

# Create generator with fixed seed for reproducibility
generator = SyntheticDataGenerator(seed=42)

# Generate a medium-sized instance
data = generator.generate_parameters(
    num_scenes=10,
    num_actors=5,
    num_locations=3,
    num_days=20
)

# Export to Excel and JSON
generator.export_to_excel(data, "my_dataset.xlsx")
generator.export_to_json(data, "my_dataset.json")
```

### Generate Multiple Instances

```python
# Generate multiple test instances of different sizes
generator.generate_multiple_instances(
    instance_sizes=[
        (5, 3, 2),   # Small: 5 scenes, 3 actors, 2 locations
        (10, 5, 3),  # Medium: 10 scenes, 5 actors, 3 locations
        (15, 7, 4),  # Large: 15 scenes, 7 actors, 4 locations
        (20, 10, 5)  # Extra Large: 20 scenes, 10 actors, 5 locations
    ],
    output_dir="test_instances"
)
```

### Validate Generated Data

```python
from validate_synthetic_data import SyntheticDataValidator

validator = SyntheticDataValidator()
is_valid, errors, warnings = validator.validate_excel_file("my_dataset.xlsx")
validator.print_validation_report(is_valid, errors, warnings)
```

## Data Generation Methodology

The generator follows the approach from research papers:

### Paper#1 Approach
- **Transfer Costs**: Random values in range (0, 10000) with symmetry (c_ij = c_ji)
- **Actor Wages**: Random values in range (80, 1000) dollars per day
- **Actor-Scene Assignment**: 1-3 actors per scene, randomly assigned
- **Data Structure**: Completely synthetic with random generation

### Paper#2 Approach
- **Transfer Costs**: Random values in range (0, 10000) with symmetry
- **Actor Wages**: Random values in range (80, 100) CHY per day
- **Location Assignment**: Random assignment of scenes to locations
- **Time Intervals**: Random shooting and transition times

### Enhanced Parameters
- **Scene Durations**: Random values in specified range (1-5 days typical)
- **Location Costs**: Daily costs for using each location
- **Scene Precedence**: Random precedence relationships (30% probability)
- **Actor Availability**: Simplified (available all days)

## Generated Data Structure

### Excel Format
The generator creates Excel files with multiple sheets:
- `METADATA` - Generation parameters and metadata
- `SCENES` - Scene list with durations
- `ACTORS` - Actor list with daily wages
- `LOCATIONS` - Location list with daily costs
- `ACTOR_SCENES` - Actor-scene assignment matrix
- `SCENE_LOCATIONS` - Scene-location compatibility matrix
- `TRANSFER_COSTS` - Transfer cost matrix between scenes
- `SCENE_PRECEDENCE` - Precedence constraints (if any)
- `ACTOR_AVAILABILITY` - Actor availability calendar

### JSON Format
The generator also creates JSON files with the same data in programmatic format for easy integration with algorithms.

## Configuration Options

### Basic Parameters
- `num_scenes`: Number of scenes to generate
- `num_actors`: Number of actors available
- `num_locations`: Number of shooting locations
- `num_days`: Total shooting days available

### Cost Ranges
- `wage_range`: (min, max) daily wage for actors
- `transfer_cost_range`: (min, max) cost for scene transitions
- `location_cost_range`: (min, max) daily cost per location

### Scene Parameters
- `scene_duration_range`: (min, max) days required per scene

## Example Output

### Small Instance (5 scenes, 3 actors, 2 locations)
```
Scenes: ['Scene_01', 'Scene_02', 'Scene_03', 'Scene_04', 'Scene_05']
Actors: ['Actor_A', 'Actor_B', 'Actor_C']
Locations: ['Mumbai_Mountain_1', 'Delhi_Village_2']
Actor wages: {'Actor_A': 206.62, 'Actor_B': 435.4, 'Actor_C': 449.93}
Scene durations: {'Scene_01': 3, 'Scene_02': 2, 'Scene_03': 2, 'Scene_04': 3, 'Scene_05': 1}
Location costs: {'Mumbai_Mountain_1': 674.04, 'Delhi_Village_2': 212.74}
```

### Data Statistics
- Average actors per scene: 1.80
- Average locations per scene: 1.20
- Transfer cost range: 428.33 - 895.59
- Average scene duration: 2.20 days

## Validation

The validator checks for:
- ✅ **Data Structure**: Required keys and proper format
- ✅ **Parameter Ranges**: Valid wage, cost, and duration values
- ✅ **Mapping Consistency**: Valid actor-scene and scene-location assignments
- ✅ **Cost Symmetry**: Transfer costs are symmetric (c_ij = c_ji)
- ✅ **Feasibility**: Every scene has at least one actor and one compatible location

## Usage for Genetic Algorithm Development

The generated data is perfect for developing genetic algorithms:

1. **Chromosome Representation**: Use scene ordering as chromosome
2. **Fitness Function**: Calculate total cost using generated parameters
3. **Constraints**: Validate against actor availability and location compatibility
4. **Operators**: Implement crossover and mutation for scene sequences

### Example Integration

```python
# Load generated data
import pandas as pd
data = pd.read_excel("my_dataset.xlsx", sheet_name=None)

# Extract parameters for GA
scenes = data['SCENES']['Scene_ID'].tolist()
actors = data['ACTORS']['Actor_ID'].tolist()
locations = data['LOCATIONS']['Location_ID'].tolist()
actor_wages = dict(zip(data['ACTORS']['Actor_ID'], data['ACTORS']['Daily_Wage']))
transfer_costs = dict(zip(data['TRANSFER_COSTS']['From_Scene'], 
                         data['TRANSFER_COSTS']['To_Scene']))

# Use in your genetic algorithm
def fitness_function(chromosome):
    # chromosome is a list of scene indices
    total_cost = 0
    # Calculate costs using generated parameters
    # ... your GA implementation
    return total_cost
```

## Reproducibility

All generated data is reproducible using fixed seeds:
- Same seed → Same dataset
- Different seeds → Different but consistent datasets
- Metadata includes generation parameters for full reproducibility

## Comparison with Real Data

| Aspect | Synthetic Data | Real Data |
|--------|----------------|-----------|
| **Reproducibility** | Perfect (fixed seeds) | Limited |
| **Scalability** | Easy to generate any size | Fixed size |
| **Complexity** | Configurable | Fixed complexity |
| **Realism** | Moderate | High |
| **Validation** | Automated | Manual |

## Future Enhancements

- [ ] More realistic actor availability patterns
- [ ] Weather-dependent location costs
- [ ] Equipment requirements per scene
- [ ] Budget constraints and penalties
- [ ] Multi-objective optimization support

## Dependencies

- `pandas>=2.0`
- `numpy>=1.24`
- `openpyxl>=3.1`
- `matplotlib>=3.7` (for plotting, optional)

## License

This synthetic data generator is part of the Movie Scheduling Research Project and follows the same license terms.
