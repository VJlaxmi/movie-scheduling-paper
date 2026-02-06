#!/usr/bin/env python3
"""
Synthetic Data Generator for Movie Scene Scheduling Problem
Based on Paper#1 and Paper#2 approaches with enhanced parameters

This generator creates synthetic datasets following the methodology from:
- Paper#1: "Research on Scheduling of Movie Scenes" 
- Paper#2: "Scheduling Problem of Movie Scenes Based on Three Meta-Heuristic Algorithms"

Parameters Generated:
1. Set of Scenes
2. Set of Actors  
3. Set of Locations (Cities, locations)
4. Daily wage of each actor
5. Actor - Scene Mapping
6. Scene - Location Mapping
7. Location cost for each day
8. Cost of moving from Scene i to Scene j
9. Number of days required to shoot Scene i
"""

import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import json
import os
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        
    def generate_parameters(self, 
                          num_scenes: int = 10,
                          num_actors: int = 5, 
                          num_locations: int = 3,
                          num_days: int = 20,
                          wage_range: Tuple[float, float] = (80, 1000),
                          transfer_cost_range: Tuple[float, float] = (0, 10000),
                          scene_duration_range: Tuple[int, int] = (1, 5),
                          location_cost_range: Tuple[float, float] = (100, 2000)) -> Dict:
        """
        Generate all synthetic parameters for the movie scheduling problem
        
        Args:
            num_scenes: Number of scenes to generate
            num_actors: Number of actors
            num_locations: Number of shooting locations
            num_days: Total shooting days available
            wage_range: (min, max) daily wage for actors
            transfer_cost_range: (min, max) cost for scene transitions
            scene_duration_range: (min, max) days required per scene
            location_cost_range: (min, max) daily cost per location
        """
        
        print(f"Generating synthetic data with seed {self.seed}")
        print(f"Parameters: {num_scenes} scenes, {num_actors} actors, {num_locations} locations, {num_days} days")
        
        # 1. Set of Scenes
        scenes = [f"Scene_{i+1:02d}" for i in range(num_scenes)]
        
        # 2. Set of Actors
        actors = [f"Actor_{chr(65+i)}" for i in range(num_actors)]  # Actor_A, Actor_B, etc.
        
        # 3. Set of Locations (Cities, locations)
        cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
        location_types = ["Studio", "Outdoor", "Indoor", "Beach", "Mountain", "City", "Village", "Factory"]
        
        locations = []
        for i in range(num_locations):
            city = random.choice(cities)
            loc_type = random.choice(location_types)
            locations.append(f"{city}_{loc_type}_{i+1}")
        
        # 4. Daily wage of each actor (following Paper#1 approach)
        actor_wages = {}
        for actor in actors:
            # Random wage between min and max, following Paper#1 range
            wage = random.uniform(wage_range[0], wage_range[1])
            actor_wages[actor] = round(wage, 2)
        
        # 5. Actor - Scene Mapping (following Paper#1 approach)
        # Each scene has 1-3 actors (realistic for movie scenes)
        actor_scene_mapping = {}
        for scene in scenes:
            # Randomly select 1-3 actors for each scene
            num_actors_in_scene = random.randint(1, min(3, num_actors))
            selected_actors = random.sample(actors, num_actors_in_scene)
            actor_scene_mapping[scene] = selected_actors
        
        # 6. Scene - Location Mapping (following Paper#2 approach)
        # Each scene can be shot at 1-2 compatible locations
        scene_location_mapping = {}
        for scene in scenes:
            # Randomly select 1-2 locations for each scene
            num_locations_for_scene = random.randint(1, min(2, num_locations))
            selected_locations = random.sample(locations, num_locations_for_scene)
            scene_location_mapping[scene] = selected_locations
        
        # 7. Location cost for each day (new parameter)
        location_daily_costs = {}
        for location in locations:
            # Random daily cost for each location
            daily_cost = random.uniform(location_cost_range[0], location_cost_range[1])
            location_daily_costs[location] = round(daily_cost, 2)
        
        # 8. Cost of moving from Scene i to Scene j (following Paper#1/Paper#2 approach)
        # Symmetric transfer costs: c_ij = c_ji
        transfer_costs = {}
        for i, scene_i in enumerate(scenes):
            for j, scene_j in enumerate(scenes):
                if i != j:
                    # Random cost between min and max
                    cost = random.uniform(transfer_cost_range[0], transfer_cost_range[1])
                    transfer_costs[(scene_i, scene_j)] = round(cost, 2)
                    # Ensure symmetry: c_ij = c_ji
                    transfer_costs[(scene_j, scene_i)] = round(cost, 2)
                else:
                    # Same scene transition cost is 0
                    transfer_costs[(scene_i, scene_j)] = 0.0
        
        # 9. Number of days required to shoot Scene i (new parameter)
        scene_durations = {}
        for scene in scenes:
            # Random duration between min and max days
            duration = random.randint(scene_duration_range[0], scene_duration_range[1])
            scene_durations[scene] = duration
        
        # Additional parameters for completeness
        # Actor availability (simplified - available for all days)
        actor_availability = {}
        for actor in actors:
            actor_availability[actor] = list(range(1, num_days + 1))
        
        # Scene precedence (random dependencies)
        scene_precedence = []
        for i in range(num_scenes - 1):
            # 30% chance of precedence relationship
            if random.random() < 0.3:
                scene_i = scenes[i]
                scene_j = scenes[i + 1]
                scene_precedence.append((scene_i, scene_j))
        
        # Compile all data
        synthetic_data = {
            'metadata': {
                'seed': self.seed,
                'generation_time': datetime.now().isoformat(),
                'num_scenes': num_scenes,
                'num_actors': num_actors,
                'num_locations': num_locations,
                'num_days': num_days,
                'parameters': {
                    'wage_range': wage_range,
                    'transfer_cost_range': transfer_cost_range,
                    'scene_duration_range': scene_duration_range,
                    'location_cost_range': location_cost_range
                }
            },
            'scenes': scenes,
            'actors': actors,
            'locations': locations,
            'actor_wages': actor_wages,
            'actor_scene_mapping': actor_scene_mapping,
            'scene_location_mapping': scene_location_mapping,
            'location_daily_costs': location_daily_costs,
            'transfer_costs': transfer_costs,
            'scene_durations': scene_durations,
            'actor_availability': actor_availability,
            'scene_precedence': scene_precedence
        }
        
        return synthetic_data
    
    def export_to_excel(self, data: Dict, filename: str = "synthetic_movie_data.xlsx"):
        """Export synthetic data to Excel format compatible with existing code"""
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Metadata sheet
            metadata_df = pd.DataFrame([
                ['Seed', data['metadata']['seed']],
                ['Generation Time', data['metadata']['generation_time']],
                ['Number of Scenes', data['metadata']['num_scenes']],
                ['Number of Actors', data['metadata']['num_actors']],
                ['Number of Locations', data['metadata']['num_locations']],
                ['Number of Days', data['metadata']['num_days']]
            ], columns=['Parameter', 'Value'])
            metadata_df.to_excel(writer, sheet_name='METADATA', index=False)
            
            # Scenes sheet
            scenes_df = pd.DataFrame(data['scenes'], columns=['Scene_ID'])
            scenes_df['Duration_Days'] = [data['scene_durations'][s] for s in data['scenes']]
            scenes_df.to_excel(writer, sheet_name='SCENES', index=False)
            
            # Actors sheet
            actors_df = pd.DataFrame([
                {'Actor_ID': actor, 'Daily_Wage': data['actor_wages'][actor]} 
                for actor in data['actors']
            ])
            actors_df.to_excel(writer, sheet_name='ACTORS', index=False)
            
            # Locations sheet
            locations_df = pd.DataFrame([
                {'Location_ID': loc, 'Daily_Cost': data['location_daily_costs'][loc]} 
                for loc in data['locations']
            ])
            locations_df.to_excel(writer, sheet_name='LOCATIONS', index=False)
            
            # Actor-Scene mapping
            actor_scene_data = []
            for scene, actors_in_scene in data['actor_scene_mapping'].items():
                for actor in actors_in_scene:
                    actor_scene_data.append({
                        'Scene_ID': scene,
                        'Actor_ID': actor,
                        'Required': 1
                    })
            actor_scene_df = pd.DataFrame(actor_scene_data)
            actor_scene_df.to_excel(writer, sheet_name='ACTOR_SCENES', index=False)
            
            # Scene-Location mapping
            scene_location_data = []
            for scene, locations_for_scene in data['scene_location_mapping'].items():
                for location in locations_for_scene:
                    scene_location_data.append({
                        'Scene_ID': scene,
                        'Location_ID': location,
                        'Compatible': 1
                    })
            scene_location_df = pd.DataFrame(scene_location_data)
            scene_location_df.to_excel(writer, sheet_name='SCENE_LOCATIONS', index=False)
            
            # Transfer costs matrix
            transfer_data = []
            for (scene_i, scene_j), cost in data['transfer_costs'].items():
                transfer_data.append({
                    'From_Scene': scene_i,
                    'To_Scene': scene_j,
                    'Transfer_Cost': cost
                })
            transfer_df = pd.DataFrame(transfer_data)
            transfer_df.to_excel(writer, sheet_name='TRANSFER_COSTS', index=False)
            
            # Scene precedence
            if data['scene_precedence']:
                precedence_df = pd.DataFrame(data['scene_precedence'], 
                                          columns=['Predecessor_Scene', 'Successor_Scene'])
                precedence_df.to_excel(writer, sheet_name='SCENE_PRECEDENCE', index=False)
            
            # Actor availability (simplified - all days)
            availability_data = []
            for actor, days in data['actor_availability'].items():
                for day in days:
                    availability_data.append({
                        'Actor_ID': actor,
                        'Day': day,
                        'Available': 1
                    })
            availability_df = pd.DataFrame(availability_data)
            availability_df.to_excel(writer, sheet_name='ACTOR_AVAILABILITY', index=False)
        
        print(f"Data exported to {filename}")
        return filename
    
    def export_to_json(self, data: Dict, filename: str = "synthetic_movie_data.json"):
        """Export synthetic data to JSON format for easy programmatic access"""
        # Convert tuple keys to strings for JSON compatibility
        json_data = data.copy()
        if 'transfer_costs' in json_data:
            json_data['transfer_costs'] = {f"{k[0]}|{k[1]}": v for k, v in data['transfer_costs'].items()}
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Data exported to {filename}")
        return filename
    
    def generate_multiple_instances(self, 
                                  instance_sizes: List[Tuple[int, int, int]] = None,
                                  output_dir: str = "synthetic_instances") -> List[str]:
        """
        Generate multiple test instances of different sizes
        
        Args:
            instance_sizes: List of (scenes, actors, locations) tuples
            output_dir: Directory to save instances
        """
        if instance_sizes is None:
            instance_sizes = [
                (5, 3, 2),   # Small
                (10, 5, 3),  # Medium
                (15, 7, 4),  # Large
                (20, 10, 5)  # Extra Large
            ]
        
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        for i, (num_scenes, num_actors, num_locations) in enumerate(instance_sizes):
            print(f"\nGenerating instance {i+1}: {num_scenes} scenes, {num_actors} actors, {num_locations} locations")
            
            # Generate data with different seed for each instance
            generator = SyntheticDataGenerator(seed=self.seed + i)
            data = generator.generate_parameters(
                num_scenes=num_scenes,
                num_actors=num_actors,
                num_locations=num_locations,
                num_days=num_scenes * 2  # Reasonable number of days
            )
            
            # Export both formats
            excel_file = os.path.join(output_dir, f"instance_{i+1:02d}.xlsx")
            json_file = os.path.join(output_dir, f"instance_{i+1:02d}.json")
            
            generator.export_to_excel(data, excel_file)
            generator.export_to_json(data, json_file)
            
            generated_files.extend([excel_file, json_file])
        
        print(f"\nGenerated {len(instance_sizes)} instances in {output_dir}/")
        return generated_files

def main():
    """Example usage of the synthetic data generator"""
    
    # Create generator
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate a medium-sized instance
    print("=== Generating Medium Instance ===")
    data = generator.generate_parameters(
        num_scenes=10,
        num_actors=5,
        num_locations=3,
        num_days=20
    )
    
    # Export to both formats
    generator.export_to_excel(data, "synthetic_medium.xlsx")
    generator.export_to_json(data, "synthetic_medium.json")
    
    # Generate multiple instances
    print("\n=== Generating Multiple Instances ===")
    generator.generate_multiple_instances()
    
    # Print summary
    print(f"\n=== Data Summary ===")
    print(f"Scenes: {len(data['scenes'])}")
    print(f"Actors: {len(data['actors'])}")
    print(f"Locations: {len(data['locations'])}")
    print(f"Total transfer costs: {len(data['transfer_costs'])}")
    print(f"Scene precedence rules: {len(data['scene_precedence'])}")
    
    # Show sample data
    print(f"\n=== Sample Data ===")
    print(f"Sample scenes: {data['scenes'][:3]}")
    print(f"Sample actors: {data['actors'][:3]}")
    print(f"Sample locations: {data['locations'][:3]}")
    print(f"Sample actor wages: {dict(list(data['actor_wages'].items())[:3])}")
    print(f"Sample scene durations: {dict(list(data['scene_durations'].items())[:3])}")

if __name__ == "__main__":
    main()
