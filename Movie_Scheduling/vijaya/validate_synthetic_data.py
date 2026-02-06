#!/usr/bin/env python3
"""
Data Validation for Synthetic Movie Scheduling Data
Validates the generated synthetic data for consistency and feasibility
"""

import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Set
import numpy as np

class SyntheticDataValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_data(self, data: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate synthetic data for consistency and feasibility
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate basic structure
        self._validate_basic_structure(data)
        
        # Validate parameters
        self._validate_parameters(data)
        
        # Validate mappings
        self._validate_mappings(data)
        
        # Validate costs and constraints
        self._validate_costs(data)
        
        # Validate feasibility
        self._validate_feasibility(data)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_basic_structure(self, data: Dict):
        """Validate basic data structure"""
        required_keys = ['scenes', 'actors', 'locations', 'actor_wages', 
                        'actor_scene_mapping', 'scene_location_mapping',
                        'location_daily_costs', 'transfer_costs', 'scene_durations']
        
        for key in required_keys:
            if key not in data:
                self.errors.append(f"Missing required key: {key}")
        
        if 'metadata' in data:
            metadata = data['metadata']
            if 'num_scenes' in metadata and 'scenes' in data:
                if len(data['scenes']) != metadata['num_scenes']:
                    self.errors.append(f"Scene count mismatch: metadata says {metadata['num_scenes']}, actual {len(data['scenes'])}")
    
    def _validate_parameters(self, data: Dict):
        """Validate parameter ranges and types"""
        # Validate actor wages
        if 'actor_wages' in data:
            for actor, wage in data['actor_wages'].items():
                if not isinstance(wage, (int, float)) or wage <= 0:
                    self.errors.append(f"Invalid wage for {actor}: {wage}")
                elif wage > 10000:
                    self.warnings.append(f"Very high wage for {actor}: {wage}")
        
        # Validate scene durations
        if 'scene_durations' in data:
            for scene, duration in data['scene_durations'].items():
                if not isinstance(duration, int) or duration <= 0:
                    self.errors.append(f"Invalid duration for {scene}: {duration}")
                elif duration > 10:
                    self.warnings.append(f"Very long duration for {scene}: {duration} days")
        
        # Validate location costs
        if 'location_daily_costs' in data:
            for location, cost in data['location_daily_costs'].items():
                if not isinstance(cost, (int, float)) or cost <= 0:
                    self.errors.append(f"Invalid daily cost for {location}: {cost}")
    
    def _validate_mappings(self, data: Dict):
        """Validate actor-scene and scene-location mappings"""
        # Validate actor-scene mapping
        if 'actor_scene_mapping' in data and 'actors' in data:
            actors = set(data['actors'])
            for scene, scene_actors in data['actor_scene_mapping'].items():
                if not isinstance(scene_actors, list):
                    self.errors.append(f"Invalid actor list for {scene}: {scene_actors}")
                    continue
                
                for actor in scene_actors:
                    if actor not in actors:
                        self.errors.append(f"Unknown actor {actor} in scene {scene}")
                
                if len(scene_actors) == 0:
                    self.warnings.append(f"Scene {scene} has no actors")
                elif len(scene_actors) > 5:
                    self.warnings.append(f"Scene {scene} has many actors: {len(scene_actors)}")
        
        # Validate scene-location mapping
        if 'scene_location_mapping' in data and 'locations' in data:
            locations = set(data['locations'])
            for scene, scene_locations in data['scene_location_mapping'].items():
                if not isinstance(scene_locations, list):
                    self.errors.append(f"Invalid location list for {scene}: {scene_locations}")
                    continue
                
                for location in scene_locations:
                    if location not in locations:
                        self.errors.append(f"Unknown location {location} for scene {scene}")
                
                if len(scene_locations) == 0:
                    self.warnings.append(f"Scene {scene} has no compatible locations")
    
    def _validate_costs(self, data: Dict):
        """Validate cost matrices and constraints"""
        # Validate transfer costs symmetry
        if 'transfer_costs' in data:
            for (scene_i, scene_j), cost in data['transfer_costs'].items():
                if not isinstance(cost, (int, float)) or cost < 0:
                    self.errors.append(f"Invalid transfer cost from {scene_i} to {scene_j}: {cost}")
                
                # Check symmetry
                if (scene_j, scene_i) in data['transfer_costs']:
                    reverse_cost = data['transfer_costs'][(scene_j, scene_i)]
                    if abs(cost - reverse_cost) > 1e-6:
                        self.warnings.append(f"Transfer cost asymmetry: {scene_i}->{scene_j}={cost}, {scene_j}->{scene_i}={reverse_cost}")
    
    def _validate_feasibility(self, data: Dict):
        """Validate basic feasibility constraints"""
        if 'scenes' in data and 'actor_scene_mapping' in data:
            # Check if every scene has at least one actor
            for scene in data['scenes']:
                if scene not in data['actor_scene_mapping']:
                    self.errors.append(f"Scene {scene} has no actor mapping")
                elif len(data['actor_scene_mapping'][scene]) == 0:
                    self.errors.append(f"Scene {scene} has no actors assigned")
        
        if 'scenes' in data and 'scene_location_mapping' in data:
            # Check if every scene has at least one compatible location
            for scene in data['scenes']:
                if scene not in data['scene_location_mapping']:
                    self.errors.append(f"Scene {scene} has no location mapping")
                elif len(data['scene_location_mapping'][scene]) == 0:
                    self.errors.append(f"Scene {scene} has no compatible locations")
    
    def validate_excel_file(self, filename: str) -> Tuple[bool, List[str], List[str]]:
        """Validate an Excel file generated by the synthetic data generator"""
        try:
            # Read Excel file
            excel_data = pd.read_excel(filename, sheet_name=None)
            
            # Convert to our data format for validation
            data = self._excel_to_data_dict(excel_data)
            
            return self.validate_data(data)
            
        except Exception as e:
            return False, [f"Error reading Excel file: {str(e)}"], []
    
    def _excel_to_data_dict(self, excel_data: Dict[str, pd.DataFrame]) -> Dict:
        """Convert Excel data to our internal data format"""
        data = {}
        
        # Extract basic lists
        if 'SCENES' in excel_data:
            scenes_df = excel_data['SCENES']
            data['scenes'] = scenes_df['Scene_ID'].tolist()
            data['scene_durations'] = dict(zip(scenes_df['Scene_ID'], scenes_df['Duration_Days']))
        
        if 'ACTORS' in excel_data:
            actors_df = excel_data['ACTORS']
            data['actors'] = actors_df['Actor_ID'].tolist()
            data['actor_wages'] = dict(zip(actors_df['Actor_ID'], actors_df['Daily_Wage']))
        
        if 'LOCATIONS' in excel_data:
            locations_df = excel_data['LOCATIONS']
            data['locations'] = locations_df['Location_ID'].tolist()
            data['location_daily_costs'] = dict(zip(locations_df['Location_ID'], locations_df['Daily_Cost']))
        
        # Extract mappings
        if 'ACTOR_SCENES' in excel_data:
            actor_scene_df = excel_data['ACTOR_SCENES']
            data['actor_scene_mapping'] = {}
            for _, row in actor_scene_df.iterrows():
                scene = row['Scene_ID']
                actor = row['Actor_ID']
                if scene not in data['actor_scene_mapping']:
                    data['actor_scene_mapping'][scene] = []
                data['actor_scene_mapping'][scene].append(actor)
        
        if 'SCENE_LOCATIONS' in excel_data:
            scene_location_df = excel_data['SCENE_LOCATIONS']
            data['scene_location_mapping'] = {}
            for _, row in scene_location_df.iterrows():
                scene = row['Scene_ID']
                location = row['Location_ID']
                if scene not in data['scene_location_mapping']:
                    data['scene_location_mapping'][scene] = []
                data['scene_location_mapping'][scene].append(location)
        
        # Extract transfer costs
        if 'TRANSFER_COSTS' in excel_data:
            transfer_df = excel_data['TRANSFER_COSTS']
            data['transfer_costs'] = {}
            for _, row in transfer_df.iterrows():
                key = (row['From_Scene'], row['To_Scene'])
                data['transfer_costs'][key] = row['Transfer_Cost']
        
        return data
    
    def print_validation_report(self, is_valid: bool, errors: List[str], warnings: List[str]):
        """Print a formatted validation report"""
        print("=" * 50)
        print("SYNTHETIC DATA VALIDATION REPORT")
        print("=" * 50)
        
        if is_valid:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        
        if errors:
            print(f"\n🚨 ERRORS ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        if not errors and not warnings:
            print("\n✨ No issues found!")
        
        print("=" * 50)

def main():
    """Test the validation on generated synthetic data"""
    validator = SyntheticDataValidator()
    
    # Test the medium instance
    print("Validating synthetic_medium.xlsx...")
    is_valid, errors, warnings = validator.validate_excel_file("synthetic_medium.xlsx")
    validator.print_validation_report(is_valid, errors, warnings)
    
    # Test all instances
    instances_dir = "synthetic_instances"
    if os.path.exists(instances_dir):
        print(f"\nValidating all instances in {instances_dir}/...")
        for filename in sorted(os.listdir(instances_dir)):
            if filename.endswith('.xlsx'):
                filepath = os.path.join(instances_dir, filename)
                print(f"\n--- Validating {filename} ---")
                is_valid, errors, warnings = validator.validate_excel_file(filepath)
                validator.print_validation_report(is_valid, errors, warnings)

if __name__ == "__main__":
    main()
