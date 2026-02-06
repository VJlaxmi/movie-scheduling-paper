#!/usr/bin/env python3
"""
Focused Cleanup - Remove non-essential algorithm files
Keep only the core research files needed for the paper
"""

import os
import shutil

def focused_cleanup():
    """Remove non-essential algorithm files"""
    
    print("🎯 Focused Cleanup - Removing Non-Essential Files...")
    
    # Files to remove (non-essential algorithms)
    files_to_remove = [
        "nsga_runner.py",      # Multi-objective, not needed for single-objective research
        "nsga_batch.py",       # Batch runner for NSGA-III
        "pso_runner.py",       # PSO - GA is the main metaheuristic
        "ga_runner.py",        # Original GA - superseded by improved version
        "compare_ga_versions.py",  # GA comparison - not needed for final paper
        "cleanup_project.py",  # Cleanup script itself
        "focused_cleanup.py",  # This script
    ]
    
    # Remove files
    removed_files = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
                removed_files += 1
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
        else:
            print(f"⚠️  Not found: {file_path}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"   Files removed: {removed_files}")
    
    # Essential files that remain
    essential_files = [
        # Core Research Files
        "beta_testing.py",           # MILP baseline
        "improved_ga_example.py",    # Enhanced GA (9.92% better)
        "scorer.py",                 # Unified scoring function
        
        # Data & Validation
        "synthetic_data_generator.py",
        "validate_synthetic_data.py",
        "datasets/canonical_v1.xlsx",
        "test_small.json",
        "test_medium.json", 
        "test_large.json",
        "testing_data_enriched.xlsx",
        
        # Analysis
        "compare_results.py",        # Results comparison
        
        # Documentation
        "GA_IMPROVEMENTS_ANALYSIS.md",
        "papers_data_preparation_comparison.md",
        "SYNTHETIC_DATA_README.md",
        "PROJECT_STRUCTURE.md",
        
        # Configuration
        "requirements.txt",
    ]
    
    print(f"\n📋 Essential Files Remaining ({len(essential_files)} files):")
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ Missing: {file_path}")
    
    print(f"\n🎯 Final Project Focus:")
    print(f"   📊 MILP Baseline: beta_testing.py")
    print(f"   🧬 Enhanced GA: improved_ga_example.py") 
    print(f"   📈 Unified Scoring: scorer.py")
    print(f"   📊 Results Analysis: compare_results.py")
    print(f"   📝 Research Documentation: 4 files")
    print(f"   📋 Data & Validation: 6 files")
    print(f"   ⚙️ Configuration: 1 file")
    print(f"   📁 Total: {len(essential_files)} files")

if __name__ == "__main__":
    focused_cleanup()
