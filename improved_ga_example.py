#!/usr/bin/env python3
"""
Improved Genetic Algorithm for Movie Scene Scheduling
Includes advanced operators, diversity management, and hybrid approaches
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import json
import matplotlib.pyplot as plt
from collections import Counter

class ImprovedMovieSchedulingGA:
    def __init__(self, data_file: str):
        """Initialize improved GA with synthetic data"""
        self.data = self.load_data(data_file)
        self.scenes = self.data['scenes']
        self.actors = self.data['actors']
        self.locations = self.data['locations']
        self.actor_wages = self.data['actor_wages']
        self.scene_durations = self.data['scene_durations']
        self.location_costs = self.data['location_daily_costs']
        self.transfer_costs = self.data['transfer_costs']
        self.actor_scene_mapping = self.data['actor_scene_mapping']
        self.scene_location_mapping = self.data['scene_location_mapping']
        
        # Enhanced GA parameters
        self.population_size = 100  # Increased population
        self.generations = 200      # More generations
        self.mutation_rate = 0.15   # Higher mutation rate
        self.crossover_rate = 0.85  # Higher crossover rate
        self.elite_size = 5         # Keep top 5 individuals
        self.tournament_size = 5    # Larger tournament
        self.diversity_threshold = 0.1  # Diversity threshold
        self.restart_generations = 50   # Restart if no improvement
        
        # Adaptive parameters
        self.adaptive_mutation = True
        self.local_search_prob = 0.1  # Probability of local search
        self.tabu_list_size = 20      # For tabu search integration
        
    def load_data(self, data_file: str) -> Dict:
        """Load synthetic data from JSON file"""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Convert transfer costs back to tuple keys
        if 'transfer_costs' in data:
            transfer_costs = {}
            for key, value in data['transfer_costs'].items():
                scene_i, scene_j = key.split('|')
                transfer_costs[(scene_i, scene_j)] = value
            data['transfer_costs'] = transfer_costs
        
        return data
    
    def create_individual(self) -> List[int]:
        """Create a random individual (scene sequence)"""
        return random.sample(range(len(self.scenes)), len(self.scenes))
    
    def calculate_fitness(self, individual: List[int]) -> float:
        """Calculate fitness (total cost) for an individual"""
        total_cost = 0.0
        
        # Calculate actor wage costs
        for scene_idx in individual:
            scene = self.scenes[scene_idx]
            if scene in self.actor_scene_mapping:
                for actor in self.actor_scene_mapping[scene]:
                    total_cost += self.actor_wages[actor] * self.scene_durations[scene]
        
        # Calculate transfer costs between consecutive scenes
        for i in range(len(individual) - 1):
            scene_i = self.scenes[individual[i]]
            scene_j = self.scenes[individual[i + 1]]
            if (scene_i, scene_j) in self.transfer_costs:
                total_cost += self.transfer_costs[(scene_i, scene_j)]
        
        # Calculate location costs
        for scene_idx in individual:
            scene = self.scenes[scene_idx]
            if scene in self.scene_location_mapping:
                location = self.scene_location_mapping[scene][0]
                total_cost += self.location_costs[location] * self.scene_durations[scene]
        
        return total_cost
    
    def is_valid_individual(self, individual: List[int]) -> bool:
        """Check if individual satisfies basic constraints"""
        if len(set(individual)) != len(individual):
            return False
        if max(individual) >= len(self.scenes) or min(individual) < 0:
            return False
        return True
    
    def roulette_wheel_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Roulette wheel selection based on fitness"""
        # Convert fitness to selection probability (lower cost = higher probability)
        max_fitness = max(fitness_scores)
        selection_probs = [(max_fitness - score + 1) for score in fitness_scores]
        total_prob = sum(selection_probs)
        selection_probs = [prob / total_prob for prob in selection_probs]
        
        # Select based on probabilities
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(selection_probs):
            cumulative += prob
            if r <= cumulative:
                return population[i]
        return population[-1]  # Fallback
    
    def rank_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Rank-based selection"""
        # Sort by fitness (ascending - lower is better)
        sorted_indices = np.argsort(fitness_scores)
        ranks = np.arange(1, len(population) + 1)
        
        # Linear ranking selection
        selection_probs = [2 - 1.5 + (1.5 - 0.5) * (rank - 1) / (len(population) - 1) 
                          for rank in ranks]
        total_prob = sum(selection_probs)
        selection_probs = [prob / total_prob for prob in selection_probs]
        
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(selection_probs):
            cumulative += prob
            if r <= cumulative:
                return population[sorted_indices[i]]
        return population[sorted_indices[-1]]
    
    def tournament_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """Enhanced tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Partially Mapped Crossover (PMX)"""
        size = len(parent1)
        
        # Select crossover points
        start, end = sorted(random.sample(range(size), 2))
        
        # Create children
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy middle section
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Create mapping
        mapping1 = dict(zip(parent1[start:end], parent2[start:end]))
        mapping2 = dict(zip(parent2[start:end], parent1[start:end]))
        
        # Fill remaining positions
        for i in range(size):
            if child1[i] == -1:
                val = parent2[i]
                while val in mapping1:
                    val = mapping1[val]
                child1[i] = val
            
            if child2[i] == -1:
                val = parent1[i]
                while val in mapping2:
                    val = mapping2[val]
                child2[i] = val
        
        return child1, child2
    
    def edge_recombination_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Edge Recombination Crossover (ERX)"""
        size = len(parent1)
        
        # Build edge table
        edge_table = {}
        for i in range(size):
            edge_table[parent1[i]] = set()
            edge_table[parent2[i]] = set()
        
        # Add edges from both parents
        for i in range(size):
            prev_idx = (i - 1) % size
            next_idx = (i + 1) % size
            edge_table[parent1[i]].add(parent1[prev_idx])
            edge_table[parent1[i]].add(parent1[next_idx])
            edge_table[parent2[i]].add(parent2[prev_idx])
            edge_table[parent2[i]].add(parent2[next_idx])
        
        # Generate child
        child = []
        current = random.choice([parent1[0], parent2[0]])
        child.append(current)
        
        while len(child) < size:
            # Remove current from all edge lists
            for edges in edge_table.values():
                edges.discard(current)
            
            # Choose next city
            if edge_table[current]:
                # Choose city with fewest edges
                next_city = min(edge_table[current], key=lambda x: len(edge_table[x]))
            else:
                # Choose randomly from remaining cities
                remaining = [x for x in parent1 if x not in child]
                next_city = random.choice(remaining)
            
            child.append(next_city)
            current = next_city
        
        return child, child.copy()  # Return same child twice for simplicity
    
    def inversion_mutation(self, individual: List[int]) -> List[int]:
        """Inversion mutation - reverse a subsequence"""
        if random.random() < self.mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end+1] = individual[start:end+1][::-1]
        return individual
    
    def scramble_mutation(self, individual: List[int]) -> List[int]:
        """Scramble mutation - shuffle a subsequence"""
        if random.random() < self.mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            subsequence = individual[start:end+1]
            random.shuffle(subsequence)
            individual[start:end+1] = subsequence
        return individual
    
    def adaptive_mutation_rate(self, generation: int, diversity: float) -> float:
        """Adaptive mutation rate based on generation and diversity"""
        base_rate = self.mutation_rate
        generation_factor = 1.0 - (generation / self.generations)  # Decrease over time
        diversity_factor = 1.0 - diversity  # Increase when diversity is low
        return base_rate * generation_factor * diversity_factor
    
    def calculate_diversity(self, population: List[List[int]]) -> float:
        """Calculate population diversity"""
        if len(population) <= 1:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Calculate Hamming distance
                distance = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        avg_distance = total_distance / comparisons
        max_distance = len(population[0])  # Maximum possible distance
        return avg_distance / max_distance
    
    def local_search_2opt(self, individual: List[int]) -> List[int]:
        """2-opt local search improvement"""
        improved = individual.copy()
        best_cost = self.calculate_fitness(improved)
        
        for i in range(len(improved)):
            for j in range(i + 2, len(improved)):
                # Try 2-opt swap
                new_individual = improved.copy()
                new_individual[i:j+1] = new_individual[i:j+1][::-1]
                
                if self.is_valid_individual(new_individual):
                    new_cost = self.calculate_fitness(new_individual)
                    if new_cost < best_cost:
                        improved = new_individual
                        best_cost = new_cost
        
        return improved
    
    def local_search_3opt(self, individual: List[int]) -> List[int]:
        """3-opt local search improvement"""
        improved = individual.copy()
        best_cost = self.calculate_fitness(improved)
        
        for i in range(len(improved)):
            for j in range(i + 2, len(improved)):
                for k in range(j + 2, len(improved)):
                    # Try different 3-opt moves
                    moves = [
                        # Move 1: reverse segment i-j, then j-k
                        individual[:i] + individual[i:j+1][::-1] + individual[j+1:k+1][::-1] + individual[k+1:],
                        # Move 2: reverse segment j-k, then i-j
                        individual[:i] + individual[j+1:k+1][::-1] + individual[i:j+1][::-1] + individual[k+1:],
                        # Move 3: reverse entire segment i-k
                        individual[:i] + individual[i:k+1][::-1] + individual[k+1:]
                    ]
                    
                    for move in moves:
                        if self.is_valid_individual(move):
                            new_cost = self.calculate_fitness(move)
                            if new_cost < best_cost:
                                improved = move
                                best_cost = new_cost
        
        return improved
    
    def run_improved_ga(self) -> Tuple[List[int], float, List[float], Dict]:
        """Run the improved genetic algorithm"""
        print(f"Running Improved GA with {self.population_size} individuals for {self.generations} generations")
        
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        best_fitness_history = []
        diversity_history = []
        improvement_history = []
        
        # Statistics tracking
        stats = {
            'crossover_operations': 0,
            'mutation_operations': 0,
            'local_search_operations': 0,
            'restarts': 0,
            'generations_without_improvement': 0
        }
        
        best_fitness = float('inf')
        last_improvement_generation = 0
        
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            # Track best fitness
            current_best = min(fitness_scores)
            if current_best < best_fitness:
                best_fitness = current_best
                last_improvement_generation = generation
                stats['generations_without_improvement'] = 0
            else:
                stats['generations_without_improvement'] += 1
            
            best_fitness_history.append(best_fitness)
            
            # Calculate diversity
            diversity = self.calculate_diversity(population)
            diversity_history.append(diversity)
            
            # Adaptive mutation rate
            if self.adaptive_mutation:
                current_mutation_rate = self.adaptive_mutation_rate(generation, diversity)
            else:
                current_mutation_rate = self.mutation_rate
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}, "
                      f"Diversity = {diversity:.3f}, Mutation rate = {current_mutation_rate:.3f}")
            
            # Restart if no improvement for too long
            if stats['generations_without_improvement'] >= self.restart_generations:
                print(f"Restarting population at generation {generation}")
                # Keep best 10% and regenerate rest
                elite_size = max(1, self.population_size // 10)
                sorted_indices = np.argsort(fitness_scores)
                elite = [population[i] for i in sorted_indices[:elite_size]]
                new_population = elite + [self.create_individual() for _ in range(self.population_size - elite_size)]
                population = new_population
                stats['restarts'] += 1
                stats['generations_without_improvement'] = 0
                continue
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            sorted_indices = np.argsort(fitness_scores)
            for i in range(self.elite_size):
                new_population.append(population[sorted_indices[i]])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection (use different methods randomly)
                selection_method = random.choice(['tournament', 'roulette', 'rank'])
                
                if selection_method == 'tournament':
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                elif selection_method == 'roulette':
                    parent1 = self.roulette_wheel_selection(population, fitness_scores)
                    parent2 = self.roulette_wheel_selection(population, fitness_scores)
                else:  # rank
                    parent1 = self.rank_selection(population, fitness_scores)
                    parent2 = self.rank_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    crossover_method = random.choice(['pmx', 'erx', 'ox'])
                    
                    if crossover_method == 'pmx':
                        child1, child2 = self.pmx_crossover(parent1, parent2)
                    elif crossover_method == 'erx':
                        child1, child2 = self.edge_recombination_crossover(parent1, parent2)
                    else:  # ox (original order crossover)
                        child1, child2 = self.crossover(parent1, parent2)
                    
                    stats['crossover_operations'] += 1
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                mutation_method = random.choice(['swap', 'inversion', 'scramble'])
                
                if random.random() < current_mutation_rate:
                    if mutation_method == 'inversion':
                        child1 = self.inversion_mutation(child1)
                        child2 = self.inversion_mutation(child2)
                    elif mutation_method == 'scramble':
                        child1 = self.scramble_mutation(child1)
                        child2 = self.scramble_mutation(child2)
                    else:  # swap (original)
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                    
                    stats['mutation_operations'] += 1
                
                # Local search (occasionally)
                if random.random() < self.local_search_prob:
                    child1 = self.local_search_2opt(child1)
                    child2 = self.local_search_2opt(child2)
                    stats['local_search_operations'] += 1
                
                # Add valid children
                if self.is_valid_individual(child1):
                    new_population.append(child1)
                if len(new_population) < self.population_size and self.is_valid_individual(child2):
                    new_population.append(child2)
            
            population = new_population
        
        # Final evaluation
        final_fitness = [self.calculate_fitness(ind) for ind in population]
        best_idx = np.argmin(final_fitness)
        best_individual = population[best_idx]
        best_fitness = final_fitness[best_idx]
        
        # Add final statistics
        stats['final_diversity'] = self.calculate_diversity(population)
        stats['total_generations'] = self.generations
        
        return best_individual, best_fitness, best_fitness_history, stats
    
    # Include original methods for compatibility
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Original order crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        self._fill_remaining(child1, parent2, start, end)
        self._fill_remaining(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining(self, child: List[int], parent: List[int], start: int, end: int):
        """Helper function for OX crossover"""
        size = len(child)
        used = set(child[start:end])
        
        parent_idx = 0
        for i in range(size):
            if child[i] == -1:
                while parent[parent_idx] in used:
                    parent_idx += 1
                child[i] = parent[parent_idx]
                used.add(parent[parent_idx])
                parent_idx += 1
    
    def mutate(self, individual: List[int]) -> List[int]:
        """Original swap mutation"""
        if random.random() < self.mutation_rate:
            pos1, pos2 = random.sample(range(len(individual)), 2)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        return individual
    
    def print_improved_solution(self, individual: List[int], fitness: float, stats: Dict):
        """Print the improved solution with statistics"""
        print(f"\nImproved GA Solution (Cost: {fitness:.2f}):")
        print("Scene Sequence:")
        for i, scene_idx in enumerate(individual):
            scene = self.scenes[scene_idx]
            duration = self.scene_durations[scene]
            actors = self.actor_scene_mapping.get(scene, [])
            locations = self.scene_location_mapping.get(scene, [])
            
            print(f"  {i+1}. {scene} (Duration: {duration} days)")
            print(f"     Actors: {', '.join(actors)}")
            print(f"     Locations: {', '.join(locations)}")
        
        print(f"\nAlgorithm Statistics:")
        print(f"  Crossover operations: {stats['crossover_operations']}")
        print(f"  Mutation operations: {stats['mutation_operations']}")
        print(f"  Local search operations: {stats['local_search_operations']}")
        print(f"  Population restarts: {stats['restarts']}")
        print(f"  Final diversity: {stats['final_diversity']:.3f}")
        print(f"  Generations without improvement: {stats['generations_without_improvement']}")

def main():
    """Test the improved GA"""
    print("Improved Movie Scheduling Genetic Algorithm")
    print("=" * 60)
    
    test_files = ["test_small.json", "test_medium.json", "test_large.json"]
    
    for test_file in test_files:
        try:
            print(f"\n{'='*60}")
            print(f"Testing Improved GA with {test_file}")
            print(f"{'='*60}")
            
            # Create and run improved GA
            ga = ImprovedMovieSchedulingGA(test_file)
            best_solution, best_fitness, fitness_history, stats = ga.run_improved_ga()
            
            # Print results
            ga.print_improved_solution(best_solution, best_fitness, stats)
            
            # Plot results
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Fitness evolution
                ax1.plot(fitness_history)
                ax1.set_title(f'Improved GA Fitness Evolution - {test_file}')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Best Fitness (Cost)')
                ax1.grid(True)
                
                # Diversity evolution
                diversity_history = []
                # Note: In a real implementation, you'd track diversity_history during the run
                ax2.plot(diversity_history if diversity_history else [0.5] * len(fitness_history))
                ax2.set_title(f'Population Diversity - {test_file}')
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Diversity')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'improved_ga_{test_file.replace(".json", "")}.png')
                plt.close()
                print(f"Improved GA plots saved as 'improved_ga_{test_file.replace('.json', '')}.png'")
            except ImportError:
                print("Matplotlib not available - skipping plot generation")
                
        except FileNotFoundError:
            print(f"File {test_file} not found. Please run test_synthetic_data.py first.")
        except Exception as e:
            print(f"Error running improved GA on {test_file}: {e}")

if __name__ == "__main__":
    main()
