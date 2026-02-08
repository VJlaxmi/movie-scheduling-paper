"""
Q-Learning Adaptive Operator Selection (AOS)

Uses Reinforcement Learning to dynamically select the best operator
combination at each stage of the evolutionary search.

State: (generation_progress, diversity, improvement_rate, gap_from_lb)
Actions: Choose crossover and mutation operator
Reward: Improvement in hypervolume
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class QLearningAOS:
    """
    Q-Learning based Adaptive Operator Selection for NSGA-II.

    State features (discretized):
        - generation_progress: early/mid/late
        - diversity: low/medium/high
        - improvement_rate: stagnant/slow/fast
        - gap_quality: good/moderate/poor

    Actions: (crossover_method, mutation_method)
    """

    CROSSOVER_OPS = ["pmx", "ox", "lax"]  # LAX = Location-Aware Crossover
    MUTATION_OPS = ["swap", "block_swap", "acm"]  # ACM = Actor-Continuity Mutation

    def __init__(self,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.3,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.05):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Build action space
        self.actions = [(cx, mut) for cx in self.CROSSOVER_OPS
                        for mut in self.MUTATION_OPS]
        self.num_actions = len(self.actions)

        # Q-table: state_key -> [Q-values for each action]
        self.q_table: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions)
        )

        # Statistics tracking
        self.action_counts = defaultdict(int)
        self.action_rewards = defaultdict(list)
        self.history: List[Dict] = []

    def _discretize_state(self, generation: int, max_generation: int,
                          diversity: float, improvement_rate: float,
                          quality_gap: float) -> str:
        """Discretize continuous state features into categorical state."""
        # Generation progress
        progress = generation / max(1, max_generation)
        if progress < 0.33:
            gen_state = "early"
        elif progress < 0.66:
            gen_state = "mid"
        else:
            gen_state = "late"

        # Diversity
        if diversity < 0.2:
            div_state = "low"
        elif diversity < 0.5:
            div_state = "med"
        else:
            div_state = "high"

        # Improvement rate
        if improvement_rate < 0.001:
            imp_state = "stagnant"
        elif improvement_rate < 0.05:
            imp_state = "slow"
        else:
            imp_state = "fast"

        # Quality gap (from best known)
        if quality_gap < 0.05:
            gap_state = "good"
        elif quality_gap < 0.2:
            gap_state = "mod"
        else:
            gap_state = "poor"

        return f"{gen_state}_{div_state}_{imp_state}_{gap_state}"

    def select_action(self, generation: int, max_generation: int,
                      diversity: float, improvement_rate: float,
                      quality_gap: float) -> Tuple[str, str, int]:
        """
        Select crossover and mutation operators using epsilon-greedy Q-learning.

        Returns:
            (crossover_op, mutation_op, action_index)
        """
        state = self._discretize_state(generation, max_generation,
                                       diversity, improvement_rate, quality_gap)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        else:
            q_values = self.q_table[state]
            action_idx = int(np.argmax(q_values))

        cx_op, mut_op = self.actions[action_idx]
        self.action_counts[(cx_op, mut_op)] += 1

        return cx_op, mut_op, action_idx

    def update(self, state_info: Tuple, action_idx: int,
               reward: float, next_state_info: Tuple):
        """
        Update Q-table with (state, action, reward, next_state).

        state_info: (generation, max_gen, diversity, imp_rate, gap)
        """
        state = self._discretize_state(*state_info)
        next_state = self._discretize_state(*next_state_info)

        q_values = self.q_table[state]
        next_q = self.q_table[next_state]

        # Q-learning update
        q_values[action_idx] = q_values[action_idx] + self.lr * (
            reward + self.gamma * np.max(next_q) - q_values[action_idx]
        )

        # Decay epsilon
        self.epsilon = max(self.min_epsilon,
                           self.epsilon * self.epsilon_decay)

        # Record
        cx_op, mut_op = self.actions[action_idx]
        self.action_rewards[(cx_op, mut_op)].append(reward)
        self.history.append({
            "state": state,
            "action": (cx_op, mut_op),
            "reward": reward,
            "epsilon": self.epsilon
        })

    def get_stats(self) -> Dict:
        """Return statistics about operator usage and performance."""
        stats = {}
        for action_key in self.actions:
            cx, mut = action_key
            key = f"{cx}+{mut}"
            stats[key] = {
                "count": self.action_counts[action_key],
                "avg_reward": float(np.mean(self.action_rewards[action_key]))
                if self.action_rewards[action_key] else 0.0,
                "total_reward": float(np.sum(self.action_rewards[action_key]))
                if self.action_rewards[action_key] else 0.0,
            }

        # Best operator combination
        if self.action_rewards:
            best_key = max(self.action_rewards.keys(),
                           key=lambda k: np.mean(self.action_rewards[k])
                           if self.action_rewards[k] else 0)
            stats["best_operator"] = f"{best_key[0]}+{best_key[1]}"
        else:
            stats["best_operator"] = "none"

        stats["total_actions"] = sum(self.action_counts.values())
        stats["final_epsilon"] = self.epsilon
        stats["num_states_visited"] = len(self.q_table)

        return stats
