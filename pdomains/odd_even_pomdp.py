"""
Odd-Even POMDP Implementation

This implements a variant of the classic Odd-Even POMDP where:
- Numbers range from 1 to n (hyperparameter)
- True state is either "odd" or "even" (fixed throughout episode)
- Observations are drawn from Gaussian distributions but constrained to odd/even integers
- Mean and standard deviation are hyperparameters with defaults
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OddEvenPOMDPConfig:
    """Configuration for the Odd-Even POMDP"""
    n: int = 20  # Maximum number in range [1, n]
    mean: Optional[float] = None  # Mean of Gaussian (random if None)
    std_dev: float = 2.0  # Standard deviation of Gaussian
    seed: Optional[int] = None  # Random seed for reproducibility


class OddEvenPOMDP:
    """
    Odd-Even POMDP where observations are drawn from Gaussian distributions
    constrained to odd/even integers in the range [1, n].
    """
    
    def __init__(self, config: OddEvenPOMDPConfig):
        self.config = config
        self.n = config.n
        self.std_dev = config.std_dev
        
        # Initialize random number generator
        self.rng = np.random.RandomState(config.seed)
        
        # Set mean randomly if not provided
        if config.mean is None:
            self.mean = self.rng.uniform(1, self.n)
        else:
            self.mean = config.mean
            
        # True state (fixed throughout episode)
        self.true_state = self.rng.choice(['odd', 'even'])
        
        # Generate valid odd and even numbers in range [1, n]
        self.odd_numbers = np.array([i for i in range(1, self.n + 1) if i % 2 == 1])
        self.even_numbers = np.array([i for i in range(1, self.n + 1) if i % 2 == 0])
        
        # Pre-compute probabilities for efficiency
        self._compute_probabilities()
        
    def _compute_probabilities(self):
        """Pre-compute observation probabilities for odd and even states"""
        # Compute Gaussian probabilities for odd numbers
        odd_probs = np.exp(-0.5 * ((self.odd_numbers - self.mean) / self.std_dev) ** 2)
        odd_probs = odd_probs / np.sum(odd_probs)  # Normalize
        
        # Compute Gaussian probabilities for even numbers  
        even_probs = np.exp(-0.5 * ((self.even_numbers - self.mean) / self.std_dev) ** 2)
        even_probs = even_probs / np.sum(even_probs)  # Normalize
        
        self.odd_probs = odd_probs
        self.even_probs = even_probs
        
    def get_observation(self) -> int:
        """
        Generate an observation based on the true state.
        
        Returns:
            int: An observation (odd or even integer in range [1, n])
        """
        if self.true_state == 'odd':
            return self.rng.choice(self.odd_numbers, p=self.odd_probs)
        else:  # even
            return self.rng.choice(self.even_numbers, p=self.even_probs)
    
    def get_reward(self, action: str) -> float:
        """
        Get reward for taking an action.
        
        Args:
            action: Either 'odd' or 'even'
            
        Returns:
            float: +1 if action matches true state, -1 otherwise
        """
        return 1.0 if action == self.true_state else -1.0
    
    def get_observation_probability(self, observation: int, state: str) -> float:
        """
        Get the probability of observing 'observation' given state 'state'.
        
        Args:
            observation: The observed integer
            state: Either 'odd' or 'even'
            
        Returns:
            float: Probability of observation given state
        """
        if state == 'odd':
            if observation in self.odd_numbers:
                idx = np.where(self.odd_numbers == observation)[0][0]
                return self.odd_probs[idx]
            else:
                return 0.0
        else:  # even
            if observation in self.even_numbers:
                idx = np.where(self.even_numbers == observation)[0][0]
                return self.even_probs[idx]
            else:
                return 0.0
    
    def update_belief(self, belief: np.ndarray, observation: int) -> np.ndarray:
        """
        Update belief state using Bayes' rule.
        
        Args:
            belief: Current belief state [P(odd), P(even)]
            observation: New observation
            
        Returns:
            np.ndarray: Updated belief state
        """
        # Get observation probabilities
        p_obs_given_odd = self.get_observation_probability(observation, 'odd')
        p_obs_given_even = self.get_observation_probability(observation, 'even')
        
        # Bayes' rule: P(state|obs) = P(obs|state) * P(state) / P(obs)
        # P(obs) = P(obs|odd) * P(odd) + P(obs|even) * P(even)
        p_obs = p_obs_given_odd * belief[0] + p_obs_given_even * belief[1]
        
        if p_obs == 0:
            # If observation is impossible, return uniform belief
            return np.array([0.5, 0.5])
        
        # Updated belief
        new_belief = np.array([
            p_obs_given_odd * belief[0] / p_obs,
            p_obs_given_even * belief[1] / p_obs
        ])
        
        return new_belief
    
    def get_optimal_action(self, belief: np.ndarray) -> str:
        """
        Get the optimal action given current belief state.
        
        Args:
            belief: Current belief state [P(odd), P(even)]
            
        Returns:
            str: Optimal action ('odd' or 'even')
        """
        return 'odd' if belief[0] > belief[1] else 'even'
    
    def reset(self, new_seed: Optional[int] = None):
        """
        Reset the POMDP with a new true state.
        
        Args:
            new_seed: Optional new random seed
        """
        if new_seed is not None:
            self.rng = np.random.RandomState(new_seed)
        
        # Choose new true state
        self.true_state = self.rng.choice(['odd', 'even'])
        
    def get_info(self) -> dict:
        """
        Get information about the current POMDP instance.
        
        Returns:
            dict: Information about the POMDP configuration and state
        """
        return {
            'n': self.n,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'true_state': self.true_state,
            'odd_numbers': self.odd_numbers.tolist(),
            'even_numbers': self.even_numbers.tolist(),
            'odd_probs': self.odd_probs.tolist(),
            'even_probs': self.even_probs.tolist()
        }


def run_example():
    """Example usage of the OddEvenPOMDP"""
    print("Odd-Even POMDP Example")
    print("=" * 50)
    
    # Create POMDP with default configuration
    config = OddEvenPOMDPConfig(n=10, std_dev=1.5, seed=42)
    pomdp = OddEvenPOMDP(config)
    
    print(f"Configuration: n={config.n}, mean={pomdp.mean:.2f}, std_dev={config.std_dev}")
    print(f"True state: {pomdp.true_state}")
    print(f"Odd numbers: {pomdp.odd_numbers}")
    print(f"Even numbers: {pomdp.even_numbers}")
    print()
    
    # Start with uniform belief
    belief = np.array([0.5, 0.5])
    print(f"Initial belief: P(odd)={belief[0]:.3f}, P(even)={belief[1]:.3f}")
    
    # Generate some observations and update belief
    print("\nGenerating observations and updating belief:")
    print("-" * 40)
    
    for step in range(5):
        obs = pomdp.get_observation()
        belief = pomdp.update_belief(belief, obs)
        action = pomdp.get_optimal_action(belief)
        reward = pomdp.get_reward(action)
        
        print(f"Step {step + 1}: obs={obs}, belief=[{belief[0]:.3f}, {belief[1]:.3f}], "
              f"action={action}, reward={reward}")
    
    print(f"\nTrue state was: {pomdp.true_state}")
    print(f"Final belief: P(odd)={belief[0]:.3f}, P(even)={belief[1]:.3f}")


if __name__ == "__main__":
    run_example()
