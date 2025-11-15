"""
Odd-Even POMDP Implementation

This implements a variant where:
- Numbers range from 1 to n (hyperparameter)
- Hidden parameter: either "odd" or "even" (fixed throughout episode)
- True distribution: Gaussian distribution constrained to odd/even numbers only
- Prediction task: predict the mean as an integer
- True mean: the raw mean rounded to the nearest valid number (odd/even) based on hidden parameter
- Mean and standard deviation are hyperparameters with defaults
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class OddEvenPOMDPConfig:
    """Configuration for the Odd-Even POMDP"""
    n: int = 20  # Maximum number in range [1, n]
    mean: Optional[float] = None  # Mean of Gaussian (random if None)
    std_dev: float = 2.0  # Standard deviation of Gaussian
    seed: Optional[int] = None  # Random seed for reproducibility
    belief_resolution: int = 100  # Number of discrete belief points for mode estimation


class OddEvenPOMDP:
    """
    Mean Prediction POMDP where:
    - Hidden parameter is either "odd" or "even"
    - Observations are drawn from Gaussian distributions constrained to odd/even integers
    - Agent must predict the mean as an integer
    - True mean is the raw mean rounded to the nearest valid number (odd/even)
    """
    
    def __init__(self, config: OddEvenPOMDPConfig):
        self.config = config
        self.n = config.n
        self.std_dev = config.std_dev
        self.belief_resolution = config.belief_resolution
        
        # Initialize random number generator
        self.rng = np.random.RandomState(config.seed)
        
        # Set raw mean (continuous) randomly if not provided
        if config.mean is None:
            self.raw_mean = self.rng.uniform(1, self.n)
        else:
            self.raw_mean = config.mean
            
        # Hidden parameter (fixed throughout episode)
        self.hidden_param = self.rng.choice(['odd', 'even'])
        
        # Generate valid odd and even numbers in range [1, n]
        self.odd_numbers = np.array([i for i in range(1, self.n + 1) if i % 2 == 1])
        self.even_numbers = np.array([i for i in range(1, self.n + 1) if i % 2 == 0])
        
        # Get valid numbers for current hidden parameter
        if self.hidden_param == 'odd':
            valid_nums = self.odd_numbers
        else:
            valid_nums = self.even_numbers
        
        # Compute true mean as integer: round raw_mean to nearest valid number
        self.mean = int(valid_nums[np.argmin(np.abs(valid_nums - self.raw_mean))])
        
        # Create discrete belief space over valid integers only
        self.belief_points = valid_nums.copy()
        
        # Initialize uniform belief over possible integer means
        self.belief = np.ones(len(self.belief_points)) / len(self.belief_points)
        
        # Track observation history for rendering
        self.observation_history = []
        
        # Pre-compute probabilities for efficiency
        self._compute_probabilities()
        
    def _compute_probabilities(self):
        """Pre-compute observation probabilities for the hidden parameter"""
        if self.hidden_param == 'odd':
            # Compute Gaussian probabilities for odd numbers using integer mean
            self.observation_probs = np.exp(-0.5 * ((self.odd_numbers - self.mean) / self.std_dev) ** 2)
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs)  # Normalize
            self.valid_numbers = self.odd_numbers
        else:  # even
            # Compute Gaussian probabilities for even numbers using integer mean
            self.observation_probs = np.exp(-0.5 * ((self.even_numbers - self.mean) / self.std_dev) ** 2)
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs)  # Normalize
            self.valid_numbers = self.even_numbers
        
        # Update belief points to match valid numbers
        self.belief_points = self.valid_numbers.copy()
        # Reset belief to uniform over valid integers
        self.belief = np.ones(len(self.belief_points)) / len(self.belief_points)
        
    def get_observation(self) -> int:
        """
        Generate an observation based on the hidden parameter.
        
        Returns:
            int: An observation (odd or even integer in range [1, n])
        """
        obs = self.rng.choice(self.valid_numbers, p=self.observation_probs)
        self.observation_history.append(obs)
        return obs
    
    def get_particle_set(self, num_particles: int) -> List[int]:
        """
        Get a set of particles representing the belief state.
        
        Args:
            num_particles: Number of particles to generate
        """
        return self.rng.choice(self.valid_numbers, size=num_particles, p=self.observation_probs)
    
    def get_reward(self, predicted_mean: int) -> float:
        """
        Get reward for predicting the mean.
        
        Args:
            predicted_mean: The predicted mean value (integer)
            
        Returns:
            float: Negative squared error as reward
        """
        error = predicted_mean - self.mean
        return -error ** 2  # Negative squared error (higher reward for better predictions)
    
    def _compute_observation_probability(self, observation: int, mean: int) -> float:
        """
        Compute probability of observation given a specific mean.
        
        Args:
            observation: The observed integer
            mean: The mean of the Gaussian distribution (integer)
            
        Returns:
            float: Probability of observation given mean
        """
        # Check if observation is consistent with hidden parameter
        if self.hidden_param == 'odd' and observation % 2 == 0:
            return 0.0
        if self.hidden_param == 'even' and observation % 2 == 1:
            return 0.0
            
        # Compute Gaussian probability
        prob = np.exp(-0.5 * ((observation - mean) / self.std_dev) ** 2)
        
        # Normalize over valid numbers for this hidden parameter
        normalization = np.sum(np.exp(-0.5 * ((self.valid_numbers - mean) / self.std_dev) ** 2))
        
        return prob / normalization if normalization > 0 else 0.0
    
    def update_belief(self, observation: int) -> np.ndarray:
        """
        Update belief state using Bayes' rule.
        
        Args:
            observation: New observation
            
        Returns:
            np.ndarray: Updated belief state over possible integer means
        """
        # Compute likelihood for each possible mean
        likelihoods = np.array([self._compute_observation_probability(observation, mean) 
                               for mean in self.belief_points])
        
        # Bayes' rule: P(mean|obs) = P(obs|mean) * P(mean) / P(obs)
        # P(obs) = sum over all means of P(obs|mean) * P(mean)
        p_obs = np.sum(likelihoods * self.belief)
        
        if p_obs == 0:
            # If observation is impossible, return uniform belief
            self.belief = np.ones(len(self.belief_points)) / len(self.belief_points)
        else:
            # Updated belief
            self.belief = likelihoods * self.belief / p_obs
        
        return self.belief
    
    def get_optimal_prediction(self) -> int:
        """
        Get the optimal mean prediction given current belief state.
        Returns the expected value rounded to nearest integer.
        
        Returns:
            int: Optimal mean prediction (expected value rounded to nearest integer)
        """
        expected_value = np.sum(self.belief_points * self.belief)
        # Round to nearest integer in valid numbers
        return int(self.belief_points[np.argmin(np.abs(self.belief_points - expected_value))])
    
    def get_max_likelihood_prediction(self) -> int:
        """
        Get the maximum likelihood mean prediction.
        
        Returns:
            int: Mean with highest belief probability
        """
        max_idx = np.argmax(self.belief)
        return int(self.belief_points[max_idx])
    
    def reset(self, new_seed: Optional[int] = None):
        """
        Reset the POMDP with a new hidden parameter.
        
        Args:
            new_seed: Optional new random seed
        """
        if new_seed is not None:
            self.rng = np.random.RandomState(new_seed)
        
        # Choose new hidden parameter
        self.hidden_param = self.rng.choice(['odd', 'even'])
        
        # Get valid numbers for new hidden parameter
        if self.hidden_param == 'odd':
            valid_nums = self.odd_numbers
        else:
            valid_nums = self.even_numbers
        
        # Recompute true mean as integer: round raw_mean to nearest valid number
        self.mean = int(valid_nums[np.argmin(np.abs(valid_nums - self.raw_mean))])
        
        # Clear observation history
        self.observation_history = []
        
        # Recompute probabilities (this will also update belief_points and reset belief)
        self._compute_probabilities()
        
    def get_info(self) -> dict:
        """
        Get information about the current POMDP instance.
        
        Returns:
            dict: Information about the POMDP configuration and state
        """
        return {
            'n': self.n,
            'raw_mean': self.raw_mean,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'hidden_param': self.hidden_param,
            'odd_numbers': self.odd_numbers.tolist(),
            'even_numbers': self.even_numbers.tolist(),
            'valid_numbers': self.valid_numbers.tolist(),
            'observation_probs': self.observation_probs.tolist(),
            'belief_points': self.belief_points.tolist(),
            'current_belief': self.belief.tolist()
        }
    
    def render(self, mode='human', show_ground_truth=True, max_history=20):
        """
        Render the current state of the POMDP.
        
        Args:
            mode: Rendering mode ('human' to display, 'rgb_array' to return array)
            show_ground_truth: Whether to show the true mean and hidden parameter
            max_history: Maximum number of recent observations to display
            
        Returns:
            If mode is 'rgb_array', returns numpy array of the figure.
            Otherwise returns None.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Odd-Even POMDP State', fontsize=14, fontweight='bold')
        
        # Plot 1: Belief distribution over possible means
        ax1 = axes[0, 0]
        ax1.bar(self.belief_points, self.belief, alpha=0.7, color='blue', width=0.8)
        ax1.set_xlabel('Mean Value (Integer)')
        ax1.set_ylabel('Belief Probability')
        ax1.set_title('Belief Distribution Over Integer Means')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Mark optimal and ML predictions
        optimal_pred = self.get_optimal_prediction()
        ml_pred = self.get_max_likelihood_prediction()
        ax1.axvline(optimal_pred, color='g', linestyle='--', linewidth=2, label=f'Optimal: {optimal_pred}')
        ax1.axvline(ml_pred, color='r', linestyle='--', linewidth=2, label=f'ML: {ml_pred}')
        
        if show_ground_truth:
            ax1.axvline(self.mean, color='k', linestyle=':', linewidth=2, label=f'True Mean: {self.mean}')
        
        ax1.legend()
        
        # Plot 2: Observation probabilities
        ax2 = axes[0, 1]
        ax2.bar(self.valid_numbers, self.observation_probs, alpha=0.7, color='orange')
        ax2.set_xlabel('Observation Value')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Observation Probabilities ({self.hidden_param})')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Recent observations
        ax3 = axes[1, 0]
        recent_obs = self.observation_history[-max_history:] if len(self.observation_history) > 0 else []
        if recent_obs:
            ax3.plot(range(len(recent_obs)), recent_obs, 'o-', markersize=6, linewidth=1.5)
            ax3.set_xlabel('Observation Index')
            ax3.set_ylabel('Observation Value')
            ax3.set_title(f'Recent Observations (last {len(recent_obs)})')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0.5, self.n + 0.5])
        else:
            ax3.text(0.5, 0.5, 'No observations yet', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Recent Observations')
        
        # Plot 4: Statistics and info
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = f"""
Configuration:
  • Range: [1, {self.n}]
  • Std Dev: {self.std_dev:.2f}
  • Valid Means: {len(self.belief_points)}

Current State:
  • Hidden Parameter: {self.hidden_param}
  • Valid Numbers: {len(self.valid_numbers)} ({self.hidden_param})
  • Observations: {len(self.observation_history)}

Predictions:
  • Optimal: {optimal_pred}
  • Max Likelihood: {ml_pred}
  • Reward (optimal): {self.get_reward(optimal_pred):.3f}
        """
        
        if show_ground_truth:
            info_text += f"""
Ground Truth:
  • Raw Mean: {self.raw_mean:.2f}
  • True Mean: {self.mean}
  • Error (optimal): {abs(optimal_pred - self.mean)}
            """
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        
        if mode == 'rgb_array':
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return buf
        elif mode == 'human':
            plt.show()
            return None
        else:
            plt.close(fig)
            return None


def run_example():
    """Example usage of the ModePredictionPOMDP"""
    print("Mode Prediction POMDP Example")
    print("=" * 50)
    
    # Create POMDP with default configuration
    config = OddEvenPOMDPConfig(n=10, std_dev=1.5, seed=42)
    pomdp = OddEvenPOMDP(config)
    
    print(f"Configuration: n={config.n}, raw_mean={pomdp.raw_mean:.2f}, true_mean={pomdp.mean}, std_dev={config.std_dev}")
    print(f"Hidden parameter: {pomdp.hidden_param}")
    print(f"Valid numbers: {pomdp.valid_numbers}")
    print()
    
    print("Generating observations and updating belief:")
    print("-" * 40)
    
    # Generate some observations and update belief
    for step in range(8):
        obs = pomdp.get_observation()
        pomdp.update_belief(obs)
        pomdp.render()
        optimal_pred = pomdp.get_optimal_prediction()
        ml_pred = pomdp.get_max_likelihood_prediction()
        reward = pomdp.get_reward(optimal_pred)
        
        print(f"Step {step + 1}: obs={obs}, optimal_pred={optimal_pred}, "
              f"ml_pred={ml_pred}, reward={reward:.3f}")
    
    print(f"\nRaw mean: {pomdp.raw_mean:.2f}")
    print(f"True mean: {pomdp.mean}")
    print(f"Hidden parameter: {pomdp.hidden_param}")
    print(f"Final optimal prediction: {pomdp.get_optimal_prediction()}")
    print(f"Final ML prediction: {pomdp.get_max_likelihood_prediction()}")


if __name__ == "__main__":
    run_example()