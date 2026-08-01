"""
Odd-Even POMDP Implementation

This implements a variant where:
- Numbers range from 1 to n (hyperparameter)
- There is a single hidden true_state in [1, n], fixed for the episode
- Observations are drawn from a Gaussian centered at true_state, restricted
  to integers sharing true_state's own parity (odd/even) -- parity is just
  a property of true_state, not a separate hidden variable
- Prediction task: predict true_state as an integer
- Standard deviation defaults to a value computed from n_dist_size
  (sqrt(n_dist_size)/sigma_divisor + 1), or can be pinned to a fixed
  constant via an explicit std_dev override
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.stats import norm


@dataclass
class OddEvenPOMDPConfig:
    """Configuration for the Odd-Even POMDP"""
    n_dist_size: int = 10  # Maximum number in range [1, n]
    true_state: Optional[float] = None  # Fixes true_state to this value if provided (random draw if None)
    # Standard deviation of the observation Gaussian. If left as None (the
    # default), it's computed from n_dist_size and sigma_divisor below
    # instead of being a fixed constant -- pass an explicit value here to
    # override and use a fixed std_dev regardless of n_dist_size, same as
    # the old behavior.
    std_dev: Optional[float] = None
    # std_dev = sqrt(n_dist_size)/sigma_divisor + 1 when std_dev is None.
    # Mirrors the scaling in src/odd_even_beliefmdp.py's
    # _build_observation_probability_matrix: the local discrimination
    # difficulty here (telling state s apart from its same-parity neighbor
    # s+/-2) doesn't grow with n_dist_size, only the number of distinct
    # states does, so std_dev should grow slower than linearly in
    # n_dist_size -- sqrt is the compromise that still gives bigger
    # problems proportionally wider (but not runaway) noise. The default
    # divisor reproduces the original fixed std_dev=2.0 exactly at
    # n_dist_size=10.
    sigma_divisor: float = float(np.sqrt(10))
    seed: Optional[int] = None  # Random seed for reproducibility
    n_particles: int = 100  # Number of discrete belief points for mode estimation
    true_particles: bool =  True
    resample_proportion: float = 0.5

    def __post_init__(self):
        if self.std_dev is None:
            self.std_dev = np.sqrt(self.n_dist_size) / self.sigma_divisor + 1



class OddEvenPOMDP(gym.Env):
    """
    State Prediction POMDP where:
    - There is a single hidden true_state, fixed for the episode
    - Observations are drawn from a Gaussian centered at true_state,
      constrained to integers sharing true_state's own parity
    - Agent must predict true_state as an integer
    """

    def __init__(self, config: OddEvenPOMDPConfig):
        super().__init__()
        print("Initializing OddEvenPOMDP")
        self.config = config
        self.n_dist_size = config.n_dist_size
        self.std_dev = config.std_dev
        self.sigma_divisor = config.sigma_divisor
        self.n_particles = config.n_particles
        self.true_particles = config.true_particles
        self.resample_proportion = config.resample_proportion

        # Initialize random number generator
        self.rng = np.random.RandomState(config.seed)

        # Generate valid odd and even numbers in range [1, n]
        self.odd_numbers = np.array([i for i in range(1, self.n_dist_size + 1) if i % 2 == 1])
        self.even_numbers = np.array([i for i in range(1, self.n_dist_size + 1) if i % 2 == 0])

        # valid_nums: every integer true_state could possibly be, 1..n_dist_size.
        # There's no separate "hidden_param" to pick a parity ahead of time --
        # parity is just a property of whichever true_state gets drawn below
        # (true_state % 2), derived on demand wherever it's needed.
        valid_nums = np.arange(1, self.n_dist_size + 1)

        # `true_state` is what Brendan's original code called `mean`. Renamed
        # since it's the actual hidden ground-truth state we're trying to
        # identify -- not a separate "mean" concept -- and this name is
        # easier to read and understand. Sampled directly and uniformly from
        # valid_nums (or pinned via config.true_state if provided), instead of the
        # old two-step "flip a coin for parity, then draw a continuous value
        # and round to the nearest same-parity integer", which needed a
        # redundant separately-sampled hidden_param and biased the two ends
        # of the range (rounding a continuous draw to the nearest grid point
        # gives boundary points only half the catchment width of interior
        # points).
        if config.true_state is None:
            self.true_state = int(self.rng.choice(valid_nums))
        else:
            self.true_state = int(config.true_state)

        # Create discrete belief space over ALL valid integers (both parities).
        # A real prior can't already know which parity is true -- that's
        # exactly what update_belief() is supposed to figure out from data.
        self.belief_points = np.arange(1, self.n_dist_size + 1)

        # Initialize uniform belief over all possible integer states
        self.belief = np.ones(len(self.belief_points)) / len(self.belief_points)

        # Track observation history for rendering
        self.observation_history = []

        # Pre-compute the true observation-generating distribution for efficiency
        self._compute_probabilities()

        # Initialize particles
        particles = self.rng.choice(self.valid_numbers, p=self.observation_probs, size=self.n_particles)
        self.particles = particles

        # Define action and observation spaces for gymnasium
        # Action space: discrete actions from 0 to n_dist_size-1 (predicting true_state, 0-indexed)
        self.action_space = spaces.Discrete(config.n_dist_size)

        # Observation space: particles from the POMDP
        self.observation_space = spaces.Box(
            low=1.0,
            high=float(config.n_dist_size),
            shape=(config.n_particles,),
            dtype=np.float32
        )

    def step(self, action: int):
        """
        Take a step in the POMDP.

        Args:
            action: The action to take (integer, 0-indexed, will be converted to 1-indexed for prediction)

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, info
        """
        # Convert action from 0-indexed to 1-indexed (action is a prediction of true_state)
        predicted_state = int(action) + 1

        # Generate new observation samples
        samples = self.rng.choice(self.valid_numbers, p=self.observation_probs, size=self.n_particles)

        if self.true_particles:
            self.particles = samples
        else:
            # Resample particles based on observation
            if hasattr(self, 'particles') and len(self.particles) > 0:
                gaussians = norm(loc=samples.mean(), scale=samples.std())
                weights = gaussians.pdf(self.particles)
                if weights.sum() > 0:
                    weights /= weights.sum()
                    indices = self.rng.choice(self.n_particles, size=int(self.n_particles * self.resample_proportion), replace=False, p=1-weights)
                    self.particles[indices] = self.rng.uniform(1, self.n_dist_size, size=len(indices))
            else:
                self.particles = samples

        # Get reward for the predicted state
        reward = self.get_reward(predicted_state)

        # Update observation history for rendering
        if len(samples) > 0:
            self.observation_history.append(samples[0])  # Store first sample as observation

        # Note: max_steps check removed as it's handled by the gym adapter
        terminated = False
        truncated = False

        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1

        # Convert particles to float32 for observation space
        obs = self.particles.astype(np.float32)

        # Compute normalized reward between theoretical min and max
        min_reward, max_reward = self.get_reward_bounds()
        # Avoid division by zero if bounds collapse (should not happen here)
        if max_reward == min_reward:
            normalized_reward = 0.0
        else:
            normalized_reward = (reward - min_reward) / (max_reward - min_reward)

        info = {
            'true_state': self.true_state,
            'predicted_state': predicted_state,
            'reward_min': min_reward,
            'reward_max': max_reward,
            'reward_normalized': float(normalized_reward),
        }

        return obs, reward, terminated, truncated, info
    def _init_particle_set(self):
        """Initialize particle set for mode estimation"""
        return self.rng.rand(self.n_particles) * self.n_dist_size

    def _compute_probabilities(self):
        """Pre-compute the true observation-generating distribution for true_state's parity"""
        if self.true_state % 2 == 1:
            # Compute Gaussian probabilities for odd numbers using the true state
            self.observation_probs = np.exp(-0.5 * ((self.odd_numbers - self.true_state) / self.std_dev) ** 2)
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs)  # Normalize
            self.valid_numbers = self.odd_numbers
        else:  # even
            # Compute Gaussian probabilities for even numbers using the true state
            self.observation_probs = np.exp(-0.5 * ((self.even_numbers - self.true_state) / self.std_dev) ** 2)
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs)  # Normalize
            self.valid_numbers = self.even_numbers

    def get_distribution(self) -> np.ndarray:
        """
        Get the distribution of the belief state.

        Returns:
            np.ndarray: Distribution of the belief state
        """
        return self.observation_probs

    def get_observation(self) -> int:
        """
        Generate an observation based on true_state's parity.

        Returns:
            int: An observation (odd or even integer in range [1, n])
        """
        obs = self.rng.choice(self.valid_numbers, p=self.observation_probs)
        self.observation_history.append(obs)
        return obs


    def get_reward(self, predicted_state: int) -> float:
        """
        Get reward for predicting true_state.

        Args:
            predicted_state: The predicted state value (integer)

        Returns:
            float: Negative squared error as reward
        """
        error = predicted_state - self.true_state
        return -error ** 2  # Negative squared error (higher reward for better predictions)

    def get_reward_bounds(self) -> Tuple[float, float]:
        """
        Get theoretical minimum and maximum possible reward.

        Reward is defined as - (predicted_state - true_state)^2 with true_state in [1, n_dist_size]
        and predictions also in [1, n_dist_size]. The best possible reward is 0 (perfect prediction),
        and the worst is when prediction and true state are at opposite ends of the range.

        Returns:
            Tuple[float, float]: (min_reward, max_reward)
        """
        max_reward = 0.0
        # Maximum squared error occurs between 1 and n_dist_size: (n_dist_size - 1)^2
        min_reward = -float((self.n_dist_size - 1) ** 2)
        return min_reward, max_reward

    def _compute_observation_probability(self, observation: int, candidate_state: int) -> float:
        """
        Compute probability of observation given a hypothesized true_state.

        Args:
            observation: The observed integer
            candidate_state: A hypothesized value for true_state

        Returns:
            float: Probability of observation given candidate_state
        """
        # A hypothesized state can only ever produce same-parity observations
        # (by construction of the generative model) -- judged against the
        # CANDIDATE state's own parity, since this scores every entry in
        # belief_points (both parities), not just the true one.
        candidate_is_odd = (candidate_state % 2 == 1)
        if candidate_is_odd and observation % 2 == 0:
            return 0.0
        if not candidate_is_odd and observation % 2 == 1:
            return 0.0

        # Compute Gaussian probability
        prob = np.exp(-0.5 * ((observation - candidate_state) / self.std_dev) ** 2)

        # Normalize over the valid numbers for the CANDIDATE state's own parity
        valid_for_candidate = self.odd_numbers if candidate_is_odd else self.even_numbers
        normalization = np.sum(np.exp(-0.5 * ((valid_for_candidate - candidate_state) / self.std_dev) ** 2))

        return prob / normalization if normalization > 0 else 0.0

    def update_belief(self, observation: int) -> np.ndarray:
        """
        Update belief state using Bayes' rule.

        Args:
            observation: New observation

        Returns:
            np.ndarray: Updated belief state over possible integer states
        """
        # Compute likelihood for each possible state
        likelihoods = np.array([self._compute_observation_probability(observation, candidate_state)
                               for candidate_state in self.belief_points])

        # Bayes' rule: P(state|obs) = P(obs|state) * P(state) / P(obs)
        # P(obs) = sum over all states of P(obs|state) * P(state)
        p_obs = np.sum(likelihoods * self.belief)

        if p_obs == 0:
            # This observation is impossible under every state the current
            # belief still assigns any mass to -- either the observation
            # model is wrong or something upstream fed a corrupted/
            # out-of-distribution observation. Raise rather than silently
            # resetting to uniform, matching src/odd_even_beliefmdp.py's
            # ExactBeliefUpdater -- silently recovering would hide a real
            # bug and make the two implementations diverge on this edge case.
            raise ValueError(
                f"Impossible observation ({observation}): total posterior "
                "probability is 0. Refusing to silently reset the belief "
                "to uniform -- fix the underlying bug."
            )
        else:
            # Updated belief
            self.belief = likelihoods * self.belief / p_obs

        return self.belief

    def get_optimal_prediction(self) -> int:
        """
        Get the optimal state prediction given current belief state.
        Returns the expected value rounded to nearest integer.

        Returns:
            int: Optimal state prediction (expected value rounded to nearest integer)
        """
        expected_value = np.sum(self.belief_points * self.belief)
        # Round to nearest integer in valid numbers
        return int(self.belief_points[np.argmin(np.abs(self.belief_points - expected_value))])

    def get_max_likelihood_prediction(self) -> int:
        """
        Get the maximum likelihood state prediction.

        Returns:
            int: State with highest belief probability
        """
        max_idx = np.argmax(self.belief)
        return int(self.belief_points[max_idx])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the POMDP with a new true_state.

        Args:
            seed: Optional new random seed
            options: Optional dict with reset options

        Returns:
            Tuple[np.ndarray, dict]: Observation (particles) and info dict
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # valid_nums: every integer true_state could possibly be, 1..n_dist_size
        valid_nums = np.arange(1, self.n_dist_size + 1)

        # Resample true_state each episode unless a fixed value was configured.
        if self.config.true_state is None:
            self.true_state = int(self.rng.choice(valid_nums))

        # Clear observation history
        self.observation_history = []

        # Recompute the true observation-generating probabilities for the new episode
        self._compute_probabilities()

        # Fresh uniform belief over ALL valid integers for the new episode --
        # independent of the new episode's (unknown-to-the-agent) true_state.
        self.belief_points = np.arange(1, self.n_dist_size + 1)
        self.belief = np.ones(len(self.belief_points)) / len(self.belief_points)
        self.step_count = 0

        # Initialize particles
        particles = self.rng.choice(self.valid_numbers, p=self.observation_probs, size=self.n_particles)
        self.particles = particles

        # Convert to float32 for observation space
        obs = particles.astype(np.float32)
        info = {
            'true_state': self.true_state,
        }

        return obs, info

    def get_info(self) -> dict:
        """
        Get information about the current POMDP instance.

        Returns:
            dict: Information about the POMDP configuration and state
        """
        return {
            'n_dist_size': self.n_dist_size,
            'true_state': self.true_state,
            'std_dev': self.std_dev,
            'sigma_divisor': self.sigma_divisor,
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
            show_ground_truth: Whether to show the true state
            max_history: Maximum number of recent observations to display

        Returns:
            If mode is 'rgb_array', returns numpy array of the figure.
            Otherwise returns None.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Odd-Even POMDP State', fontsize=14, fontweight='bold')

        # Plot 1: Belief distribution over possible states
        ax1 = axes[0, 0]
        ax1.bar(self.belief_points, self.belief, alpha=0.7, color='blue', width=0.8)
        ax1.set_xlabel('State Value (Integer)')
        ax1.set_ylabel('Belief Probability')
        ax1.set_title('Belief Distribution Over Integer States')
        ax1.grid(True, alpha=0.3, axis='y')

        # Mark optimal and ML predictions
        optimal_pred = self.get_optimal_prediction()
        ml_pred = self.get_max_likelihood_prediction()
        ax1.axvline(optimal_pred, color='g', linestyle='--', linewidth=2, label=f'Optimal: {optimal_pred}')
        ax1.axvline(ml_pred, color='r', linestyle='--', linewidth=2, label=f'ML: {ml_pred}')

        if show_ground_truth:
            ax1.axvline(self.true_state, color='k', linestyle=':', linewidth=2, label=f'True State: {self.true_state}')

        ax1.legend()

        # Plot 2: Observation probabilities
        parity_label = 'odd' if self.true_state % 2 == 1 else 'even'
        ax2 = axes[0, 1]
        ax2.bar(self.valid_numbers, self.observation_probs, alpha=0.7, color='orange')
        ax2.set_xlabel('Observation Value')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Observation Probabilities ({parity_label})')
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
            ax3.set_ylim([0.5, self.n_dist_size + 0.5])
        else:
            ax3.text(0.5, 0.5, 'No observations yet',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Recent Observations')

        # Plot 4: Statistics and info
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = f"""
Configuration:
  • Range: [1, {self.n_dist_size}]
  • Std Dev: {self.std_dev:.2f}
  • Valid States: {len(self.belief_points)}

Current State:
  • Parity: {parity_label}
  • Valid Numbers: {len(self.valid_numbers)} ({parity_label})
  • Observations: {len(self.observation_history)}

Predictions:
  • Optimal: {optimal_pred}
  • Max Likelihood: {ml_pred}
  • Reward (optimal): {self.get_reward(optimal_pred):.3f}
        """

        if show_ground_truth:
            info_text += f"""
Ground Truth:
  • True State: {self.true_state}
  • Error (optimal): {abs(optimal_pred - self.true_state)}
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


def visualize_particles(pomdp, particles: List[int], step: int):
    """
    Visualize particle set alongside belief distribution.

    Args:
        pomdp: The POMDP instance
        particles: List of particle values
        step: Current step number
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Particle Sampling Visualization - Step {step}', fontsize=14, fontweight='bold')

    # Plot 1: Particle histogram
    ax1 = axes[0]
    # Convert to numpy array if needed and get unique values with counts
    particles_arr = np.array(particles)
    unique_vals, counts = np.unique(particles_arr, return_counts=True)
    particle_values = sorted(unique_vals)
    particle_freqs = [counts[unique_vals == v][0] / len(particles) for v in particle_values]

    ax1.bar(particle_values, particle_freqs, alpha=0.7, color='purple', width=0.8, label='Particle Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Particle Set Distribution (n={len(particles)})')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # Mark true state
    ax1.axvline(pomdp.true_state, color='k', linestyle=':', linewidth=2, label=f'True State: {pomdp.true_state}')
    ax1.legend()

    # Plot 2: Comparison with belief
    ax2 = axes[1]

    # Belief distribution
    ax2.bar(pomdp.belief_points, pomdp.belief, alpha=0.5, color='blue', width=0.8, label='Belief Distribution')

    # Particle distribution (normalized)
    particles_arr = np.array(particles)
    particle_probs = np.zeros(len(pomdp.belief_points))
    for i, bp in enumerate(pomdp.belief_points):
        particle_probs[i] = np.sum(particles_arr == bp) / len(particles)

    ax2.bar(pomdp.belief_points, particle_probs, alpha=0.7, color='purple', width=0.6, label='Particle Distribution')

    ax2.set_xlabel('State Value')
    ax2.set_ylabel('Probability')
    ax2.set_title('Belief vs Particle Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Mark predictions
    optimal_pred = pomdp.get_optimal_prediction()
    ml_pred = pomdp.get_max_likelihood_prediction()
    ax2.axvline(optimal_pred, color='g', linestyle='--', linewidth=1.5, label=f'Optimal: {optimal_pred}')
    ax2.axvline(ml_pred, color='r', linestyle='--', linewidth=1.5, label=f'ML: {ml_pred}')
    ax2.axvline(pomdp.true_state, color='k', linestyle=':', linewidth=2, label=f'True State: {pomdp.true_state}')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_example():
    """Example usage of the OddEvenPOMDP with particle sampling"""
    print("Odd-Even POMDP Example with Particle Sampling")
    print("=" * 60)

    # Create POMDP with default configuration
    # config = OddEvenPOMDPConfig(n_dist_size=10, std_dev=1.5, seed=42)
    config = OddEvenPOMDPConfig(n_dist_size=10, seed=42)
    pomdp = OddEvenPOMDP(config)

    print(f"Configuration: n_dist_size={config.n_dist_size}, true_state={pomdp.true_state}, std_dev={config.std_dev}")
    print(f"Valid numbers: {pomdp.valid_numbers}")
    print()

    # Test particle sampling before any observations
    print("Initial particle sampling (before observations):")
    print("-" * 60)
    num_particles = 1000
    initial_particles = pomdp.get_particle_set(num_particles)
    print(f"Sampled {num_particles} particles")
    unique_particles, counts = np.unique(initial_particles, return_counts=True)
    particle_dist = {int(v): int(c) for v, c in zip(unique_particles, counts)}
    print(f"Particle values: {sorted(particle_dist.keys())}")
    print(f"Particle distribution: {particle_dist}")
    print()

    # Visualize initial particles
    visualize_particles(pomdp, initial_particles, 0)

    print("Generating observations, updating belief, and sampling particles:")
    print("-" * 60)

    # Generate some observations and update belief
    for step in range(8):
        obs = pomdp.get_observation()
        pomdp.update_belief(obs)

        # Sample particles after belief update
        particles = pomdp.get_particle_set(num_particles)

        optimal_pred = pomdp.get_optimal_prediction()
        ml_pred = pomdp.get_max_likelihood_prediction()
        reward = pomdp.get_reward(optimal_pred)

        # Compute particle-based prediction (mean of particles)
        particle_mean = np.mean(particles)
        particle_mean_int = int(pomdp.belief_points[np.argmin(np.abs(pomdp.belief_points - particle_mean))])

        print(f"Step {step + 1}:")
        print(f"  Observation: {obs}")
        print(f"  Optimal prediction: {optimal_pred}, ML prediction: {ml_pred}, Particle mean: {particle_mean_int}")
        print(f"  Reward (optimal): {reward:.3f}")
        unique_p, counts_p = np.unique(particles, return_counts=True)
        particle_dist_step = {int(v): int(c) for v, c in zip(unique_p, counts_p)}
        print(f"  Particle distribution: {particle_dist_step}")
        print()

        # Visualize particles every few steps
        if (step + 1) % 2 == 0 or step == 7:
            visualize_particles(pomdp, particles, step + 1)

    print("Final Summary:")
    print("-" * 60)
    print(f"True state: {pomdp.true_state}")
    print(f"Final optimal prediction: {pomdp.get_optimal_prediction()}")
    print(f"Final ML prediction: {pomdp.get_max_likelihood_prediction()}")

    # Final particle sampling
    final_particles = pomdp.get_particle_set(num_particles)
    final_particle_mean = np.mean(final_particles)
    final_particle_mean_int = int(pomdp.belief_points[np.argmin(np.abs(pomdp.belief_points - final_particle_mean))])
    print(f"Final particle mean: {final_particle_mean_int}")
    unique_final, counts_final = np.unique(final_particles, return_counts=True)
    final_particle_dist = {int(v): int(c) for v, c in zip(unique_final, counts_final)}
    print(f"Final particle distribution: {final_particle_dist}")


if __name__ == "__main__":
    run_example()
