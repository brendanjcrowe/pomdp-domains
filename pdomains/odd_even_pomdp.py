"""
Odd-Even POMDP Implementation

This implements a variant where:
- Numbers range from 1 to n (hyperparameter)
- There is a single hidden true_state in [1, n], fixed for the episode
- Observations are drawn from a Gaussian centered at true_state, restricted
  to integers sharing true_state's own parity (odd/even) -- parity is just
  a property of true_state, not a separate hidden variable
- Prediction task: predict true_state as an integer, scored 1.0 for an
  exact hit and 0.0 otherwise. (This was -(pred - s*)^2 until
  2026-09-03; under that rule the posterior mean was an exact
  sufficient statistic for the optimal action, so no belief encoder
  richer than mean+variance could be distinguished. See get_reward.)
- Standard deviation defaults to a value computed from n_dist_size
  (sqrt(n_dist_size)/sigma_divisor + 1), or can be pinned to a fixed
  constant via an explicit std_dev override

Each step emits `obs_per_step` observations (default 1) and folds every one
of them into the env's own exact posterior, which travels in `info` -- never
in the agent's observation. The registered ids are pdomains-odd-even-10-v0,
pdomains-odd-even-50-v0 and pdomains-odd-even-50-long-v0; every episode cap
lives on the registration.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


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
    # How many independent observations step() emits per timestep.
    #
    # ONE is the only value that leaves a belief problem to solve. The
    # observations are i.i.d. given true_state, so k of them shrink the
    # standard error of the mean by sqrt(k): at n_dist_size=50 the old
    # default of 100 gave a standard error of 0.28 against a state grid of
    # spacing 1, so a memoryless policy pinned the state from a single step
    # and no belief encoder could be told apart from any other.
    #
    # This is NOT the belief encoder's particle count. That count belongs to
    # the particle filter, which lives on the wrapper, not on the env.
    obs_per_step: Optional[int] = None      # resolved to 1 in __post_init__
    # DEPRECATED alias for obs_per_step. The old name meant three different
    # things at once -- observations per step, the observation-space shape,
    # and a resampling size -- so it could not be read without ambiguity.
    n_particles: Optional[int] = None
    # Observation mode. True (the default) emits fresh draws from the true
    # observation model: the honest POMDP observation. False emits a sample
    # from the env's OWN exact posterior instead. That is a diagnostic
    # oracle-belief mode which hands the agent the posterior, so it must
    # never be used for an encoder comparison -- and never with an external
    # particle filter, which would read those samples as fresh observations
    # and count the same evidence twice.
    true_particles: bool = True
    # Unused. The resampling branch this configured never implemented a
    # valid Bayes step; the field is kept so existing callers construct.
    resample_proportion: float = 0.5

    def __post_init__(self):
        if self.std_dev is None:
            self.std_dev = np.sqrt(self.n_dist_size) / self.sigma_divisor + 1

        # Resolve the obs_per_step / n_particles alias. Disagreement raises:
        # picking one of two contradicting values silently would change the
        # difficulty of the entire task without saying so.
        if (self.obs_per_step is not None and self.n_particles is not None
                and int(self.obs_per_step) != int(self.n_particles)):
            raise ValueError(
                f"obs_per_step={self.obs_per_step} contradicts the "
                f"deprecated alias n_particles={self.n_particles}. Pass "
                "obs_per_step only."
            )
        if self.obs_per_step is None:
            if self.n_particles is None:
                self.obs_per_step = 1
            else:
                warnings.warn(
                    "OddEvenPOMDPConfig.n_particles is deprecated; it names "
                    "the emitted-observation count, so use obs_per_step.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                self.obs_per_step = int(self.n_particles)
        self.obs_per_step = int(self.obs_per_step)
        if self.obs_per_step < 1:
            raise ValueError(
                f"obs_per_step must be >= 1, got {self.obs_per_step}")
        # Keep the alias readable and consistent for anything still on it.
        self.n_particles = self.obs_per_step



class OddEvenPOMDP(gym.Env):
    """
    State Prediction POMDP where:
    - There is a single hidden true_state, fixed for the episode
    - Observations are drawn from a Gaussian centered at true_state,
      constrained to integers sharing true_state's own parity
    - Agent must predict true_state as an integer
    """

    # Declared so gym.make(..., render_mode=...) and the passive env checker
    # can see what render() supports.
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: OddEvenPOMDPConfig):
        super().__init__()
        self.config = config
        self.n_dist_size = config.n_dist_size
        self.std_dev = config.std_dev
        self.sigma_divisor = config.sigma_divisor
        self.obs_per_step = config.obs_per_step
        self.true_particles = config.true_particles

        # Seed gymnasium's own generator rather than keeping a private
        # RandomState. `self.rng` is an alias for it (see the property
        # below), so every draw in this env comes from the one stream that
        # super().reset(seed=...) reseeds. Two consequences worth stating:
        # env.np_random -- the attribute every standard tool reaches for --
        # is now the env's real randomness, and reset(seed=k) reproduces
        # episode k exactly, because on this env the hidden state is drawn at
        # reset, so the seed IS the episode. An eval loop must therefore pass
        # seed + episode_index, never a constant seed.
        if config.seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(
                config.seed)

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
        self.step_count = 0

        # Pre-compute the true observation-generating distribution for efficiency
        self._compute_probabilities()

        # Define action and observation spaces for gymnasium
        # Action space: discrete actions from 0 to n_dist_size-1 (predicting true_state, 0-indexed)
        self.action_space = spaces.Discrete(config.n_dist_size)

        # Observation space: the observations this step emits. Its shape
        # follows obs_per_step, which is the only quantity that decides it.
        self.observation_space = spaces.Box(
            low=1.0,
            high=float(config.n_dist_size),
            shape=(self.obs_per_step,),
            dtype=np.float32
        )

    @property
    def rng(self) -> np.random.Generator:
        """Alias for gymnasium's seeded generator.

        Kept as a name so callers that reach for `env.rng` keep working,
        while there is only ONE stream to seed.
        """
        return self.np_random

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

        # Emit this step's observations and fold EVERY one of them into the
        # env's own posterior. update_belief() was correct but unreachable
        # from step(), so env.belief stayed exactly the uniform prior for a
        # whole episode -- which silently turned get_optimal_prediction() and
        # get_max_likelihood_prediction(), and so any oracle or probe label
        # built on them, into a constant.
        observations = self._draw_observations(self.obs_per_step)
        for observation in observations:
            self.update_belief(int(observation))

        # Get reward for the predicted state
        reward = self.get_reward(predicted_state)

        # Update observation history for rendering
        self.observation_history.extend(int(o) for o in observations)

        # No terminal state: the task is prediction at every step, so the
        # horizon comes from the gym registration's max_episode_steps.
        terminated = False
        truncated = False

        self.step_count += 1

        if self.true_particles:
            emitted = observations
        else:
            # Diagnostic oracle-belief mode: emit a sample from the env's own
            # posterior instead of raw observations. This is the only correct
            # reading of what the old resampling branch was reaching for -- it
            # called rng.choice(p=1-weights) on something that is not a
            # distribution, wrote float draws into an integer array, and built
            # a Gaussian whose scale is 0 whenever every draw agrees. Sampling
            # the exact posterior is the same idea done as a valid operation.
            emitted = self.get_particle_set(self.obs_per_step)
        obs = np.asarray(emitted, dtype=np.float32)

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
            'step_count': self.step_count,
            'observations': np.asarray(observations, dtype=np.int64),
            # The posterior travels in info, NOT in the observation: the
            # agent has to earn it through its own belief encoder. It is here
            # so probe labels and the greedy-argmax oracle can read the exact
            # belief that the encoder arms are being compared against.
            'belief': self.belief.copy(),
            'belief_points': self.belief_points.copy(),
            'optimal_prediction': self.get_optimal_prediction(),
            'max_likelihood_prediction': self.get_max_likelihood_prediction(),
        }

        return obs, reward, terminated, truncated, info

    def _draw_observations(self, count: int) -> np.ndarray:
        """Draw `count` i.i.d. observations from the true observation model."""
        return self.rng.choice(
            self.valid_numbers, p=self.observation_probs, size=int(count))

    def get_particle_set(self, num_particles: int) -> np.ndarray:
        """Sample candidate states from the CURRENT belief.

        A weighted posterior turned into an unweighted sample. Used by
        run_example(), and by the true_particles=False observation mode.
        Note that a particle filter for this env should carry the mass in the
        weights instead: at n_dist_size=50 the posterior's effective sample
        size falls to about 1 of 50, so an unweighted sample throws away
        nearly all of the belief's resolution.
        """
        return self.rng.choice(
            self.belief_points, p=self.belief, size=int(num_particles))

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
        Get reward for predicting true_state: 1.0 for an exact hit, else 0.0.

        WHY THIS AND NOT SQUARED ERROR. The old reward was -(pred - s*)^2.
        Under it the expected reward of action a is

            E[R | a] = -sum_s p(s) (a - s)^2 = -(a - mu)^2 - sigma^2

        and the sigma^2 term does not depend on a. So the optimal action was
        round(mu) and NOTHING about the belief beyond its first moment could
        change the decision -- verified exhaustively: over 12,400 beliefs
        round(mu) was the argmax action 1.0000 of the time, losing 0.0 reward.
        That made a mean+variance encoding Bayes-optimal and left a richer
        belief encoder (CGF, Set Transformer, Deep Sets) no room to win, which
        is exactly what the 3M-step runs showed -- CGF and Gaussian tied to
        within seed noise. The domain's multimodal same-parity comb was real;
        the scoring rule simply discarded it.

        Under 0/1 exact match the optimal action is the posterior MODE, so the
        decision depends on where the mass actually sits. On the same 12,400
        beliefs round(mu) is the WRONG action 33.7% of the time and gives up
        0.186 hit-probability on average (worst case 0.756). That is the room
        a belief encoder needs in order to be measurable.

        Args:
            predicted_state: The predicted state value (integer)

        Returns:
            float: 1.0 if predicted_state == true_state, else 0.0
        """
        return 1.0 if int(predicted_state) == int(self.true_state) else 0.0

    def get_reward_bounds(self) -> Tuple[float, float]:
        """
        Get theoretical minimum and maximum possible reward.

        Reward is 0/1 exact match, so the bounds do not depend on n_dist_size:
        0.0 for any miss and 1.0 for a hit. `reward_normalized` in info is
        therefore the same number as the raw reward here, which is intended --
        it is kept so the info key does not change shape for consumers.

        Returns:
            Tuple[float, float]: (min_reward, max_reward)
        """
        return 0.0, 1.0

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
        Get the Bayes-optimal state prediction under the CURRENT reward.

        The reward is 0/1 exact match, so E[R | a] = p(a) and the optimal
        action is the posterior MODE. This used to return the rounded
        posterior mean, which was optimal under the old squared-error reward
        and is wrong here -- on this domain the rounded mean differs from the
        mode in 33.7% of beliefs, so an oracle built on it would be beatable
        by a third of its own decisions and would understate the ceiling every
        encoder arm is measured against.

        This is now the same quantity as get_max_likelihood_prediction(). Both
        names are kept: callers ask for "the best action" and "the MAP state"
        for different reasons, and they would diverge again if the reward
        changed. Whenever get_reward() changes, THIS function must be
        re-derived with it.

        Returns:
            int: argmax_s P(s | observations)
        """
        return int(self.belief_points[np.argmax(self.belief)])

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
            Tuple[np.ndarray, dict]: Observation and info dict
        """
        # This reseeds self.np_random -- and therefore self.rng -- whenever a
        # seed is given. The hidden state is drawn below, so reset(seed=k)
        # replays episode k byte for byte. That is correct, and it is the trap
        # this env sets: an eval loop that passes one constant seed to every
        # reset measures a single episode N times.
        super().reset(seed=seed)

        # valid_nums: every integer true_state could possibly be, 1..n_dist_size
        valid_nums = np.arange(1, self.n_dist_size + 1)

        # Resample true_state each episode unless a fixed value was configured.
        # The else mirrors __init__: without it a value pinned AFTER
        # construction (e.g. by a visualisation that wants to watch a chosen
        # state) was silently ignored and the previous episode's state reused.
        if self.config.true_state is None:
            self.true_state = int(self.rng.choice(valid_nums))
        else:
            self.true_state = int(self.config.true_state)

        # Clear observation history
        self.observation_history = []

        # Recompute the true observation-generating probabilities for the new episode
        self._compute_probabilities()

        # Fresh uniform PRIOR over ALL valid integers for the new episode --
        # independent of the new episode's (unknown-to-the-agent) true_state.
        # The reset observation below turns it into the posterior b0.
        self.belief_points = np.arange(1, self.n_dist_size + 1)
        self.belief = np.ones(len(self.belief_points)) / len(self.belief_points)
        self.step_count = 0

        # b0 = P(s | o0), the standard POMDP convention -- and here an
        # expensive one to get wrong. This observation used to be returned and
        # then thrown away, leaving the prior uniform: a T-step episode drew
        # T + 1 observations and used T, and the wasted one was the only thing
        # that could inform the FIRST action. At n_dist_size=50 over a 50-step
        # cap, step 1 alone accounts for 81% of the optimal policy's pooled
        # mean reward, so the headline metric was 81% decided by a step at
        # which no belief encoder had any information at all -- measured, the
        # optimal policy scores -4.956 per step with o0 discarded against
        # -0.941 with it folded in.
        #
        # Any particle filter for this env MUST consume its initial_env_obs
        # the same way, or its belief sits one update behind the env's with no
        # error raised. OddEven{ExactSupport,Bootstrap}ParticleFilter do.
        observations = self._draw_observations(self.obs_per_step)
        for observation in observations:
            self.update_belief(int(observation))

        emitted = (observations if self.true_particles
                   else self.get_particle_set(self.obs_per_step))
        obs = np.asarray(emitted, dtype=np.float32)
        info = {
            'true_state': self.true_state,
            'step_count': self.step_count,
            'observations': np.asarray(observations, dtype=np.int64),
            'belief': self.belief.copy(),
            'belief_points': self.belief_points.copy(),
            'optimal_prediction': self.get_optimal_prediction(),
            'max_likelihood_prediction': self.get_max_likelihood_prediction(),
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
            'obs_per_step': self.obs_per_step,
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


def make_odd_even_pomdp(**kwargs) -> OddEvenPOMDP:
    """gymnasium entry point for the registered Odd-Even ids.

    OddEvenPOMDP.__init__ takes a single config object, while gym.make()
    forwards keyword arguments, so the registration needs this shim. Every
    OddEvenPOMDPConfig field is therefore settable as a registration kwarg
    or a gym.make() kwarg.
    """
    return OddEvenPOMDP(OddEvenPOMDPConfig(**kwargs))


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
