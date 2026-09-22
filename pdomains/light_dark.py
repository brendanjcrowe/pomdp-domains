"""One-dimensional Light-Dark POMDP.

A Python port of the ``LightDark1D`` problem from POMDPs.jl (originally used to test
MCVI), kept faithful to that implementation's dynamics, reward and defaults.

Model
-----
::

       -3 -2 -1  0  1  2  3  4  5
    ...|  |  |  |  |  |  |  |  |  ...
                 G     S        L

The hidden state is a scalar position ``y``. ``G`` is the goal (``y = 0``), ``S`` the
mean of the initial-state distribution (``y ~ N(2, 3)``) and ``L`` the light source at
``y = 5``. Each step the agent moves left, moves right, or *declares* that it is at the
goal, which ends the episode: ``+10`` if ``|y| < 1`` at that moment, ``-10`` otherwise.

Observations are noisy measurements of the position, ``o ~ N(y, sigma(y))`` with::

    sigma(y) = |y - 5| / sqrt(2) + 1e-2

so the measurement is nearly exact under the light at ``y = 5`` and progressively
useless away from it. The agent never observes its own position, so the optimal policy
is an information-gathering detour: travel to the light, localise, then travel back to
the goal and declare. A policy that heads straight for the goal arrives with a belief
too wide to declare on, which is what makes this a belief problem rather than a control
problem.

The belief is unimodal (a diffusing, sharpening blob), so a Gaussian summary of it is a
good approximation -- unlike Multimodal Search, this domain does not separate belief
ENCODERS. Its value is as a control: an env where the optimal policy provably needs the
belief's SPREAD, not just its mean.

Actions are ``Discrete(3)`` -- index 0/1/2 maps to move left / declare / move right via
:data:`ACTIONS`, matching the Julia ``-1:1`` action set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Action index -> displacement, so index 1 (displacement 0) is the terminating declare
# action, exactly as in the Julia `actions(::LightDark1D) = -1:1`.
ACTIONS = (-1, 0, 1)
DECLARE = 1


@dataclass
class LightDark1DConfig:
    """Configuration for :class:`LightDark1DEnv`; defaults are the Julia ``LightDark1D()``."""

    discount_factor: float = 0.9      # reported on the env, not applied by it (the agent discounts)
    correct_r: float = 10.0           # declaring inside the goal tolerance
    incorrect_r: float = -10.0        # declaring outside it
    step_size: float = 1.0            # displacement per move
    movement_cost: float = 0.0        # see `symmetric_movement_cost`
    goal_tolerance: float = 1.0       # declare is correct when |y| < this
    light_pos: float = 5.0            # observation noise vanishes here
    sigma_slope: float = float(1.0 / np.sqrt(2.0))
    sigma_min: float = 1e-2           # noise floor AT the light, so sigma > 0 everywhere
    init_mean: float = 2.0            # y0 ~ N(init_mean, init_std)
    init_std: float = 3.0
    # The Julia reward for a move is `-movement_cost * a`, which is ASYMMETRIC: with a
    # positive movement_cost it charges for moving right and PAYS for moving left. That
    # is harmless at the default of 0.0, but it is a reward bug the moment the cost is
    # turned on, so this flag switches to `-movement_cost * |a|`. Left False to keep
    # `LightDark1DConfig()` bit-identical to `LightDark1D()`.
    symmetric_movement_cost: bool = False
    # Optional hard bound on |y|. None (the default) matches Julia's unbounded line; the
    # episode cap on the registration is what keeps a runaway agent finite.
    position_limit: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.goal_tolerance <= 0:
            raise ValueError(f"goal_tolerance must be positive, got {self.goal_tolerance}")
        if self.sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive (sigma=0 is a degenerate "
                             f"observation model), got {self.sigma_min}")
        if self.init_std <= 0:
            raise ValueError(f"init_std must be positive, got {self.init_std}")
        if self.position_limit is not None and self.position_limit <= self.goal_tolerance:
            raise ValueError("position_limit must leave room for the goal region: "
                             f"{self.position_limit} <= {self.goal_tolerance}")


def observation_sigma(y, config: Optional[LightDark1DConfig] = None):
    """``sigma(y) = |y - light| * slope + floor`` -- the Julia ``default_sigma``.

    Vectorised over ``y`` so a particle filter can score a whole particle set at once.
    """
    c = config or LightDark1DConfig()
    return np.abs(np.asarray(y, dtype=np.float64) - c.light_pos) * c.sigma_slope + c.sigma_min


class LightDark1DEnv(gym.Env):
    """Localise under a light source, then declare at the goal."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[LightDark1DConfig] = None):
        super().__init__()
        self.config = config or LightDark1DConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.discount_factor = self.config.discount_factor
        self.y = 0.0
        self.step_count = 0

    # --- model ----------------------------------------------------------------
    def sigma(self, y: float) -> float:
        """Observation standard deviation at position ``y``."""
        return float(observation_sigma(y, self.config))

    def _observation(self) -> np.ndarray:
        """``o ~ N(y, sigma(y))`` -- the only window the agent has onto its position."""
        return np.array([self.y + self.rng.normal() * self.sigma(self.y)], dtype=np.float32)

    def _at_goal(self) -> bool:
        return bool(abs(self.y) < self.config.goal_tolerance)

    # --- gym API --------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.y = float(c.init_mean + self.rng.normal() * c.init_std)
        if c.position_limit is not None:
            self.y = float(np.clip(self.y, -c.position_limit, c.position_limit))
        self.step_count = 0
        return self._observation(), self._info(declared=False, correct=False)

    def step(self, action):
        c = self.config
        idx = int(action)
        if not self.action_space.contains(idx):
            raise ValueError(f"action must be one of {list(range(len(ACTIONS)))}, got {action}")
        move = ACTIONS[idx]
        self.step_count += 1

        # Reward is a function of the state BEFORE the move (Julia `reward(p, s, a)`), so
        # declaring is scored at the position the agent declared from.
        declared = idx == DECLARE
        if declared:
            correct = self._at_goal()
            reward = c.correct_r if correct else c.incorrect_r
        else:
            correct = False
            cost = abs(move) if c.symmetric_movement_cost else move
            reward = -c.movement_cost * cost
            self.y = float(self.y + move * c.step_size)
            if c.position_limit is not None:
                self.y = float(np.clip(self.y, -c.position_limit, c.position_limit))

        return self._observation(), float(reward), declared, False, self._info(declared, correct)

    def _info(self, declared: bool, correct: bool) -> dict:
        return {
            "state": self.y,                 # privileged: for filters/diagnostics, never the agent
            "sigma": self.sigma(self.y),
            "light_pos": self.config.light_pos,
            "at_goal": self._at_goal(),
            "declared": declared,
            "success": bool(declared and correct),
            "step_count": self.step_count,
        }


def make_env(**config_kwargs) -> "LightDark1DEnv":
    """Entry point of ``pdomains-light-dark-1d-v0``: ``gym.make`` hands registration and call
    kwargs to the entry point, and the class takes a config object, so build it here."""
    return LightDark1DEnv(LightDark1DConfig(**config_kwargs))


# --- belief heuristics ----------------------------------------------------------------
# Ports of the Julia DummyHeuristic1DPolicy / SmartHeuristic1DPolicy. Both read a belief
# through its mean and standard deviation only, which makes them the natural probe
# policies for this domain: the dummy one ignores the spread and the smart one uses it.

def dummy_heuristic_action(mean: float, std: float, threshold: float = 0.1) -> int:
    """Drive the belief mean to the goal and declare once the belief is tight there.

    Never seeks the light, so it declares only if the belief happens to sharpen on its
    own -- the "mean is enough" baseline.
    """
    return _heuristic(mean, std, target=0.0, threshold=threshold)


def smart_heuristic_action(mean: float, std: float, threshold: float = 0.1) -> int:
    """Head for the light while the belief is wide, then for the goal once it is tight.

    This is the information-gathering policy the domain is built to reward.
    """
    target = 5.0 if std > threshold else 0.0
    return _heuristic(mean, std, target=target, threshold=threshold)


def _heuristic(mean: float, std: float, target: float, threshold: float) -> int:
    if std < threshold and -0.5 < mean < 0.5:
        return DECLARE
    if mean < target:
        return ACTIONS.index(1)     # right
    if mean > target:
        return ACTIONS.index(-1)    # left
    # Exactly on a target that is not declare-worthy (e.g. parked at the light with a
    # belief still too wide). There is no stay action, so step off and come back: every
    # step yields an observation, so oscillating here keeps sharpening the belief. The
    # Julia original leaves `a` undefined in this branch and would raise.
    return ACTIONS.index(-1)
