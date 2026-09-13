"""Cluster-Hunt, least-mass and most-var: 2D belief-cloud hunt tasks.

Moved verbatim from ``src/hunt_tasks/env/{cluster_hunt,min_mass_hunt}.py`` of the parent repo
(rl_for_beliefmdps) on 2026-09-13 (plan section 9 of its ``refactor_plans.md``) so the tasks can
be registered as ``pdomains-*`` env ids and run on the shared RL harness. The originals stay where
they are with their own ids (``ClusterHunt-v0``, ``MinMassHunt-v0``) for the recorded experiments
in ``domain_mds/{cluster_hunt,least_mass,hunt_tasks_pretraining}.md``.

Three deviations from the originals: (1) the ``gym.register`` calls at the bottom of each file are
not here -- ``pdomains/__init__.py`` registers ``pdomains-cluster-hunt-v0`` (the recorded RL
config), ``pdomains-least-mass-v0`` and ``pdomains-most-var-v0``; (2) :class:`MinMassHuntConfig`
gains ``target_rule`` (``"min_mass"``, the original task, or ``"max_var"``: the target is the
WIDEST cluster) and ``sigma_margin``; (3) under ``"max_var"`` the widths are drawn by rejection
until the widest exceeds every other by ``sigma_margin`` and the particle counts are as equal as
possible, so mass carries no information about the target. Under ``"min_mass"`` every random draw
happens in the original order, so trajectories are bit-identical to the original env for the same
seed (``tests/test_hunt_envs.py``).

The particle cloud IS the belief on these tasks: the env redraws it from the live clusters at
every step; nothing the agent does changes it apart from collecting a cluster (Cluster-Hunt).

--- Cluster-Hunt (original docstring) ---
Cluster-Hunt: a 2D belief-MDP where the agent must visit every belief mode.

Hidden state
    Five Gaussian cluster centres mu_j in [0,20]^2, widths sigma_j, and an
    alive flag per cluster.

Observation
    The agent's own position (exact) plus a cloud of 100 particles drawn from
    the alive clusters. The particle cloud IS the belief; there is no separate
    observation model. The agent is never told where the clusters are, nor how
    many are left.

Action
    Continuous [vx, vy] in [-1,1]^2, scaled by v_max. Single-integrator
    dynamics: p' = clip(p + dt * v_max * a).

Task
    Reach every cluster. When the agent comes within `hit_radius` of a live
    cluster, that cluster is collected and its particle mass is redistributed
    over the clusters that remain. The particle count stays at 100 throughout,
    so the belief goes from five modes of 20 particles to one mode of 100.

Reward
    +collect_reward per cluster collected, a small per-step cost, and
    potential-based shaping towards the nearest live cluster. The shaping
    potential uses the true centres, which the agent cannot see; it enters the
    reward only. Potential-based shaping does not change the optimal policy.

--- Least-mass (original docstring) ---
Least-mass Cluster-Hunt: the agent must reach the LIGHTEST cluster.

Difference from ClusterHunt: the clusters carry uneven particle counts, and the
episode ends the moment the agent enters ANY cluster.

  * the lightest cluster -> +success_reward, terminated, solved
  * any other cluster    -> -wrong_penalty,  terminated, failed
  * running out of time  -> -timeout_penalty, truncated

So a policy cannot succeed by steering at nearby mass. The lightest cluster is
the least prominent thing in the cloud, and touching a heavy one ends the episode
badly. The encoder has to separate the modes and compare their masses.

Reward shaping is potential-based on the distance to the lightest cluster. It
uses the true centre, which never enters the observation, and potential-based
shaping does not change the optimal policy. At evaluation there is no shaping, so
a policy that cannot identify the target from its observation cannot fake it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

#: Arena side, centre and the scale the observation divides by (shared by both tasks).
DOMAIN = 20.0
CENTER = 10.0
SCALE = 10.0

TARGET_RULES = ("min_mass", "max_var")


# =============================================================================================
# Cluster-Hunt (moved verbatim from cluster_hunt.py)
# =============================================================================================

@dataclass
class ClusterHuntConfig:
    n_clusters: int = 5
    n_particles: int = 100
    mean_lo: float = 2.0
    mean_hi: float = 18.0
    min_sep: float = 3.0
    sigma_lo: float = 0.30
    sigma_hi: float = 0.70
    v_max: float = 1.0
    dt: float = 1.0
    hit_radius: float = 1.0
    max_steps: int = 200
    collect_reward: float = 10.0
    step_cost: float = 0.01
    shaping_coef: float = 1.0
    progress_bonus: float = 20.0   # potential credit per cluster already taken
    n_active: int = 5              # curriculum: how many clusters actually spawn
    gamma: float = 0.99
    resample_every_step: bool = True
    include_oracle: bool = False   # adds a privileged "oracle" obs key

    def to_dict(self) -> dict:
        return asdict(self)


class ClusterHuntEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[ClusterHuntConfig] = None, **kwargs):
        super().__init__()
        cfg = config or ClusterHuntConfig()
        for k, v in kwargs.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown ClusterHunt option {k!r}")
            setattr(cfg, k, v)
        self.cfg = cfg

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs = {
            "agent": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "particles": spaces.Box(-1.0, 1.0, shape=(cfg.n_particles, 2),
                                    dtype=np.float32),
        }
        if cfg.include_oracle:
            # (dx, dy, alive) per cluster, sorted by distance. Privileged.
            obs["oracle"] = spaces.Box(-2.0, 2.0, shape=(cfg.n_clusters, 3),
                                       dtype=np.float32)
        self.observation_space = spaces.Dict(obs)

        self._rng = np.random.default_rng()
        self.centers = np.zeros((cfg.n_clusters, 2))
        self.sigmas = np.zeros(cfg.n_clusters)
        self.alive = np.ones(cfg.n_clusters, dtype=bool)
        self.pos = np.zeros(2)
        self.t = 0

    # ------------------------------------------------------------------ setup
    def set_hit_radius(self, r: float) -> None:
        """Used by the curriculum callback."""
        self.cfg.hit_radius = float(r)

    def set_n_active(self, n: int) -> None:
        """Curriculum on task size. Fewer clusters is an easier encoding job:
        with one cluster the cloud has a single mode and its centroid is the
        answer. The particle count stays at 100 either way."""
        self.cfg.n_active = int(np.clip(n, 1, self.cfg.n_clusters))

    def _sample_centers(self) -> np.ndarray:
        c = self.cfg
        for _ in range(2000):
            cand = self._rng.uniform(c.mean_lo, c.mean_hi, size=(c.n_clusters, 2))
            d = np.linalg.norm(cand[:, None] - cand[None], axis=-1)
            iu = np.triu_indices(c.n_clusters, 1)
            if d[iu].min() >= c.min_sep:
                return cand
        return cand  # fallback; separation is a soft requirement

    def _particle_counts(self) -> np.ndarray:
        """Split n_particles as evenly as possible over the live clusters."""
        c = self.cfg
        live = np.flatnonzero(self.alive)
        counts = np.zeros(c.n_clusters, dtype=int)
        if len(live) == 0:
            return counts
        base, rem = divmod(c.n_particles, len(live))
        counts[live] = base
        counts[live[:rem]] += 1
        return counts

    def _draw_particles(self) -> np.ndarray:
        c = self.cfg
        counts = self._particle_counts()
        if counts.sum() == 0:
            return np.full((c.n_particles, 2), CENTER)
        idx = np.repeat(np.arange(c.n_clusters), counts)
        pts = (self.centers[idx]
               + self.sigmas[idx, None] * self._rng.standard_normal((len(idx), 2)))
        pts = np.clip(pts, 0.0, DOMAIN)
        self._rng.shuffle(pts, axis=0)
        return pts

    # ------------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        c = self.cfg
        self.n_active = int(np.clip(c.n_active, 1, c.n_clusters))
        self.centers = self._sample_centers()
        self.sigmas = self._rng.uniform(c.sigma_lo, c.sigma_hi, size=c.n_clusters)
        self.alive = np.zeros(c.n_clusters, dtype=bool)
        self.alive[: self.n_active] = True
        self.pos = self._rng.uniform(0.0, DOMAIN, size=2)
        self.t = 0
        self.particles = self._draw_particles()
        return self._obs(), {"collected": 0, "n_alive": int(self.alive.sum()),
                             "n_active": self.n_active}

    def _potential(self) -> float:
        """Distance to the nearest live cluster, plus credit for clusters
        already collected. The second term stops collecting from looking like a
        loss: without it the potential jumps down by the distance to the next
        cluster at the exact moment a cluster is taken."""
        c = self.cfg
        taken = self.n_active - int(self.alive.sum())
        prog = c.progress_bonus * taken
        if not self.alive.any():
            return c.shaping_coef * prog
        d = np.linalg.norm(self.centers[self.alive] - self.pos, axis=-1)
        return c.shaping_coef * (prog - float(d.min()))

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        phi_before = self._potential()

        self.pos = np.clip(self.pos + c.dt * c.v_max * a, 0.0, DOMAIN)
        self.t += 1

        collected = 0
        if self.alive.any():
            d = np.linalg.norm(self.centers - self.pos, axis=-1)
            hit = self.alive & (d < c.hit_radius)
            if hit.any():
                # Collect the nearest hit cluster only, so a single step cannot
                # sweep up several clusters at once.
                j = int(np.flatnonzero(hit)[np.argmin(d[hit])])
                self.alive[j] = False
                collected = 1

        if collected or c.resample_every_step:
            self.particles = self._draw_particles()

        phi_after = self._potential()
        reward = (c.collect_reward * collected
                  - c.step_cost
                  + c.gamma * phi_after - phi_before)

        terminated = not self.alive.any()
        truncated = (self.t >= c.max_steps) and not terminated
        info = {"collected": collected,
                "n_alive": int(self.alive.sum()),
                "n_active": self.n_active,
                "n_collected_total": int(self.n_active - self.alive.sum())}
        if terminated or truncated:
            info["episode_collected"] = int(self.n_active - self.alive.sum())
            info["episode_steps"] = self.t
            info["solved"] = bool(terminated)
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> dict:
        o = {
            "agent": ((self.pos - CENTER) / SCALE).astype(np.float32),
            "particles": ((self.particles - CENTER) / SCALE).astype(np.float32),
        }
        if self.cfg.include_oracle:
            rel = (self.centers - self.pos) / SCALE
            d = np.linalg.norm(self.centers - self.pos, axis=-1)
            d = np.where(self.alive, d, np.inf)
            order = np.argsort(d)
            arr = np.concatenate([rel[order], self.alive[order, None]], axis=1)
            arr[~self.alive[order]] = 0.0
            o["oracle"] = arr.astype(np.float32)
        return o


def make_env(**kwargs):
    return ClusterHuntEnv(**kwargs)


# =============================================================================================
# Least-mass / most-var (moved verbatim from min_mass_hunt.py; target_rule added)
# =============================================================================================
@dataclass
class MinMassHuntConfig:
    n_clusters: int = 5
    n_particles: int = 100
    mean_lo: float = 2.0
    mean_hi: float = 18.0
    min_sep: float = 3.0
    sigma_lo: float = 0.30
    sigma_hi: float = 0.70
    n_min_lo: int = 6
    n_min_hi: int = 15
    mass_margin: int = 5
    v_max: float = 1.0
    dt: float = 1.0
    hit_radius: float = 0.6
    max_steps: int = 60
    success_reward: float = 20.0
    wrong_penalty: float = 5.0
    timeout_penalty: float = 20.0
    step_cost: float = 0.01
    step_cost_growth: float = 0.0   # extra per-step cost, scaled by t/max_steps
    reach_coef: float = 0.0         # shaping toward the NEAREST cluster
    shaping_coef: float = 1.0
    gamma: float = 0.99
    resample_every_step: bool = True
    n_active: int = 5            # curriculum: how many clusters spawn
    include_oracle: bool = False
    #: "min_mass": the target is the LIGHTEST cluster (the original task). "max_var": the
    #: target is the WIDEST cluster (most-var, 2026-09-13); counts are then as equal as possible.
    target_rule: str = "min_mass"
    #: max_var only: the widest cluster is wider than every other by at least this much.
    sigma_margin: float = 0.2

    def to_dict(self):
        return asdict(self)


class MinMassHuntEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[MinMassHuntConfig] = None, **kw):
        super().__init__()
        cfg = config or MinMassHuntConfig()
        for k, v in kw.items():
            if not hasattr(cfg, k):
                raise TypeError(f"unknown option {k!r}")
            setattr(cfg, k, v)
        if cfg.target_rule not in TARGET_RULES:
            raise ValueError(f"target_rule must be one of {TARGET_RULES}, got {cfg.target_rule!r}")
        self.cfg = cfg
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs = {"agent": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
               "particles": spaces.Box(-1.0, 1.0, (cfg.n_particles, 2), dtype=np.float32)}
        if cfg.include_oracle:
            obs["oracle"] = spaces.Box(-2.0, 2.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs)
        self._rng = np.random.default_rng()

    def set_n_active(self, n: int) -> None:
        self.cfg.n_active = int(np.clip(n, 2, self.cfg.n_clusters))

    def set_hit_radius(self, r: float) -> None:
        self.cfg.hit_radius = float(r)

    def _sample_centers(self, k):
        c = self.cfg
        for _ in range(2000):
            cand = self._rng.uniform(c.mean_lo, c.mean_hi, size=(k, 2))
            d = np.linalg.norm(cand[:, None] - cand[None], axis=-1)
            iu = np.triu_indices(k, 1)
            if k < 2 or d[iu].min() >= c.min_sep:
                return cand
        return cand

    def _sample_counts(self, k):
        """Unique minimum, guaranteed by a margin."""
        c = self.cfg
        for _ in range(200):
            n_min = int(self._rng.integers(c.n_min_lo, c.n_min_hi + 1))
            base = n_min + c.mass_margin
            surplus = c.n_particles - n_min - (k - 1) * base
            if surplus < 0:
                continue
            cuts = np.sort(self._rng.integers(0, surplus + 1, size=max(k - 2, 0)))
            parts = np.diff(np.concatenate([[0], cuts, [surplus]])) if k > 1 else []
            counts = np.concatenate([[n_min], base + np.asarray(parts, dtype=int)])
            return counts.astype(int)
        raise RuntimeError("could not sample counts")

    def _sample_sigmas(self, k):
        """max_var: widths with a unique maximum, guaranteed by sigma_margin (rejection)."""
        c = self.cfg
        for _ in range(2000):
            sig = self._rng.uniform(c.sigma_lo, c.sigma_hi, size=k)
            top = np.sort(sig)
            if k < 2 or top[-1] - top[-2] >= c.sigma_margin:
                return sig
        raise RuntimeError("could not sample sigmas with the required margin; widen "
                           "[sigma_lo, sigma_hi] or lower sigma_margin")

    def _equal_counts(self, k):
        """max_var: n_particles split as equally as possible; the remainder (< k particles)
        goes to clusters chosen at random, so it carries no information about the target."""
        base, rem = divmod(self.cfg.n_particles, k)
        counts = np.full(k, base, dtype=int)
        if rem:
            counts[self._rng.choice(k, size=rem, replace=False)] += 1
        return counts

    def _draw(self):
        idx = np.repeat(np.arange(self.k), self.counts)
        p = (self.centers[idx]
             + self.sigmas[idx, None] * self._rng.standard_normal((len(idx), 2)))
        p = np.clip(p, 0.0, DOMAIN)
        self._rng.shuffle(p, axis=0)
        return p

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        c = self.cfg
        self.k = int(np.clip(c.n_active, 2, c.n_clusters))
        self.centers = self._sample_centers(self.k)
        if c.target_rule == "min_mass":
            # The original task. Draw order unchanged: sigmas, counts, permutation, position.
            self.sigmas = self._rng.uniform(c.sigma_lo, c.sigma_hi, size=self.k)
            counts = self._sample_counts(self.k)
            perm = self._rng.permutation(self.k)
            self.counts = counts[perm]
            self.target = int(np.argmin(self.counts))
        else:
            # most-var: the widest cluster, unique by sigma_margin; mass carries no information.
            self.sigmas = self._sample_sigmas(self.k)
            self.counts = self._equal_counts(self.k)
            self.target = int(np.argmax(self.sigmas))
        self.pos = self._rng.uniform(0.0, DOMAIN, size=2)
        self.t = 0
        self.particles = self._draw()
        return self._obs(), {"target": self.target}

    def _potential(self):
        """Two terms, both potential-based.

        shaping_coef pulls toward the TRUE lightest cluster, which the agent
        cannot see. reach_coef pulls toward the NEAREST cluster of any kind. The
        second exists because the measured failure is not picking the wrong
        cluster, it is picking none: a forced-commit evaluation showed the
        pretrained policy is beside the correct cluster on 52% of its timeouts,
        against a 20% chance rate. A pull toward committing at all costs nothing
        in policy-invariance terms and may convert those stalls into decisions.
        """
        c = self.cfg
        d = float(np.linalg.norm(self.centers[self.target] - self.pos))
        phi = -c.shaping_coef * d
        if c.reach_coef:
            dn = float(np.linalg.norm(self.centers[:self.k] - self.pos, axis=-1).min())
            phi -= c.reach_coef * dn
        return phi

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        phi0 = self._potential()
        self.pos = np.clip(self.pos + c.dt * c.v_max * a, 0.0, DOMAIN)
        self.t += 1
        if c.resample_every_step:
            self.particles = self._draw()

        d = np.linalg.norm(self.centers - self.pos, axis=-1)
        hit = int(np.argmin(d)) if d.min() < c.hit_radius else -1

        # A per-step cost that grows with time makes stalling progressively
        # worse than committing, without changing what the outcomes are worth.
        cost = c.step_cost + c.step_cost_growth * (self.t / max(c.max_steps, 1))
        reward = -cost + c.gamma * self._potential() - phi0
        terminated = truncated = False
        solved = False
        if hit >= 0:
            terminated = True
            if hit == self.target:
                reward += c.success_reward
                solved = True
            else:
                reward -= c.wrong_penalty
        elif self.t >= c.max_steps:
            truncated = True
            reward -= c.timeout_penalty

        info = {"target": self.target, "hit": hit}
        if terminated or truncated:
            info.update(solved=solved,
                        outcome=("correct" if solved else
                                 "wrong" if hit >= 0 else "timeout"),
                        episode_steps=self.t)
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        o = {"agent": ((self.pos - CENTER) / SCALE).astype(np.float32),
             "particles": ((self.particles - CENTER) / SCALE).astype(np.float32)}
        if self.cfg.include_oracle:
            rel = (self.centers[self.target] - self.pos) / SCALE
            o["oracle"] = np.concatenate([rel, [1.0]]).astype(np.float32)
        return o
