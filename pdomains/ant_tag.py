import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from gymnasium.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

class AntTagEnv(gym.Env):

    def __init__(self, seed=None, num_frames_skip=15, rendering=False,
                 model_name: str = "ant_tag_small.xml",
                 cage_max: float = 4.5,
                 visible_radius: float = 3.0,
                 tag_radius: float = 1.5,
                 target_step: float = 0.5):
        """`model_name` / `cage_max` are keyword-only in practice and default
        to the historical hardcoded values, so every existing caller is
        unaffected. They exist so arena-scaled variants (e.g.
        CounterweightedDenAntTagEnv, which needs the 14x14
        ``ant_tag_large.xml``) can reuse this constructor verbatim.

        `visible_radius` / `tag_radius` likewise default to the historical
        3.0 / 1.5. They are constructor arguments so a registration can
        tighten the sensing/tagging geometry (``pdomains-ant-tag-smart-hard-v0``
        uses 1.0 / 0.6) without a subclass, and the rendered range-marker
        sites in the MuJoCo asset are resized to match. `target_step` (the
        target's per-step displacement, historically 0.5) is exposed for the
        same reason; the particle filters read it off the live env."""

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-6,6])
        initial_joint_ranges[1] = np.array([-6,6])

        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.extra_dim = 2 # xy coordinates of the target

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.data.qpos) + len(self.data.qvel) + self.extra_dim # State will include (i) joint angles and (ii) joint velocities, extra info
        self.action_dim = len(self.model.actuator_ctrlrange) # low-level action dim

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

        self.cage_max_x = float(cage_max)
        self.cage_max_y = float(cage_max)

        # Implement visualization if necessary
        self.visualize = rendering  # Visualization boolean
        self.viewer = None
        if self.visualize:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.num_frames_skip = num_frames_skip

        # For Gym interface
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32            
        )

        if not tag_radius > 0.0:
            raise ValueError("tag_radius must be positive")
        if not visible_radius > tag_radius:
            raise ValueError(
                "visible_radius must exceed tag_radius (urgency formula "
                "divides by their difference)")
        self.visible_radius = float(visible_radius)
        self.tag_radius = float(tag_radius)
        self._sync_range_marker_sites()
        self.min_distance = 5.0
        if not target_step > 0.0:
            raise ValueError("target_step must be positive")
        self.target_step = float(target_step)

        self.seed(seed)

    def _sync_range_marker_sites(self) -> None:
        """Resize the asset's rendered visibility/tag discs to the live radii.

        The ``visible_area`` / ``tag_area`` mocap bodies are visual only and
        carry no game logic; this keeps renders honest when a variant
        changes the radii away from the values baked into the XML."""
        for body_name, radius in (
                ("visible_area", self.visible_radius),
                ("tag_area", self.tag_radius)):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            site_ids = np.flatnonzero(self.model.site_bodyid == body_id)
            if site_ids.size != 1:
                raise RuntimeError(
                    f"Expected exactly one site on {body_name!r}, found "
                    f"{site_ids.size}")
            self.model.site_size[site_ids[0], 0] = radius

    # Get state, which concatenates joint positions and velocities
    def _get_obs(self, target_pos_visible):
        if target_pos_visible:
            return np.concatenate((self.data.qpos, self.data.qvel, self.data.mocap_pos[0][:2]), dtype=np.float32)
        else:
            return np.concatenate((self.data.qpos, self.data.qvel, np.zeros(2)), dtype=np.float32)

    def reset(self, seed=None, options=None):

        if seed is not None:
            self.seed(seed)

        # Reset controls
        self.data.ctrl[:] = 0

        # Set initial joint positions and velocities
        for i in range(len(self.data.qpos)):
            self.data.qpos[i] = self.np_random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.data.qvel)):
            self.data.qvel[i] = self.np_random.uniform(self.initial_state_space[len(self.data.qpos) + i][0],self.initial_state_space[len(self.data.qpos) + i][1])

        init_position_ok = False

        while (not init_position_ok):
            # Initialize the target's position
            target_pos = self.np_random.uniform(low=[-self.cage_max_x, -self.cage_max_y], high=[self.cage_max_x, self.cage_max_y])
            ant_pos = self.np_random.uniform(low=[-self.cage_max_x, -self.cage_max_y], high=[self.cage_max_x, self.cage_max_y])

            d2target = np.linalg.norm(ant_pos - target_pos)

            if d2target > self.min_distance:
                init_position_ok = True

        self.data.mocap_pos[0][:2] = target_pos

        self.data.qpos[:2] = ant_pos

        # Move 2 spheres along the ant
        self.data.mocap_pos[1][:2] = ant_pos
        self.data.mocap_pos[2][:2] = ant_pos

        mujoco.mj_step(self.model, self.data)

        # Updated for gymnasium: return observation and info
        return self._get_obs(False), {}

    def _move_target(self, ant_pos, current_target_pos):
        target2ant_vec = ant_pos - current_target_pos
        target2ant_vec = target2ant_vec / np.linalg.norm(target2ant_vec)

        per_vec_1 = [target2ant_vec[1], -target2ant_vec[0]]
        per_vec_2 = [-target2ant_vec[1], target2ant_vec[0]]
        opposite_vec = -target2ant_vec

        vec_list = [per_vec_1, per_vec_2, opposite_vec, np.zeros(2)]

        chosen_vec_idx = self.np_random.choice(np.arange(4), p=[0.25, 0.25, 0.25, 0.25])

        chosen_vec = np.array(vec_list[chosen_vec_idx]) * self.target_step + current_target_pos

        if abs(chosen_vec[0]) > self.cage_max_x or abs(chosen_vec[1]) > self.cage_max_y:
            chosen_vec = current_target_pos

        self.data.mocap_pos[0][:2] = chosen_vec

    def _do_reveal_target(self):

        ant_pos = self.data.qpos[:2]
        target_pos = self.data.mocap_pos[0][:2]

        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target < self.visible_radius):
            reveal_target_pos = True
        else:
            reveal_target_pos = False

        return reveal_target_pos

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.data.ctrl[:] = action

        # TODO: Change the position of the target based on a fixed policy
        ant_pos = self.data.qpos[:2]
        target_pos = self.data.mocap_pos[0][:2]

        self._move_target(ant_pos, target_pos)

        # Move 2 spheres along the ant
        self.data.mocap_pos[1][:2] = ant_pos
        self.data.mocap_pos[2][:2] = ant_pos

        for _ in range(self.num_frames_skip):
            mujoco.mj_step(self.model, self.data)
            if self.visualize and self.viewer is not None:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    # Relaunch viewer if it was closed
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    self.viewer.sync()

        ant_pos = self.data.qpos[:2]

        done = False
        env_reward = -1

        target_pos = self.data.mocap_pos[0][:2]

        # + reward and terminate the episode if can tag the target
        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target <= self.tag_radius):
            env_reward = 0
            done = True

        reveal_target_pos = self._do_reveal_target()

        # Updated for gymnasium: split done into terminated and truncated
        terminated = done
        truncated = False

        return self._get_obs(reveal_target_pos), env_reward, terminated, truncated, {}


    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def get_target_pos(self) -> npt.NDArray[np.float32]:
        """Returns the current 2d pose of the target"""
        return self.data.mocap_pos[0][:2].copy()


class SmartAntTagEnv(AntTagEnv):
    """AntTag variant with a distance-aware target: flees more often as the ant
    closes in, and slides along the cage wall instead of freezing against it.

    `evasion_scale` (default 1.0) dials how much of that "smartness" is
    active: 0.0 reduces the flee/stay behavior to the base AntTagEnv's
    flat 25/25/25/25 target, 1.0 is full smart behavior. Wall
    sliding is unaffected by evasion_scale since it's a dynamics fix, not a
    difficulty knob. A curriculum (see 4_train_rl_frozen.py's
    CurriculumVisibilityWrapper.set_evasion_scale) can anneal this over
    training instead of exposing the smart target at full strength from
    the start.

    `target_speed_scale` (default 0.0 = OFF) additionally lets the target
    move FASTER as it gets cornered: step size becomes
    target_step * (1 + urgency * target_speed_scale), so 1.0 reproduces the
    old hardcoded "up to 2x speed when cornered" behavior.

    It defaults to OFF because urgency is keyed to visible_radius -- it is
    nonzero only when dist < visible_radius, i.e. exactly when the target is
    VISIBLE. So a speed boost adds no belief-tracking difficulty whatsoever
    (while the ant is blind, urgency is 0 and this env is identical to
    AntTagEnv apart from wall sliding); it only makes the terminal,
    fully-observed chase harder. That is pure control difficulty, and control
    is already the binding constraint: the trained locomotion policy's
    measured per-step displacement is ~0.17 mean / 0.42 max, which is below
    the target's *baseline* 0.5 step, and far below the 1.0 it would reach at
    urgency=1 with scale 1.0. Prefer shrinking visible_radius to make the
    task harder in a way that actually stresses the belief representation.
    """

    def __init__(self, *args, target_speed_scale: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.evasion_scale = 1.0
        self.target_speed_scale = target_speed_scale

    def _move_target(self, ant_pos, current_target_pos):
        target2ant_vec = ant_pos - current_target_pos
        dist = np.linalg.norm(target2ant_vec)
        target2ant_vec = target2ant_vec / dist

        per_vec_1 = np.array([target2ant_vec[1], -target2ant_vec[0]])
        per_vec_2 = np.array([-target2ant_vec[1], target2ant_vec[0]])
        opposite_vec = -target2ant_vec
        stay_vec = np.zeros(2)

        vec_list = [per_vec_1, per_vec_2, opposite_vec, stay_vec]

        # 0 when the ant is at/beyond visible_radius (flat/random, same as AntTagEnv),
        # 1 when the ant is within tag_radius (flee hard and fast)
        urgency = np.clip(
            (self.visible_radius - dist) / (self.visible_radius - self.tag_radius), 0.0, 1.0
        ) * self.evasion_scale

        p_flee = 0.25 + urgency * 0.45   # 0.25 -> 0.70
        p_stay = 0.25 - urgency * 0.20   # 0.25 -> 0.05
        p_side = (1.0 - p_flee - p_stay) / 2.0
        probs = [p_side, p_side, p_flee, p_stay]

        chosen_vec_idx = self.np_random.choice(np.arange(4), p=probs)
        # Constant speed by default (target_speed_scale=0.0): the target flees
        # more OFTEN when cornered, but never faster. Set target_speed_scale=1.0
        # for the old "up to 2x speed when cornered" behavior -- see class docstring.
        step_size = self.target_step * (1.0 + urgency * self.target_speed_scale)

        candidate_pos = np.array(vec_list[chosen_vec_idx]) * step_size + current_target_pos

        # Slide along the wall instead of freezing when a move would exit the cage
        candidate_pos[0] = np.clip(candidate_pos[0], -self.cage_max_x, self.cage_max_x)
        candidate_pos[1] = np.clip(candidate_pos[1], -self.cage_max_y, self.cage_max_y)

        self.data.mocap_pos[0][:2] = candidate_pos


class GhostAntTagEnv(SmartAntTagEnv):
    """SmartAntTag plus a long-range, unreliable 'ping' sensor.

    While the target is NOT within the live visibility radius, each step
    emits with probability `ping_prob` a single 2D ping in info["ghost_ping"]:
    the true target position + N(0, ping_sigma^2) noise with probability
    `ping_beta`, or a uniform-random arena location otherwise (a false alarm).
    The ant cannot tell which. info["ghost_ping_is_true"] records the ground
    truth for offline analysis ONLY -- it must never be fed to the policy or
    the particle filter.

    Pings ride the info dict, not the observation: the 31-D obs is unchanged,
    and the ping's information reaches the agent only through the particle
    filter (GhostAntTagParticleFilter's clutter-robust mixture update).

    Suppression uses `current_visibility_radius` when a curriculum wrapper
    has published it (same live-value pattern as the PF interaction mapper),
    falling back to the env's fixed visible_radius. During curriculum
    warm-start (radius=100) the target is always visible, so pings never
    fire and this env is bit-identical to SmartAntTagEnv.
    """

    def __init__(self, *args, ping_prob: float = 0.04, ping_beta: float = 0.35,
                 ping_sigma: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.ping_prob = ping_prob
        self.ping_beta = ping_beta
        self.ping_sigma = ping_sigma

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["ghost_ping"] = None
        info["ghost_ping_is_true"] = None

        if not terminated:
            ant_pos = self.data.qpos[:2]
            target_pos = self.data.mocap_pos[0][:2]
            radius = float(getattr(self, "current_visibility_radius",
                                   self.visible_radius))
            dist = float(np.linalg.norm(ant_pos - target_pos))
            if dist >= radius and self.np_random.random() < self.ping_prob:
                if self.np_random.random() < self.ping_beta:
                    ping = target_pos + self.np_random.normal(
                        0.0, self.ping_sigma, size=2)
                    ping = np.clip(
                        ping,
                        [-self.cage_max_x, -self.cage_max_y],
                        [self.cage_max_x, self.cage_max_y],
                    )
                    is_true = True
                else:
                    ping = self.np_random.uniform(
                        low=[-self.cage_max_x, -self.cage_max_y],
                        high=[self.cage_max_x, self.cage_max_y],
                    )
                    is_true = False
                info["ghost_ping"] = ping.astype(np.float32)
                info["ghost_ping_is_true"] = is_true

        return obs, reward, terminated, truncated, info


class TwinDenAntTagEnv(SmartAntTagEnv):
    """SmartAntTag with two mirrored hideout 'dens' and a per-episode
    tight/loose assignment — a moment-matched belief construction.

    The target always commits to its NEAREST den (a deterministic function of
    position, so the motion model stays Markov in position and a single
    position-particle filter can represent it): outside the den's leash
    radius it walks straight to the den center at target_step; inside, it
    executes the inherited SmartAntTag urgency-flee step, projected back onto
    the leash disc. One den per episode is 'tight' (leash den_radius_tight)
    and the other 'loose' (den_radius_loose); `tight_den` in {0,1} is
    resampled uniformly each reset.

    `tight_den` is a motion-model parameter: it is forwarded to the particle
    filter (via ant_tag_pf_interaction_mapper, same live-attribute pattern
    as evasion_scale) but NEVER placed in the observation — the agent can
    learn it only from the particle geometry. info["den_tight"] and
    info["den_committed"] are ground truth for offline analysis ONLY;
    den_committed (which den the target actually chose) must never be fed
    to the policy or the filter.

    Geometry is chosen so both leash discs lie fully inside the arena and
    the two dens are point reflections of each other through the origin:
    with equal den weights and the ant out of urgency range, swapping
    tight_den maps the belief through x -> -x exactly, leaving the pooled
    mean (0) and covariance invariant while flipping all odd moments — the
    decision bit lives in the third cumulant along the den diagonal.

    target_speed_scale is unsupported (the leashed motion ignores it); it
    must remain 0.0.
    """

    def __init__(self, *args, den_dist: float = 2.7,
                 den_radius_tight: float = 0.4,
                 den_radius_loose: float = 1.4, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isclose(self.target_speed_scale, 0.0):
            raise ValueError(
                "TwinDenAntTagEnv does not support target_speed_scale != 0.0"
            )
        self.den_positions = np.array(
            [[-den_dist, -den_dist], [den_dist, den_dist]], dtype=np.float64
        )
        self.den_radius_tight = float(den_radius_tight)
        self.den_radius_loose = float(den_radius_loose)
        self.tight_den = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.tight_den = int(self.np_random.integers(2))
        return obs, info

    def _committed_den(self, pos):
        d = np.linalg.norm(self.den_positions - pos, axis=1)
        return int(np.argmin(d)), float(np.min(d))

    def _move_target(self, ant_pos, current_target_pos):
        den, dist_to_den = self._committed_den(current_target_pos)
        den_pos = self.den_positions[den]
        radius = (self.den_radius_tight if den == self.tight_den
                  else self.den_radius_loose)

        if dist_to_den > radius:
            # Transit: deterministic walk to the den center, landing exactly.
            step = min(self.target_step, dist_to_den)
            direction = (den_pos - current_target_pos) / max(dist_to_den, 1e-8)
            candidate = current_target_pos + step * direction
        else:
            # In-den: inherited smart urgency-flee step, leashed to the disc.
            target2ant = ant_pos - current_target_pos
            dist = np.linalg.norm(target2ant)
            target2ant = target2ant / max(dist, 1e-8)
            per_vec_1 = np.array([target2ant[1], -target2ant[0]])
            per_vec_2 = -per_vec_1
            opposite_vec = -target2ant
            stay_vec = np.zeros(2)
            urgency = np.clip(
                (self.visible_radius - dist)
                / (self.visible_radius - self.tag_radius), 0.0, 1.0
            ) * self.evasion_scale
            p_flee = 0.25 + urgency * 0.45
            p_stay = 0.25 - urgency * 0.20
            p_side = (1.0 - p_flee - p_stay) / 2.0
            idx = self.np_random.choice(
                np.arange(4), p=[p_side, p_side, p_flee, p_stay])
            direction = [per_vec_1, per_vec_2, opposite_vec, stay_vec][idx]
            candidate = current_target_pos + np.array(direction) * self.target_step
            off = candidate - den_pos
            norm = np.linalg.norm(off)
            if norm > radius:
                candidate = den_pos + off * (radius / norm)

        candidate = np.clip(
            candidate,
            [-self.cage_max_x, -self.cage_max_y],
            [self.cage_max_x, self.cage_max_y],
        )
        self.data.mocap_pos[0][:2] = candidate

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["den_tight"] = self.tight_den
        den, _ = self._committed_den(self.data.mocap_pos[0][:2].copy())
        info["den_committed"] = den
        return obs, reward, terminated, truncated, info

class CounterweightedDenAntTagEnv(SmartAntTagEnv):
    """Counterweighted dens: heavy-near den (occupied w.p. w=f/(h+f)) and
    light-far den on the opposite diagonal side, mirror bit resampled each
    episode. Pooled belief mean is pinned at 0 BY CONSTRUCTION (w*h=(1-w)*f);
    the mirror bit lives only in odd moments. Target spawns settled in its
    den (no transit -- that deletes Twin-Den's free interception channel).
    Optional spook alarm: entering the EMPTY den's trigger zone releases the
    target into full SmartAntTag evasion, irreversibly.

    info["cden_occupied"] is ground truth for offline analysis ONLY and must
    never reach the policy or the PF; cden_heavy_pos/light_pos/w_heavy/
    spooked/spook_pos ARE PF-known motion/observation-model parameters
    (mapper live-attribute pattern). target_speed_scale must remain 0.0.
    """

    def __init__(self, *args, cden_h: float = 2.4, cden_f: float = 6.75,
                 cden_r: float = 0.4, spook: bool = True,
                 spook_radius: float = 2.2, ant_clearance: float = 2.7,
                 visible_radius: float = 1.8, tag_radius: float = 1.5,
                 **kwargs):
        # Radii validation and the range-marker site resize live in AntTagEnv.
        super().__init__(*args, model_name="ant_tag_large.xml",
                         cage_max=7.0, visible_radius=visible_radius,
                         tag_radius=tag_radius, **kwargs)
        if not np.isclose(self.target_speed_scale, 0.0):
            raise ValueError(
                "CounterweightedDenAntTagEnv requires target_speed_scale == 0.0")
        self.cden_h, self.cden_f, self.cden_r = (
            float(cden_h), float(cden_f), float(cden_r))
        # Mean-pinning is an EQUATION, not a tuned number: w*h == (1-w)*f.
        self.cden_w_heavy = self.cden_f / (self.cden_h + self.cden_f)
        self.cden_spook_enabled = bool(spook)
        self.cden_spook_radius = float(spook_radius)
        self.ant_clearance = float(ant_clearance)
        self._diag = np.array([1.0, 1.0]) / np.sqrt(2.0)
        # The four STATIC candidate centers, in the order [-h, +h, -f, +f].
        self.cden_candidates = np.stack([s * d * self._diag
            for d in (self.cden_h, self.cden_f) for s in (-1.0, 1.0)])
        self.cden_heavy_side = 0
        self.cden_heavy_pos = -self.cden_h * self._diag
        self.cden_light_pos = self.cden_f * self._diag
        self.cden_spooked = False
        self.cden_spook_pos = None
        self._occupied_is_heavy = True

    def reset(self, seed=None, options=None):
        # Base placement runs first (it seeds/reinitializes the joints), then
        # every position it chose is overridden below.
        obs, info = super().reset(seed=seed, options=options)
        s = int(self.np_random.integers(2))
        sign = -1.0 if s == 0 else 1.0
        self.cden_heavy_side = s
        self.cden_heavy_pos = sign * self.cden_h * self._diag
        self.cden_light_pos = -sign * self.cden_f * self._diag
        self.cden_spooked = False
        self.cden_spook_pos = None
        self._occupied_is_heavy = bool(
            self.np_random.random() < self.cden_w_heavy)
        center = (self.cden_heavy_pos if self._occupied_is_heavy
                  else self.cden_light_pos)
        rr = self.cden_r * np.sqrt(self.np_random.random())
        th = self.np_random.uniform(0.0, 2.0 * np.pi)
        target_pos = center + rr * np.array([np.cos(th), np.sin(th)])
        # Decoupled ant spawn: >= clearance from ALL FOUR CANDIDATE centers.
        # Rejecting around only the two ACTUAL centers would make the spawn
        # density depend on the mirror bit -- a direct base-observation leak.
        while True:
            ant_pos = self.np_random.uniform(
                low=[-self.cage_max_x, -self.cage_max_y],
                high=[self.cage_max_x, self.cage_max_y])
            if np.min(np.linalg.norm(self.cden_candidates - ant_pos,
                                     axis=1)) >= self.ant_clearance:
                break
        self.data.qpos[:2] = ant_pos
        self.data.mocap_pos[0][:2] = target_pos
        self.data.mocap_pos[1][:2] = ant_pos
        self.data.mocap_pos[2][:2] = ant_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(False), info

    def _move_target(self, ant_pos, current_target_pos):
        if self.cden_spooked:
            # Alarm raised: the leash dissolves, full SmartAntTag evasion.
            return SmartAntTagEnv._move_target(self, ant_pos,
                                               current_target_pos)
        dens = np.stack([self.cden_heavy_pos, self.cden_light_pos])
        d = np.linalg.norm(dens - current_target_pos, axis=1)
        den_pos = dens[int(np.argmin(d))]
        radius = self.cden_r
        # --- in-den: inherited smart urgency-flee step, leashed to the disc.
        # KEEP TEXTUALLY PARALLEL with
        # CounterweightedDenAntTagParticleFilter.predict's in-den branch. ---
        target2ant = ant_pos - current_target_pos
        dist = np.linalg.norm(target2ant)
        target2ant = target2ant / max(dist, 1e-8)
        per_vec_1 = np.array([target2ant[1], -target2ant[0]])
        per_vec_2 = -per_vec_1
        opposite_vec = -target2ant
        toward_vec = target2ant
        stay_vec = np.zeros(2)
        urgency = np.clip(
            (self.visible_radius - dist)
            / (self.visible_radius - self.tag_radius), 0.0, 1.0
        ) * self.evasion_scale
        # ISOTROPIC-AT-ZERO-URGENCY (differs from SmartAntTag/TwinDen on
        # purpose; see below). The inherited option set is
        # {perp_left, perp_right, away, STAY} -- there is no "toward", so the
        # inherited flat 25/25/25/25 is NOT isotropic: the perpendiculars
        # cancel but `away` has no counterweight, leaving a net drift of
        # 0.25 * target_step = 0.125 u/step directly away from the ant even
        # when urgency is EXACTLY 0. In this env that drift is a leak: it
        # displaces each den's belief mode along the ant->den bearing, and
        # because the two dens sit on opposite diagonal sides with unequal
        # weights w and 1-w, the displacements do not cancel and the pooled
        # mean acquires a term that FLIPS WITH THE MIRROR BIT -- readable by
        # a mean+covariance encoder that is supposed to be blind to it.
        #
        # Fix: reassign the urgency-0 `stay` mass to a new `toward` option, so
        # at urgency 0 the target wanders on an isotropic 4-point star
        # (mean 0, covariance 0.5*step^2*I) -- the natural model for a target
        # that cannot detect the ant at all (urgency 0 <=> ant beyond
        # visible_radius). At urgency 1 the split is 0.125/0.125/0.70/0/0.05,
        # i.e. IDENTICAL to the inherited cornered behavior; only the
        # no-threat end of the interpolation changed.
        p_flee = 0.25 + urgency * 0.45
        p_toward = 0.25 * (1.0 - urgency)
        p_stay = 0.05 * urgency
        p_side = (1.0 - p_flee - p_toward - p_stay) / 2.0
        idx = self.np_random.choice(
            np.arange(5), p=[p_side, p_side, p_flee, p_toward, p_stay])
        direction = [per_vec_1, per_vec_2, opposite_vec,
                     toward_vec, stay_vec][idx]
        candidate = current_target_pos + np.array(direction) * self.target_step
        off = candidate - den_pos
        norm = np.linalg.norm(off)
        if norm > radius:
            candidate = den_pos + off * (radius / norm)
        candidate = np.clip(
            candidate,
            [-self.cage_max_x, -self.cage_max_y],
            [self.cage_max_x, self.cage_max_y],
        )
        self.data.mocap_pos[0][:2] = candidate

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # One-shot, irreversible spook trigger, evaluated post-physics. The
        # one-step reaction lag (this step's target motion already happened)
        # is accepted and documented.
        if (self.cden_spook_enabled and not self.cden_spooked
                and not terminated):
            empty = (self.cden_light_pos if self._occupied_is_heavy
                     else self.cden_heavy_pos)
            if np.linalg.norm(self.data.qpos[:2] - empty) < self.cden_spook_radius:
                self.cden_spooked = True
                self.cden_spook_pos = empty.copy()
        info["cden_heavy_side"] = self.cden_heavy_side
        info["cden_occupied"] = "heavy" if self._occupied_is_heavy else "light"
        info["cden_spooked"] = self.cden_spooked
        return obs, reward, terminated, truncated, info


class TerminalPhantomCounterweightedDenAntTagEnv(
        CounterweightedDenAntTagEnv):
    """Counterweighted dens with an irreversible arrangement commitment.

    Each episode activates one of the two mirrored den arrangements.  The
    active heavy and light dens retain positive prior mass; the heavy and
    light candidates belonging to the *inactive* arrangement are phantom
    terminal regions.  Entering either phantom region ends the episode with
    ``phantom_penalty``.  The trigger radius defaults to the earliest possible
    visual-overlap distance (``visible_radius + cden_r``), preventing a policy
    from safely probing a zero-mass candidate and then redirecting.

    The ant starts in a small centered disc.  Besides making the two
    arrangements exactly symmetric from the physical observation, this keeps
    a direct commitment to the active heavy den from crossing a phantom merely
    because the ant happened to spawn on the far side of the arena.  Because
    mean-pinning places the active light den beyond the mirrored heavy point,
    a direct light-den route crosses that phantom circle and must detour; this
    is an intentional consequence of making both inactive candidates hazards.

    ``info["is_success"]`` distinguishes a real tag from a phantom terminal.
    The phantom locations and hit labels in ``info`` are diagnostic ground
    truth only; the PF interaction mapper does not forward them to the policy.
    """

    def __init__(self, *args, phantom_terminal_radius: float | None = None,
                 phantom_penalty: float = -300.0,
                 central_spawn_radius: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if phantom_terminal_radius is None:
            phantom_terminal_radius = self.visible_radius + self.cden_r
        if phantom_terminal_radius <= 0.0:
            raise ValueError("phantom_terminal_radius must be positive")
        if central_spawn_radius < 0.0:
            raise ValueError("central_spawn_radius must be non-negative")

        self.phantom_terminal_radius = float(phantom_terminal_radius)
        self.phantom_penalty = float(phantom_penalty)
        self.central_spawn_radius = float(central_spawn_radius)

        # The nearest phantom is a heavy candidate at radial distance h.
        # Every allowed spawn must begin strictly outside its terminal zone.
        if self.central_spawn_radius + self.phantom_terminal_radius >= self.cden_h:
            raise ValueError(
                "central spawn disc overlaps a phantom terminal region: "
                "central_spawn_radius + phantom_terminal_radius must be "
                "smaller than cden_h")
        # Also keep the nearest possible target outside the physical sensor at
        # reset; otherwise target visibility could leak occupancy immediately.
        if (self.central_spawn_radius + self.cden_r + self.visible_radius
                >= self.cden_h):
            raise ValueError(
                "central spawn can initially see a target in the heavy den: "
                "central_spawn_radius + cden_r + visible_radius must be "
                "smaller than cden_h")

        self.cden_phantom_positions = np.stack([
            -self.cden_heavy_pos,
            -self.cden_light_pos,
        ])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # The inactive arrangement is the point reflection of the active one.
        # Order: [phantom heavy, phantom light].
        self.cden_phantom_positions = np.stack([
            -self.cden_heavy_pos,
            -self.cden_light_pos,
        ])

        rr = self.central_spawn_radius * np.sqrt(self.np_random.random())
        th = self.np_random.uniform(0.0, 2.0 * np.pi)
        ant_pos = rr * np.array([np.cos(th), np.sin(th)])
        self.data.qpos[:2] = ant_pos
        self.data.mocap_pos[1][:2] = ant_pos
        self.data.mocap_pos[2][:2] = ant_pos
        mujoco.mj_forward(self.model, self.data)

        info.update({
            "cden_heavy_side": self.cden_heavy_side,
            "cden_occupied": (
                "heavy" if self._occupied_is_heavy else "light"),
            "cden_phantom_positions": self.cden_phantom_positions.copy(),
            "cden_phantom_hit": False,
            "cden_phantom_index": None,
            "termination_reason": None,
            "is_success": False,
        })
        return self._get_obs(False), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ant_pos = self.data.qpos[:2]
        phantom_distances = np.linalg.norm(
            self.cden_phantom_positions - ant_pos[None, :], axis=1)
        phantom_index = int(np.argmin(phantom_distances))
        phantom_hit = bool(
            not terminated
            and phantom_distances[phantom_index]
            < self.phantom_terminal_radius)

        if phantom_hit:
            terminated = True
            reward = self.phantom_penalty
            termination_reason = "phantom_den"
            is_success = False
        elif terminated:
            # CounterweightedDenAntTagEnv has no non-tag terminal condition.
            termination_reason = "tag"
            is_success = True
            phantom_index = None
        else:
            termination_reason = None
            is_success = False
            phantom_index = None

        info.update({
            "cden_phantom_positions": self.cden_phantom_positions.copy(),
            "cden_phantom_hit": phantom_hit,
            "cden_phantom_index": phantom_index,
            "cden_phantom_min_distance": float(np.min(phantom_distances)),
            "termination_reason": termination_reason,
            "is_success": is_success,
        })
        return obs, reward, terminated, truncated, info
