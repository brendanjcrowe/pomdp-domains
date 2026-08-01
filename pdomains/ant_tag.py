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

    def __init__(self, seed=None, num_frames_skip=15, rendering=False):

        model_name = "ant_tag_small.xml"

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

        self.cage_max_x = 4.5
        self.cage_max_y = 4.5

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

        self.visible_radius = 3.0
        self.tag_radius = 1.5
        self.min_distance = 5.0
        self.target_step = 0.5

        self.seed(seed)

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