import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

GREEN = [0, 1, 0, 0.5]
RED = [1, 0, 0, 0.5]

class AntEnv(gym.Env):

    def __init__(self, seed=None, num_frames_skip=15, rendering=False):

        #################### START CONFIGS #######################

        model_name = "ant_heaven_hell.xml"

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

        self.state_dim = len(self.data.qpos) + len(self.data.qvel) + 1 # State will include (i) joint angles and (ii) joint velocities, extra info
        self.action_dim = len(self.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.model.actuator_ctrlrange[:,1].copy() # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges

        self.initial_state_space = initial_state_space

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

        self.heaven_hell = [[-6.25, 6.0], [6.25, 6.0]]
        self.priest_pos = [0.0, 6.0]
        self.radius = 2.0

        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'left_area')
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'right_area')

        self.json_writer = None
        self.list_pos = []

        self.seed(seed)

    # Get state, which concatenates joint positions and velocities
    def _get_obs(self, reveal_heaven_direction, at_reset=False):
        if at_reset:
            return np.concatenate((self.data.qpos, self.data.qvel, np.zeros(1)))

        heaven_direction = self.heaven_direction * np.ones(1) if reveal_heaven_direction else np.zeros(1)

        return np.concatenate((self.data.qpos, self.data.qvel, heaven_direction))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        # Reset controls
        self.data.ctrl[:] = 0

        # Set initial joint positions and velocities
        for i in range(len(self.data.qpos)):
            self.data.qpos[i] = self.np_random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.data.qvel)):
            self.data.qvel[i] = self.np_random.uniform(self.initial_state_space[len(self.data.qpos) + i][0],self.initial_state_space[len(self.data.qpos) + i][1])

        # Initialize ant's position
        self.data.qpos[0] = self.np_random.uniform(-1.0, 1.0)
        self.data.qpos[1] = self.np_random.uniform(0.0, 1.0)

        # Randomize the side of heaven
        coin_face = self.np_random.random() >= 0.5

        # -1: heaven on left, 1: heaven on the right
        self.heaven_pos = self.heaven_hell[coin_face]
        self.hell_pos = self.heaven_hell[not coin_face]

        self.heaven_direction = np.sign(self.heaven_pos[0])
        
        # Changing the color of heaven/hell areas
        if self.heaven_direction > 0:

            # print("Heaven on the right")

            # heaven on the right 
            self.model.site_rgba[self.right_id] = GREEN
            self.model.site_rgba[self.left_id] = RED
        else:
            # print("Heaven on the left")

            # heaven on the left
            self.model.site_rgba[self.left_id] = GREEN
            self.model.site_rgba[self.right_id] = RED
            
        mujoco.mj_step(self.model, self.data)

        # Updated for gymnasium: return observation and info
        return self._get_obs(False, at_reset=True), {}

    def _do_reveal_target(self):

        ant_pos = self.data.qpos[:2]

        d2priest = np.linalg.norm(ant_pos - self.priest_pos)
        if (d2priest < self.radius):
            reveal_heaven_direction = True
        else:
            reveal_heaven_direction = False

        return reveal_heaven_direction

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            mujoco.mj_step(self.model, self.data)
            if self.visualize and self.viewer is not None:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    self.viewer.sync()

        ant_pos = self.data.qpos[:2]

        d2heaven = np.linalg.norm(ant_pos - self.heaven_pos)

        done = False
        env_reward = -1

        # + reward and terminate the episode if going to heaven
        if (d2heaven <= self.radius):
            env_reward = 0
            done = True

        d2hell = np.linalg.norm(ant_pos - self.hell_pos)

        # terminate the episode if going to  hell
        if (d2hell <= self.radius):
            env_reward = -10.0
            done = True

        reveal_heaven_direction = self._do_reveal_target()

        # Updated for gymnasium: split done into terminated and truncated
        terminated = done
        truncated = False

        return self._get_obs(reveal_heaven_direction), env_reward, terminated, truncated, {}

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]
