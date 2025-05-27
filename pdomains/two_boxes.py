import operator
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


class BoxEnv(gym.Env):

    def __init__(self, rendering=False, seed=None):
        """
        """

        super(BoxEnv, self).__init__()

        # X range
        self.x_left_limit = 0
        self.x_right_limit = 100
        self.x_g_left_limit = self.x_left_limit + 10
        self.x_g_right_limit = self.x_right_limit - 10

        self.go_to_left = 0.0

        # Boundary
        self.lboundary = 40
        self.rboundary = 60

        self.action_dim = 1

        #################### END CONFIGS #######################

        # mujoco-py
        xml_path = Path(__file__).resolve().parent / 'assets' / 'two_boxes.xml'
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None  # Initializes only when self.render() is called.
        self.rendering = rendering

        # Constants
        self.FINGER_TIP_OFFSET = 0.375

        # MuJoCo
        # bodies
        self.gripah_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripah-base')
        self.small_box_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'small_box')
        self.small_box_2_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'small_box_2')

        self.big_box_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'big_box')
        self.big_box_2_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'big_box_2')

        self.lregion_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'left_boundary')
        self.rregion_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'right_boundary')        

        # geoms
        self.wide_finger_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'geom:wide-finger')
        self.wide_finger_tip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'geom:wide-finger-tip')
        # joints
        self.slide_x_c_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'slide:gripah-base-x')
        self.hinge_wide_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge:wide-finger')
        self.hinge_narrow_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge:narrow-finger')
        # actuators
        self.velocity_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'velocity:x')
        self.velocity_narrow_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'velocity:narrow-finger')
        self.position_narrow_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'position:narrow-finger')

        self.model.jnt_range[self.slide_x_c_id][0] = self.x_left_limit
        self.model.jnt_range[self.slide_x_c_id][1] = self.x_right_limit
        self._place_grid_marks()

        # Gripah
        self.default_velocity = 15
        self.step_length = 100
        self.low_stiffness = 200

        self.qpos_nfinger = 0

        self.x_box = None

        self.state_dim = 2
        self.low_obs_dim = 2

        self.prepare_high_obs_fn = lambda state: state

        # TODO: Tune this
        self.min_x_g_box_distance = 6

        # Action
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        self.x_g = 0
        self.theta = 0

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,), dtype=np.float32)

        # The finger is always soft
        self.model.jnt_stiffness[self.hinge_wide_finger_id] = self.low_stiffness

        # numpy random
        self.np_random = None
        self.seed(seed)

    def reset(self):
        """
        """

        # Resets the mujoco env
        mujoco.mj_resetData(self.model, self.data)

        self.box_1 = 40
        self.box_2 = 60

        self.x_g = 50

        # ok = False
        # while(not ok):
        #     self.x_g = self.np_random.uniform(self.x_g_left_limit, self.x_g_right_limit)
        #     if abs(self.x_g - self.box_1) >= self.min_x_g_box_distance and abs(self.x_g - self.box_2) >= self.min_x_g_box_distance:
        #         ok = True

        # Assigns the parameters to mujoco-py
        option = self.np_random.integers(4)

        # Both boxes are small
        if option == 0:
            # Pass this batch by going to the right
            self.go_to_left = False

            self.model.body_pos[self.small_box_bid][0] = self.box_1
            self.model.body_pos[self.small_box_2_bid][0] = self.box_2

            self.model.body_pos[self.big_box_bid][0] = -500
            self.model.body_pos[self.big_box_2_bid][0] = -500

        # Small box on the left, big box on the right, Not pass
        if option == 1:
            self.go_to_left = True

            self.model.body_pos[self.small_box_bid][0] = self.box_1
            self.model.body_pos[self.big_box_bid][0] = self.box_2

            self.model.body_pos[self.small_box_2_bid][0] = -500
            self.model.body_pos[self.big_box_2_bid][0] = -500

        # Small box on the right, big box on the left, Not pass
        if option == 2:
            self.go_to_left = True

            self.model.body_pos[self.small_box_bid][0] = self.box_2
            self.model.body_pos[self.big_box_bid][0] = self.box_1

            self.model.body_pos[self.small_box_2_bid][0] = -500
            self.model.body_pos[self.big_box_2_bid][0] = -500

        # Both boxes are big, Pass
        if option == 3:
            self.go_to_left = False

            self.model.body_pos[self.big_box_2_bid][0] = self.box_2
            self.model.body_pos[self.big_box_bid][0] = self.box_1

            self.model.body_pos[self.small_box_2_bid][0] = -500
            self.model.body_pos[self.small_box_bid][0] = -500

        # qpos
        self.data.qpos[self.slide_x_c_id] = self.x_g + self.FINGER_TIP_OFFSET
        self._control_narrow_finger(theta_target=0.9, teleport=True)

        self._update_state()

        # Updated for gymnasium: return observation and info
        return self._get_obs(), {}

    def step(self, action):
        """
        Steps the simulation with the given action and returns the observations.

        :param action: (movement)
        :return: the observations of the environment
        """

        env_reward = -1
        done = False

        self._move_gripper(action)

        self._update_state()

        if self.x_g <= self.x_g_left_limit:
            if self.go_to_left:
                env_reward = 0.0
            else:
                env_reward = -5.0 # negative reward

        if self.x_g >= self.x_g_right_limit:
            if not self.go_to_left:
                env_reward = 0.0
            else:
                env_reward = -5.0

            done = True

        # Updated for gymnasium: split done into terminated and truncated
        terminated = done
        truncated = False

        return self._get_obs(), env_reward, terminated, truncated, {}

    def _get_obs(self):
        
        return np.array((self.x_g / self.x_right_limit, 
                self.theta))

    def render(self, mode='human'):
        if self.rendering:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Settings for camera can be done through the viewer handle's cam attribute
                # For example: self.viewer.cam.distance = 150 (if available and needed)
                # The passive viewer might require locking before changing cam attributes: with self.viewer.lock(): self.viewer.cam.distance = 150
                # For now, I will keep the camera adjustments commented out as their direct migration can be complex.
                # self.viewer.cam.distance = 150
                # self.viewer.cam.azimuth = 90
                # self.viewer.cam.elevation = -15

            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.sync()

    def close(self):
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :seed the seed for the random number generator(s)
        """

        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def _update_state(self):
        """
        Samples the data from sensors and updates the state.
        """

        self.x_g = self._get_raw_x_g()
        self.theta = self._get_theta()

    def _move_gripper(self, movement):
        if self.x_g <= self.x_g_left_limit or self.x_g >= self.x_g_right_limit:
            return        

        self._control_slider_x(movement)

    def _control_slider_x(self, scale):
        """
        Controls the x slider of the gripah.
        """
        self.data.ctrl[self.velocity_x_id] = self.default_velocity * scale

    def _control_narrow_finger(self, theta_target, teleport=False):
        """
        Controls the narrow finger of the gripah.

        :param theta_target: the target angular position of the narrow finger
        :param teleport: if True, the narrow finger position is set immediately;
                         otherwise, the narrow finger is controlled by velocity actuator.
        """

        if teleport:
            # Teleports the narrow finger to the specified position.
            self.data.qpos[self.hinge_narrow_finger_id] = theta_target
            # Updates the actuator according to the new position immediately.
            self.data.ctrl[self.position_narrow_finger_id] = theta_target
        else:
            # Controls the narrow finger by setting the velocity.
            theta_error = theta_target - self._get_narrow_finger_angle()
            velocity = np.sign(theta_error) * self.default_velocity

            # If the error is small enough, directly set the velocity to 0.
            if np.abs(theta_error) < 0.05:
                velocity = 0

            # Sets the narrow finger velocity.
            self.data.ctrl[self.velocity_narrow_finger_id] = velocity

    def _get_theta(self):
        """
        Gets the state of the narrow finger.
        """

        return self.data.sensordata[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'touch:wide-finger-tip')]

    def _get_wide_finger_angle(self):
        """
        Gets the angle of the wide finger.
        """

        return self.data.qpos[self.hinge_wide_finger_id]

    def _get_narrow_finger_angle(self):
        """
        Gets the angle of the narrow finger.
        """

        return self.data.qpos[self.hinge_narrow_finger_id]

    def _get_narrow_finger_stiffness(self):
        """
        Gets the stiffness of the narrow finger.
        """

        return self.model.jnt_stiffness[self.hinge_narrow_finger_id]

    def _get_raw_x_g(self):
        """
        Gets the x coordinate of the finger tip.
        """

        return self.data.geom_xpos[self.wide_finger_tip_geom_id][0]

    def _get_gripah_raw_state(self):
        """
        Returns the state of the gripah:
            x coordinate of the gripah,
            angle of the wide finger and
            angle of the narrow finger
        """

        return (self.data.qpos[self.slide_x_c_id],
                self._get_wide_finger_angle(),
                self._get_narrow_finger_angle())

    def _place_grid_marks(self):
        """
        Places the grid marks in the scene.
        """

        x_pos = np.arange(self.x_left_limit, self.x_right_limit + 1, 10)
        y_pos = -10
        z_pos = -10

        for i, x in enumerate(x_pos):
            self.model.geom_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'grid_mark_{i}')][0] = x
            self.model.geom_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'grid_mark_{i}')][1] = y_pos
            self.model.geom_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'grid_mark_{i}')][2] = z_pos

        self.model.site_pos[self.lregion_bid][0] = self.lboundary
        self.model.site_pos[self.rregion_bid][0] = self.rboundary

    def _is_in_collision(self, body1_id, body2_id):
        """
        Checks if two bodies are in collision.

        :param body1_id: the id of the first body
        :param body2_id: the id of the second body
        :return: True if the two bodies are in collision, False otherwise
        """

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Checks if the two bodies are in collision
            if (contact.geom1 == self.model.geom_parentid[body1_id] and contact.geom2 == self.model.geom_parentid[body2_id]) or \
               (contact.geom2 == self.model.geom_parentid[body1_id] and contact.geom1 == self.model.geom_parentid[body2_id]):
                return True

        return False
