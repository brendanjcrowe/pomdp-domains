import sys
import unittest
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
#from pdomains.car_flag import CarEnv


# Add mock classes to avoid MuJoCo dependency
class MockMjSim:
    def __init__(self, *args, **kwargs):
        self.data = MagicMock()
        self.data.qpos = np.zeros(10)
        self.data.qvel = np.zeros(10)
        self.data.ctrl = np.zeros(5)
        self.data.mocap_pos = [np.zeros(3) for _ in range(3)]
        self.model = MagicMock()
        self.model.actuator_ctrlrange = np.array([[-1, 1] for _ in range(5)])
        self.model.site_rgba = np.zeros((5, 4))
        
    def step(self):
        pass
    
    def reset(self):
        pass

class MockMjViewer:
    def __init__(self, *args, **kwargs):
        pass
    
    def render(self):
        pass

class MockLoad:
    def __call__(self, *args, **kwargs):
        return MagicMock()

# Mock the mujoco_py module
# This mock is for the OLD mujoco_py, which might not be relevant if all envs are migrated to new mujoco
# However, if any test *itself* tries to import mujoco_py, this might still be useful.
# For the new `mujoco` library, tests should ideally run with it installed.
sys.modules['mujoco_py'] = MagicMock()
sys.modules['mujoco_py'].MjSim = MockMjSim
sys.modules['mujoco_py'].MjViewer = MockMjViewer
sys.modules['mujoco_py'].load_model_from_path = MockLoad()

from pdomains.ant_heaven_hell import AntEnv

# Import our environments after mocking mujoco_py
from pdomains.ant_tag import AntTagEnv

# from pdomains.car_flag import CarEnv # Temporarily commented out due to rendering import issues
from pdomains.two_boxes import BoxEnv


class TestGymnasiumRefactoring(unittest.TestCase):
    
    def test_step_returns_five_values(self):
        """Test that step() returns 5 values as per gymnasium's API."""
        envs = [
            AntTagEnv(rendering=False),
            AntEnv(rendering=False),
            BoxEnv(rendering=False),
            #CarEnv(rendering=False) # Temporarily commented out
        ]
        
        for env in envs:
            with self.subTest(env=env.__class__.__name__):
                # Reset the environment
                obs, info = env.reset()
                self.assertIsInstance(info, dict)
                
                # Take a step
                action = env.action_space.sample()
                result = env.step(action)
                
                # Check that it returns 5 values
                self.assertEqual(len(result), 5)
                obs, reward, terminated, truncated, info = result
                
                # Check types
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIsInstance(info, dict)

if __name__ == "__main__":
    unittest.main() 