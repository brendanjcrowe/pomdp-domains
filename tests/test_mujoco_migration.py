import numpy as np
import pytest

# Try to import all relevant environments
try:
    from pdomains.ant_tag import AntTagEnv
    MUJOCO_ENVS_AVAILABLE = True
except ImportError:
    AntTagEnv = None
    MUJOCO_ENVS_AVAILABLE = False # Flag to skip tests if core mujoco envs are not found

try:
    from pdomains.ur5_reacher import UR5Env
except ImportError:
    UR5Env = None
    # If AntTagEnv was available but UR5Env is not, it might indicate a partial setup.
    # For simplicity, we'll rely on the MUJOCO_ENVS_AVAILABLE from AntTagEnv for now.

try:
    from pdomains.two_boxes import BoxEnv
except ImportError:
    BoxEnv = None

try:
    from pdomains.ant_heaven_hell import (
        AntEnv as AntHeavenHellEnv,  # Alias to avoid name clash
    )
except ImportError:
    AntHeavenHellEnv = None

# List of MuJoCo environments to test
# We filter out None values in case some environments couldn't be imported.
MUJOCO_ENV_CLASSES = [env for env in [AntTagEnv, UR5Env, BoxEnv, AntHeavenHellEnv] if env is not None]

@pytest.mark.skipif(not MUJOCO_ENVS_AVAILABLE, reason="MuJoCo environments not available (e.g., mujoco library missing or pdomains not found)")
@pytest.mark.parametrize("env_class", MUJOCO_ENV_CLASSES)
def test_mujoco_env_instantiation_step_render(env_class):
    """Tests instantiation, reset, step, and rendering of a MuJoCo environment."""
    env = None
    try:
        # Instantiate with rendering True to test viewer initialization
        env = env_class(rendering=True)
        assert env is not None, "Environment instantiation failed"

        obs, info = env.reset()
        assert obs is not None, "Reset failed to return an observation"

        # Determine a sample valid action
        action_space = env.action_space
        sample_action = action_space.sample()
        
        # Ensure the action is in the correct format (e.g. np.array if Box space)
        if not isinstance(sample_action, np.ndarray):
            sample_action = np.array([sample_action])
        
        # If action space is Box and has a shape, ensure sample_action matches
        if hasattr(action_space, 'shape') and action_space.shape is not None and sample_action.shape != action_space.shape:
             if len(action_space.shape) > 0 and action_space.shape[0] == sample_action.size:
                 sample_action = sample_action.reshape(action_space.shape)
             else:
                # Fallback for more complex shapes, or if simple reshape won't work.
                # This might happen with nested spaces, though less common for basic mujoco envs.
                # For now, we'll print a warning and proceed, but this might need refinement
                # if specific environments have very complex action structures.
                print(f"Warning: Sample action shape {sample_action.shape} might not perfectly match action space {action_space.shape}. Attempting to proceed.")


        obs, reward, terminated, truncated, info = env.step(sample_action)
        assert obs is not None, "Step failed to return an observation"

        # Test rendering part by trying to close viewer if it exists
        if hasattr(env, 'viewer') and env.viewer is not None:
            if hasattr(env.viewer, 'is_running') and env.viewer.is_running():
                env.viewer.close()
            elif hasattr(env.viewer, 'close') and not hasattr(env.viewer, 'is_running'): # For viewers without is_running()
                env.viewer.close()
        
        # Explicitly call the environment's close method if it exists (good practice for Gymnasium envs)
        if hasattr(env, 'close') and callable(env.close):
            env.close()

    except Exception as e:
        pytest.fail(f"Test failed for {env_class.__name__} with error: {e}")
    finally:
        # Ensure viewer is closed even if an assertion fails mid-test
        if env is not None and hasattr(env, 'viewer') and env.viewer is not None:
            if hasattr(env.viewer, 'is_running') and env.viewer.is_running():
                env.viewer.close()
            elif hasattr(env.viewer, 'close') and not hasattr(env.viewer, 'is_running'): # For viewers without is_running()
                try:
                    env.viewer.close()
                except Exception as e_close:
                    print(f"Error closing viewer during finally block for {env_class.__name__}: {e_close}")
        if env is not None and hasattr(env, 'close') and callable(env.close):
            try:
                env.close() # General env close
            except Exception as e_env_close:
                print(f"Error calling env.close() during finally block for {env_class.__name__}: {e_env_close}")

if not MUJOCO_ENV_CLASSES:
    print("\nWarning: No MuJoCo environment classes were found or imported successfully for testing.\n"\
          "Please ensure `mujoco` is installed and `pdomains` environments are accessible.")

# Example of how to run this test with pytest:
# pytest tests/test_mujoco_migration.py 