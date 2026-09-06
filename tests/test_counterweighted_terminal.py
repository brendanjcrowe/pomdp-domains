import gymnasium as gym
import mujoco
import numpy as np

import pdomains  # noqa: F401 - register environments


ENV_ID = "pdomains-ant-tag-cdens-terminal-v0"


def _teleport_ant(raw, pos):
    pos = np.asarray(pos, dtype=np.float64)
    raw.data.qpos[:2] = pos
    raw.data.qvel[:] = 0.0
    raw.data.mocap_pos[1][:2] = pos
    raw.data.mocap_pos[2][:2] = pos
    mujoco.mj_forward(raw.model, raw.data)


def _find_seed(env, *, side, occupied_is_heavy):
    raw = env.unwrapped
    for seed in range(10_000):
        env.reset(seed=seed)
        if (raw.cden_heavy_side == side
                and raw._occupied_is_heavy == occupied_is_heavy):
            return seed
    raise AssertionError("could not find requested latent configuration")


def test_registered_geometry_spawn_and_moment_aliasing():
    env = gym.make(ENV_ID)
    raw = env.unwrapped

    assert raw.phantom_penalty == -300.0

    side_counts = np.zeros(2, dtype=int)
    occupied_heavy = 0
    starts = {0: [], 1: []}
    for seed in range(400):
        obs, info = env.reset(seed=seed)
        side = raw.cden_heavy_side
        side_counts[side] += 1
        occupied_heavy += int(raw._occupied_is_heavy)
        starts[side].append(raw.data.qpos[:2].copy())

        assert np.linalg.norm(raw.data.qpos[:2]) <= (
            raw.central_spawn_radius + 1e-10)
        assert np.min(np.linalg.norm(
            raw.cden_phantom_positions - raw.data.qpos[:2], axis=1
        )) > raw.phantom_terminal_radius
        np.testing.assert_allclose(
            raw.cden_phantom_positions,
            np.stack([-raw.cden_heavy_pos, -raw.cden_light_pos]),
        )
        np.testing.assert_array_equal(obs[-2:], np.zeros(2))
        assert info["is_success"] is False

    # Both episode latents are genuinely sampled, and neither changes the
    # centered physical spawn distribution in a detectable gross way.
    assert np.all(side_counts > 150)
    assert 0.68 < occupied_heavy / 400 < 0.79
    np.testing.assert_allclose(
        np.mean(starts[0], axis=0), np.mean(starts[1], axis=0), atol=0.08)

    # Population Gaussian moments are exactly identical under mirroring.
    u = np.array([1.0, 1.0]) / np.sqrt(2.0)
    w = raw.cden_w_heavy
    disc_cov = (raw.cden_r ** 2 / 4.0) * np.eye(2)
    moments = []
    characteristic_values = []
    for sign in (-1.0, 1.0):
        heavy = sign * raw.cden_h * u
        light = -sign * raw.cden_f * u
        mean = w * heavy + (1.0 - w) * light
        cov = (w * np.outer(heavy, heavy)
               + (1.0 - w) * np.outer(light, light)
               + disc_cov)
        moments.append((mean, cov))

        # A one-dimensional characteristic-function sample along the den
        # axis.  Symmetric within-den noise contributes the same real factor
        # to both arrangements, so the imaginary sign flip remains.
        frequency = 0.5
        phi = (w * np.exp(1j * frequency * sign * raw.cden_h)
               + (1.0 - w)
               * np.exp(-1j * frequency * sign * raw.cden_f))
        characteristic_values.append(phi)

    np.testing.assert_allclose(moments[0][0], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(moments[1][0], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(moments[0][1], moments[1][1], atol=1e-12)
    np.testing.assert_allclose(
        characteristic_values[0], np.conj(characteristic_values[1]),
        atol=1e-12)
    assert abs(characteristic_values[0].imag) > 0.1

    env.close()


def test_both_inactive_arrangement_candidates_are_terminal_failures():
    env = gym.make(ENV_ID)
    raw = env.unwrapped
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    for side in (0, 1):
        seed = _find_seed(env, side=side, occupied_is_heavy=True)
        for phantom_index in (0, 1):
            env.reset(seed=seed)
            _teleport_ant(raw, raw.cden_phantom_positions[phantom_index])
            _, reward, terminated, truncated, info = env.step(zero_action)

            assert terminated is True
            assert truncated is False
            assert reward == raw.phantom_penalty
            assert info["cden_phantom_hit"] is True
            assert info["cden_phantom_index"] == phantom_index
            assert info["termination_reason"] == "phantom_den"
            assert info["is_success"] is False

    env.close()


def test_active_empty_den_spooks_but_does_not_terminal_fail():
    env = gym.make(ENV_ID)
    raw = env.unwrapped
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    for side in (0, 1):
        # A light-occupied episode makes the active heavy den empty.
        seed = _find_seed(env, side=side, occupied_is_heavy=False)
        env.reset(seed=seed)
        _teleport_ant(raw, raw.cden_heavy_pos)
        _, _, terminated, truncated, info = env.step(zero_action)

        assert terminated is False
        assert truncated is False
        assert info["cden_spooked"] is True
        assert info["cden_phantom_hit"] is False
        assert info["termination_reason"] is None
        assert info["is_success"] is False

    env.close()


def test_real_tag_is_terminal_success_not_phantom_failure():
    env = gym.make(ENV_ID)
    raw = env.unwrapped
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    for side in (0, 1):
        seed = _find_seed(env, side=side, occupied_is_heavy=True)
        env.reset(seed=seed)
        _teleport_ant(raw, raw.get_target_pos())
        _, reward, terminated, truncated, info = env.step(zero_action)

        assert terminated is True
        assert truncated is False
        assert reward == 0.0
        assert info["cden_phantom_hit"] is False
        assert info["termination_reason"] == "tag"
        assert info["is_success"] is True

    env.close()
