"""Light-Dark 1D (2026-09-19): faithfulness to the POMDPs.jl LightDark1D it ports, and
the behavioural property the domain exists for (the light detour beats a direct run)."""
import numpy as np
import pytest
import gymnasium as gym

import pdomains  # noqa: F401 - register environments
from pdomains.light_dark import (
    ACTIONS,
    DECLARE,
    LightDark1DConfig,
    LightDark1DEnv,
    dummy_heuristic_action,
    make_env,
    observation_sigma,
    smart_heuristic_action,
)

LEFT, RIGHT = ACTIONS.index(-1), ACTIONS.index(1)


def test_registration_and_spaces():
    env = gym.make("pdomains-light-dark-1d-v0")
    assert env.action_space.n == 3
    assert env.observation_space.shape == (1,)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert set(info) >= {"state", "sigma", "at_goal", "declared", "success"}


def test_defaults_match_the_julia_constructor():
    """LightDark1D() = LightDark1D(0.9, 10.0, -10.0, 1.0, 0.0, default_sigma)."""
    c = LightDark1DConfig()
    assert (c.discount_factor, c.correct_r, c.incorrect_r) == (0.9, 10.0, -10.0)
    assert (c.step_size, c.movement_cost) == (1.0, 0.0)
    assert (c.init_mean, c.init_std) == (2.0, 3.0)


def test_sigma_matches_default_sigma():
    """default_sigma(x) = abs(x - 5)/sqrt(2) + 1e-2, vectorised."""
    ys = np.array([-3.0, 0.0, 2.0, 5.0, 9.0])
    expected = np.abs(ys - 5.0) / np.sqrt(2.0) + 1e-2
    np.testing.assert_allclose(observation_sigma(ys), expected)
    assert observation_sigma(5.0) == pytest.approx(1e-2)     # noise floor at the light


def test_initial_state_distribution():
    env = LightDark1DEnv(LightDark1DConfig(seed=0))
    ys = []
    for _ in range(4000):
        env.reset()
        ys.append(env.y)
    assert np.mean(ys) == pytest.approx(2.0, abs=0.2)
    assert np.std(ys) == pytest.approx(3.0, abs=0.2)


def test_moves_translate_the_state_and_do_not_terminate():
    env = LightDark1DEnv(LightDark1DConfig(seed=1))
    env.reset(seed=1)
    y0 = env.y
    _, reward, terminated, truncated, _ = env.step(RIGHT)
    assert env.y == pytest.approx(y0 + 1.0)
    assert (reward, terminated, truncated) == (0.0, False, False)
    env.step(LEFT)
    assert env.y == pytest.approx(y0)


@pytest.mark.parametrize("y,expected", [(0.0, 10.0), (0.99, 10.0), (-0.99, 10.0),
                                        (1.0, -10.0), (-1.01, -10.0), (4.0, -10.0)])
def test_declare_is_scored_at_the_declaring_position(y, expected):
    env = LightDark1DEnv(LightDark1DConfig(seed=0))
    env.reset(seed=0)
    env.y = y
    _, reward, terminated, truncated, info = env.step(DECLARE)
    assert reward == expected
    assert terminated and not truncated
    assert info["declared"] and info["success"] == (expected > 0)
    assert env.y == pytest.approx(y)         # declaring does not move the agent


def test_movement_cost_is_asymmetric_by_default_and_symmetric_on_request():
    """The Julia reward for a move is -movement_cost * a, which PAYS for moving left.
    Faithful by default; `symmetric_movement_cost` is the opt-in fix."""
    env = LightDark1DEnv(LightDark1DConfig(movement_cost=1.0, seed=0))
    env.reset(seed=0)
    assert env.step(RIGHT)[1] == pytest.approx(-1.0)
    assert env.step(LEFT)[1] == pytest.approx(1.0)

    env = LightDark1DEnv(LightDark1DConfig(movement_cost=1.0, symmetric_movement_cost=True, seed=0))
    env.reset(seed=0)
    assert env.step(RIGHT)[1] == pytest.approx(-1.0)
    assert env.step(LEFT)[1] == pytest.approx(-1.0)


def test_observation_noise_follows_the_light():
    """Observations are near-exact at the light and useless far from it."""
    env = LightDark1DEnv(LightDark1DConfig(seed=0))
    env.reset(seed=0)
    errors = {}
    for y in (5.0, 0.0):
        env.y = y
        errors[y] = np.std([env._observation()[0] - y for _ in range(2000)])
    assert errors[5.0] == pytest.approx(observation_sigma(5.0), rel=0.15)
    assert errors[0.0] == pytest.approx(observation_sigma(0.0), rel=0.15)
    assert errors[5.0] < 0.05 < errors[0.0]


def test_seeded_episodes_are_reproducible():
    def rollout(seed):
        env = make_env()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        for a in (RIGHT, RIGHT, LEFT, DECLARE):
            obs, r, term, _, _ = env.step(a)
            out.append((obs, r, term))
        return out
    a, b = rollout(7), rollout(7)
    assert repr(a) == repr(b)
    assert repr(a) != repr(rollout(8))


def test_registration_truncates_at_fifty_steps():
    env = gym.make("pdomains-light-dark-1d-v0")
    env.reset(seed=0)
    for step in range(1, 60):
        _, _, terminated, truncated, _ = env.step(RIGHT)   # never declares
        assert not terminated
        if truncated:
            break
    assert step == 50


def test_invalid_action_is_rejected():
    env = make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(3)


@pytest.mark.parametrize("kwargs", [{"step_size": 0.0}, {"goal_tolerance": -1.0},
                                    {"sigma_min": 0.0}, {"init_std": 0.0},
                                    {"position_limit": 0.5}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        LightDark1DConfig(**kwargs)


# --- the behavioural property the domain exists for -----------------------------------

def _filter_step(particles, move, obs, config):
    """One bootstrap-filter step: deterministic motion, then weight by N(y, sigma(y))."""
    particles = particles + move * config.step_size
    sigma = observation_sigma(particles, config)
    logw = -0.5 * ((obs - particles) / sigma) ** 2 - np.log(sigma)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    idx = np.random.default_rng(int(abs(obs) * 1e6) % 2**31).choice(len(particles), len(particles), p=w)
    return particles[idx]


def _run_heuristic(policy, seed, n_particles=500, max_steps=50):
    config = LightDark1DConfig()
    env = LightDark1DEnv(config)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    particles = rng.normal(config.init_mean, config.init_std, n_particles)
    particles = _filter_step(particles, 0, obs[0], config)
    total, discount = 0.0, 1.0
    for _ in range(max_steps):
        a = policy(float(particles.mean()), float(particles.std()))
        obs, reward, terminated, _, info = env.step(a)
        total += discount * reward
        discount *= config.discount_factor
        if terminated:
            return total, info["success"]
        particles = _filter_step(particles, ACTIONS[a], obs[0], config)
    return total, False


def test_light_seeking_policy_beats_the_mean_only_policy():
    """The domain's whole point: a policy that reads the belief's SPREAD detours to the
    light, localises and declares; one that reads only the mean cannot localise."""
    smart = [_run_heuristic(smart_heuristic_action, s) for s in range(20)]
    dummy = [_run_heuristic(dummy_heuristic_action, s) for s in range(20)]
    smart_success = np.mean([s for _, s in smart])
    dummy_success = np.mean([s for _, s in dummy])
    assert smart_success > 0.8, f"smart policy only succeeded {smart_success:.2f} of the time"
    assert np.mean([r for r, _ in smart]) > np.mean([r for r, _ in dummy])
    assert smart_success > dummy_success
