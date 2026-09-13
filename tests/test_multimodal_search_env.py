"""Multimodal Search moved into pdomains (2026-09-13): bit-identical to the set_transformer copy,
and the registration."""
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import pdomains  # noqa: F401 - register environments
from pdomains.multimodal_search import BASE_OBS_DIM, MODE_OBS_DIM, MultimodalSearchConfig, MultimodalSearchEnv, make_env

_CANDIDATES = [Path(__file__).resolve().parents[2] / "set_transformer" / "set_transformer" / "rl" / "envs" / "multimodal_search.py",
               Path.home() / "Documents" / "Research" / "rl_for_beliefmdps" / "set_transformer" / "set_transformer"
               / "rl" / "envs" / "multimodal_search.py"]
_ORIGINAL = next((p for p in _CANDIDATES if p.is_file()), None)


def _load_original():
    spec = importlib.util.spec_from_file_location("_orig_multimodal_search", _ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rollout(env, seed, n_steps, action_rng):
    out = []
    obs, info = env.reset(seed=seed)
    out.append((obs, None, False, False, info))
    for _ in range(n_steps):
        a = action_rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        out.append((obs, r, term, trunc, info))
        if term or trunc:
            break
    return out


def _plain(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}


@pytest.mark.skipif(_ORIGINAL is None, reason="set_transformer copy of the env not beside this checkout")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_moved_env_is_bit_identical_to_the_set_transformer_copy(seed):
    orig = _load_original()
    for kw in ({}, dict(k_min=3, k_max=5, visibility_radius=2.0)):
        a = _rollout(MultimodalSearchEnv(MultimodalSearchConfig(**kw)), seed, 42, np.random.default_rng(seed))
        b = _rollout(orig.MultimodalSearchEnv(orig.MultimodalSearchConfig(**kw)), seed, 42, np.random.default_rng(seed))
        assert len(a) == len(b)
        for (oa, ra, ta, ua, ia), (ob, rb, tb, ub, ib) in zip(a, b):
            np.testing.assert_array_equal(oa, ob)
            assert ra == rb and ta == tb and ua == ub and _plain(ia) == _plain(ib)
    # the registered env (through TimeLimit) plays the same episode as the bare original
    a = _rollout(gym.make("pdomains-multimodal-search-v0"), seed, 42, np.random.default_rng(seed))
    b = _rollout(orig.MultimodalSearchEnv(orig.MultimodalSearchConfig()), seed, 42, np.random.default_rng(seed))
    assert len(a) == len(b)
    for (oa, *ra), (ob, *rb) in zip(a, b):
        np.testing.assert_array_equal(oa, ob)
        assert ra[:3] == rb[:3] and _plain(ra[3]) == _plain(rb[3])


def test_registration_cap_42_and_default_configuration():
    assert gym.spec("pdomains-multimodal-search-v0").max_episode_steps == 42
    env = gym.make("pdomains-multimodal-search-v0")
    cfg = env.unwrapped.config if hasattr(env.unwrapped, "config") else env.unwrapped.cfg
    assert cfg.max_steps == 42 and cfg.k_max == 10 and cfg.arena_half == 14.0
    obs, _ = env.reset(seed=0)
    assert obs.shape == (BASE_OBS_DIM + MODE_OBS_DIM * cfg.k_max,) == (67,)
    n = 0
    for _ in range(100):
        _, _, term, trunc, _ = env.step(np.zeros(2, np.float32))
        n += 1
        if term or trunc:
            break
    assert n <= 42
    # the factory forwards keyword arguments into the config
    small = make_env(k_min=2, k_max=4)
    small_cfg = small.config if hasattr(small, "config") else small.cfg
    assert small_cfg.k_max == 4 and small.observation_space.shape == (BASE_OBS_DIM + MODE_OBS_DIM * 4,)
