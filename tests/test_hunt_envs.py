"""The hunt envs moved into pdomains (2026-09-13): bit-identical to the originals under the
original rule, the most-var rule's guarantees, and the three registrations."""
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import pdomains  # noqa: F401 - register environments
from pdomains.hunt import ClusterHuntEnv, MinMassHuntConfig, MinMassHuntEnv

# The originals live in the parent repo (rl_for_beliefmdps/src/hunt_tasks/env); this checkout may
# be a worktree elsewhere, so look in both places and skip when neither exists.
_CANDIDATES = [Path(__file__).resolve().parents[2] / "src" / "hunt_tasks" / "env",
               Path.home() / "Documents" / "Research" / "rl_for_beliefmdps" / "src" / "hunt_tasks" / "env"]
_ORIGINALS = next((p for p in _CANDIDATES if p.is_dir()), None)


_ORIGINAL_CACHE = {}


def _load_original(name):
    """Execute the original file once per session (each execution re-registers its old gym id)."""
    if name in _ORIGINAL_CACHE:
        return _ORIGINAL_CACHE[name]
    module = _ORIGINAL_CACHE[name] = _exec_original(name)
    return module


def _exec_original(name):
    spec = importlib.util.spec_from_file_location(f"_orig_{name}", _ORIGINALS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


def _rollout(env, seed, n_steps, action_rng):
    """obs / reward / terminated / truncated / info over a fixed action sequence."""
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


def _assert_same(a, b):
    assert len(a) == len(b)
    for (oa, ra, ta, ua, ia), (ob, rb, tb, ub, ib) in zip(a, b):
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        assert ra == rb and ta == tb and ua == ub
        assert {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ia.items()} == \
               {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ib.items()}


@pytest.mark.skipif(_ORIGINALS is None, reason="src/hunt_tasks/env not beside this checkout")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_moved_envs_are_bit_identical_to_the_originals(seed):
    orig_ch = _load_original("cluster_hunt")
    orig_mm = _load_original("min_mass_hunt")
    for new_cls, old_cls, kw in ((ClusterHuntEnv, orig_ch.ClusterHuntEnv, dict(hit_radius=0.6, min_sep=2.5, max_steps=60)),
                                 (ClusterHuntEnv, orig_ch.ClusterHuntEnv, {}),
                                 (MinMassHuntEnv, orig_mm.MinMassHuntEnv, {}),
                                 (MinMassHuntEnv, orig_mm.MinMassHuntEnv, dict(n_active=3))):
        a = _rollout(new_cls(**kw), seed, 60, np.random.default_rng(100 + seed))
        b = _rollout(old_cls(**kw), seed, 60, np.random.default_rng(100 + seed))
        _assert_same(a, b)
    # the registered env (through TimeLimit) plays the same episode as the bare class
    reg = gym.make("pdomains-least-mass-v0")
    a = _rollout(reg, seed, 60, np.random.default_rng(seed))
    b = _rollout(orig_mm.MinMassHuntEnv(), seed, 60, np.random.default_rng(seed))
    _assert_same(a, b)


def test_most_var_target_is_the_widest_cluster_with_the_margin_and_equal_counts():
    env = MinMassHuntEnv(target_rule="max_var", sigma_hi=0.9, sigma_margin=0.2)
    rng = np.random.default_rng(0)
    for _ in range(500):
        env.set_n_active(int(rng.integers(2, 6)))
        _, info = env.reset(seed=int(rng.integers(1 << 30)))
        sig = np.sort(env.sigmas)
        assert env.target == int(np.argmax(env.sigmas)) == info["target"]
        assert sig[-1] - sig[-2] >= 0.2 - 1e-12
        assert 0.3 <= sig[0] and sig[-1] <= 0.9
        assert env.counts.sum() == 100 and env.counts.max() - env.counts.min() <= 1
        assert len(env.particles) == 100
    # the original rule is untouched: lightest cluster, uneven counts with the mass margin
    env = MinMassHuntEnv()
    env.reset(seed=3)
    assert env.target == int(np.argmin(env.counts))
    c = np.sort(env.counts)
    assert c[1] - c[0] >= env.cfg.mass_margin
    with pytest.raises(ValueError, match="target_rule"):
        MinMassHuntEnv(target_rule="max_mass")
    assert MinMassHuntConfig().target_rule == "min_mass" and MinMassHuntConfig().sigma_margin == 0.2


def test_registrations_resolve_with_cap_60_and_the_recorded_configs():
    for env_id in ("pdomains-cluster-hunt-v0", "pdomains-least-mass-v0", "pdomains-most-var-v0"):
        assert gym.spec(env_id).max_episode_steps == 60
        env = gym.make(env_id)
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"agent", "particles"} and obs["particles"].shape == (100, 2)
        assert env.unwrapped.cfg.max_steps == 60
        # the env's own truncation and the registration cap coincide: the episode ends at 60
        n = 0
        for _ in range(100):
            _, _, term, trunc, _ = env.step(np.zeros(2, np.float32))
            n += 1
            if term or trunc:
                break
        assert n <= 60
    ch = gym.make("pdomains-cluster-hunt-v0").unwrapped.cfg
    assert (ch.hit_radius, ch.min_sep, ch.max_steps) == (0.6, 2.5, 60)
    mv = gym.make("pdomains-most-var-v0").unwrapped.cfg
    assert (mv.target_rule, mv.sigma_hi, mv.sigma_margin, mv.sigma_lo) == ("max_var", 0.9, 0.2, 0.3)
    assert gym.make("pdomains-least-mass-v0").unwrapped.cfg.target_rule == "min_mass"
