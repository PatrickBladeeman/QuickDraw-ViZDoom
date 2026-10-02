"""Native-geometry controls for the Basic goal-conditioning environment."""

import pytest


pytest.importorskip("vizdoom")

from quickdraw_vizdoom.envs.native_basic import NativeBasicV1Env  # noqa: E402


def _native_oracle(seed: int) -> tuple[bool, int]:
    with NativeBasicV1Env() as env:
        _, info = env.reset(seed=seed)
        assert info["native_geometry"]["target_visible"]
        for _ in range(120):
            error = info["alignment_error"]
            action = (
                1 if error > 8 else 2 if error < -8 else 0,
                0 if abs(error) > 8 else 1,
            )
            _, _, terminated, truncated, info = env.step(action)
            assert ("target_hit" in info["events"]) == bool(
                info["native_target_killed"]
            )
            if terminated or truncated:
                return terminated, info["native_kill_count"]
    return False, 0


def test_native_oracle_kills_the_real_target_for_nine_seeds():
    outcomes = [_native_oracle(seed) for seed in range(36000, 36009)]
    assert outcomes == [(True, 1)] * 9


def test_unaligned_shot_is_not_a_wrapper_false_positive():
    with NativeBasicV1Env() as env:
        env.reset(seed=36000)
        _, _, terminated, truncated, info = env.step((0, 1))
    assert not terminated and not truncated
    assert "target_hit" not in info["events"]
    assert info["native_kill_count"] == 0


def test_native_kill_race_does_not_require_a_removed_target_object():
    actions = [
        (2, 1),
        (0, 0),
        (1, 1),
        (2, 0),
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (0, 1),
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 0),
    ]
    with NativeBasicV1Env() as env:
        env.reset(seed=100000)
        for action in actions:
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    assert terminated and not truncated
    assert info["native_target_killed"]
    assert not info["engine_episode_finished"]
    assert not info["native_geometry_available"]
