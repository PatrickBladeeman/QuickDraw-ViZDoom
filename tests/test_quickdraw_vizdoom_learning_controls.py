"""Focused controls for the fixed-goal parity learner."""

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from quickdraw_vizdoom import learning  # noqa: E402


def _masks():
    return np.zeros(3, bool), np.zeros(2, bool)


def _info(**extra):
    return {
        "position_slot": 0,
        "target_slot": 1,
        "next_unavailable_masks": _masks(),
        **extra,
    }


def _physical(*, terminated=False):
    observation = np.zeros((84, 84, 4), np.float32)
    return learning.PhysicalTransition(
        observation,
        (2, 0),
        0.25,
        observation,
        terminated,
        False,
        _masks(),
        _info(),
        _info(position_slot=1),
    )


class _FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.goal_conditioning = True
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observations, goals):
        return self.values.expand(len(observations), -1)


def test_fixed_hit_environment_reward_uses_one_row_and_environment_boundary():
    row = _physical()
    assert learning.relabel_transition(
        row, fixed_goal=learning.BasicGoal.HIT_TARGET
    ) == ((row, learning.BasicGoal.HIT_TARGET),)
    online = _FixedQ([0, 9, 0, 0, 0, 0])
    target = _FixedQ([0, 7, 0, 0, 0, 0])
    target_value = learning.compute_goal_td_targets(
        online,
        target,
        [(row, learning.BasicGoal.HIT_TARGET)],
        gamma=0.5,
        environment_reward=True,
    )
    assert target_value.item() == pytest.approx(3.75)
    terminal_value = learning.compute_goal_td_targets(
        online,
        target,
        [(_physical(terminated=True), learning.BasicGoal.HIT_TARGET)],
        gamma=0.5,
        environment_reward=True,
    )
    assert terminal_value.item() == pytest.approx(0.25)


def test_fixed_hit_training_stays_fixed_and_records_parity_rows(monkeypatch):
    goals = []
    controls = []

    class Env:
        def reset(self, *, seed):
            return np.zeros((84, 84, 4), np.float32), _info(seed=seed)

        def step(self, action):
            return np.zeros((84, 84, 4), np.float32), -0.01, False, False, _info()

    def select(*_args, goal, **_kwargs):
        goals.append(goal)
        return 2, 0

    def train(*_args, **kwargs):
        controls.append(kwargs)
        return 0.0

    monkeypatch.setattr(learning, "select_goal_action", select)
    monkeypatch.setattr(learning, "goal_train_step", train)
    summary, _ = learning.run_goal_conditioned_training(
        Env(),
        steps=3,
        warmup=1,
        batch_size=1,
        seed=7,
        device="cpu",
        fixed_goal=learning.BasicGoal.HIT_TARGET,
        environment_reward=True,
    )

    assert goals == [learning.BasicGoal.HIT_TARGET] * 3
    assert summary["goal_names"] == ["HIT_TARGET"]
    assert summary["fixed_goal"] == "HIT_TARGET"
    assert summary["environment_reward"]
    assert summary["relabeled_row_count"] == summary["physical_replay_count"] == 3
    assert summary["replay_counts"]["relabeled_goal"] == {"HIT_TARGET": 3}
    assert all(control["environment_reward"] is True for control in controls)


def test_native_aligned_flag_overrides_legacy_slot_fallback():
    assert learning.goal_achieved(
        learning.BasicGoal.ALIGN_WITHOUT_FIRE,
        (2, 0),
        {"aligned": True},
    )
    assert not learning.goal_achieved(
        learning.BasicGoal.ALIGN_WITHOUT_FIRE,
        (2, 0),
        {"aligned": False, "position_slot": 1, "target_slot": 1},
    )
