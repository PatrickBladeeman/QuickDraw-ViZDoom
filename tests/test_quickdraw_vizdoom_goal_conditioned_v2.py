"""Focused tests for Basic's two-goal joint-action learner."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from quickdraw_vizdoom import learning


def _masks():
    return np.zeros(3, dtype=bool), np.zeros(2, dtype=bool)


def _info(position=1, target=2, events=()):
    return {
        "position_slot": position,
        "target_slot": target,
        "events": list(events),
    }


def _physical(
    *, action=(1, 0), next_info=None, next_masks=None, terminated=False, truncated=False
):
    return learning.PhysicalTransition(
        np.zeros((84, 84, 4), dtype=np.uint8),
        action,
        2.5,
        np.ones((84, 84, 4), dtype=np.uint8),
        terminated,
        truncated,
        _masks() if next_masks is None else next_masks,
        _info(),
        _info() if next_info is None else next_info,
    )


def test_goal_rewards_and_environment_boundaries():
    aligned = _info(position=2, target=2)
    hit = _info(position=2, target=2, events=("target_hit",))
    miss = _info(position=1, target=2, events=("missed_shot",))

    assert learning.goal_reward(learning.BasicGoal.ALIGN_WITHOUT_FIRE, (1, 0), aligned) == 1.0
    assert learning.goal_reward(learning.BasicGoal.ALIGN_WITHOUT_FIRE, (2, 0), aligned) == 1.0
    assert learning.goal_reward(learning.BasicGoal.ALIGN_WITHOUT_FIRE, (1, 1), hit) == -1.0
    assert learning.goal_reward(learning.BasicGoal.HIT_TARGET, (1, 1), hit) == 1.0
    assert learning.goal_reward(learning.BasicGoal.HIT_TARGET, (0, 1), miss) == -0.1
    assert learning.goal_reward(learning.BasicGoal.HIT_TARGET, (1, 0), {}) == -0.01
    assert learning.goal_reward(
        learning.BasicGoal.HIT_TARGET, (1, 0), {}, terminated=True
    ) == -0.01
    assert learning.goal_reward(
        learning.BasicGoal.HIT_TARGET, (1, 0), {}, truncated=True
    ) == -0.01
    assert learning.goal_done(learning.BasicGoal.HIT_TARGET, (1, 0), {}, terminated=True)
    assert not learning.goal_done(
        learning.BasicGoal.HIT_TARGET, (1, 0), {}, truncated=True
    )


def test_behavior_goal_persists_and_alternates_at_success_or_reset(monkeypatch):
    class FakeEnv:
        steps = 0
        resets = 0

        def reset(self, *, seed):
            self.resets += 1
            position = 2 if self.resets == 1 else 0
            return np.zeros((84, 84, 4), dtype=np.uint8), {
                **_info(position=position),
                "seed": seed,
                "next_unavailable_masks": _masks(),
            }

        def step(self, action):
            transitions = [
                (_info(position=1), False),
                (_info(position=1), True),
                (_info(position=1), False),
                (_info(position=1, events=("target_hit",)), False),
                (_info(position=1), False),
            ]
            info, terminated = transitions[self.steps]
            self.steps += 1
            return (
                np.zeros((84, 84, 4), dtype=np.uint8),
                0.25,
                terminated,
                False,
                {**info, "next_unavailable_masks": _masks()},
            )

    selected_goals = []

    def select(_network, _observation, _masks, _epsilon, _rng, _device, *, goal, **_):
        selected_goals.append(goal)
        return (1, 0)

    monkeypatch.setattr(learning, "select_goal_action", select)
    monkeypatch.setattr(learning, "goal_train_step", lambda *args, **kwargs: 0.0)
    env = FakeEnv()
    learning.run_goal_conditioned_training(
        env, steps=5, warmup=1, batch_size=1, seed=10, device="cpu", epsilon=0.0
    )

    assert selected_goals == [
        learning.BasicGoal.ALIGN_WITHOUT_FIRE,
        learning.BasicGoal.ALIGN_WITHOUT_FIRE,
        learning.BasicGoal.HIT_TARGET,
        learning.BasicGoal.HIT_TARGET,
        learning.BasicGoal.ALIGN_WITHOUT_FIRE,
    ]
    assert env.resets == 2


def test_one_physical_transition_is_immutable_and_relabels_once_per_goal():
    observation = np.full((84, 84, 4), 7, dtype=np.uint8)
    next_observation = np.full((84, 84, 4), 9, dtype=np.uint8)
    masks = _masks()
    payload = learning.PhysicalTransition(
        observation,
        (1, 1),
        4.5,
        next_observation,
        False,
        False,
        masks,
        _info(),
        _info(position=2, target=2, events=("target_hit",)),
    )
    observation.fill(0)
    next_observation.fill(0)
    masks[0].fill(True)

    rows = learning.relabel_transition(payload)

    assert len(rows) == 2
    assert [row[1] for row in rows] == list(learning.BasicGoal)
    assert rows[0][0] is rows[1][0] is payload
    assert rows[0][0].observation.tobytes() == rows[1][0].observation.tobytes()
    assert rows[0][0].observation[0, 0, 0] == 7
    assert rows[0][0].next_observation.tobytes() == rows[1][0].next_observation.tobytes()
    assert rows[0][0].next_observation[0, 0, 0] == 9
    assert rows[0][0].action == rows[1][0].action == (1, 1)
    assert rows[0][0].environment_reward == rows[1][0].environment_reward == 4.5
    assert rows[0][0].terminated == rows[1][0].terminated is False
    assert rows[0][0].truncated == rows[1][0].truncated is False
    for first, second in zip(rows[0][0].next_masks, rows[1][0].next_masks):
        assert first.tobytes() == second.tobytes()
    assert [learning.goal_reward(goal, physical.action, physical.next_info) for physical, goal in rows] == [-1.0, 1.0]
    assert [learning.goal_achieved(goal, physical.action, physical.next_info) for physical, goal in rows] == [False, True]
    with pytest.raises(ValueError):
        payload.observation[0, 0, 0] = 1


class _TableQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))
        self.goal_ids = []

    def forward(self, observations, goals):
        goals = torch.as_tensor(goals, device=self.values.device)
        ids = goals.argmax(dim=-1) if goals.ndim == 2 else goals.argmax().view(1)
        self.goal_ids.extend(ids.tolist())
        return self.values[ids]


def test_double_dqn_bootstraps_with_the_row_goal():
    online = _TableQ(
        [[0, 0, 0, 0, 10, 0], [20, 0, 0, 0, 0, 0]]
    )
    target = _TableQ(
        [[0, 0, 0, 0, 2, 0], [40, 0, 0, 0, 0, 0]]
    )
    row = (_physical(truncated=True), learning.BasicGoal.ALIGN_WITHOUT_FIRE)

    value = learning.compute_goal_td_targets(online, target, [row], gamma=0.9)

    assert value.item() == pytest.approx(-0.01 + 0.9 * 2)
    assert online.goal_ids == target.goal_ids == [0]


def test_joint_legal_mask_applies_to_greedy_selection_and_bootstrap():
    online = _TableQ([[100, 1, 5, 2, 3, 99], [0, 0, 0, 0, 0, 0]])
    target = _TableQ([[1000, 1, 7, 2, 3, 999], [0, 0, 0, 0, 0, 0]])
    masks = (np.array([True, False, True]), np.array([False, True]))
    observation = np.zeros((84, 84, 4), dtype=np.uint8)

    action = learning.select_goal_action(
        online,
        observation,
        masks,
        epsilon=0.0,
        rng=np.random.default_rng(1),
        goal=learning.BasicGoal.ALIGN_WITHOUT_FIRE,
    )
    row = (_physical(next_masks=masks), learning.BasicGoal.ALIGN_WITHOUT_FIRE)
    target_value = learning.compute_goal_td_targets(online, target, [row], gamma=0.5)

    assert action == (1, 0)
    assert target_value.item() == pytest.approx(-0.01 + 0.5 * 7)


class _ZeroEncoder(nn.Module):
    def forward(self, observations):
        return observations.new_zeros((observations.shape[0], 64))


def test_terminal_replay_learns_opposite_actions_for_opposite_goals():
    network = learning.GoalConditionedQNetwork()
    network.encoder = _ZeroEncoder()
    target = learning.GoalConditionedQNetwork()
    target.encoder = _ZeroEncoder()
    target.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    replay = deque(maxlen=10_000)
    aligned = _info(position=2, target=2)
    hit = _info(position=2, target=2, events=("target_hit",))
    replay.append(_physical(action=(1, 0), next_info=aligned, terminated=True))
    replay.append(_physical(action=(1, 1), next_info=hit, terminated=True))
    rng = np.random.default_rng(4)

    for _ in range(100):
        loss = learning.goal_train_step(
            network,
            optimizer,
            replay,
            batch_size=2,
            rng=rng,
            target_network=target,
        )

    observation = np.zeros((84, 84, 4), dtype=np.uint8)
    assert np.isfinite(loss)
    assert learning.select_goal_action(
        network, observation, _masks(), 0.0, rng, goal=learning.BasicGoal.ALIGN_WITHOUT_FIRE
    ) == (1, 0)
    assert learning.select_goal_action(
        network, observation, _masks(), 0.0, rng, goal=learning.BasicGoal.HIT_TARGET
    ) == (1, 1)
