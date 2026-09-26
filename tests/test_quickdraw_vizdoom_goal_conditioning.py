"""Focused checks for the Basic categorical-goal baseline."""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from quickdraw_vizdoom import goal_conditioning  # noqa: E402
from quickdraw_vizdoom.goal_conditioning import (  # noqa: E402
    Goal,
    rule_goal,
    shuffled_goal,
)
from quickdraw_vizdoom.learning import (  # noqa: E402
    BranchingQNetwork,
    compute_td_targets,
    run_training,
    train_step,
)
from quickdraw_vizdoom.qa import goal_conditioning as qa_goal_conditioning  # noqa: E402
from quickdraw_vizdoom.qa.results import Outcome  # noqa: E402
from quickdraw_vizdoom.qa.runner import run_profile  # noqa: E402


def _observation(value: float = 0.0) -> np.ndarray:
    return np.full((84, 84, 4), value, dtype=np.float32)


@pytest.mark.parametrize(
    "position,target,expected",
    [
        (-1, 0, Goal.ALIGN_RIGHT),
        (1, 0, Goal.ALIGN_LEFT),
        (0, 0, Goal.TAKE_SHOT),
    ],
)
def test_rule_and_shuffled_goals(position, target, expected):
    info = {"position_slot": position, "target_slot": target}

    assert rule_goal(info) is expected
    assert shuffled_goal(info) is not expected
    assert {
        shuffled_goal({**info, "seed": 1, "decision": decision}) for decision in (0, 1)
    } == set(Goal) - {expected}


def test_all_goal_ids_keep_existing_action_branches_and_no_goal_path():
    observations = torch.zeros(2, 84, 84, 4)
    conditioned = BranchingQNetwork(goal_count=3)
    for goal in Goal:
        movement, combat = conditioned(observations, torch.tensor([goal, goal]))
        assert movement.shape == (2, 3)
        assert combat.shape == (2, 2)

    movement, combat = BranchingQNetwork()(observations)
    assert movement.shape == (2, 3)
    assert combat.shape == (2, 2)
    with pytest.raises(ValueError, match="Goal IDs"):
        conditioned(observations, torch.tensor([0, 3]))


def test_replay_uses_current_goal_for_update_and_next_goal_for_td_target():
    class TrackingNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.calls = []

        def forward(self, observations, goals):
            self.calls.append(goals.detach().cpu().tolist())
            values = self.bias + goals.float().unsqueeze(1)
            return values.expand(-1, 3), values.expand(-1, 2)

    masks = (np.zeros(3, dtype=bool), np.zeros(2, dtype=bool))
    transitions = [
        (
            _observation(),
            (0, 0),
            1.0,
            _observation(),
            False,
            False,
            masks,
            goal,
            next_goal,
        )
        for goal, next_goal in ((0, 1), (2, 0))
    ]
    network = TrackingNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    targets = compute_td_targets(network, transitions, gamma=0.5)
    assert targets.tolist() == pytest.approx([2.0, 1.0])
    network.calls.clear()

    loss = train_step(
        network, optimizer, transitions, 2, np.random.default_rng(1), gamma=0.5
    )

    assert np.isfinite(loss)
    assert set(network.calls[0]) == {0, 2}
    assert set(network.calls[1]) == {0, 1}


def test_training_latches_next_goal_and_records_goal_events(monkeypatch):
    captured = []

    class Env:
        def __init__(self):
            self.decision = 0

        def reset(self, seed):
            self.decision = 0
            return _observation(), {
                "seed": seed,
                "episode_index": 0,
                "decision": 0,
                "position_slot": 0,
                "target_slot": 1,
                "next_unavailable_masks": [[False] * 3, [False] * 2],
            }

        def step(self, action):
            self.decision += 1
            terminal = self.decision == 2
            return (
                _observation(),
                1.0,
                terminal,
                False,
                {
                    "episode_index": 0,
                    "decision": self.decision,
                    "position_slot": 1,
                    "target_slot": 1,
                    "shots_fired": int(action[1] == 1),
                    "events": ["target_hit"] if terminal else ["decision"],
                    "next_unavailable_masks": [[False] * 3, [False] * 2],
                },
            )

    def update(_network, _optimizer, replay, *_args, **_kwargs):
        captured.append(list(replay))
        return 0.1

    monkeypatch.setattr("quickdraw_vizdoom.learning.train_step", update)
    result, network = run_training(
        Env(),
        steps=2,
        warmup=1,
        batch_size=1,
        seed=1,
        device="cpu",
        goal_selector=rule_goal,
        goal_source="correct_goal",
    )

    assert network.goal_count == 3
    assert captured[0][0][7:] == (Goal.ALIGN_RIGHT, Goal.TAKE_SHOT)
    assert captured[-1][-1][7:] == (Goal.TAKE_SHOT, Goal.TAKE_SHOT)
    first, second = result["transition_metrics"]
    assert first["goal_success"] and first["goal_changed"]
    assert "goal_change" in first["events"]
    assert second["goal_success"] and second["goal_source"] == "correct_goal"


def test_comparison_uses_each_condition_and_seed_once(monkeypatch):
    calls = []

    def session(**kwargs):
        calls.append(kwargs)
        return {
            "seed": kwargs["seed"],
            "updates": 1,
            "last_loss": 0.1,
            "cumulative_episode_return": 1.0,
            "decisions_to_hit": [1],
            "time_to_goal": [1],
            "shots_hit": 1,
            "total_shots": 1,
            "goal_attempts": int(kwargs["condition"] != "no_goal"),
            "goal_completions": int(kwargs["condition"] != "no_goal"),
            "held_out_return": 1.0,
            "held_out_target_hits": 1,
            "evaluation_decisions_to_hit": [1],
            "evaluation_episodes": 1,
            "evaluation_updates": 0,
            "goal_selection_counts": {},
            "goal_transition_counts": {},
        }

    monkeypatch.setattr(goal_conditioning, "run_session", session)
    result = goal_conditioning.run_comparison(
        seeds=(1, 2, 3), steps=1, warmup=1, batch_size=1, evaluation_episodes=1
    )

    assert [(call["condition"], call["seed"]) for call in calls] == [
        (condition, seed)
        for condition in goal_conditioning.CONDITIONS
        for seed in (1, 2, 3)
    ]
    assert set(result["conditions"]) == set(goal_conditioning.CONDITIONS)
    assert result["correct_vs_shuffled"]["research_gate_passed"] is False


def test_registered_profile_runs_all_nine_sessions(monkeypatch):
    calls = []

    def session(**kwargs):
        calls.append(kwargs)
        condition, seed = kwargs["condition"], kwargs["seed"]
        record = {
            "goal": None if condition == "no_goal" else 0,
            "next_goal": None if condition == "no_goal" else 0,
            "goal_source": condition,
            "goal_success": False,
            "goal_changed": False,
            "events": [],
        }
        return {
            "trainer": "branching_q_learning",
            "condition": condition,
            "seed": seed,
            "decisions": 512,
            "updates": 481,
            "last_loss": 0.1,
            "cumulative_episode_return": 1.0,
            "mean_decisions_to_hit": 2.0,
            "shot_percentage": 50.0,
            "goal_completion_rate": 0.5,
            "held_out_return": 1.0,
            "held_out_target_hit_rate": 1.0,
            "evaluation_updates": 0,
            "evaluation_parameters_unchanged": True,
            "evaluation_episodes": 100,
            "training_seed_schedule": [seed],
            "evaluation_seed_schedule": [36000],
            "goal_selection_counts": {},
            "goal_transition_counts": {},
            "transition_metrics": [record],
            "evaluation_transition_metrics": [record],
            "process_launches": 1,
            "training_sessions": 1,
            "external_calls": 0,
        }

    monkeypatch.setattr(
        qa_goal_conditioning,
        "_load_runtime",
        lambda: (session, RuntimeError),
    )
    contract = (
        goal_conditioning.DEFAULT_CONFIG_PATH.parents[2] / "contracts/basic-v1.json"
    )
    report = run_profile(contract, "goal-conditioning")

    assert report.outcome is Outcome.PASS
    assert len(calls) == len(report.training_sessions) == 9
