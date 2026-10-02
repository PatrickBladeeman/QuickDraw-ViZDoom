"""Checks for the matched no-goal learner and paired adviser experiment."""

import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from quickdraw_vizdoom import adviser_evaluation as experiment  # noqa: E402
from quickdraw_vizdoom import learning  # noqa: E402
from quickdraw_vizdoom.goal_teachers import make_teacher  # noqa: E402
from quickdraw_vizdoom.learning_components import (  # noqa: E402
    BasicGoal,
    GoalConditionedQNetwork,
    PhysicalTransition,
    compute_goal_td_targets,
    goal_one_hot,
    goal_train_step,
    relabel_transition,
)


def _info(position=0, **extra):
    return {
        "position_slot": position,
        "target_slot": 2,
        "decision": position,
        "remaining_ammunition": 300,
        "next_unavailable_masks": (np.zeros(3, bool), np.zeros(2, bool)),
        **extra,
    }


class TinyPolicy(torch.nn.Module):
    def __init__(self, goal_conditioning=True):
        super().__init__()
        self.goal_conditioning = goal_conditioning
        self.weight = torch.nn.Parameter(torch.zeros(1))


class ThreeStepEnv:
    def __init__(self):
        self.resets, self.actions, self.closed = [], [], False

    def reset(self, *, seed):
        self.resets.append(seed)
        self.position = 0
        return np.zeros((84, 84, 4), np.float32), _info(seed=seed)

    def step(self, action):
        self.actions.append(action)
        self.position += int(action == (2, 0))
        hit = action == (0, 1) and self.position == 2
        return (
            np.zeros((84, 84, 4), np.float32),
            0.99 if hit else -0.01,
            hit,
            False,
            _info(self.position, events=["target_hit"] if hit else []),
        )

    def close(self):
        self.closed = True


def test_native_goal_summary_and_handoff_use_actual_geometry(monkeypatch):
    class NativeEnv(ThreeStepEnv):
        def reset(self, *, seed):
            observation, info = super().reset(seed=seed)
            info.update(native_geometry={}, aligned=False, alignment_error=16.0)
            return observation, info

        def step(self, action):
            observation, reward, hit, truncated, info = super().step(action)
            info.update(
                aligned=self.position == 2,
                alignment_error=16.0 - 8.0 * self.position,
                native_kill_count=int(hit),
            )
            return observation, reward, hit, truncated, info

    monkeypatch.setattr(
        experiment,
        "select_goal_action",
        lambda *args, goal, **kwargs: (
            (2, 0) if goal == BasicGoal.ALIGN_WITHOUT_FIRE else (0, 1)
        ),
    )
    report = experiment.evaluate_adviser(
        TinyPolicy(), NativeEnv(), [36000], 32001, make_teacher("rule")
    )
    episode = report["episodes"][0]
    assert episode["native_kill_count"] == 1
    assert episode["alignment_decision"] == 2
    assert episode["align_to_hit_handoff"] and episode["handoff_hit"]
    assert episode["requested_goal_success"]


def test_native_summary_rejects_nonfinite_geometry():
    from quickdraw_vizdoom.goal_teachers import teacher_summary

    with pytest.raises(ValueError, match="finite"):
        teacher_summary({"aligned": False, "alignment_error": float("nan")})


def test_no_goal_uses_matched_network_and_environment_reward_ddqn():
    conditioned, plain = GoalConditionedQNetwork(), GoalConditionedQNetwork(False)
    assert conditioned.state_dict().keys() == plain.state_dict().keys()
    assert sum(p.numel() for p in conditioned.parameters()) == sum(
        p.numel() for p in plain.parameters()
    )
    assert plain(torch.zeros(84, 84, 4)).shape == (1, 6)
    with pytest.raises(ValueError):
        plain(torch.zeros(84, 84, 4), goal_one_hot(BasicGoal.HIT_TARGET))

    class FixedQ(TinyPolicy):
        def __init__(self, values):
            super().__init__(False)
            self.values = torch.tensor(values, dtype=torch.float32)

        def forward(self, observations, goals=None):
            assert goals is None
            return self.values.expand(len(observations), -1)

    observation = np.zeros((84, 84, 4), np.float32)
    rows = [
        PhysicalTransition(
            observation,
            (0, 0),
            0.25,
            observation,
            terminal,
            truncated,
            (np.array([False, True, False]), np.zeros(2, bool)),
            _info(),
            _info(2),
        )
        for terminal, truncated in [
            (False, False),
            (True, False),
            (False, True),
        ]
    ]
    assert relabel_transition(rows[0], goal_conditioning=False) == ((rows[0], None),)
    targets = compute_goal_td_targets(
        FixedQ([1, 2, 100, 100, 3, 4]),
        FixedQ([10, 20, 30, 40, 50, 60]),
        [(physical, None) for physical in rows],
        gamma=0.5,
    )
    assert targets.tolist() == [30.25, 0.25, 30.25]
    target = GoalConditionedQNetwork(False)
    target.load_state_dict(plain.state_dict())
    before = experiment._hash(plain)
    loss = goal_train_step(
        plain,
        torch.optim.Adam(plain.parameters(), lr=1e-3),
        [rows[1]],
        1,
        np.random.default_rng(1),
        target_network=target,
    )
    assert np.isfinite(loss) and experiment._hash(plain) != before


def test_no_goal_training_never_selects_or_relables_goals(monkeypatch):
    goals = []

    def select(*_args, goal, **_kwargs):
        goals.append(goal)
        return 2, 0

    monkeypatch.setattr(learning, "select_goal_action", select)
    monkeypatch.setattr(learning, "goal_train_step", lambda *_args, **_kwargs: 0.1)
    summary, network = learning.run_goal_conditioned_training(
        ThreeStepEnv(), 2, 1, 1, 7, "cpu", goal_conditioning=False
    )
    assert goals == [None, None] and not network.goal_conditioning
    assert summary["optimizer_updates"] == 2
    assert summary["relabeled_row_count"] == summary["physical_replay_count"] == 2
    assert not summary["goal_conditioned"] and summary["goal_names"] == []
    with pytest.raises(ValueError):
        learning.run_goal_conditioned_training(
            ThreeStepEnv(),
            2,
            1,
            1,
            7,
            "cpu",
            goal_conditioning=False,
            behavior_goal_selector=lambda *_: BasicGoal.HIT_TARGET,
        )


def test_adviser_runs_only_at_goal_boundaries_and_keeps_policy_frozen(
    monkeypatch,
):
    selected = []

    def select(*_args, goal, **_kwargs):
        selected.append(goal)
        return (2, 0) if goal == BasicGoal.ALIGN_WITHOUT_FIRE else (0, 1)

    monkeypatch.setattr(experiment, "select_goal_action", select)
    audit, env = [], ThreeStepEnv()
    result = experiment.evaluate_adviser(
        TinyPolicy(),
        env,
        (40, 41),
        7,
        make_teacher("rule", audit=audit),
        audit,
    )
    assert (
        selected
        == [
            BasicGoal.ALIGN_WITHOUT_FIRE,
            BasicGoal.ALIGN_WITHOUT_FIRE,
            BasicGoal.HIT_TARGET,
        ]
        * 2
    )
    assert env.resets == [40, 41]
    assert [r["decisions_to_hit"] for r in result["episodes"]] == [3, 3]
    assert [call["decision"] for call in audit] == [0, 2, 0, 2]
    assert [call["episode_seed"] for call in audit] == [40, 40, 41, 41]
    assert result["evaluation_updates"] == 0
    assert result["parameter_hash_before"] == result["parameter_hash_after"]


def test_adviser_failure_is_failed_episode_without_fallback():
    env, audit = ThreeStepEnv(), []

    def failed(_info, _rng):
        audit.append({"error": "timeout", "latency_seconds": 0.1})
        raise ValueError("timeout")

    result = experiment.evaluate_adviser(TinyPolicy(), env, (40,), 7, failed, audit)
    row = result["episodes"][0]
    assert not row["target_hit"] and row["teacher_error"] == "timeout"
    assert row["decisions"] == 0 and row["decision_cost"] == 300
    assert env.actions == []
    summary = experiment._aggregate(result["episodes"], audit)
    assert summary["teacher_failed_episodes"] == summary["teacher_failed_calls"] == 1


def test_saved_policies_are_reused_across_arms_and_runs(monkeypatch, tmp_path):
    training_calls, evaluations, envs = [], [], []

    def train(
        _env, steps, _warmup, _batch, seed, _device, *, goal_conditioning, **_kwargs
    ):
        training_calls.append((seed, goal_conditioning))
        return {
            "training_episode_seeds": [100000],
            "decisions": steps,
        }, TinyPolicy(goal_conditioning)

    def evaluate(network, _env, seeds, training_seed, teacher=None, audit=None):
        evaluations.append((training_seed, network, teacher))
        return {
            "episodes": [
                {
                    "seed": seed,
                    "target_hit": True,
                    "decisions": 3,
                    "decision_cost": 3,
                    "environment_return": 0.97,
                    "teacher_error": None,
                }
                for seed in seeds
            ]
        }

    def env():
        instance = ThreeStepEnv()
        envs.append(instance)
        return instance

    monkeypatch.setattr(experiment, "run_goal_conditioned_training", train)
    monkeypatch.setattr(experiment, "GoalConditionedQNetwork", TinyPolicy)
    monkeypatch.setattr(experiment, "evaluate_adviser", evaluate)
    monkeypatch.setattr(experiment, "BasicV1Env", env)
    output = tmp_path / "baseline.json"
    result = experiment.run_experiment(
        output_path=output,
        arms=("no_goal", "random", "rule"),
        evaluation_episodes=2,
    )
    assert len(training_calls) == 6 and all(instance.closed for instance in envs)
    for start in range(0, len(evaluations), 3):
        assert evaluations[start][1] is not evaluations[start + 1][1]
        assert evaluations[start + 1][1] is evaluations[start + 2][1]
    assert all(metrics["episodes"] == 6 for metrics in result["aggregate"].values())
    assert json.loads(output.read_text())["checkpoint"] == result["checkpoint"]
    assert (
        result["paired_comparisons"]["random_vs_rule"][
            "hit_rate_delta_first_minus_second"
        ]
        == 0
    )

    monkeypatch.setattr(
        experiment,
        "make_teacher",
        lambda *_args, **_kwargs: lambda *_: BasicGoal.HIT_TARGET,
    )
    experiment.run_experiment(
        output_path=tmp_path / "llm.json",
        arms=("random", "rule", "llm"),
        evaluation_episodes=2,
        checkpoint_path=result["checkpoint"],
    )
    assert len(training_calls) == 6  # Loading never trains again.
    with pytest.raises(FileExistsError):
        experiment.run_experiment(output_path=output, arms=("no_goal",))
