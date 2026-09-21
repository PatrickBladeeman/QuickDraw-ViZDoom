"""Shuffled-seed Basic-arena training checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quickdraw_vizdoom.envs.micro_learning_generalization import (
    MicroLearningGeneralizationV1Env,
    permuted_episode_seeds,
)
from quickdraw_vizdoom.qa import micro_learning_generalization as qa_generalization
from quickdraw_vizdoom.qa.results import Outcome
from quickdraw_vizdoom.qa.runner import run_profile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/micro-learning-generalization-v1.json"


def test_basic_arena_keeps_the_episode_alive_for_multiple_decisions():
    with MicroLearningGeneralizationV1Env() as env:
        observation, info = env.reset(seed=32001)
        assert observation.shape == (84, 84, 4)
        assert np.isfinite(observation).all()
        assert info["next_unavailable_masks"] == [[False, False, False], [False, False]]
        for movement in (1, 2):
            observation, reward, terminated, truncated, info = env.step((movement, 0))
            assert observation.shape == (84, 84, 4)
            assert reward == pytest.approx(-0.01)
            assert not terminated and not truncated
        assert info["decision"] == 2
        assert info["visual_split"] == "training"
        assert env.reset_records["training"][0]["seed"] == 32001


def test_watcher_reports_episode_return_steps_and_shot_rate(monkeypatch, capsys):
    from quickdraw_vizdoom import watch_generalization

    transition = (
        np.zeros((84, 84, 4), np.float32),
        -0.03,
        True,
        False,
        {
            "visual_split": "training",
            "visual_variant": "target-slot-0",
            "target_slot": 0,
            "semantic_action": ["Stay", "Shoot"],
            "event_counters": {"shots_fired": 2, "target_hits": 1},
            "decision": 4,
        },
    )
    monkeypatch.setattr(
        MicroLearningGeneralizationV1Env,
        "step",
        lambda self, action: transition,
    )
    with watch_generalization.WatchEnv(delay=0) as env:
        env.exploring = True
        env.step((0, 1))

    assert "episode hit return -0.03 steps 4 shots 1/2 (50%)" in capsys.readouterr().out


def test_seed_schedules_are_permuted_balanced_and_not_alternating():
    for start, count in ((32001, 512), (34001, 512), (36000, 100)):
        seeds = permuted_episode_seeds(start, count, permutation_seed=start)
        assert set(seeds) == set(range(start, start + count))
        assert sum(seed % 2 for seed in seeds) == count // 2
        assert any(a % 2 == b % 2 for a, b in zip(seeds, seeds[1:]))
        assert not all(a % 2 != b % 2 for a, b in zip(seeds, seeds[1:]))


def _valid_result(seed: int) -> dict:
    training_schedule = list(permuted_episode_seeds(seed, 512, permutation_seed=seed))
    return {
        "decisions": 512,
        "updates": 481,
        "terminal": 1,
        "truncated": 0,
        "training_episodes": 1,
        "last_loss": 0.01,
        "training_seed_start": seed,
        "training_seed_end": seed + 511,
        "training_episode_seeds": training_schedule[:1],
        "training_seed_schedule": training_schedule,
        "evaluation_seed_start": 36000,
        "evaluation_seed_end": 36099,
        "evaluation_episode_seeds": list(
            permuted_episode_seeds(36000, 100, permutation_seed=seed)
        ),
        "evaluation_episodes": 100,
        "evaluation_decisions": 100,
        "evaluation_terminal": 100,
        "evaluation_truncated": 0,
        "evaluation_updates": 0,
        "parameters_unchanged": True,
        "greedy_successes": 100,
        "greedy_success_rate": 1.0,
        "training_visual_variants": ["target-slot-0"],
        "held_out_visual_variants": ["target-slot-0"],
        "training_reset_frame_hashes": ["training-hash"],
        "held_out_reset_frame_hashes": ["held-out-hash"],
        "reset_seed_overlap": [],
        "frame_hash_overlap": [],
        "variant_overlap": ["target-slot-0"],
        "process_launches": 1,
        "training_sessions": 1,
        "trace_collections": 0,
        "external_calls": 0,
        "checkpoints": 0,
    }


def test_qa_runs_two_sessions_and_allows_same_arena_variants(monkeypatch):
    calls = []

    def run_session(**kwargs):
        calls.append(kwargs)
        return _valid_result(kwargs["seed"])

    monkeypatch.setattr(
        qa_generalization, "_load_runtime", lambda: (run_session, RuntimeError)
    )
    report = run_profile(CONTRACT, "micro-learning-generalization")
    assert report.outcome is Outcome.PASS
    assert [call["seed"] for call in calls] == [32001, 34001]
    json.loads(json.dumps(report.to_dict(), allow_nan=False))


@pytest.mark.parametrize(
    "mutation",
    [
        {"greedy_success_rate": 0.79},
        {"updates": 0},
        {"evaluation_updates": 1},
        {"reset_seed_overlap": [32001]},
    ],
)
def test_qa_rejects_session_failures(monkeypatch, mutation):
    monkeypatch.setattr(
        qa_generalization,
        "_load_runtime",
        lambda: (
            lambda **kwargs: {**_valid_result(kwargs["seed"]), **mutation},
            RuntimeError,
        ),
    )
    assert (
        run_profile(CONTRACT, "micro-learning-generalization").outcome is Outcome.FAIL
    )


def test_missing_training_dependency_is_infrastructure_invalid(monkeypatch):
    def missing_dependency():
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr(qa_generalization, "_load_runtime", missing_dependency)
    assert (
        run_profile(CONTRACT, "micro-learning-generalization").outcome
        is Outcome.INFRA_INVALID
    )
