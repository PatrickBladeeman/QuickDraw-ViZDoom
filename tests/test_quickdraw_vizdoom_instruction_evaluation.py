"""Instruction scoring, finite plans, and the existing HTTP boundary."""

import json

import numpy as np
import pytest

from quickdraw_vizdoom import goal_teachers
from quickdraw_vizdoom import instruction_evaluation as benchmark


def test_development_grammar_and_plan_validation():
    cases = benchmark.read_cases(benchmark.DEVELOPMENT)
    assert len(cases) == 20
    for case in cases:
        assert benchmark.rule_plan(case["instruction"]) == benchmark.PLANS[case["task"]]
    for plan in benchmark.PLANS.values():
        assert benchmark.validate_plan({"plan": [goal.name for goal in plan]}) == plan
    for invalid in (
        {"plan": []},
        {"plan": ["OTHER"]},
        {"plan": "HIT_TARGET"},
        {"plan": ["HIT_TARGET", "ALIGN_WITHOUT_FIRE"]},
        {"plan": ["HIT_TARGET"], "extra": 1},
    ):
        with pytest.raises(ValueError):
            benchmark.validate_plan(invalid)


def test_constraints_are_scored_from_execution_not_canonical_plan():
    row = {
        "infrastructure_invalid": False,
        "native_kill_count": 1,
        "aligned_final": False,
        "shots": 1,
        "early_shots": 0,
    }
    assert benchmark.task_success("hit", row)
    assert benchmark.task_success("align_then_hit", row)
    assert not benchmark.task_success("preserve", row)
    assert not benchmark.task_success("align_then_hit", {**row, "early_shots": 1})
    alive = {**row, "native_kill_count": 0, "aligned_final": True, "shots": 0}
    assert benchmark.task_success("preserve", alive)
    assert not benchmark.task_success("preserve", {**alive, "shots": 1})
    assert not benchmark.task_success(
        "preserve", {**alive, "infrastructure_invalid": True}
    )


def test_plan_stops_and_uses_actual_shots_for_ordering():
    class Env:
        def reset(self, *, seed):
            self.count = 0
            return np.zeros(1), {
                "aligned": False,
                "alignment_error": 10,
                "remaining_ammunition": 10,
            }

        def step(self, action):
            self.count += 1
            hit = action[1] == 1
            return (
                np.zeros(1),
                0.0,
                hit,
                False,
                {
                    "aligned": not hit,
                    "alignment_error": 0,
                    "remaining_ammunition": 9 if hit else 10,
                    "native_kill_count": int(hit),
                    "events": ["target_hit"] if hit else [],
                },
            )

    env = Env()
    preserve = benchmark.rollout(env, 1, benchmark.PLANS["preserve"])
    assert preserve["decisions"] == 1 and preserve["shots"] == 0
    assert benchmark.task_success("preserve", preserve)
    sequence = benchmark.rollout(env, 1, benchmark.PLANS["align_then_hit"])
    assert sequence["decisions"] == 2 and sequence["early_shots"] == 0
    assert sequence["plan_completed"] and benchmark.task_success(
        "align_then_hit", sequence
    )


def test_shared_request_never_sends_gold_labels_or_authentication_to_audit(monkeypatch):
    observed = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size):
            assert size == 65537
            return json.dumps(
                {"choices": [{"message": {"content": '{"plan":["HIT_TARGET"]}'}}]}
            ).encode()

    def urlopen(request, timeout):
        observed.append(json.loads(request.data))
        return Reply()

    monkeypatch.setenv("OPENROUTER_API_KEY", "private-key")
    monkeypatch.setattr(goal_teachers.urllib.request, "urlopen", urlopen)
    record = {}
    result = goal_teachers.request_json(
        benchmark.PROMPT,
        {"instruction": "Hit the target."},
        url=goal_teachers.OPENROUTER_URL,
        model="test",
        timeout=10,
        record=record,
    )
    assert benchmark.validate_plan(result) == benchmark.PLANS["hit"]
    assert json.loads(observed[0]["messages"][1]["content"]) == {
        "instruction": "Hit the target."
    }
    assert "private-key" not in json.dumps(record)


def test_confidence_interval_counts_families_not_repeated_layouts():
    rows = [
        {"family": family, "arm": arm, "success_rate": float(arm == "llm")}
        for family in ("a", "b", "c")
        for arm in ("llm", "rule")
        for _ in range(20)
    ]
    result = benchmark.paired_interval(rows, "llm", "rule")
    assert result["family_count"] == 3
    assert result["success_rate_delta"] == 1
    assert result["paired_family_bootstrap_97_5_interval"] == [1, 1]


def test_undersized_heldout_set_is_rejected_before_requests(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "test",
                "family": "new-family",
                "instruction": "Please hit it.",
                "task": "hit",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="ten families"):
        benchmark.read_cases(path)


def test_native_scripted_plans_satisfy_all_three_tasks():
    pytest.importorskip("vizdoom")
    with benchmark.NativeBasicV1Env() as env:
        for task, plan in benchmark.PLANS.items():
            row = benchmark.rollout(env, 36000, plan)
            assert benchmark.task_success(task, row)
