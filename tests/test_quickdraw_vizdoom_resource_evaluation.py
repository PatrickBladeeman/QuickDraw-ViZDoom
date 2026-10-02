"""Grounded resource dependencies, adviser parity and held-out scoring."""

import copy
import io
import json

import pytest

from quickdraw_vizdoom import goal_teachers
from quickdraw_vizdoom.envs.resource_arena import ResourceArena
from quickdraw_vizdoom.resource_evaluation import (
    adviser_state,
    complete,
    episode,
    generate_cases,
    paired_interval,
    reference_plan,
    rule_goal,
)


def test_native_last_round_unlocks_pickup_and_preserves_target_identity():
    case = {
        "seed": 71000,
        "initial_ammo": 1,
        "cache_ammo": 2,
        "cache_requires": "B",
        "required_targets": ["A", "C"],
        "targets": [{"id": "A", "y": -96}, {"id": "B", "y": 48}, {"id": "C", "y": 144}],
    }
    with ResourceArena(case) as env:
        initial = adviser_state(env.state)
        frozen = copy.deepcopy(initial)
        reference = reference_plan(initial)
        assert reference["goal_count"] == 4
        assert (
            reference["goals"][0]
            == rule_goal(initial)
            == {"goal": "ELIMINATE", "target_id": "B"}
        )
        assert initial == frozen  # Planning must not mutate the shared input.
        assert env.execute(rule_goal(initial)) == "achieved"
        assert env.state["ammo"] == 0 and env.state["native_kills"] == 1
        assert env.events[-1]["killed"] == ["B"] and not complete(env.state)
        assert rule_goal(env.state) == {"goal": "COLLECT_AMMO"}
        assert env.execute(rule_goal(env.state)) == "achieved"
        assert env.state["ammo"] == 2 and env.state["cache"]["collected"]
        for _ in range(2):
            assert env.execute(rule_goal(env.state)) == "achieved"
        assert complete(env.state) and env.state["native_kills"] == 3
        assert [event["ammo_delta"] for event in env.events] == [-1, 2, -1, -1]
        assert sum(len(event["killed"]) for event in env.events) == 3


def test_deadlock_reference_and_balanced_disjoint_cases():
    development = generate_cases("development", 12)
    heldout = generate_cases("heldout", 12)
    confirmation = generate_cases("confirmation", 12)
    assert {c["seed"] for c in confirmation}.isdisjoint(
        c["seed"] for c in development + heldout
    )
    assert {t["y"] for c in confirmation for t in c["targets"]}.isdisjoint(
        {t["y"] for c in development + heldout for t in c["targets"]}
    )
    assert {c["seed"] for c in development}.isdisjoint(c["seed"] for c in heldout)
    assert {t["y"] for c in development for t in c["targets"]}.isdisjoint(
        {t["y"] for c in heldout for t in c["targets"]}
    )
    assert all(
        sum(c["stratum"] == s for c in heldout) == 4
        for s in ("direct", "required_gate", "optional_gate")
    )
    with ResourceArena(heldout[2]) as env:
        wrong = next(
            t for t in env.state["targets"] if t["id"] != heldout[2]["cache_requires"]
        )
        assert (
            env.execute({"goal": "ELIMINATE", "target_id": wrong["id"]}) == "achieved"
        )
        assert env.state["ammo"] == 0 and reference_plan(env.state) is None
        assert not complete(env.state)
    with pytest.raises(ValueError):
        generate_cases("heldout", 4)


def test_llm_schema_and_failure_scoring_use_identical_state(monkeypatch):
    case = generate_cases("heldout", 3)[0]
    captured = []

    def transport(prompt, state, **kwargs):
        captured.append(copy.deepcopy(state))
        assert kwargs["response_schema"]["properties"]["choice"]["enum"] == list(
            range(len(state["available_goals"]))
        )
        return {"choice": state["available_goals"].index(rule_goal(state))}

    monkeypatch.setattr("quickdraw_vizdoom.resource_evaluation.request_json", transport)
    result = episode(
        case, "llm", model="test", url="http://localhost/v1/chat/completions", timeout=1
    )
    assert result["success"]
    assert captured == [item["state"] for item in result["trace"]]
    monkeypatch.setattr(
        "quickdraw_vizdoom.resource_evaluation.request_json",
        lambda *args, **kwargs: {"choice": -1},
    )
    failed = episode(
        case, "llm", model="test", url="http://localhost/v1/chat/completions", timeout=1
    )
    assert not failed["success"] and failed["failure"] == "adviser_failure"
    assert len(failed["adviser_audit"]) == 1
    interval = paired_interval([failed], [result])
    assert interval["difference"] == -1 and interval["baseline_only_successes"] == 1
    assert interval["paired_case_bootstrap_95_percent"] == [-1, -1]


def test_structured_choice_is_required_at_the_provider_boundary(monkeypatch):
    sent = []
    schema = {
        "type": "object",
        "properties": {"choice": {"type": "integer", "enum": [0, 1]}},
        "required": ["choice"],
        "additionalProperties": False,
    }

    def response(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps({"choices": [{"message": {"content": '{"choice":1}'}}]}).encode()
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    monkeypatch.setattr(goal_teachers.urllib.request, "urlopen", response)
    record = {}
    content = goal_teachers.request_json(
        "choose",
        {},
        url=goal_teachers.OPENROUTER_URL,
        model="test",
        timeout=1,
        record=record,
        response_schema=schema,
    )
    assert content == {"choice": 1}
    assert sent[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "adviser_choice", "strict": True, "schema": schema},
    }
    assert sent[0]["provider"] == {"require_parameters": True}
    assert "secret-test-key" not in json.dumps(record)
