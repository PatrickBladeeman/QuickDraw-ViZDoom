"""Focused checks for optional behavior-goal teachers."""

from __future__ import annotations

import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from quickdraw_vizdoom import goal_teachers, learning  # noqa: E402
from quickdraw_vizdoom.learning_components import BasicGoal  # noqa: E402


def _masks():
    return np.zeros(3, dtype=bool), np.zeros(2, dtype=bool)


def _info(position=0, target=1, **extra):
    return {
        "position_slot": position,
        "target_slot": target,
        "decision": 3,
        "remaining_ammunition": 4,
        **extra,
    }


def test_rule_and_seeded_random_teachers_choose_only_basic_goals():
    rule = goal_teachers.make_teacher("rule")
    assert (
        rule(_info(position=0, target=1), np.random.default_rng(1))
        is BasicGoal.ALIGN_WITHOUT_FIRE
    )
    assert (
        rule(_info(position=1, target=1), np.random.default_rng(1))
        is BasicGoal.HIT_TARGET
    )

    random = goal_teachers.make_teacher("random")
    first, second = np.random.default_rng(9), np.random.default_rng(9)
    assert [random(_info(), first) for _ in range(4)] == [
        random(_info(), second) for _ in range(4)
    ]


def test_llm_teacher_posts_only_symbolic_state_and_audits_reply(monkeypatch):
    requests = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=None):
            assert size == 65537
            return (
                b'{"choices":[{"message":{"content":"{\\"goal\\":\\"HIT_TARGET\\"}"}}]}'
            )

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Reply()

    monkeypatch.setattr(goal_teachers.urllib.request, "urlopen", urlopen)
    audit = []
    teacher = goal_teachers.make_teacher(
        "llm",
        url="http://127.0.0.1:8080/v1/chat/completions",
        model="tiny",
        audit=audit,
    )

    assert (
        teacher(_info(private="never send"), np.random.default_rng(1))
        is BasicGoal.HIT_TARGET
    )
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert timeout == 10.0 and body["model"] == "tiny" and body["temperature"] == 0
    assert body["messages"][-1]["content"] == json.dumps(
        goal_teachers.teacher_summary(_info()), sort_keys=True
    )
    assert "private" not in json.dumps(body)
    assert audit[0]["request"]["model"] == "tiny"
    assert audit[0]["state"] == goal_teachers.teacher_summary(_info())
    assert "raw_response" in audit[0] and "latency_seconds" in audit[0]


@pytest.mark.parametrize(
    "reply,error",
    [
        (
            b'{"choices":[{"message":{"content":"{\\"goal\\":\\"OTHER\\"}"}}]}',
            ValueError,
        ),
        (TimeoutError("slow"), ValueError),
    ],
)
def test_llm_teacher_rejects_bad_replies_and_transport_errors(
    monkeypatch, reply, error
):
    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, _size=None):
            return reply

    def urlopen(*_args, **_kwargs):
        if isinstance(reply, Exception):
            raise reply
        return Reply()

    monkeypatch.setattr(goal_teachers.urllib.request, "urlopen", urlopen)
    audit = []
    with pytest.raises(error):
        goal_teachers.make_teacher(
            "llm",
            url="http://127.0.0.1:8080/v1/chat/completions",
            model="tiny",
            audit=audit,
        )(_info(), np.random.default_rng(1))
    assert "error" in audit[0]


def test_openrouter_authentication_is_preflighted_and_never_audited(monkeypatch):
    requests = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=None):
            assert size == 65537
            return (
                b'{"choices":[{"message":{"content":"{\\"goal\\":\\"HIT_TARGET\\"}"}}]}'
            )

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Reply()

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setattr(goal_teachers.urllib.request, "urlopen", urlopen)
    audit = []
    teacher = goal_teachers.make_teacher(
        "llm",
        url=goal_teachers.OPENROUTER_URL,
        model=goal_teachers.DEFAULT_LLM_MODEL,
        audit=audit,
    )
    assert teacher(_info(), np.random.default_rng(1)) is BasicGoal.HIT_TARGET
    request, _ = requests[0]
    assert request.get_header("Authorization") == "Bearer secret-key"
    body = json.loads(request.data)
    assert body["reasoning"] == {"enabled": False}
    assert body["response_format"] == {"type": "json_object"}
    assert "secret-key" not in json.dumps(audit)

    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError):
        goal_teachers.make_teacher(
            "llm", url=goal_teachers.OPENROUTER_URL, model="tiny"
        )
    for invalid_key in ("bad-\u2022-key", "bad key", "bad\nkey"):
        monkeypatch.setenv("OPENROUTER_API_KEY", invalid_key)
        with pytest.raises(ValueError, match="visible ASCII"):
            goal_teachers.make_teacher(
                "llm", url=goal_teachers.OPENROUTER_URL, model="tiny"
            )
    assert len(requests) == 1
    with pytest.raises(ValueError):
        goal_teachers.make_teacher("llm", url="https://example.test", model="tiny")


def test_training_uses_teacher_only_at_reset_or_goal_boundary(monkeypatch):
    class Env:
        def __init__(self):
            self.step_count = self.resets = 0

        def reset(self, *, seed):
            self.resets += 1
            return np.zeros((84, 84, 4), dtype=np.uint8), {
                **_info(position=0, target=1, seed=seed),
                "next_unavailable_masks": _masks(),
            }

        def step(self, _action):
            states = [
                (_info(0, 1), False),
                (_info(1, 1), False),
                (_info(0, 1), True),
            ]
            info, terminated = states[self.step_count]
            self.step_count += 1
            return (
                np.zeros((84, 84, 4), dtype=np.uint8),
                0.0,
                terminated,
                False,
                {**info, "next_unavailable_masks": _masks()},
            )

    calls, selected = [], []

    def teacher(info, rng):
        calls.append((dict(info), rng.integers(2)))
        return BasicGoal.ALIGN_WITHOUT_FIRE

    def select(*_args, goal, **_kwargs):
        selected.append(goal)
        return 1, 0

    monkeypatch.setattr(learning, "select_goal_action", select)
    monkeypatch.setattr(learning, "goal_train_step", lambda *_args, **_kwargs: 0.0)
    learning.run_goal_conditioned_training(
        Env(),
        steps=3,
        warmup=1,
        batch_size=1,
        seed=10,
        device="cpu",
        epsilon=0.0,
        behavior_goal_selector=teacher,
    )

    assert selected == [BasicGoal.ALIGN_WITHOUT_FIRE] * 3
    assert len(calls) == 2  # Initial reset and the second transition's achieved goal.


def test_training_rejects_a_teacher_value_outside_basic_goal(monkeypatch):
    class Env:
        def reset(self, *, seed):
            return np.zeros((84, 84, 4), dtype=np.uint8), {
                **_info(seed=seed),
                "next_unavailable_masks": _masks(),
            }

    monkeypatch.setattr(
        learning, "select_goal_action", lambda *_args, **_kwargs: (1, 0)
    )
    with pytest.raises(ValueError):
        learning.run_goal_conditioned_training(
            Env(),
            steps=1,
            warmup=1,
            batch_size=1,
            seed=10,
            device="cpu",
            behavior_goal_selector=lambda _info, _rng: "not a goal",
        )


def test_comparison_shares_evaluation_schedule_and_retains_failed_condition(
    monkeypatch, tmp_path
):
    calls = []

    def report(*, evaluation_seeds, behavior_goal_selector):
        calls.append(
            (
                evaluation_seeds,
                behavior_goal_selector(_info(), np.random.default_rng(1)),
            )
        )
        if len(calls) == 2:
            raise RuntimeError("deliberate failure")
        return {
            "aggregate": {
                "correct_goal_success_rate": 1.0,
                "swapped_goal_success_rate": 0.0,
            }
        }

    monkeypatch.setattr(goal_teachers, "run_goal_conditioned_report", report)
    output = tmp_path / "comparison.json"
    result = goal_teachers.run_comparison(
        teachers=("rule", "random"), evaluation_episodes=2, output_path=output
    )

    assert [seeds for seeds, _ in calls] == [
        goal_teachers.EVALUATION_SEEDS[:2],
        goal_teachers.EVALUATION_SEEDS[:2],
    ]
    assert result["conditions"]["rule"]["status"] == "completed"
    assert result["conditions"]["random"]["status"] == "failed"
    assert output.exists()
    with pytest.raises(FileExistsError):
        goal_teachers.run_comparison(teachers=("rule",), output_path=output)
