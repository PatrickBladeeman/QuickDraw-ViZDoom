"""Bounded micro-learning contract, runtime, and QA checks."""

import builtins
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quickdraw_vizdoom.envs.micro_learning import MicroLearningV1Env
from quickdraw_vizdoom.qa import micro_learning as qa_micro
from quickdraw_vizdoom.qa.results import Outcome
from quickdraw_vizdoom.qa.runner import run_profile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/micro-learning-v1.json"


def test_real_classes_masks_rewards_and_terminal_targets():
    torch = pytest.importorskip("torch")
    from quickdraw_vizdoom.learning import BranchingQNetwork, compute_td_targets

    frames, labels, transitions = {}, [], []
    with MicroLearningV1Env() as env:
        for seed in range(36000, 36004):
            observation, info = env.reset(seed=seed)
            present = bool(seed % 2)
            labels.append(info["target_present"])
            assert info["target_present"] == present
            assert info["real_observation"]
            assert observation.shape == (84, 84, 4)
            assert observation.dtype == np.float32
            assert (
                observation.any() and 0 <= observation.min() <= observation.max() <= 1
            )
            assert all(
                np.array_equal(observation[:, :, 0], observation[:, :, i])
                for i in range(4)
            )
            if present in frames:
                np.testing.assert_array_equal(frames[present], observation)
            frames[present] = observation
            assert info["next_unavailable_masks"] == [
                [False, True, True],
                [False, False],
            ]
            for action in ((1, 0), (2, 1)):
                with pytest.raises(ValueError, match="unavailable"):
                    env.step(action)
            # Idle gives both a correct absent and an incorrect present example.
            final, reward, terminal, truncated, info = env.step((0, 0))
            assert reward == (-1.0 if present else 1.0)
            assert terminal and not truncated and not info["bootstrap"]
            assert info["correct"] == (not present)
            assert info["branch_action"] == [0, 0]
            assert (
                info["real_observation"]
                and info["final_frame_source"] == "vizdoom_state"
            )
            np.testing.assert_array_equal(final[:, :, :3], observation[:, :, 1:])
            transitions.append(
                (
                    observation,
                    (0, 0),
                    reward,
                    final,
                    terminal,
                    truncated,
                    env.action_masks(),
                )
            )
            with pytest.raises(RuntimeError):
                env.step((0, 0))
        for seed in (36000, 36001):
            env.reset(seed=seed)
            assert env.step((0, 1))[1] == (1.0 if seed % 2 else -1.0)
    assert labels.count(True) == labels.count(False) == 2
    assert not np.array_equal(frames[False], frames[True])
    targets = compute_td_targets(BranchingQNetwork(), transitions)
    assert torch.equal(targets, torch.tensor([1.0, -1.0, 1.0, -1.0]))


def test_reset_rejects_missing_zero_or_identical_visual_signal(monkeypatch):
    class Game:
        def set_seed(self, seed):
            pass

        def set_doom_map(self, name):
            pass

        def new_episode(self):
            pass

        def get_state(self):
            return self.state

        def close(self):
            pass

    game = Game()
    env = MicroLearningV1Env()
    env._game = game
    game.state = None
    with pytest.raises(ValueError, match="real observation"):
        env.reset(seed=0)
    game.state = type(
        "State", (), {"screen_buffer": np.zeros((120, 160), dtype=np.uint8)}
    )()
    with pytest.raises(ValueError, match="nonzero"):
        env.reset(seed=0)
    game.state.screen_buffer.fill(100)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="distinct"):
        env.reset(seed=1)


def test_fresh_training_state_and_frozen_image_only_evaluation(monkeypatch):
    torch = pytest.importorskip("torch")
    from quickdraw_vizdoom import learning, micro_learning

    environments, updates, inputs = [], [], []

    class Env:
        def __init__(self, _path):
            environments.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def reset(self, seed):
            return np.full((84, 84, 4), 0.5, np.float32), {
                "target_present": "privileged metadata must not reach the policy",
                "next_unavailable_masks": [[False, True, True], [False, False]],
            }

        def step(self, action):
            observation, info = self.reset(0)
            return observation, 1.0, True, False, {**info, "real_observation": True}

    def update(network, optimizer, replay, batch_size, rng, **kwargs):
        updates.append((network, optimizer, replay, rng))
        assert len(replay) == 1 and batch_size == 1
        return 0.1

    forward = learning.BranchingQNetwork.forward

    def checked_forward(self, observation):
        inputs.append(observation)
        assert isinstance(observation, torch.Tensor)
        assert observation.shape == (84, 84, 4) and observation.dtype == torch.float32
        return forward(self, observation)

    monkeypatch.setattr(micro_learning, "MicroLearningV1Env", Env)
    monkeypatch.setattr(learning, "train_step", update)
    monkeypatch.setattr(learning.BranchingQNetwork, "forward", checked_forward)
    monkeypatch.setattr(
        torch.optim.Adam,
        "step",
        lambda *_a, **_k: pytest.fail("Evaluation updated parameters"),
    )
    for seed in (32001, 34001):
        result = micro_learning.run_session(
            config_path="unused",
            seed=seed,
            steps=1,
            warmup=1,
            batch_size=1,
            epsilon=0.2,
            gamma=0.99,
            device="cpu",
            evaluation_seed_start=36000,
            evaluation_episodes=2,
        )
        assert result["parameters_unchanged"] and result["evaluation_updates"] == 0
    assert len(environments) == len(updates) == 2 and inputs
    assert environments[0] is not environments[1]
    assert all(first is not second for first, second in zip(*updates))
    with pytest.raises(ValueError, match="overlap"):
        micro_learning.run_session(
            config_path="unused",
            seed=36000,
            steps=1,
            warmup=1,
            batch_size=1,
            epsilon=0.2,
            gamma=0.99,
            device="cpu",
            evaluation_seed_start=36000,
            evaluation_episodes=2,
        )
    assert len(environments) == 2


def valid_result(seed):
    return {
        "decisions": 512,
        "updates": 481,
        "terminal": 512,
        "truncated": 0,
        "last_loss": 0.01,
        "training_seed_start": seed,
        "training_seed_end": seed + 511,
        "evaluation_seed_start": 36000,
        "evaluation_seed_end": 36099,
        "evaluation_episodes": 100,
        "evaluation_updates": 0,
        "parameters_unchanged": True,
        "greedy_successes": 100,
        "greedy_success_rate": 1.0,
        "random_baseline_rate": 0.5,
        "improvement_percentage_points": 50.0,
        "process_launches": 1,
        "training_sessions": 1,
        "trace_collections": 0,
        "external_calls": 0,
    }


class StartupError(RuntimeError):
    pass


def test_qa_runs_exactly_two_registered_sessions_and_reports_both(monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return valid_result(kwargs["seed"])

    monkeypatch.setattr(qa_micro, "_load_runtime", lambda: (run, StartupError))
    report = run_profile(CONTRACT, "micro-learning")
    assert report.outcome is Outcome.PASS
    assert calls == [
        dict(
            config_path=ROOT / "scenarios/quickdraw/micro-learning-v1.cfg",
            seed=seed,
            steps=512,
            warmup=32,
            batch_size=32,
            epsilon=0.2,
            gamma=0.99,
            device="cpu",
            evaluation_seed_start=36000,
            evaluation_episodes=100,
        )
        for seed in (32001, 34001)
    ]
    serialized = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert serialized["outcome"] == "PASS"
    assert [s["seed"] for s in serialized["training_sessions"]] == [32001, 34001]


@pytest.mark.parametrize(
    "mutation",
    [
        {
            "greedy_successes": 89,
            "greedy_success_rate": 0.89,
            "improvement_percentage_points": 39,
        },
        {"updates": 0},
        {"last_loss": float("nan")},
        {"last_loss": float("inf")},
        {"last_loss": -1.0},
        {"last_loss": 10**400},
        {"training_seed_end": 36000},
        {"evaluation_seed_start": 32001},
        {"process_launches": 2},
        {"decisions": 513},
        {"evaluation_episodes": 101},
        {"evaluation_updates": 1},
        {"parameters_unchanged": False},
        {"greedy_success_rate": float("nan")},
        {"random_baseline_rate": 0.4},
        {"updates": True},
        {"improvement_percentage_points": 25},
    ],
)
def test_invalid_session_fails(monkeypatch, mutation):
    monkeypatch.setattr(
        qa_micro,
        "_load_runtime",
        lambda: (
            lambda **kwargs: {**valid_result(kwargs["seed"]), **mutation},
            StartupError,
        ),
    )
    report = run_profile(CONTRACT, "micro-learning")
    assert report.outcome is Outcome.FAIL
    assert len(report.training_sessions) == 2
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("failure", ["dependency", "startup", "artifact"])
def test_infrastructure_failures(monkeypatch, failure):
    def missing_dependency():
        raise ModuleNotFoundError("torch")

    def startup(**_kwargs):
        raise StartupError("ViZDoom startup failed")

    if failure == "dependency":
        monkeypatch.setattr(qa_micro, "_load_runtime", missing_dependency)
    elif failure == "startup":
        monkeypatch.setattr(qa_micro, "_load_runtime", lambda: (startup, StartupError))
    else:
        read = Path.read_bytes

        def missing(path):
            if path.suffix == ".wad":
                raise FileNotFoundError(str(path))
            return read(path)

        monkeypatch.setattr(Path, "read_bytes", missing)
    assert run_profile(CONTRACT, "micro-learning").outcome is Outcome.INFRA_INVALID


@pytest.mark.parametrize(
    "field,value",
    [
        ("seeds", [36000, 34001]),
        ("steps", 513),
        ("warmup", 513),
        ("budgets", {}),
        ("device", "cuda"),
        ("evaluation_episodes", 101),
    ],
)
def test_malformed_profile_fails_before_runtime(tmp_path, monkeypatch, field, value):
    contract = json.loads(CONTRACT.read_text())
    contract["qa"]["profiles"]["micro-learning"][field] = value
    (tmp_path / "schemas").mkdir()
    schema = ROOT / "contracts/schemas/micro-learning-v1.schema.json"
    (tmp_path / "schemas" / schema.name).write_text(schema.read_text())
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(
        qa_micro, "_load_runtime", lambda: pytest.fail("Runtime was imported")
    )
    assert run_profile(path, "micro-learning").outcome is Outcome.FAIL


def test_basic_profiles_do_not_import_micro_learning(monkeypatch):
    original = builtins.__import__

    def reject(name, *args, **kwargs):
        if "micro_learning" in name or name == "torch" or name.startswith("torch."):
            raise AssertionError("Basic imported an unselected learning path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject)
    from quickdraw_vizdoom.qa import runner

    monkeypatch.setattr(
        runner,
        "_load_training_runtime",
        lambda: (
            lambda **_kwargs: {
                "decisions": 16,
                "updates": 13,
                "terminal": 0,
                "truncated": 0,
                "last_loss": 0.1,
            },
            StartupError,
        ),
    )
    for profile in ("dev", "integration", "training-smoke"):
        assert (
            run_profile(ROOT / "contracts/basic-v1.json", profile).outcome
            is Outcome.PASS
        )
    assert (
        run_profile(ROOT / "contracts/basic-v1.json", "micro-learning").outcome
        is Outcome.FAIL
    )


def test_qa_import_is_lazy_in_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import quickdraw_vizdoom.qa; "
            "assert not any('micro_learning' in name or name == 'torch' "
            "or name == 'quickdraw_vizdoom.learning' for name in sys.modules)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
