"""The native game viewer changes presentation, not agent decisions."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from quickdraw_vizdoom.envs.micro_learning import MicroLearningV1Env


def test_native_window_is_opt_in(monkeypatch):
    calls = []

    class Game:
        def load_config(self, path):
            pass

        def set_window_visible(self, visible):
            calls.append(visible)

        def add_game_args(self, args):
            calls.append(args)

        def set_seed(self, seed):
            pass

        def init(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "vizdoom", SimpleNamespace(DoomGame=Game))
    for visible in (False, True):
        with MicroLearningV1Env(window_visible=visible) as env:
            env._load_game(32001)
    assert calls == [False, True, "+vid_winscale 4"]


def test_watch_preserves_transition_and_reuses_training_and_evaluation(
    monkeypatch, capsys
):
    pytest.importorskip("torch")
    from quickdraw_vizdoom import watch

    transition = (
        np.ones((84, 84, 4), np.float32),
        1.0,
        True,
        False,
        {
            "correct": True,
            "target_present": True,
            "semantic_action": ["Stay", "Shoot"],
        },
    )
    actions, delays, calls = [], [], []

    def step(self, action):
        actions.append(action)
        return transition

    def train(env, **kwargs):
        on_selection = kwargs.pop("on_selection")
        assert on_selection == env.record_selection
        on_selection(True)
        calls.append(kwargs)
        assert env._window_visible
        assert env.step((0, 1)) is transition
        return {"updates": 481, "last_loss": 0.01}, "network"

    def evaluate(network, env, **kwargs):
        assert network == "network" and env.phase == "Evaluation"
        assert env.decisions == 0 and not env.recent
        assert not env.exploring
        assert kwargs == {"seed_start": 36000, "episodes": 100}
        assert env.step((0, 1)) is transition
        return {"greedy_successes": 100}

    monkeypatch.setattr(MicroLearningV1Env, "step", step)
    monkeypatch.setattr(watch.time, "sleep", delays.append)
    monkeypatch.setattr(watch, "run_training", train)
    monkeypatch.setattr(watch, "evaluate", evaluate)
    assert watch.main(["--delay", "0.25"]) == 0
    assert actions == [(0, 1), (0, 1)] and delays == [0.25, 0.25]
    assert calls == [
        dict(
            steps=512,
            warmup=32,
            batch_size=32,
            seed=32001,
            device="cpu",
            epsilon=0.2,
            gamma=0.99,
        )
    ]
    output = capsys.readouterr().out
    assert "Shoot | reward +1 | source=random" in output
    assert "Shoot | reward +1 | source=greedy" in output
    for delay in ("nan", "inf", "-1"):
        with pytest.raises(SystemExit) as error:
            watch.main(["--delay", delay])
        assert error.value.code == 2


def test_real_training_and_evaluation_log_the_selected_source(capsys):
    pytest.importorskip("torch")
    from quickdraw_vizdoom.watch import WatchEnv, evaluate, run_training

    with WatchEnv(delay=0) as env:
        env._window_visible = False
        _, network = run_training(
            env,
            steps=4,
            warmup=4,
            batch_size=4,
            seed=32001,
            device="cpu",
            epsilon=1.0,
            on_selection=env.record_selection,
        )
        env.phase = "Evaluation"
        env.exploring = False
        evaluate(
            network,
            env,
            seed_start=36000,
            episodes=2,
        )
    lines = capsys.readouterr().out.splitlines()
    training = [line for line in lines if line.startswith("Training")]
    evaluation = [line for line in lines if line.startswith("Evaluation")]
    assert len(training) == 4 and len(evaluation) == 2
    assert all("source=random" in line for line in training)
    assert all("source=greedy" in line for line in evaluation)
