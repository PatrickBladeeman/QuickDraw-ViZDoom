"""Tests for the minimal Basic v1 PyTorch learning loop."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from quickdraw_vizdoom.learning import (  # noqa: E402
    BranchingQNetwork,
    compute_td_targets,
    select_action,
    train_step,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _observation(value: float = 0.0) -> np.ndarray:
    return np.full((84, 84, 4), value, dtype=np.float32)


def test_network_shapes_and_masked_action_selection() -> None:
    network = BranchingQNetwork()
    movement, combat = network(torch.zeros(2, 84, 84, 4))

    assert movement.shape == (2, 3)
    assert combat.shape == (2, 2)

    action = select_action(
        network,
        _observation(),
        ([True, False, True], [True, False]),
        epsilon=1.0,
        rng=np.random.default_rng(1),
    )
    assert action == (1, 1)
    assert all(isinstance(value, int) for value in action)


def test_replay_update_is_finite_and_changes_a_parameter() -> None:
    rng = np.random.default_rng(2)
    network = BranchingQNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    replay = [
        (
            rng.random((84, 84, 4), dtype=np.float32),
            (index % 3, index % 2),
            1.0,
            rng.random((84, 84, 4), dtype=np.float32),
            False,
            False,
            (
                np.zeros(3, dtype=bool),
                np.zeros(2, dtype=bool),
            ),
        )
        for index in range(4)
    ]
    before = [parameter.detach().clone() for parameter in network.parameters()]

    loss = train_step(network, optimizer, replay, 4, rng)

    assert np.isfinite(loss)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, network.parameters())
    )


def test_terminated_does_not_bootstrap_but_truncated_does() -> None:
    class FixedNetwork(BranchingQNetwork):
        def forward(self, observations):
            batch = observations.shape[0]
            return (
                torch.tensor([1.0, 2.0, 3.0]).expand(batch, -1),
                torch.tensor([4.0, 5.0]).expand(batch, -1),
            )

    masks = (np.array([False, False, True]), np.array([False, True]))
    transitions = [
        (_observation(), (0, 0), 1.0, _observation(), True, False, masks),
        (_observation(), (0, 0), 1.0, _observation(), False, True, masks),
    ]

    targets = compute_td_targets(FixedNetwork(), transitions, gamma=0.5)

    assert targets.tolist() == pytest.approx([1.0, 4.0])


def test_real_smoke_command_steps_and_updates() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "quickdraw_vizdoom.learning",
            "--steps",
            "4",
            "--warmup",
            "4",
            "--batch-size",
            "4",
            "--seed",
            "31001",
            "--device",
            "cpu",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS decisions=4 updates=1" in result.stdout
