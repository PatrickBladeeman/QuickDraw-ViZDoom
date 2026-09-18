"""Minimal branch-action DQN smoke loop for the Basic v1 environment."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quickdraw_vizdoom.envs import BasicV1Env


BranchAction = tuple[int, int]
Masks = tuple[np.ndarray, np.ndarray]
Transition = tuple[
    np.ndarray,
    BranchAction,
    float,
    np.ndarray,
    bool,
    bool,
    Masks,
]


class EnvironmentStartupError(RuntimeError):
    """Raised when a smoke session cannot start its environment."""


class BranchingQNetwork(nn.Module):
    """A shared image encoder with movement and combat Q-value heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 64),
            nn.ReLU(),
        )
        self.movement_head = nn.Linear(64, 3)
        self.combat_head = nn.Linear(64, 2)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if observations.ndim != 4:
            raise ValueError("Observations must be HWC or batched NHWC.")
        if observations.shape[-1] == 4:
            observations = observations.permute(0, 3, 1, 2)
        if observations.shape[1:] != (4, 84, 84):
            raise ValueError("Observations must have shape (84, 84, 4).")
        encoded = self.encoder(observations.float().contiguous())
        return self.movement_head(encoded), self.combat_head(encoded)


def normalize_masks(masks: Sequence[Sequence[bool] | np.ndarray]) -> Masks:
    normalized = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if (
        len(normalized) != 2
        or normalized[0].shape != (3,)
        or normalized[1].shape != (2,)
    ):
        raise ValueError("Masks must match the movement (3) and combat (2) branches.")
    if any(mask.all() for mask in normalized):
        raise ValueError("Each action branch must have at least one available action.")
    return normalized[0], normalized[1]


def select_action(
    network: BranchingQNetwork,
    observation: np.ndarray,
    masks: Sequence[Sequence[bool] | np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
) -> BranchAction:
    """Choose an epsilon-greedy tuple without selecting unavailable actions."""

    movement_mask, combat_mask = normalize_masks(masks)
    if rng.random() < epsilon:
        return tuple(
            int(rng.choice(np.flatnonzero(~mask)))
            for mask in (movement_mask, combat_mask)
        )  # type: ignore[return-value]

    with torch.no_grad():
        values = network(torch.as_tensor(observation, device=device))
    return tuple(
        int(
            branch_values[0]
            .masked_fill(torch.as_tensor(mask, device=device), -torch.inf)
            .argmax()
            .item()
        )
        for branch_values, mask in zip(values, (movement_mask, combat_mask))
    )  # type: ignore[return-value]


def compute_td_targets(
    network: BranchingQNetwork,
    transitions: Sequence[Transition],
    gamma: float = 0.99,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute masked branch-sum targets, bootstrapping through truncation."""

    rewards = torch.tensor([item[2] for item in transitions], device=device)
    terminated = torch.tensor(
        [item[4] for item in transitions], dtype=torch.bool, device=device
    )
    next_observations = torch.as_tensor(
        np.stack([item[3] for item in transitions]), device=device
    )
    masks = [normalize_masks(item[6]) for item in transitions]
    movement_masks = torch.as_tensor(
        np.stack([item[0] for item in masks]), device=device
    )
    combat_masks = torch.as_tensor(np.stack([item[1] for item in masks]), device=device)

    with torch.no_grad():
        movement, combat = network(next_observations)
        next_values = movement.masked_fill(movement_masks, -torch.inf).max(1).values
        next_values += combat.masked_fill(combat_masks, -torch.inf).max(1).values
        return rewards + gamma * (~terminated).float() * next_values


def train_step(
    network: BranchingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[Transition],
    batch_size: int,
    rng: np.random.Generator,
    gamma: float = 0.99,
    device: torch.device | str = "cpu",
) -> float:
    """Sample replay and perform one optimizer update."""

    if len(replay) < batch_size:
        raise ValueError("Replay does not contain a full batch.")
    indices = rng.choice(len(replay), size=batch_size, replace=False)
    batch = [replay[int(index)] for index in indices]
    observations = torch.as_tensor(np.stack([item[0] for item in batch]), device=device)
    actions = torch.tensor([item[1] for item in batch], device=device)

    movement, combat = network(observations)
    selected = movement.gather(1, actions[:, :1]).squeeze(1)
    selected += combat.gather(1, actions[:, 1:]).squeeze(1)
    targets = compute_td_targets(network, batch, gamma=gamma, device=device)
    loss = F.smooth_l1_loss(selected, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    value = float(loss.detach().cpu())
    if not np.isfinite(value):
        raise RuntimeError("Optimizer produced a non-finite loss.")
    return value


def run_smoke(
    steps: int,
    warmup: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Run a seeded rollout with real replay updates."""

    threshold = max(warmup, batch_size)
    if steps < threshold or batch_size < 1 or warmup < 0:
        raise ValueError("steps must reach warmup and batch-size must be positive.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    network = BranchingQNetwork().to(torch_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    replay: deque[Transition] = deque(maxlen=1_000)
    updates = terminals = truncations = episode = 0
    last_loss: float | None = None

    try:
        env = BasicV1Env()
    except Exception as error:
        raise EnvironmentStartupError(f"Environment startup failed: {error}") from error

    with env:
        try:
            observation, info = env.reset(seed=seed)
        except Exception as error:
            raise EnvironmentStartupError(
                f"Environment startup failed: {error}"
            ) from error
        masks = normalize_masks(info["next_unavailable_masks"])
        for _ in range(steps):
            action = select_action(
                network, observation, masks, epsilon=0.2, rng=rng, device=torch_device
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_masks = normalize_masks(info["next_unavailable_masks"])
            replay.append(
                (
                    observation.copy(),
                    action,
                    reward,
                    next_observation.copy(),
                    terminated,
                    truncated,
                    next_masks,
                )
            )
            if len(replay) >= threshold:
                last_loss = train_step(
                    network,
                    optimizer,
                    replay,
                    batch_size,
                    rng,
                    device=torch_device,
                )
                updates += 1

            if terminated or truncated:
                terminals += int(terminated)
                truncations += int(truncated)
                episode += 1
                try:
                    observation, info = env.reset(seed=seed + episode)
                except Exception as error:
                    raise EnvironmentStartupError(
                        f"Environment startup failed: {error}"
                    ) from error
                masks = normalize_masks(info["next_unavailable_masks"])
            else:
                observation, masks = next_observation, next_masks

    if last_loss is None:
        raise RuntimeError("Smoke run completed without an optimizer update.")
    return {
        "decisions": steps,
        "updates": updates,
        "terminal": terminals,
        "truncated": truncations,
        "last_loss": last_loss,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run_smoke(**vars(args))
    print(
        "PASS "
        f"decisions={result['decisions']} updates={result['updates']} "
        f"terminal={result['terminal']} truncated={result['truncated']} "
        f"last_loss={result['last_loss']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
