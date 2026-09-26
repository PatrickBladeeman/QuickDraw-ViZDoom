"""Minimal branch-action DQN smoke loop for the Basic v1 environment."""

from __future__ import annotations

import argparse
import operator
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quickdraw_vizdoom.envs import BasicV1Env


BranchAction = tuple[int, int]
Masks = tuple[np.ndarray, np.ndarray]
UnconditionedTransition = tuple[
    np.ndarray,
    BranchAction,
    float,
    np.ndarray,
    bool,
    bool,
    Masks,
]
ConditionedTransition = tuple[
    np.ndarray,
    BranchAction,
    float,
    np.ndarray,
    bool,
    bool,
    Masks,
    int,
    int,
]
Transition = UnconditionedTransition | ConditionedTransition
GoalSelector = Callable[[Mapping[str, Any]], int]


class EnvironmentStartupError(RuntimeError):
    """Raised when a smoke session cannot start its environment."""


class BranchingQNetwork(nn.Module):
    """A shared image encoder with movement and combat Q-value heads."""

    def __init__(self, goal_count: int = 0) -> None:
        super().__init__()
        if goal_count < 0:
            raise ValueError("goal_count must not be negative.")
        self.goal_count = goal_count
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 64),
            nn.ReLU(),
        )
        self.movement_head = nn.Linear(64 + goal_count, 3)
        self.combat_head = nn.Linear(64 + goal_count, 2)

    def forward(
        self, observations: torch.Tensor, goals: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if observations.ndim != 4:
            raise ValueError("Observations must be HWC or batched NHWC.")
        if observations.shape[-1] == 4:
            observations = observations.permute(0, 3, 1, 2)
        if observations.shape[1:] != (4, 84, 84):
            raise ValueError("Observations must have shape (84, 84, 4).")
        encoded = self.encoder(observations.float().contiguous())
        if self.goal_count:
            if goals is None:
                raise ValueError("A categorical goal is required by this network.")
            goals = torch.as_tensor(goals, device=encoded.device)
            if goals.ndim == 0:
                goals = goals.expand(encoded.shape[0])
            if goals.ndim != 1 or goals.shape[0] != encoded.shape[0]:
                raise ValueError(
                    "Goals must contain one categorical ID per observation."
                )
            if torch.any(goals != goals.long()) or torch.any(
                (goals < 0) | (goals >= self.goal_count)
            ):
                raise ValueError(f"Goal IDs must be in [0, {self.goal_count}).")
            encoded = torch.cat(
                (encoded, F.one_hot(goals.long(), self.goal_count).float()), dim=1
            )
        elif goals is not None:
            raise ValueError("The unconditioned network does not accept goals.")
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
    *,
    goal: int | None = None,
    on_selection: Callable[[bool], None] | None = None,
) -> BranchAction:
    """Choose a legal action; optionally report whether exploration was used."""

    movement_mask, combat_mask = normalize_masks(masks)
    exploring = rng.random() < epsilon
    if on_selection is not None:
        on_selection(exploring)
    if exploring:
        return tuple(
            int(rng.choice(np.flatnonzero(~mask)))
            for mask in (movement_mask, combat_mask)
        )  # type: ignore[return-value]

    with torch.no_grad():
        tensor = torch.as_tensor(observation, device=device)
        values = (
            network(tensor) if goal is None else network(tensor, torch.tensor(goal))
        )
    return tuple(
        int(
            branch_values[0]
            .masked_fill(torch.as_tensor(mask, device=device), -torch.inf)
            .argmax()
            .item()
        )
        for branch_values, mask in zip(values, (movement_mask, combat_mask))
    )  # type: ignore[return-value]


def _transition_goals(
    transitions: Sequence[Transition],
    index: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    lengths = {len(transition) for transition in transitions}
    if lengths == {7}:
        return None
    if lengths != {9}:
        raise ValueError(
            "Replay batches cannot mix conditioned and unconditioned data."
        )
    return torch.tensor(
        [transition[index] for transition in transitions], device=device
    )


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
    next_goals = _transition_goals(transitions, 8, device)

    with torch.no_grad():
        values = (
            network(next_observations)
            if next_goals is None
            else network(next_observations, next_goals)
        )
        movement, combat = values
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
    goals = _transition_goals(batch, 7, device)

    values = network(observations) if goals is None else network(observations, goals)
    movement, combat = values
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


def _select_goal(selector: GoalSelector | None, info: Mapping[str, Any]) -> int | None:
    if selector is None:
        return None
    try:
        goal = operator.index(selector(info))
    except (TypeError, ValueError) as error:
        raise ValueError("Goal selectors must return an integer goal ID.") from error
    if goal not in range(3):
        raise ValueError("Goal IDs must be 0, 1, or 2.")
    return goal


def transition_metric(
    *,
    action: BranchAction,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    goal: int | None,
    next_goal: int | None,
    goal_source: str,
    goal_age: int,
) -> dict[str, Any]:
    """Build the policy-safe telemetry record for one completed decision."""

    events = list(info.get("events", ()))
    changed = goal != next_goal
    if changed:
        events.append("goal_change")
    success = bool(
        goal is not None
        and (
            (goal in (0, 1) and info.get("position_slot") == info.get("target_slot"))
            or (goal == 2 and "target_hit" in events)
        )
    )
    return {
        "episode": info.get("episode_index"),
        "decision": info.get("decision"),
        "action": list(action),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "shot": bool(info.get("shots_fired", action[1] == 1)),
        "target_hit": "target_hit" in events,
        "goal": goal,
        "next_goal": next_goal,
        "goal_source": goal_source,
        "goal_success": success,
        "goal_changed": changed,
        "time_to_goal": goal_age if success else None,
        "events": events,
    }


def summarize_metrics(
    transitions: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Summarize decision and episode telemetry with finite empty-case values."""

    goal_transitions = [item for item in transitions if item["goal"] is not None]
    successes = [item for item in goal_transitions if item["goal_success"]]
    shots = sum(int(item["shot"]) for item in transitions)
    hits = sum(int(item["target_hit"]) for item in transitions)
    decisions_to_hit = [
        int(item["decisions_to_hit"])
        for item in episodes
        if item["decisions_to_hit"] is not None
    ]
    times_to_goal = [int(item["time_to_goal"]) for item in successes]
    selections = Counter(
        "none" if item["goal"] is None else str(item["goal"]) for item in transitions
    )
    changes = Counter(
        f"{item['goal']}->{item['next_goal']}"
        for item in transitions
        if item["goal_changed"]
    )

    def key(name: str) -> str:
        return f"{prefix}{name}"

    return {
        key("transition_metrics"): list(transitions),
        key("episode_metrics"): list(episodes),
        key("episode_returns"): [float(item["return"]) for item in episodes],
        key("cumulative_episode_return"): float(
            sum(float(item["reward"]) for item in transitions)
        ),
        key("mean_episode_return"): float(
            np.mean([item["return"] for item in episodes]) if episodes else 0.0
        ),
        key("decisions_to_hit"): decisions_to_hit,
        key("mean_decisions_to_hit"): float(
            np.mean(decisions_to_hit) if decisions_to_hit else 0.0
        ),
        key("shots_hit"): hits,
        key("total_shots"): shots,
        key("shot_percentage"): float(100 * hits / shots if shots else 0.0),
        key("goal_completions"): len(successes),
        key("goal_attempts"): len(goal_transitions),
        key("goal_completion_rate"): float(
            len(successes) / len(goal_transitions) if goal_transitions else 0.0
        ),
        key("time_to_goal"): times_to_goal,
        key("mean_time_to_goal"): float(
            np.mean(times_to_goal) if times_to_goal else 0.0
        ),
        key("goal_selection_counts"): dict(selections),
        key("goal_transition_counts"): dict(changes),
    }


def run_smoke(
    steps: int,
    warmup: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Run a seeded rollout with real replay updates."""

    try:
        env = BasicV1Env()
    except Exception as error:
        raise EnvironmentStartupError(f"Environment startup failed: {error}") from error
    with env:
        result, _network = run_training(env, steps, warmup, batch_size, seed, device)
    return result


def run_training(
    env: BasicV1Env,
    steps: int,
    warmup: int,
    batch_size: int,
    seed: int,
    device: str,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    *,
    on_selection: Callable[[bool], None] | None = None,
    episode_seeds: Sequence[int] | None = None,
    goal_selector: GoalSelector | None = None,
    goal_source: str = "no_goal",
) -> tuple[dict[str, Any], BranchingQNetwork]:
    """Train in the caller-owned environment and return the in-memory policy."""

    threshold = max(warmup, batch_size)
    if steps < threshold or batch_size < 1 or warmup < 0:
        raise ValueError("steps must reach warmup and batch-size must be positive.")
    if episode_seeds is not None and len(episode_seeds) != steps:
        raise ValueError("episode_seeds must contain one seed per decision.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    network = BranchingQNetwork(goal_count=3 if goal_selector else 0).to(torch_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    replay: deque[Transition] = deque(maxlen=1_000)
    updates = terminals = truncations = episode = 0
    last_loss: float | None = None
    transition_metrics: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_decisions = episode_shots = episode_hits = 0
    goal_age = 1
    reset_seeds_used: list[int] = []

    reset_seeds = tuple(episode_seeds) if episode_seeds is not None else None
    try:
        observation, info = env.reset(
            seed=seed if reset_seeds is None else reset_seeds[0]
        )
    except ValueError:
        raise
    except Exception as error:
        raise EnvironmentStartupError(f"Environment startup failed: {error}") from error
    reset_seeds_used.append(int(info.get("seed", seed)))
    masks = normalize_masks(info["next_unavailable_masks"])
    goal = _select_goal(goal_selector, info)
    for decision in range(steps):
        action = select_action(
            network,
            observation,
            masks,
            epsilon=epsilon,
            rng=rng,
            device=torch_device,
            goal=goal,
            on_selection=on_selection,
        )
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_masks = normalize_masks(info["next_unavailable_masks"])
        next_goal = _select_goal(goal_selector, info)
        transition: Transition = (
            observation.copy(),
            action,
            reward,
            next_observation.copy(),
            terminated,
            truncated,
            next_masks,
        )
        if goal is not None and next_goal is not None:
            transition = (*transition, goal, next_goal)
        replay.append(transition)
        metric = transition_metric(
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            goal=goal,
            next_goal=next_goal,
            goal_source=goal_source,
            goal_age=goal_age,
        )
        transition_metrics.append(metric)
        episode_return += float(reward)
        episode_decisions += 1
        episode_shots += int(metric["shot"])
        episode_hits += int(metric["target_hit"])
        if len(replay) >= threshold:
            last_loss = train_step(
                network,
                optimizer,
                replay,
                batch_size,
                rng,
                gamma=gamma,
                device=torch_device,
            )
            updates += 1

        if terminated or truncated:
            terminals += int(terminated)
            truncations += int(truncated)
        if terminated or truncated or decision == steps - 1:
            episode_metrics.append(
                {
                    "return": episode_return,
                    "decisions": episode_decisions,
                    "decisions_to_hit": episode_decisions if episode_hits else None,
                    "shots_hit": episode_hits,
                    "total_shots": episode_shots,
                    "target_hit": bool(episode_hits),
                    "completed": bool(terminated or truncated),
                }
            )

        if terminated or truncated:
            episode += 1
            if decision == steps - 1:
                break
            try:
                observation, info = env.reset(
                    seed=seed + episode if reset_seeds is None else reset_seeds[episode]
                )
            except ValueError:
                raise
            except Exception as error:
                raise EnvironmentStartupError(
                    f"Environment startup failed: {error}"
                ) from error
            reset_seeds_used.append(
                int(
                    info.get(
                        "seed",
                        reset_seeds[episode] if reset_seeds else seed + episode,
                    )
                )
            )
            masks = normalize_masks(info["next_unavailable_masks"])
            goal = _select_goal(goal_selector, info)
            goal_age = 1
            episode_return = 0.0
            episode_decisions = episode_shots = episode_hits = 0
        else:
            observation, masks, goal = next_observation, next_masks, next_goal
            goal_age = (
                1 if metric["goal_success"] or metric["goal_changed"] else goal_age + 1
            )

    if last_loss is None:
        raise RuntimeError("Smoke run completed without an optimizer update.")
    result = {
        "decisions": steps,
        "updates": updates,
        "terminal": terminals,
        "truncated": truncations,
        "last_loss": last_loss,
        "training_episode_seeds": reset_seeds_used,
        **summarize_metrics(transition_metrics, episode_metrics),
    }
    return result, network


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
