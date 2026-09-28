"""Shared networks, replay types, and update steps for the Basic trainers."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BranchAction = tuple[int, int]
Masks = tuple[np.ndarray, np.ndarray]
JOINT_ACTIONS: tuple[BranchAction, ...] = tuple(
    (movement, combat) for movement in range(3) for combat in range(2)
)
GOAL_COUNT = 2
GOAL_TARGET_SYNC_INTERVAL = 100


class BasicGoal(IntEnum):
    """The two requested outcomes supported by the Basic trainer."""

    ALIGN_WITHOUT_FIRE = 0
    HIT_TARGET = 1


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
GoalPotential = Callable[[Mapping[str, Any], int | None], float]

def _image_encoder() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 9 * 9, 64),
        nn.ReLU(),
    )


def _prepare_observations(observations: torch.Tensor) -> torch.Tensor:
    observations = torch.as_tensor(observations)
    if observations.ndim == 3:
        observations = observations.unsqueeze(0)
    if observations.ndim != 4:
        raise ValueError("Observations must be HWC or batched NHWC.")
    if observations.shape[-1] == 4:
        observations = observations.permute(0, 3, 1, 2)
    if observations.shape[1:] != (4, 84, 84):
        raise ValueError("Observations must have shape (84, 84, 4).")
    return observations


class EnvironmentStartupError(RuntimeError):
    """Raised when a smoke session cannot start its environment."""

class BranchingQNetwork(nn.Module):
    """A shared image encoder with movement and combat Q-value heads."""

    def __init__(self, goal_count: int = 0) -> None:
        super().__init__()
        if goal_count < 0:
            raise ValueError("goal_count must not be negative.")
        self.goal_count = goal_count
        self.encoder = _image_encoder()
        self.movement_head = nn.Linear(64 + goal_count, 3)
        self.combat_head = nn.Linear(64 + goal_count, 2)

    def forward(
        self, observations: torch.Tensor, goals: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations = _prepare_observations(observations)
        encoded = self.encoder(observations.float().contiguous())
        if self.goal_count:
            if goals is None:
                raise ValueError("A categorical goal is required by this network.")
            goal_ids = torch.as_tensor(goals, device=encoded.device)
            if goal_ids.ndim == 0:
                goal_ids = goal_ids.expand(encoded.shape[0])
            if goal_ids.ndim != 1 or goal_ids.shape[0] != encoded.shape[0]:
                raise ValueError(
                    "Goals must contain one categorical ID per observation."
                )
            if torch.any(goal_ids != goal_ids.long()) or torch.any(
                (goal_ids < 0) | (goal_ids >= self.goal_count)
            ):
                raise ValueError(f"Goal IDs must be in [0, {self.goal_count}).")
            encoded = torch.cat(
                (encoded, F.one_hot(goal_ids.long(), self.goal_count).float()), dim=1
            )
        elif goals is not None:
            raise ValueError("The unconditioned network does not accept goals.")
        return self.movement_head(encoded), self.combat_head(encoded)


class GoalConditionedQNetwork(nn.Module):
    """Six-way joint-action Q network conditioned on a two-way goal."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _image_encoder()
        self.goal_hidden = nn.Linear(64 + GOAL_COUNT, 64)
        self.joint_head = nn.Linear(64, len(JOINT_ACTIONS))

    def forward(
        self,
        observations: torch.Tensor,
        goals: torch.Tensor | Sequence[Sequence[float]],
    ) -> torch.Tensor:
        observations = _prepare_observations(observations)
        encoded = self.encoder(observations.float().contiguous())
        vectors = torch.as_tensor(goals, device=encoded.device).float()
        if vectors.ndim == 1 and encoded.shape[0] == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.shape != (encoded.shape[0], GOAL_COUNT) or not torch.isfinite(vectors).all():
            raise ValueError("Goals must be finite two-element one-hot vectors.")
        if torch.any((vectors != 0) & (vectors != 1)) or not torch.all(vectors.sum(1) == 1):
            raise ValueError("Goals must be one-hot vectors.")
        hidden = F.relu(self.goal_hidden(torch.cat((encoded, vectors), dim=1)))
        return self.joint_head(hidden)


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


def joint_action_mask(
    masks: Sequence[Sequence[bool] | np.ndarray],
) -> np.ndarray:
    """Return a six-action mask where True means the joint action is illegal."""

    movement, combat = normalize_masks(masks)
    return np.asarray(
        [movement[movement_index] or combat[combat_index]
         for movement_index, combat_index in JOINT_ACTIONS],
        dtype=bool,
    )


def joint_action_index(action: BranchAction) -> int:
    try:
        movement, combat = operator.index(action[0]), operator.index(action[1])
    except (IndexError, TypeError) as error:
        raise ValueError("Joint actions must contain movement and combat indices.") from error
    if (movement, combat) not in JOINT_ACTIONS:
        raise ValueError("Joint action is outside the Basic action space.")
    return movement * 2 + combat


def _coerce_goal(goal: int | BasicGoal) -> BasicGoal:
    try:
        return BasicGoal(operator.index(goal))
    except (TypeError, ValueError) as error:
        raise ValueError("Goal must be ALIGN_WITHOUT_FIRE or HIT_TARGET.") from error


def goal_one_hot(
    goal: int | BasicGoal,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    selected = _coerce_goal(goal)
    return F.one_hot(
        torch.tensor(int(selected), device=device), num_classes=GOAL_COUNT
    ).float()


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    return value


def _readonly_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PhysicalTransition:
    """One immutable environment transition before goal relabeling."""

    observation: np.ndarray
    action: BranchAction
    environment_reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    next_masks: Masks
    current_info: Mapping[str, Any]
    next_info: Mapping[str, Any]
    behavior_goal: BasicGoal | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.environment_reward)):
            raise ValueError("Environment rewards must be finite.")
        object.__setattr__(self, "observation", _readonly_array(self.observation))
        object.__setattr__(self, "next_observation", _readonly_array(self.next_observation))
        object.__setattr__(self, "action", JOINT_ACTIONS[joint_action_index(self.action)])
        object.__setattr__(
            self,
            "next_masks",
            tuple(_readonly_array(mask) for mask in normalize_masks(self.next_masks)),
        )
        object.__setattr__(self, "environment_reward", float(self.environment_reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "current_info", _freeze_value(self.current_info))
        object.__setattr__(self, "next_info", _freeze_value(self.next_info))
        if self.behavior_goal is not None:
            object.__setattr__(self, "behavior_goal", _coerce_goal(self.behavior_goal))

def _event_present(info: Mapping[str, Any], event: str) -> bool:
    return bool(info.get(event, False)) or event in info.get("events", ())


def goal_achieved(
    goal: int | BasicGoal,
    action: BranchAction,
    next_info: Mapping[str, Any],
) -> bool:
    selected = _coerce_goal(goal)
    if selected is BasicGoal.ALIGN_WITHOUT_FIRE:
        return (
            action[1] == 0
            and next_info.get("position_slot") == next_info.get("target_slot")
        )
    return _event_present(next_info, "target_hit")


def goal_reward(
    goal: int | BasicGoal,
    action: BranchAction,
    next_info: Mapping[str, Any],
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> float:
    """Recompute the exact requested-goal reward for a physical transition."""

    del terminated, truncated
    selected = _coerce_goal(goal)
    if selected is BasicGoal.ALIGN_WITHOUT_FIRE:
        if action[1] == 1:
            return -1.0
        if goal_achieved(selected, action, next_info):
            return 1.0
        return -0.01
    if _event_present(next_info, "target_hit"):
        return 1.0
    if _event_present(next_info, "missed_shot"):
        return -0.1
    return -0.01


def goal_done(
    goal: int | BasicGoal,
    action: BranchAction,
    next_info: Mapping[str, Any],
    terminated: bool = False,
    truncated: bool = False,
) -> bool:
    """Return the goal TD boundary; truncation intentionally still bootstraps."""

    del truncated
    return bool(terminated or goal_achieved(goal, action, next_info))


GoalRow = tuple[PhysicalTransition, BasicGoal]


def relabel_transition(physical: PhysicalTransition) -> tuple[GoalRow, GoalRow]:
    return tuple((physical, goal) for goal in BasicGoal)  # type: ignore[return-value]


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


def select_goal_action(
    network: GoalConditionedQNetwork,
    observation: np.ndarray,
    masks: Sequence[Sequence[bool] | np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
    *,
    goal: int | BasicGoal,
) -> BranchAction:
    """Choose one legal joint action with a fixed requested goal."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between zero and one.")
    unavailable = joint_action_mask(masks)
    exploring = rng.random() < epsilon
    if exploring:
        index = int(rng.choice(np.flatnonzero(~unavailable)))
        return JOINT_ACTIONS[index]
    with torch.no_grad():
        observation_tensor = torch.as_tensor(observation, device=device)
        values = network(
            observation_tensor,
            goal_one_hot(goal, device=device),
        )[0]
        selected = values.masked_fill(
            torch.as_tensor(unavailable, device=device), -torch.inf
        ).argmax()
    return JOINT_ACTIONS[int(selected.item())]


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


def compute_goal_td_targets(
    online_network: GoalConditionedQNetwork,
    target_network: GoalConditionedQNetwork,
    transitions: Sequence[GoalRow],
    gamma: float = 0.99,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute masked Double-DQN targets without changing the row's goal."""

    observations = torch.as_tensor(
        np.stack([row[0].next_observation for row in transitions]), device=device
    )
    goals = torch.stack([goal_one_hot(row[1], device=device) for row in transitions])
    masks = torch.as_tensor(
        np.stack([joint_action_mask(row[0].next_masks) for row in transitions]), device=device
    )
    rewards = torch.as_tensor(
        [goal_reward(row[1], row[0].action, row[0].next_info) for row in transitions],
        dtype=torch.float32,
        device=device,
    )
    done = torch.as_tensor(
        [goal_done(row[1], row[0].action, row[0].next_info, row[0].terminated)
         for row in transitions],
        dtype=torch.bool,
        device=device,
    )
    with torch.no_grad():
        online_values = online_network(observations, goals)
        next_actions = online_values.masked_fill(masks, -torch.inf).argmax(dim=1)
        target_values = target_network(observations, goals)
        next_values = target_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + gamma * (~done).float() * next_values
    if not torch.isfinite(targets).all():
        raise RuntimeError("Goal TD targets must be finite.")
    return targets


def goal_train_step(
    network: GoalConditionedQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: Sequence[PhysicalTransition],
    batch_size: int,
    rng: np.random.Generator,
    gamma: float = 0.99,
    device: torch.device | str = "cpu",
    *,
    target_network: GoalConditionedQNetwork,
) -> float:
    """Perform one optimizer update over both counterfactual rows per sample."""

    if len(replay) < batch_size:
        raise ValueError("Replay does not contain a full batch.")
    rows = [
        row
        for index in rng.choice(len(replay), batch_size, replace=False)
        for row in relabel_transition(replay[int(index)])
    ]
    observations = torch.as_tensor(
        np.stack([row[0].observation for row in rows]), device=device
    )
    goals = torch.stack([goal_one_hot(row[1], device=device) for row in rows])
    actions = torch.as_tensor(
        [joint_action_index(row[0].action) for row in rows], device=device
    )
    values = network(observations, goals)
    selected = values.gather(1, actions.unsqueeze(1)).squeeze(1)
    targets = compute_goal_td_targets(
        network, target_network, rows, gamma=gamma, device=device
    )
    loss = F.smooth_l1_loss(selected, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    value = float(loss.detach().cpu())
    if not np.isfinite(value):
        raise RuntimeError("Optimizer produced a non-finite goal loss.")
    return value
