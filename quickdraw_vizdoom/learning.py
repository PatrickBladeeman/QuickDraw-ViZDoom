"""Minimal branch-action DQN smoke loop for the Basic v1 environment."""

from __future__ import annotations

import argparse
import operator
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from itertools import cycle
from typing import Any

import numpy as np
import torch

from quickdraw_vizdoom.envs import BasicV1Env
from quickdraw_vizdoom.learning_components import (
    BasicGoal,
    BranchAction,
    BranchingQNetwork,
    ConditionedTransition,
    EnvironmentStartupError,
    GoalConditionedQNetwork,
    GoalPotential,
    GoalRow,
    GoalSelector,
    GOAL_COUNT,
    GOAL_TARGET_SYNC_INTERVAL,
    JOINT_ACTIONS,
    Masks,
    PhysicalTransition,
    Transition,
    UnconditionedTransition,
    _image_encoder,
    _prepare_observations,
    _transition_goals,
    compute_goal_td_targets,
    compute_td_targets,
    goal_achieved,
    goal_done,
    goal_one_hot,
    goal_reward,
    goal_train_step,
    joint_action_index,
    joint_action_mask,
    normalize_masks,
    relabel_transition,
    select_action,
    select_goal_action,
    train_step,
)


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


def potential_shaping_reward(
    potential: GoalPotential,
    info: Mapping[str, Any],
    goal: int | None,
    next_info: Mapping[str, Any],
    next_goal: int | None,
    *,
    terminated: bool,
    gamma: float,
) -> float:
    """Return the fixed training-only potential shaping reward."""

    current = float(potential(info, goal))
    following = 0.0 if terminated else float(potential(next_info, next_goal))
    reward = 0.1 * (gamma * following - current)
    if not np.isfinite(reward):
        raise ValueError("Goal-shaping rewards must be finite.")
    return reward


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
    goal_shaping_reward: float = 0.0,
    training_reward: float | None = None,
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
    environment_reward = float(reward)
    shaping_reward = float(goal_shaping_reward)
    learned_reward = (
        environment_reward + shaping_reward
        if training_reward is None
        else float(training_reward)
    )
    if not all(
        np.isfinite(value)
        for value in (environment_reward, shaping_reward, learned_reward)
    ):
        raise ValueError("Reward components must be finite.")
    return {
        "episode": info.get("episode_index"),
        "decision": info.get("decision"),
        "action": list(action),
        "reward": environment_reward,
        "environment_reward": environment_reward,
        "goal_shaping_reward": shaping_reward,
        "training_reward": learned_reward,
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
        "native_kill_delta": int(info.get("native_kill_delta", 0)),
        "native_episode_finished": bool(info.get("engine_episode_finished", False)),
        "infrastructure_invalid": bool(info.get("infrastructure_invalid", False)),
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

    environment_reward = sum(
        float(item.get("environment_reward", item["reward"])) for item in transitions
    )
    shaping_reward = sum(
        float(item.get("goal_shaping_reward", 0.0)) for item in transitions
    )
    training_reward = sum(
        float(item.get("training_reward", item["reward"])) for item in transitions
    )

    return {
        key("transition_metrics"): list(transitions),
        key("episode_metrics"): list(episodes),
        key("episode_returns"): [float(item["return"]) for item in episodes],
        key("cumulative_episode_return"): float(environment_reward),
        key("cumulative_environment_reward"): float(environment_reward),
        key("cumulative_goal_shaping_reward"): float(shaping_reward),
        key("cumulative_training_reward"): float(training_reward),
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
    goal_potential: GoalPotential | None = None,
) -> tuple[dict[str, Any], BranchingQNetwork]:
    """Train in the caller-owned environment and return the in-memory policy."""

    threshold = max(warmup, batch_size)
    if steps < threshold or batch_size < 1 or warmup < 0:
        raise ValueError("steps must reach warmup and batch-size must be positive.")
    if episode_seeds is not None and len(episode_seeds) != steps:
        raise ValueError("episode_seeds must contain one seed per decision.")
    if goal_potential is not None and goal_selector is None:
        raise ValueError("Goal potential requires a categorical goal selector.")

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
        current_info = info
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        next_masks = normalize_masks(next_info["next_unavailable_masks"])
        next_goal = _select_goal(goal_selector, next_info)
        shaping_reward = (
            0.0
            if goal_potential is None
            else potential_shaping_reward(
                goal_potential,
                current_info,
                goal,
                next_info,
                next_goal,
                terminated=terminated,
                gamma=gamma,
            )
        )
        training_reward = float(reward) + shaping_reward
        transition: Transition = (
            observation.copy(),
            action,
            training_reward,
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
            info=next_info,
            goal=goal,
            next_goal=next_goal,
            goal_source=goal_source,
            goal_age=goal_age,
            goal_shaping_reward=shaping_reward,
            training_reward=training_reward,
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
            observation, masks, goal, info = (
                next_observation,
                next_masks,
                next_goal,
                next_info,
            )
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


def _goal_replay_counts(
    payloads: Sequence[PhysicalTransition], *, goal_conditioning: bool = True
) -> dict[str, dict[str, int]]:
    behavior, relabeled, actions, rewards, successes = (Counter() for _ in range(5))
    for payload in payloads:
        if payload.behavior_goal is not None:
            behavior[payload.behavior_goal.name] += 1
        if goal_conditioning:
            for goal in BasicGoal:
                relabeled[goal.name] += 1
                actions[str(payload.action)] += 1
                rewards[str(goal_reward(goal, payload.action, payload.next_info))] += 1
                successes[
                    str(goal_achieved(goal, payload.action, payload.next_info)).lower()
                ] += 1
        else:
            relabeled["none"] += 1
            actions[str(payload.action)] += 1
            rewards[str(payload.environment_reward)] += 1
            successes[str(payload.terminated).lower()] += 1
    return {
        "behavior_goal": dict(behavior),
        "relabeled_goal": dict(relabeled),
        "joint_action": dict(actions),
        "goal_reward": dict(rewards),
        "goal_success": dict(successes),
    }


def run_goal_conditioned_training(
    env: BasicV1Env,
    steps: int,
    warmup: int,
    batch_size: int,
    seed: int,
    device: str,
    epsilon: float = 0.2,
    gamma: float = 0.99,
    *,
    episode_seeds: Sequence[int] | None = None,
    behavior_goal_selector: (
        Callable[[Mapping[str, Any], np.random.Generator], BasicGoal] | None
    ) = None,
    goal_conditioning: bool = True,
) -> tuple[dict[str, Any], GoalConditionedQNetwork]:
    """Train Basic with a matched joint-action Double-DQN path.

    Target synchronization is deliberately fixed at
    ``GOAL_TARGET_SYNC_INTERVAL`` optimizer updates for every run.
    Truncations keep the repository's existing bootstrap behavior.
    """

    threshold = max(warmup, batch_size)
    if steps < threshold or batch_size < 1 or warmup < 0:
        raise ValueError("steps must reach warmup and batch-size must be positive.")
    if episode_seeds is not None and len(episode_seeds) != steps:
        raise ValueError("episode_seeds must contain one seed per decision.")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between zero and one.")
    if not goal_conditioning and behavior_goal_selector is not None:
        raise ValueError("An unconditioned run cannot use a behavior goal selector.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device)
    network = GoalConditionedQNetwork(goal_conditioning=goal_conditioning).to(
        torch_device
    )
    target_network = GoalConditionedQNetwork(goal_conditioning=goal_conditioning).to(
        torch_device
    )
    target_network.load_state_dict(network.state_dict())
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    replay: deque[PhysicalTransition] = deque(maxlen=10_000)
    goals = cycle(BasicGoal) if goal_conditioning else None
    goal_rng = np.random.default_rng(seed + 1)
    goal_selections: list[str] = []

    def select_behavior_goal(info: Mapping[str, Any]) -> BasicGoal | None:
        if not goal_conditioning:
            return None
        selected = (
            next(goals)
            if behavior_goal_selector is None
            else behavior_goal_selector(info, goal_rng)
        )
        try:
            goal = BasicGoal(operator.index(selected))
        except (TypeError, ValueError) as error:
            raise ValueError("Behavior teacher must return a BasicGoal ID.") from error
        goal_selections.append(goal.name)
        return goal

    reset_seeds = tuple(episode_seeds) if episode_seeds is not None else None
    reset_index = 0
    reset_seeds_used: list[int] = []

    def reset() -> tuple[np.ndarray, Mapping[str, Any]]:
        nonlocal reset_index
        selected_seed = (
            seed + reset_index if reset_seeds is None else reset_seeds[reset_index]
        )
        try:
            value = env.reset(seed=selected_seed)
        except ValueError:
            raise
        except Exception as error:
            raise EnvironmentStartupError(
                f"Environment startup failed: {error}"
            ) from error
        reset_index += 1
        observation, info = value
        reset_seeds_used.append(int(info.get("seed", selected_seed)))
        return observation, info

    observation, info = reset()
    masks = info["next_unavailable_masks"]
    behavior_goal = select_behavior_goal(info)
    updates = target_syncs = terminals = truncations = 0
    last_loss: float | None = None
    environment_return = 0.0

    for decision in range(steps):
        action = select_goal_action(
            network,
            observation,
            masks,
            epsilon,
            rng,
            torch_device,
            goal=behavior_goal,
        )
        current_info = info
        next_observation, environment_reward, terminated, truncated, next_info = (
            env.step(action)
        )
        next_masks = next_info["next_unavailable_masks"]
        payload = PhysicalTransition(
            observation,
            action,
            environment_reward,
            next_observation,
            terminated,
            truncated,
            next_masks,
            current_info,
            next_info,
            behavior_goal,
        )
        replay.append(payload)
        environment_return += float(environment_reward)

        if len(replay) >= threshold:
            last_loss = goal_train_step(
                network,
                optimizer,
                replay,
                batch_size,
                rng,
                gamma=gamma,
                device=torch_device,
                target_network=target_network,
            )
            updates += 1
            if updates % GOAL_TARGET_SYNC_INTERVAL == 0:
                target_network.load_state_dict(network.state_dict())
                target_syncs += 1

        if terminated or truncated:
            terminals += int(terminated)
            truncations += int(truncated)
            if decision == steps - 1:
                break
            observation, info = reset()
            masks = info["next_unavailable_masks"]
            behavior_goal = select_behavior_goal(info)
        else:
            observation, info, masks = next_observation, next_info, next_masks
            if (
                goal_conditioning
                and decision < steps - 1
                and goal_achieved(behavior_goal, action, next_info)
            ):
                behavior_goal = select_behavior_goal(info)

    if last_loss is None:
        raise RuntimeError(
            "Goal-conditioned run completed without an optimizer update."
        )
    payloads = list(replay)
    counts = _goal_replay_counts(payloads, goal_conditioning=goal_conditioning)
    summary = {
        "trainer": (
            "goal_conditioned_joint_double_dqn"
            if goal_conditioning
            else "unconditioned_joint_double_dqn"
        ),
        "goal_conditioned": goal_conditioning,
        "goal_names": [goal.name for goal in BasicGoal] if goal_conditioning else [],
        "decisions": steps,
        "optimizer_updates": updates,
        "target_network_sync_interval": GOAL_TARGET_SYNC_INTERVAL,
        "target_network_sync_count": target_syncs,
        "terminal": terminals,
        "truncated": truncations,
        "last_loss": last_loss,
        "training_episode_seeds": reset_seeds_used,
        "physical_replay_count": len(payloads),
        "relabeled_row_count": (
            len(BasicGoal) * len(payloads) if goal_conditioning else len(payloads)
        ),
        "replay_counts": counts,
        "cumulative_environment_reward": environment_return,
    }
    if goal_conditioning and behavior_goal_selector is not None:
        summary["behavior_goal_seed"] = seed + 1
        summary["behavior_goal_selection_counts"] = dict(Counter(goal_selections))
    return summary, network


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
