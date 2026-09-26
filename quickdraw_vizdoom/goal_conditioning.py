"""Registered goal-conditioned Basic baseline and held-out comparison."""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quickdraw_vizdoom.envs.basic import DEFAULT_CONFIG_PATH, BasicV1Env
from quickdraw_vizdoom.envs.micro_learning_generalization import permuted_episode_seeds
from quickdraw_vizdoom.learning import (
    BranchingQNetwork,
    EnvironmentStartupError,
    normalize_masks,
    run_training,
    select_action,
    summarize_metrics,
    transition_metric,
)


class Goal(IntEnum):
    ALIGN_RIGHT = 0
    ALIGN_LEFT = 1
    TAKE_SHOT = 2


CONDITIONS = ("no_goal", "correct_goal", "shuffled_goal")
TRAINING_SEEDS = (32001, 34001, 35001)
TRAINING_DECISIONS = 512
EVALUATION_SEED_START = 36000
EVALUATION_EPISODES = 100
WARMUP = 32
BATCH_SIZE = 32
EPSILON = 0.2
GAMMA = 0.99
DEVICE = "cpu"


def rule_goal(info: Mapping[str, Any]) -> Goal:
    """Return the Basic rule teacher's categorical goal."""

    try:
        position = operator.index(info["position_slot"])
        target = operator.index(info["target_slot"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Rule goals require integer position_slot and target_slot."
        ) from error
    if position < target:
        return Goal.ALIGN_RIGHT
    if position > target:
        return Goal.ALIGN_LEFT
    return Goal.TAKE_SHOT


def shuffled_goal(info: Mapping[str, Any]) -> Goal:
    """Return a deterministic choice between the two incorrect goals."""

    correct = rule_goal(info)
    offset = 1 + (int(info.get("seed", 0)) + int(info.get("decision", 0))) % 2
    return Goal((correct + offset) % len(Goal))


def selector_for(condition: str) -> Callable[[Mapping[str, Any]], int] | None:
    if condition == "no_goal":
        return None
    if condition == "correct_goal":
        return rule_goal
    if condition == "shuffled_goal":
        return shuffled_goal
    raise ValueError(f"Unknown comparison condition: {condition!r}.")


def evaluate(
    network: BranchingQNetwork,
    env: BasicV1Env,
    condition: str,
    episode_seeds: Sequence[int],
    device: str = DEVICE,
) -> dict[str, Any]:
    """Greedily evaluate a frozen policy on held-out episode seeds."""

    seeds = tuple(int(seed) for seed in episode_seeds)
    if not seeds:
        raise ValueError("Evaluation requires at least one episode seed.")
    selector = selector_for(condition)
    before = [parameter.detach().clone() for parameter in network.parameters()]
    rng = np.random.default_rng(seeds[0])
    transitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    network.eval()
    with torch.no_grad():
        for seed in seeds:
            try:
                observation, info = env.reset(seed=seed)
            except ValueError:
                raise
            except Exception as error:
                raise EnvironmentStartupError(str(error)) from error
            masks = normalize_masks(info["next_unavailable_masks"])
            goal = selector(info) if selector else None
            goal_age = 1
            episode_return = 0.0
            episode_decisions = episode_shots = episode_hits = 0
            while True:
                action = select_action(
                    network,
                    observation,
                    masks,
                    epsilon=0,
                    rng=rng,
                    device=device,
                    goal=goal,
                )
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_goal = selector(info) if selector else None
                metric = transition_metric(
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    goal=None if goal is None else int(goal),
                    next_goal=None if next_goal is None else int(next_goal),
                    goal_source=condition,
                    goal_age=goal_age,
                )
                transitions.append(metric)
                episode_return += float(reward)
                episode_decisions += 1
                episode_shots += int(metric["shot"])
                episode_hits += int(metric["target_hit"])
                if terminated or truncated:
                    episodes.append(
                        {
                            "return": episode_return,
                            "decisions": episode_decisions,
                            "decisions_to_hit": (
                                episode_decisions if episode_hits else None
                            ),
                            "shots_hit": episode_hits,
                            "total_shots": episode_shots,
                            "target_hit": bool(episode_hits),
                            "completed": True,
                        }
                    )
                    break
                observation = next_observation
                masks = normalize_masks(info["next_unavailable_masks"])
                goal = next_goal
                goal_age = (
                    1
                    if metric["goal_success"] or metric["goal_changed"]
                    else goal_age + 1
                )

    unchanged = all(
        torch.equal(previous, current)
        for previous, current in zip(before, network.parameters())
    )
    summary = summarize_metrics(transitions, episodes, prefix="evaluation_")
    target_hits = sum(int(episode["target_hit"]) for episode in episodes)
    summary.update(
        {
            "evaluation_episodes": len(seeds),
            "evaluation_episode_seeds": list(seeds),
            "evaluation_updates": 0,
            "evaluation_parameters_unchanged": unchanged,
            "held_out_return": summary["evaluation_mean_episode_return"],
            "held_out_target_hits": target_hits,
            "held_out_target_hit_rate": target_hits / len(seeds),
        }
    )
    return summary


def run_session(
    *,
    condition: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    seed: int,
    steps: int = TRAINING_DECISIONS,
    warmup: int = WARMUP,
    batch_size: int = BATCH_SIZE,
    epsilon: float = EPSILON,
    gamma: float = GAMMA,
    device: str = DEVICE,
    evaluation_seed_start: int = EVALUATION_SEED_START,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    """Train and evaluate one registered condition/seed pair."""

    selector = selector_for(condition)
    training_seeds = permuted_episode_seeds(seed, steps, permutation_seed=seed)
    evaluation_seeds = permuted_episode_seeds(
        evaluation_seed_start,
        evaluation_episodes,
        permutation_seed=evaluation_seed_start,
    )
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("Training and held-out seed schedules overlap.")
    try:
        env = BasicV1Env(config_path)
    except Exception as error:
        raise EnvironmentStartupError(str(error)) from error
    with env:
        summary, network = run_training(
            env,
            steps,
            warmup,
            batch_size,
            seed,
            device,
            epsilon=epsilon,
            gamma=gamma,
            episode_seeds=training_seeds,
            goal_selector=selector,
            goal_source=condition,
        )
        summary.update(evaluate(network, env, condition, evaluation_seeds, device))
    summary.update(
        {
            "trainer": "branching_q_learning",
            "condition": condition,
            "seed": seed,
            "training_seed_schedule": list(training_seeds),
            "evaluation_seed_schedule": list(evaluation_seeds),
            "process_launches": 1,
            "training_sessions": 1,
            "external_calls": 0,
        }
    )
    return summary


def aggregate_sessions(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not sessions:
        raise ValueError("At least one session is required for aggregation.")
    shots = sum(int(session["total_shots"]) for session in sessions)
    hits = sum(int(session["shots_hit"]) for session in sessions)
    goal_attempts = sum(int(session["goal_attempts"]) for session in sessions)
    goal_completions = sum(int(session["goal_completions"]) for session in sessions)
    evaluation_episodes = sum(
        int(session["evaluation_episodes"]) for session in sessions
    )
    held_out_hits = sum(int(session["held_out_target_hits"]) for session in sessions)
    decisions_to_hit = [
        int(value) for session in sessions for value in session["decisions_to_hit"]
    ]
    held_out_decisions_to_hit = [
        int(value)
        for session in sessions
        for value in session["evaluation_decisions_to_hit"]
    ]
    times_to_goal = [
        int(value) for session in sessions for value in session["time_to_goal"]
    ]
    selections: Counter[str] = Counter()
    changes: Counter[str] = Counter()
    for session in sessions:
        selections.update(session["goal_selection_counts"])
        changes.update(session["goal_transition_counts"])
    losses = [float(session["last_loss"]) for session in sessions]
    return {
        "seeds": [int(session["seed"]) for session in sessions],
        "mean_cumulative_episode_return": float(
            np.mean([session["cumulative_episode_return"] for session in sessions])
        ),
        "mean_decisions_to_hit": float(
            np.mean(decisions_to_hit) if decisions_to_hit else 0.0
        ),
        "mean_time_to_goal": float(np.mean(times_to_goal) if times_to_goal else 0.0),
        "shots_hit": hits,
        "total_shots": shots,
        "shot_percentage": float(100 * hits / shots if shots else 0.0),
        "goal_completions": goal_completions,
        "goal_attempts": goal_attempts,
        "goal_completion_rate": float(
            goal_completions / goal_attempts if goal_attempts else 0.0
        ),
        "held_out_return": float(
            np.mean([session["held_out_return"] for session in sessions])
        ),
        "held_out_target_hits": held_out_hits,
        "held_out_episodes": evaluation_episodes,
        "held_out_target_hit_rate": held_out_hits / evaluation_episodes,
        "held_out_mean_decisions_to_hit": float(
            np.mean(held_out_decisions_to_hit) if held_out_decisions_to_hit else 0.0
        ),
        "optimizer_updates": sum(int(session["updates"]) for session in sessions),
        "final_losses": losses,
        "finite_final_loss": all(math.isfinite(loss) for loss in losses),
        "evaluation_updates": sum(
            int(session["evaluation_updates"]) for session in sessions
        ),
        "goal_selection_counts": dict(selections),
        "goal_transition_counts": dict(changes),
    }


def run_comparison(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    seeds: Sequence[int] = TRAINING_SEEDS,
    conditions: Sequence[str] = CONDITIONS,
    steps: int = TRAINING_DECISIONS,
    warmup: int = WARMUP,
    batch_size: int = BATCH_SIZE,
    epsilon: float = EPSILON,
    gamma: float = GAMMA,
    device: str = DEVICE,
    evaluation_seed_start: int = EVALUATION_SEED_START,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    """Run the three-condition comparison and aggregate results by condition."""

    seed_values = tuple(int(seed) for seed in seeds)
    condition_values = tuple(conditions)
    if len(seed_values) != len(set(seed_values)) or not seed_values:
        raise ValueError("Training seeds must be non-empty and unique.")
    if len(condition_values) != len(set(condition_values)) or not condition_values:
        raise ValueError("Conditions must be non-empty and unique.")
    results: dict[str, Any] = {}
    for condition in condition_values:
        sessions = [
            run_session(
                condition=condition,
                config_path=config_path,
                seed=seed,
                steps=steps,
                warmup=warmup,
                batch_size=batch_size,
                epsilon=epsilon,
                gamma=gamma,
                device=device,
                evaluation_seed_start=evaluation_seed_start,
                evaluation_episodes=evaluation_episodes,
            )
            for seed in seed_values
        ]
        results[condition] = {
            "sessions": sessions,
            "aggregate": aggregate_sessions(sessions),
        }

    comparison: dict[str, Any] = {}
    if "correct_goal" in results and "shuffled_goal" in results:
        correct = results["correct_goal"]["aggregate"]
        shuffled = results["shuffled_goal"]["aggregate"]
        return_advantage = correct["held_out_return"] - shuffled["held_out_return"]
        efficiency_advantage = 0.0
        if correct["held_out_target_hits"] and shuffled["held_out_target_hits"]:
            efficiency_advantage = (
                shuffled["held_out_mean_decisions_to_hit"]
                - correct["held_out_mean_decisions_to_hit"]
            )
        comparison = {
            "correct_minus_shuffled_held_out_return": return_advantage,
            "correct_minus_shuffled_target_hit_rate": (
                correct["held_out_target_hit_rate"]
                - shuffled["held_out_target_hit_rate"]
            ),
            "correct_efficiency_advantage_decisions": efficiency_advantage,
            "research_gate_passed": return_advantage > 0
            or (
                correct["held_out_target_hit_rate"]
                >= shuffled["held_out_target_hit_rate"]
                and efficiency_advantage > 0
            ),
        }
    return {
        "trainer": "branching_q_learning",
        "settings": {
            "conditions": list(condition_values),
            "training_seeds": list(seed_values),
            "training_decisions": steps,
            "warmup": warmup,
            "batch_size": batch_size,
            "epsilon": epsilon,
            "gamma": gamma,
            "device": device,
            "evaluation_seed_schedule": list(
                permuted_episode_seeds(
                    evaluation_seed_start,
                    evaluation_episodes,
                    permutation_seed=evaluation_seed_start,
                )
            ),
            "evaluation_episodes": evaluation_episodes,
        },
        "conditions": results,
        "correct_vs_shuffled": comparison,
    }
