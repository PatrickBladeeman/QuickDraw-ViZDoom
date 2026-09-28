"""Versioned Basic goal-conditioned training report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quickdraw_vizdoom.envs import BasicV1Env
from quickdraw_vizdoom.envs.basic import DEFAULT_CONFIG_PATH
from quickdraw_vizdoom.learning import (
    BasicGoal,
    GoalConditionedQNetwork,
    JOINT_ACTIONS,
    goal_achieved,
    joint_action_mask,
    run_goal_conditioned_training,
)


REPORT_VERSION = "quickdraw.goal-conditioned-basic.v1"
TRAINING_SEEDS = (32001, 34001, 35001)
EVALUATION_SEEDS = tuple(range(36000, 36100))
TRAINING_DECISIONS, WARMUP, BATCH_SIZE = 512, 32, 32
EPSILON, GAMMA = 0.2, 0.99


def _hash(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in network.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _actions(network, observation, masks):
    device = next(network.parameters()).device
    values = network(
        torch.as_tensor(np.stack((observation, observation)), device=device),
        torch.eye(len(BasicGoal), device=device),
    ).masked_fill(torch.as_tensor(joint_action_mask(masks), device=device), -torch.inf)
    return tuple(JOINT_ACTIONS[i] for i in values.argmax(1).tolist())


def evaluate_goal_conditioned(
    network: GoalConditionedQNetwork,
    env: BasicV1Env,
    episode_seeds: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    """Measure correct and swapped goals with frozen parameters."""
    seeds = tuple(map(int, episode_seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique.")
    before = _hash(network)
    episodes = {g.name: [] for g in BasicGoal}
    swapped = {g.name: [] for g in BasicGoal}
    actions = {g.name: Counter() for g in BasicGoal}
    paired = []
    network.eval()

    with torch.no_grad():
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            target, position = int(info["target_slot"]), int(info["position_slot"])
            movement = 2 if target >= position else 1
            samples = []
            if target == position:
                samples.append(("aligned", observation, info))
                movement = 1 if position > -4 else 2
                observation, _, _, _, info = env.step((movement, 0))
                samples.append(("one_slot_away", observation, info))
            else:
                while abs(target - position) > 1:
                    observation, _, _, _, info = env.step((movement, 0))
                    position = int(info["position_slot"])
                samples.append(("one_slot_away", observation, info))
                observation, _, _, _, info = env.step((movement, 0))
                samples.append(("aligned", observation, info))
            for state, observation, info in samples:
                delta = int(info["target_slot"]) - int(info["position_slot"])
                move = 0 if delta == 0 else 2 if delta > 0 else 1
                selected = _actions(network, observation, info["next_unavailable_masks"])
                paired.append({
                    "seed": seed,
                    "state": state,
                    "selected": {g.name: list(selected[int(g)]) for g in BasicGoal},
                    "expected": {
                        BasicGoal.ALIGN_WITHOUT_FIRE.name: [move, 0],
                        BasicGoal.HIT_TARGET.name: [move, 1],
                    },
                })

            for goal in BasicGoal:
                for supplied, rows in ((goal, episodes), (BasicGoal(1 - int(goal)), swapped)):
                    observation, info = env.reset(seed=seed)
                    masks, total, success = info["next_unavailable_masks"], 0.0, False
                    for _ in range(300):
                        selected = _actions(network, observation, masks)
                        action = selected[int(supplied)]
                        if supplied == goal:
                            actions[goal.name][str(action)] += 1
                        observation, reward, terminated, truncated, next_info = env.step(action)
                        total += float(reward)
                        success = goal_achieved(goal, action, next_info)
                        if success or terminated or truncated:
                            break
                        info, masks = next_info, next_info["next_unavailable_masks"]
                    rows[goal.name].append({
                        "seed": seed, "environment_return": total, "goal_success": success
                    })

    after = _hash(network)
    per_goal = {
        g.name: {
            "per_seed": episodes[g.name],
            "swapped_per_seed": swapped[g.name],
            "environment_return_mean": float(np.mean([x["environment_return"] for x in episodes[g.name]])),
            "requested_goal_success_rate": float(np.mean([x["goal_success"] for x in episodes[g.name]])),
            "swapped_goal_success_rate": float(np.mean([x["goal_success"] for x in swapped[g.name]])),
            "selected_joint_action_counts": dict(actions[g.name]),
        }
        for g in BasicGoal
    }
    return {
        "evaluation_seed_schedule": list(seeds),
        "evaluation_updates": 0,
        "evaluation_target_network_synchronizations": 0,
        "parameter_hash_before_evaluation": before,
        "parameter_hash_after_evaluation": after,
        "evaluation_parameters_unchanged": before == after,
        "per_goal": per_goal,
        "paired_counterfactual_selections": paired,
        "correct_goal_success_rate": float(np.mean([
            x["requested_goal_success_rate"] for x in per_goal.values()
        ])),
        "swapped_goal_success_rate": float(np.mean([
            x["swapped_goal_success_rate"] for x in per_goal.values()
        ])),
    }


def run_goal_conditioned_report(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
    evaluation_seeds: tuple[int, ...] = EVALUATION_SEEDS,
    output_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    training_seeds, evaluation_seeds = tuple(map(int, training_seeds)), tuple(map(int, evaluation_seeds))
    schedules = {seed: tuple(range(seed, seed + TRAINING_DECISIONS)) for seed in training_seeds}
    used = [seed for schedule in schedules.values() for seed in schedule]
    if len(training_seeds) < 3 or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("At least three unique training seeds are required.")
    if not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique.")
    if len(set(used)) != len(used) or set(used) & set(evaluation_seeds):
        raise ValueError("Training reset seeds and evaluation seeds must be disjoint.")

    sessions = []
    for seed in training_seeds:
        env = BasicV1Env(config_path)
        try:
            result, network = run_goal_conditioned_training(
                env, TRAINING_DECISIONS, WARMUP, BATCH_SIZE, seed, device,
                epsilon=EPSILON, gamma=GAMMA, episode_seeds=schedules[seed],
            )
            evaluation = evaluate_goal_conditioned(network, env, evaluation_seeds)
        finally:
            env.close()
        sessions.append({
            "training_seed": seed,
            "training_seed_schedule": list(schedules[seed]),
            "training": {
                "reset_seeds": result["training_episode_seeds"],
                "environment_return": result["cumulative_environment_reward"],
                "optimizer_updates": result["optimizer_updates"],
                "target_network_sync_interval": result["target_network_sync_interval"],
                "target_network_sync_count": result["target_network_sync_count"],
                "physical_replay_count": result["physical_replay_count"],
                "relabeled_row_count": result["relabeled_row_count"],
                "replay_counts": result["replay_counts"],
            },
            "evaluation": evaluation,
        })

    aggregate_goals = {}
    for goal in BasicGoal:
        runs = [s["evaluation"]["per_goal"][goal.name] for s in sessions]
        counts = Counter()
        for run in runs:
            counts.update(run["selected_joint_action_counts"])
        aggregate_goals[goal.name] = {
            "environment_return_mean": float(np.mean([
                x["environment_return_mean"] for x in runs
            ])),
            "requested_goal_success_rate": float(np.mean([
                x["requested_goal_success_rate"] for x in runs
            ])),
            "swapped_goal_success_rate": float(np.mean([
                x["swapped_goal_success_rate"] for x in runs
            ])),
            "selected_joint_action_counts": dict(counts),
        }
    aggregate = {
        "training_environment_returns": [
            {"seed": s["training_seed"], "return": s["training"]["environment_return"]}
            for s in sessions
        ],
        "training_environment_return_mean": float(np.mean([
            s["training"]["environment_return"] for s in sessions
        ])),
        "optimizer_updates": sum(s["training"]["optimizer_updates"] for s in sessions),
        "target_network_synchronizations": sum(
            s["training"]["target_network_sync_count"] for s in sessions
        ),
        "physical_replay_count": sum(s["training"]["physical_replay_count"] for s in sessions),
        "relabeled_row_count": sum(s["training"]["relabeled_row_count"] for s in sessions),
        "per_goal": aggregate_goals,
        "correct_goal_success_rate": float(np.mean([
            x["requested_goal_success_rate"] for x in aggregate_goals.values()
        ])),
        "swapped_goal_success_rate": float(np.mean([
            x["swapped_goal_success_rate"] for x in aggregate_goals.values()
        ])),
    }
    report = {
        "report_version": REPORT_VERSION,
        "scope": "two-goal Basic task only",
        "hyperparameters": {
            "training_decisions": TRAINING_DECISIONS, "warmup": WARMUP,
            "batch_size": BATCH_SIZE, "epsilon": EPSILON, "gamma": GAMMA,
            "target_sync_interval": sessions[0]["training"]["target_network_sync_interval"],
        },
        "training_seeds": list(training_seeds),
        "evaluation_seed_schedule": list(evaluation_seeds),
        "aggregate": aggregate,
        "sessions": sessions,
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
