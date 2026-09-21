"""Shuffled-seed training and frozen evaluation in the Basic arena."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from quickdraw_vizdoom.envs.micro_learning_generalization import (
    MicroLearningGeneralizationV1Env,
    permuted_episode_seeds,
    validate_seed_schedule,
)
from quickdraw_vizdoom.learning import (
    EnvironmentStartupError,
    run_training,
    select_action,
)


def _records(env, visual_split: str) -> list[dict]:
    return [
        record
        for record in getattr(env, "reset_records", {}).get(visual_split, ())
        if isinstance(record, dict)
    ]


def _set_visual_split(env, visual_split: str) -> None:
    env.set_visual_split(visual_split)


def evaluate(network, env, episode_seeds: Sequence[int]) -> dict:
    """Run greedy multi-step episodes without optimizer updates."""

    seeds = tuple(int(seed) for seed in episode_seeds)
    if not seeds:
        raise ValueError("Evaluation requires at least one episode.")
    before = [parameter.detach().clone() for parameter in network.parameters()]
    network.eval()
    rng = np.random.default_rng(seeds[0])
    successes = decisions = terminals = truncations = 0
    slots: list[int] = []
    with torch.no_grad():
        for seed in seeds:
            try:
                observation, reset_info = env.reset(seed=seed)
            except ValueError:
                raise
            except Exception as error:
                raise EnvironmentStartupError(str(error)) from error
            if reset_info.get("visual_split") != "held-out":
                raise ValueError("Evaluation used the training visual split.")
            done = False
            while not done:
                action = select_action(
                    network,
                    observation,
                    reset_info["next_unavailable_masks"],
                    epsilon=0,
                    rng=rng,
                    device="cpu",
                )
                observation, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                if not info.get("real_observation", True):
                    raise ValueError("Evaluation requires real observations.")
                if terminated or truncated:
                    terminals += int(terminated)
                    truncations += int(truncated)
                    successes += int(
                        terminated and info.get("terminal_reason") == "target_hit"
                    )
                    slots.append(int(info["target_slot"]))
                    done = True
                else:
                    reset_info = info
    unchanged = all(
        torch.equal(previous, current)
        for previous, current in zip(before, network.parameters())
    )
    return {
        "evaluation_episodes": len(seeds),
        "evaluation_seed_start": min(seeds),
        "evaluation_seed_end": max(seeds),
        "evaluation_episode_seeds": list(seeds),
        "evaluation_decisions": decisions,
        "evaluation_terminal": terminals,
        "evaluation_truncated": truncations,
        "evaluation_updates": 0,
        "parameters_unchanged": unchanged,
        "greedy_successes": successes,
        "greedy_success_rate": successes / len(seeds),
        "target_slots": sorted(set(slots)),
        "evaluation_reset_frame_hashes": list(
            record["reset_frame_hash"] for record in _records(env, "held-out")
        ),
        "evaluation_visual_variants": sorted(
            set(record["visual_variant"] for record in _records(env, "held-out"))
        ),
    }


def run_session(
    *,
    config_path,
    seed,
    steps,
    warmup,
    batch_size,
    epsilon,
    gamma,
    device,
    evaluation_seed_start,
    evaluation_episodes,
):
    training_seeds = permuted_episode_seeds(seed, steps, permutation_seed=seed)
    evaluation_seeds = permuted_episode_seeds(
        evaluation_seed_start,
        evaluation_episodes,
        permutation_seed=seed,
    )
    if steps >= 512:
        validate_seed_schedule(training_seeds, seed)
    if evaluation_episodes >= 100:
        validate_seed_schedule(evaluation_seeds, evaluation_seed_start)
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("Training and evaluation reset seeds overlap.")
    try:
        env = MicroLearningGeneralizationV1Env(config_path)
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
        )
        training_records = _records(env, "training")
        _set_visual_split(env, "held-out")
        summary.update(evaluate(network, env, evaluation_seeds))
    training_seeds_used = [record["seed"] for record in training_records]
    training_hashes = [record["reset_frame_hash"] for record in training_records]
    held_out_hashes = summary["evaluation_reset_frame_hashes"]
    training_variants = sorted(
        {record["visual_variant"] for record in training_records}
    )
    held_out_variants = summary["evaluation_visual_variants"]
    summary.update(
        {
            "training_seed_start": seed,
            "training_seed_end": seed + steps - 1,
            "training_episode_seeds": training_seeds_used,
            "training_seed_schedule": list(training_seeds),
            "training_episodes": len(training_records),
            "training_visual_variants": training_variants,
            "training_reset_frame_hashes": training_hashes,
            "held_out_visual_variants": held_out_variants,
            "held_out_reset_frame_hashes": held_out_hashes,
            "reset_seed_overlap": sorted(
                set(training_seeds_used) & set(evaluation_seeds)
            ),
            "frame_hash_overlap": sorted(set(training_hashes) & set(held_out_hashes)),
            "variant_overlap": sorted(set(training_variants) & set(held_out_variants)),
            "process_launches": 1,
            "training_sessions": 1,
            "trace_collections": 0,
            "external_calls": 0,
            "checkpoints": 0,
        }
    )
    return summary
