"""Fixed-budget training and frozen evaluation using the existing learning path."""

import numpy as np
import torch

from quickdraw_vizdoom.envs.micro_learning import MicroLearningV1Env
from quickdraw_vizdoom.learning import (
    EnvironmentStartupError,
    run_training,
    select_action,
)


def evaluate(network, env, seed_start, episodes):
    """Greedy evaluation has no optimizer and verifies unchanged parameters."""
    before = [parameter.detach().clone() for parameter in network.parameters()]
    network.eval()
    successes = 0
    rng = np.random.default_rng(seed_start)
    with torch.no_grad():
        for seed in range(seed_start, seed_start + episodes):
            try:
                observation, info = env.reset(seed=seed)
            except ValueError:
                raise
            except Exception as error:
                raise EnvironmentStartupError(str(error)) from error
            action = select_action(
                network,
                observation,
                info["next_unavailable_masks"],
                epsilon=0,
                rng=rng,
                device="cpu",
            )
            _, reward, terminated, truncated, info = env.step(action)
            if not terminated or truncated or not info["real_observation"]:
                raise ValueError("Evaluation requires a real one-decision terminal.")
            successes += int(reward == 1.0)
    unchanged = all(
        torch.equal(previous, current)
        for previous, current in zip(before, network.parameters())
    )
    return {
        "evaluation_episodes": episodes,
        "evaluation_seed_start": seed_start,
        "evaluation_seed_end": seed_start + episodes - 1,
        "evaluation_updates": 0,
        "parameters_unchanged": unchanged,
        "greedy_successes": successes,
        "greedy_success_rate": successes / episodes,
        "random_baseline_rate": 0.5,
        "improvement_percentage_points": (successes / episodes - 0.5) * 100,
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
    training_seeds = range(seed, seed + steps)
    evaluation_seeds = range(
        evaluation_seed_start, evaluation_seed_start + evaluation_episodes
    )
    if set(training_seeds).intersection(evaluation_seeds):
        raise ValueError("Training and held-out seeds overlap.")
    try:
        env = MicroLearningV1Env(config_path)
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
            epsilon,
            gamma,
        )
        summary.update(
            evaluate(network, env, evaluation_seed_start, evaluation_episodes)
        )
    summary.update(
        {
            "training_seed_start": seed,
            "training_seed_end": seed + steps - 1,
            "process_launches": 1,
            "training_sessions": 1,
            "trace_collections": 0,
            "external_calls": 0,
        }
    )
    return summary
