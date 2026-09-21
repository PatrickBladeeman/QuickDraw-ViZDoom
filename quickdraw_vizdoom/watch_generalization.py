"""Watch one visual-generalization training session in the Doom window."""

import argparse
import math
import time

from quickdraw_vizdoom.envs.micro_learning_generalization import (
    MicroLearningGeneralizationV1Env,
    permuted_episode_seeds,
)
from quickdraw_vizdoom.learning import run_training
from quickdraw_vizdoom.micro_learning_generalization import evaluate


class WatchEnv(MicroLearningGeneralizationV1Env):
    def __init__(self, delay: float):
        super().__init__(window_visible=True)
        self.delay = delay
        self.phase = "Training"
        self.decisions = 0
        self.episode_return = 0.0
        self.exploring = False

    def reset(self, *args, **kwargs):
        self.episode_return = 0.0
        return super().reset(*args, **kwargs)

    def record_selection(self, exploring: bool):
        self.exploring = exploring

    def step(self, branch_action):
        result = super().step(branch_action)
        _, reward, terminated, truncated, info = result
        self.decisions += 1
        self.episode_return += float(reward)
        episode_summary = ""
        if terminated or truncated:
            counters = info["event_counters"]
            shots = int(counters["shots_fired"])
            hits = int(counters["target_hits"])
            shot_rate = f"{hits / shots:.0%}" if shots else "n/a"
            outcome = "hit" if terminated else "timeout"
            episode_summary = (
                f" | episode {outcome} return {self.episode_return:+.2f}"
                f" steps {info['decision']} shots {hits}/{shots} ({shot_rate})"
            )
        print(
            f"{self.phase} {self.decisions:3d} | "
            f"{info['visual_split']}/{info['visual_variant']} | "
            f"target-slot {info['target_slot']:2d} | "
            f"{info['semantic_action'][1]:5s} | reward {reward:+.2f} | "
            f"source={'random' if self.exploring else 'greedy'}"
            f"{episode_summary}",
            flush=True,
        )
        time.sleep(self.delay)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.4)
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be finite and nonnegative")

    training_seeds = permuted_episode_seeds(32001, 512, permutation_seed=32001)
    evaluation_seeds = permuted_episode_seeds(36000, 100, permutation_seed=32001)
    print(
        "Watching generalization: 512 training decisions, then 100 held-out evaluations."
    )
    try:
        with WatchEnv(args.delay) as env:
            summary, network = run_training(
                env,
                512,
                32,
                32,
                32001,
                "cpu",
                epsilon=0.2,
                gamma=0.99,
                on_selection=env.record_selection,
                episode_seeds=training_seeds,
            )
            print(
                f"Training finished: {summary['updates']} updates; loss {summary['last_loss']:.6g}."
            )
            env.set_visual_split("held-out")
            env.phase = "Evaluation"
            env.decisions = 0
            env.exploring = False
            result = evaluate(network, env, evaluation_seeds)
            print(f"Greedy held-out successes: {result['greedy_successes']}/100.")
    except KeyboardInterrupt:
        print("Stopped watching.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
