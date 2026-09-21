"""Watch the agent train in the actual Doom window (a demonstration, not QA)."""

import argparse
import math
import time
from collections import deque

from quickdraw_vizdoom.envs.micro_learning import MicroLearningV1Env
from quickdraw_vizdoom.learning import run_training
from quickdraw_vizdoom.micro_learning import evaluate


class WatchEnv(MicroLearningV1Env):
    def __init__(self, delay: float):
        super().__init__(window_visible=True)
        self.delay = delay
        self.phase = "Training"
        self.decisions = 0
        self.recent = deque(maxlen=32)
        self.exploring = False

    def record_selection(self, exploring: bool):
        self.exploring = exploring

    def step(self, branch_action):
        result = super().step(branch_action)
        _, reward, _, _, info = result
        self.decisions += 1
        self.recent.append(info["correct"])
        print(
            f"{self.phase} {self.decisions:3d} | "
            f"target {'present' if info['target_present'] else 'absent ':7s} | "
            f"{info['semantic_action'][1]:5s} | reward {reward:+.0f} | "
            f"source={'random' if self.exploring else 'greedy'} | "
            f"last {len(self.recent)} accuracy {sum(self.recent) / len(self.recent):.0%}",
            flush=True,
        )
        time.sleep(self.delay)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delay", type=float, default=0.4, help="Seconds per decision (default: 0.4)."
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be finite and nonnegative")
    print(
        "Watching micro-learning: 512 training decisions, then 100 greedy evaluations."
    )
    print("The agent controls Doom. Training explores 20% of the time. Ctrl+C stops.")
    try:
        with WatchEnv(args.delay) as env:
            summary, network = run_training(
                env,
                steps=512,
                warmup=32,
                batch_size=32,
                seed=32001,
                device="cpu",
                epsilon=0.2,
                gamma=0.99,
                on_selection=env.record_selection,
            )
            print(
                f"Training finished: {summary['updates']} updates; loss {summary['last_loss']:.6g}."
            )
            env.phase = "Evaluation"
            env.decisions = 0
            env.recent.clear()
            env.exploring = False  # evaluate() is greedy by contract (epsilon=0).
            result = evaluate(network, env, seed_start=36000, episodes=100)
            print(f"Greedy held-out successes: {result['greedy_successes']}/100.")
    except KeyboardInterrupt:
        print("Stopped watching.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
