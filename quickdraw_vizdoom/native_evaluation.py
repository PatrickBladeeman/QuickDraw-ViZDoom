"""Matched parity and hierarchy controls on the native Doom target."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from quickdraw_vizdoom.adviser_evaluation import (
    _aggregate,
    _paired_results,
    evaluate_adviser,
)
from quickdraw_vizdoom.envs.native_basic import NativeBasicV1Env
from quickdraw_vizdoom.goal_conditioned_report import (
    BATCH_SIZE,
    EPSILON,
    EVALUATION_SEEDS,
    GAMMA,
    WARMUP,
    _hash,
)
from quickdraw_vizdoom.goal_teachers import (
    DEFAULT_LLM_MODEL,
    NATIVE_PROMPT,
    OPENROUTER_URL,
    make_teacher,
)
from quickdraw_vizdoom.learning import run_goal_conditioned_training
from quickdraw_vizdoom.learning_components import BasicGoal, GoalConditionedQNetwork


VERSION = "quickdraw.native-controls.v1"
TRAINING_SEEDS = (32001, 33001, 34001, 35001, 37001)
ARMS = {
    "no_goal": ("no_goal", None),
    "fixed_hit": ("fixed_hit", "hit"),
    "alternating_hit": ("alternating", "hit"),
    "alternating_rule": ("alternating", "rule"),
    "alternating_random": ("alternating", "random"),
    "rule_hit": ("rule", "hit"),
    "rule_rule": ("rule", "rule"),
    "llm": ("alternating", "llm"),
}


def run_experiment(
    *,
    output_path,
    training_decisions=4096,
    training_seeds=TRAINING_SEEDS,
    evaluation_episodes=100,
    diagnostic_episodes=20,
    arms=None,
    checkpoint_path=None,
    llm_url=OPENROUTER_URL,
    llm_model=DEFAULT_LLM_MODEL,
    timeout=10.0,
):
    arms = tuple(arm for arm in ARMS if arm != "llm") if arms is None else tuple(arms)
    seeds = tuple(map(int, training_seeds))
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Arms must be nonempty, unique and supported.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique.")
    if training_decisions < max(WARMUP, BATCH_SIZE):
        raise ValueError("Training decisions must reach replay warmup.")
    if (
        not 1 <= evaluation_episodes <= 100
        or not 0 <= diagnostic_episodes <= evaluation_episodes
    ):
        raise ValueError(
            "Use 1-100 evaluation episodes and a diagnostic subset within it."
        )
    path = Path(output_path)
    saved = (
        path.with_name(path.stem + "-policies.pt")
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    if (
        path.resolve() == saved.resolve()
        or path.exists()
        or (checkpoint_path is None and saved.exists())
    ):
        raise FileExistsError(
            "Use distinct, new output paths to preserve research records."
        )
    audits = {arm: [] for arm in arms}
    teachers = {
        arm: make_teacher(
            kind, url=llm_url, model=llm_model, timeout=timeout, audit=audits[arm]
        )
        for arm, (_, kind) in ARMS.items()
        if arm in arms and kind is not None
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is None:
        bundle = {
            "version": VERSION,
            "training_decisions": training_decisions,
            "sessions": [],
        }
        for seed in seeds:
            start = 100000 + seed * training_decisions
            schedule = tuple(range(start, start + training_decisions))
            session = {
                "training_seed": seed,
                "training_reset_schedule": list(schedule),
                "policies": {},
            }
            for name in ("no_goal", "fixed_hit", "alternating", "rule"):
                print(f"Training {name}, seed={seed}...", flush=True)
                started = time.perf_counter()
                with NativeBasicV1Env() as env:
                    summary, network = run_goal_conditioned_training(
                        env,
                        training_decisions,
                        WARMUP,
                        BATCH_SIZE,
                        seed,
                        "cpu",
                        epsilon=EPSILON,
                        gamma=GAMMA,
                        episode_seeds=schedule,
                        goal_conditioning=name != "no_goal",
                        fixed_goal=(
                            BasicGoal.HIT_TARGET if name == "fixed_hit" else None
                        ),
                        environment_reward=name == "fixed_hit",
                        behavior_goal_selector=(
                            make_teacher("rule") if name == "rule" else None
                        ),
                    )
                summary["training_wall_seconds"] = time.perf_counter() - started
                summary["sampled_training_rows"] = (
                    summary["optimizer_updates"]
                    * BATCH_SIZE
                    * (2 if name in ("alternating", "rule") else 1)
                )
                session["policies"][name] = {
                    "training": summary,
                    "state_dict": network.state_dict(),
                    "parameter_hash": _hash(network),
                }
            bundle["sessions"].append(session)
            torch.save(bundle, saved)
    else:
        bundle = torch.load(saved, map_location="cpu", weights_only=True)
        if bundle.get("version") != VERSION or not bundle.get("sessions"):
            raise ValueError("Checkpoint must contain native-control policies.")
    evaluation_seeds = EVALUATION_SEEDS[:evaluation_episodes]
    for session in bundle["sessions"]:
        for policy in session["policies"].values():
            if set(policy["training"]["training_episode_seeds"]) & set(
                evaluation_seeds
            ):
                raise ValueError(
                    "Training and evaluation reset seeds must be disjoint."
                )
    report = {
        "report_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Native Basic parity and hierarchy controls; no LLM-specific benefit assumed",
        "task": {
            "hit_definition": "native KILLCOUNT increase only",
            "alignment_tolerance": NativeBasicV1Env.alignment_tolerance,
            "decision_tics": 4,
        },
        "design": "Four policies per training seed. No-goal/fixed-HIT share environment rewards and one label per transition. Alternating/rule policies share goal rewards and two labels. Advisers compare frozen weights. Planned reset schedules match; actual on-policy resets may differ.",
        "parity_initialization": "Same parameter seed and architecture; constant one-hot versus zeros can change initial Q values and finite-budget optimization.",
        "checkpoint": str(saved.resolve()),
        "training_seeds": [s["training_seed"] for s in bundle["sessions"]],
        "hyperparameters": {
            "training_decisions": bundle["training_decisions"],
            "warmup": WARMUP,
            "physical_batch_size": BATCH_SIZE,
            "epsilon": EPSILON,
            "gamma": GAMMA,
            "learning_rate": 0.001,
            "replay_capacity": 10000,
            "target_sync_updates": 100,
        },
        "evaluation_seed_schedule": list(evaluation_seeds),
        "diagnostic_seed_schedule": list(evaluation_seeds[:diagnostic_episodes]),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "teacher_input": "same privileged native aligned/alignment_error/decision/ammunition fields",
        "llm_prompt": NATIVE_PROMPT if "llm" in arms else None,
        "llm_requested_model": llm_model if "llm" in arms else None,
        "llm_endpoint": llm_url if "llm" in arms else None,
        "sessions": [],
        "aggregate": {},
        "teacher_audit": audits,
    }
    for session in bundle["sessions"]:
        result = {
            "training_seed": session["training_seed"],
            "training": {},
            "conditions": {},
            "goal_diagnostics": {},
        }
        networks = {}
        for name, policy in session["policies"].items():
            network = GoalConditionedQNetwork(name != "no_goal")
            network.load_state_dict(policy["state_dict"])
            if _hash(network) != policy["parameter_hash"]:
                raise ValueError("Checkpoint parameter hash mismatch.")
            networks[name] = network
            result["training"][name] = policy["training"]
        for arm in arms:
            print(f"Evaluating {arm}, seed={session['training_seed']}...", flush=True)
            with NativeBasicV1Env() as env:
                result["conditions"][arm] = evaluate_adviser(
                    networks[ARMS[arm][0]],
                    env,
                    evaluation_seeds,
                    session["training_seed"],
                    teachers.get(arm),
                    audits[arm],
                    trace_episodes=1,
                )
        for name in ("alternating", "rule") if diagnostic_episodes else ():
            diagnostics = result["goal_diagnostics"][name] = {}
            for requested in BasicGoal:
                for swapped in (False, True):
                    supplied = BasicGoal(1 - int(requested)) if swapped else requested
                    with NativeBasicV1Env() as env:
                        evaluation = evaluate_adviser(
                            networks[name],
                            env,
                            evaluation_seeds[:diagnostic_episodes],
                            session["training_seed"],
                            lambda info, rng, supplied=supplied: supplied,
                            requested_goal=requested,
                        )
                    count = sum(
                        row["requested_goal_success"] for row in evaluation["episodes"]
                    )
                    evaluation.update(
                        successes=count, success_rate=count / diagnostic_episodes
                    )
                    diagnostics[
                        f"{requested.name}_{'swapped' if swapped else 'correct'}"
                    ] = evaluation
        report["sessions"].append(result)
        report["aggregate"] = {
            arm: _aggregate(
                [
                    row
                    for s in report["sessions"]
                    for row in s["conditions"][arm]["episodes"]
                ],
                audits[arm],
            )
            for arm in arms
        }
        for arm, aggregate in report["aggregate"].items():
            episodes = [
                row
                for s in report["sessions"]
                for row in s["conditions"][arm]["episodes"]
            ]
            handoffs = [row for row in episodes if row["align_to_hit_handoff"]]
            aggregate.update(
                native_hits=aggregate["hits"],
                handoff_episodes=len(handoffs),
                handoff_hits=sum(row["target_hit"] for row in handoffs),
                handoff_hit_rate=(
                    None
                    if not handoffs
                    else sum(row["target_hit"] for row in handoffs) / len(handoffs)
                ),
                infrastructure_invalid_episodes=sum(
                    row["infrastructure_invalid"] for row in episodes
                ),
                provider_api_calls=len(audits[arm]) if ARMS[arm][1] == "llm" else 0,
                constant_goal_selections=(
                    len(audits[arm]) if ARMS[arm][1] == "hit" else 0
                ),
            )
        report["goal_diagnostic_aggregate"] = {
            name: {
                condition: {
                    "successes": sum(
                        s["goal_diagnostics"][name][condition]["successes"]
                        for s in report["sessions"]
                    ),
                    "episodes": len(report["sessions"]) * diagnostic_episodes,
                    "success_rate": sum(
                        s["goal_diagnostics"][name][condition]["successes"]
                        for s in report["sessions"]
                    )
                    / (len(report["sessions"]) * diagnostic_episodes),
                }
                for condition in result["goal_diagnostics"][name]
            }
            for name in result["goal_diagnostics"]
        }
        report["paired_comparisons"] = _paired_results(report["sessions"], arms)
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--training-decisions", type=int, default=4096)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=TRAINING_SEEDS)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--diagnostic-episodes", type=int, default=20)
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--llm-url", default=OPENROUTER_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    arguments = vars(args)
    arguments["output_path"] = arguments.pop("output")
    arguments["checkpoint_path"] = arguments.pop("checkpoint")
    report = run_experiment(**arguments)
    for arm, result in report["aggregate"].items():
        print(
            f"{arm}: native hits={result['hits']}/{result['episodes']} ({result['hit_rate']:.1%}), decision cost={result['decision_cost_mean']:.2f}"
        )
    return int(
        any(
            r["teacher_failed_episodes"] or r["infrastructure_invalid_episodes"]
            for r in report["aggregate"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
