"""Compare no goals, random goals, rules and an LLM on frozen policies."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from quickdraw_vizdoom.envs import BasicV1Env
from quickdraw_vizdoom.goal_conditioned_report import (
    BATCH_SIZE,
    EPSILON,
    EVALUATION_SEEDS,
    GAMMA,
    TRAINING_DECISIONS,
    TRAINING_SEEDS,
    WARMUP,
    _hash,
)
from quickdraw_vizdoom.goal_teachers import (
    DEFAULT_LLM_MODEL,
    OPENROUTER_URL,
    PROMPT,
    make_teacher,
)
from quickdraw_vizdoom.learning import run_goal_conditioned_training
from quickdraw_vizdoom.learning_components import (
    BasicGoal,
    GoalConditionedQNetwork,
    goal_achieved,
    goal_one_hot,
    select_goal_action,
)


VERSION = "quickdraw.frozen-adviser.v1"
ARMS = ("no_goal", "random", "rule", "llm")
DECISION_LIMIT = 300


def evaluate_adviser(
    network,
    env,
    episode_seeds,
    training_seed,
    teacher=None,
    audit=None,
    *,
    requested_goal=None,
    trace_episodes=0,
):
    """Run the target-hit task; teacher errors count as failed episodes."""
    if network.goal_conditioning != (teacher is not None):
        raise ValueError(
            "Conditioned policies require an adviser; no-goal policies do not."
        )
    seeds = tuple(map(int, episode_seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique.")
    before = _hash(network)
    network.eval()
    device = next(network.parameters()).device
    rows = []
    outcome_goal = (
        BasicGoal.HIT_TARGET if requested_goal is None else BasicGoal(requested_goal)
    )
    with torch.no_grad():
        for episode_index, seed in enumerate(seeds):
            observation, info = env.reset(seed=seed)
            native = "native_geometry" in info
            alignment_decision, steps = None, []
            rng_seed = int(training_seed) * 100_000 + seed
            rng = np.random.default_rng(rng_seed)
            policy_rng = np.random.default_rng(seed)
            total, decisions, hit, error = 0.0, 0, False, None
            success = False
            goal, need_goal, trace = None, teacher is not None, []
            for _ in range(DECISION_LIMIT):
                if need_goal:
                    start = 0 if audit is None else len(audit)
                    timestamp = datetime.now(timezone.utc).isoformat()
                    try:
                        goal = BasicGoal(teacher(info, rng))
                        trace.append({"decision": decisions, "goal": goal.name})
                        need_goal = False
                    except ValueError as failure:
                        error = str(failure)
                    finally:
                        if audit is not None:
                            for call in audit[start:]:
                                call.update(
                                    training_seed=training_seed,
                                    episode_seed=seed,
                                    decision=decisions,
                                    timestamp_utc=timestamp,
                                )
                    if error is not None:
                        break
                action = select_goal_action(
                    network,
                    observation,
                    info["next_unavailable_masks"],
                    0.0,
                    policy_rng,
                    device,
                    goal=goal,
                )
                if episode_index < trace_episodes:
                    values = (
                        network(
                            torch.as_tensor(observation, device=device),
                            None if goal is None else goal_one_hot(goal, device=device),
                        )
                        .squeeze(0)
                        .tolist()
                    )
                    step = {
                        "decision": decisions,
                        "goal": None if goal is None else goal.name,
                        "alignment_error": info.get("alignment_error"),
                        "action": list(action),
                        "q_values": values,
                    }
                observation, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                total += float(reward)
                hit = goal_achieved(BasicGoal.HIT_TARGET, action, info)
                if native and hit != (info.get("native_kill_count", 0) > 0):
                    raise RuntimeError(
                        "Native hit event and engine kill count disagree."
                    )
                success = goal_achieved(outcome_goal, action, info)
                if alignment_decision is None and goal_achieved(
                    BasicGoal.ALIGN_WITHOUT_FIRE, action, info
                ):
                    alignment_decision = decisions
                if episode_index < trace_episodes:
                    step.update(
                        next_alignment_error=info.get("alignment_error"),
                        events=info.get("events", []),
                        native_kill_count=info.get("native_kill_count"),
                    )
                    steps.append(step)
                if success or terminated or truncated:
                    break
                need_goal = teacher is not None and goal_achieved(goal, action, info)
            rows.append(
                {
                    "seed": seed,
                    "teacher_rng_seed": (rng_seed if teacher is not None else None),
                    "target_hit": hit,
                    "decisions": decisions,
                    "decisions_to_hit": decisions if hit else None,
                    "decision_cost": decisions if hit else DECISION_LIMIT,
                    "environment_return": total,
                    "teacher_error": error,
                    "goal_trace": trace,
                }
            )
            if native or requested_goal is not None:
                row = rows[-1]
                handoff = any(
                    left["goal"] == BasicGoal.ALIGN_WITHOUT_FIRE.name
                    and right["goal"] == BasicGoal.HIT_TARGET.name
                    for left, right in zip(trace, trace[1:])
                )
                row.update(
                    requested_goal=outcome_goal.name,
                    requested_goal_success=error is None and success,
                    native_kill_count=info.get("native_kill_count", 0),
                    alignment_decision=alignment_decision,
                    align_to_hit_handoff=handoff,
                    handoff_hit=handoff and hit,
                    infrastructure_invalid="infrastructure_invalid"
                    in info.get("events", []),
                )
            if episode_index < trace_episodes:
                rows[-1]["step_trace"] = steps
    after = _hash(network)
    if before != after:
        raise RuntimeError("Evaluation changed policy parameters.")
    return {
        "episodes": rows,
        "evaluation_updates": 0,
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "evaluation_parameters_unchanged": True,
    }


def _aggregate(episodes, audit):
    hits = [row for row in episodes if row["target_hit"]]
    costs = [
        call.get("usage", {}).get("cost")
        for call in audit
        if isinstance(call.get("usage"), dict)
    ]
    costs = [cost for cost in costs if isinstance(cost, (int, float))]
    return {
        "episodes": len(episodes),
        "hits": len(hits),
        "hit_rate": len(hits) / len(episodes),
        "environment_return_mean": float(
            np.mean([r["environment_return"] for r in episodes])
        ),
        "decisions_to_hit_mean": (
            float(np.mean([r["decisions"] for r in hits])) if hits else None
        ),
        "decision_cost_mean": float(np.mean([r["decision_cost"] for r in episodes])),
        "teacher_failed_episodes": sum(
            r["teacher_error"] is not None for r in episodes
        ),
        "teacher_calls": len(audit),
        "teacher_failed_calls": sum("error" in call for call in audit),
        "teacher_latency_seconds_total": sum(call["latency_seconds"] for call in audit),
        "provider_reported_cost": sum(costs) if costs else None,
        "provider_cost_calls_reported": len(costs),
    }


def _paired_results(sessions, arms):
    comparisons = {}
    for first, second in combinations(arms, 2):
        per_seed = []
        for session in sessions:
            left = session["conditions"][first]["episodes"]
            right = session["conditions"][second]["episodes"]
            if [r["seed"] for r in left] != [r["seed"] for r in right]:
                raise RuntimeError("Unpaired evaluation schedules.")
            wins = sum(
                a["target_hit"] and not b["target_hit"] for a, b in zip(left, right)
            )
            losses = sum(
                b["target_hit"] and not a["target_hit"] for a, b in zip(left, right)
            )
            per_seed.append(
                {
                    "training_seed": session["training_seed"],
                    "first_only_hits": wins,
                    "second_only_hits": losses,
                    "hit_rate_delta_first_minus_second": (wins - losses) / len(left),
                    "decision_cost_delta_first_minus_second": float(
                        np.mean(
                            [
                                a["decision_cost"] - b["decision_cost"]
                                for a, b in zip(left, right)
                            ]
                        )
                    ),
                }
            )
        comparisons[f"{first}_vs_{second}"] = {
            "per_training_seed": per_seed,
            "hit_rate_delta_first_minus_second": float(
                np.mean([row["hit_rate_delta_first_minus_second"] for row in per_seed])
            ),
        }
    return comparisons


def run_experiment(
    *,
    output_path,
    arms=ARMS,
    evaluation_episodes=100,
    training_decisions=TRAINING_DECISIONS,
    checkpoint_path=None,
    llm_url=OPENROUTER_URL,
    llm_model=DEFAULT_LLM_MODEL,
    timeout=10.0,
):
    """Train and save paired policies, then compare advisers."""
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Arms must be nonempty, unique and supported.")
    if not 1 <= evaluation_episodes <= len(EVALUATION_SEEDS):
        raise ValueError("Evaluation episodes must be between 1 and 100.")
    if training_decisions < max(WARMUP, BATCH_SIZE):
        raise ValueError("Training decisions must reach the replay warmup.")
    path = Path(output_path)
    saved = (
        path.with_name(path.stem + "-policies.pt")
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    if path.exists() or (checkpoint_path is None and saved.exists()):
        raise FileExistsError(
            "Use a new output path to preserve earlier results/checkpoints."
        )
    if path.resolve() == saved.resolve():
        raise ValueError("Report and checkpoint paths must differ.")
    audits = {arm: [] for arm in arms}
    teachers = {
        arm: make_teacher(
            arm,
            url=llm_url,
            model=llm_model,
            timeout=timeout,
            audit=audits[arm],
        )
        for arm in arms
        if arm != "no_goal"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is None:
        bundle = {
            "version": VERSION,
            "training_decisions": training_decisions,
            "sessions": [],
        }
        for index, seed in enumerate(TRAINING_SEEDS):
            schedule = tuple(
                range(
                    100_000 + index * training_decisions,
                    100_000 + (index + 1) * training_decisions,
                )
            )
            session = {"training_seed": seed, "policies": {}}
            for name, conditioned in (
                ("no_goal", False),
                ("goal_conditioned", True),
            ):
                print(f"Training {name}, seed={seed}...", flush=True)
                env = BasicV1Env()
                try:
                    training, network = run_goal_conditioned_training(
                        env,
                        training_decisions,
                        WARMUP,
                        BATCH_SIZE,
                        seed,
                        "cpu",
                        epsilon=EPSILON,
                        gamma=GAMMA,
                        episode_seeds=schedule,
                        goal_conditioning=conditioned,
                    )
                finally:
                    env.close()
                session["policies"][name] = {
                    "training": training,
                    "state_dict": network.state_dict(),
                    "parameter_hash": _hash(network),
                }
            bundle["sessions"].append(session)
        torch.save(bundle, saved)
    else:
        bundle = torch.load(saved, map_location="cpu", weights_only=True)
        if bundle["version"] != VERSION or len(bundle["sessions"]) < 3:
            raise ValueError("Checkpoint must contain three paired training seeds.")
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
        "scope": "Basic pilot of adviser effectiveness during execution",
        "design": "Three paired training seeds. No-goal DDQN uses environment "
        "reward; the goal policy uses balanced goals, goal rewards and "
        "exhaustive relabeling. All advisers share its frozen weights. "
        "This isolates adviser choice during execution.",
        "no_goal_comparison": "Whole systems differ in reward/relabeling and "
        "symbolic adviser inputs. Not an ablation of goal input alone.",
        "teacher_input": "same privileged slot/decision/ammunition fields",
        "primary_metric": "target-hit rate; adviser errors count as failures",
        "decision_cost_definition": "decisions to hit; 300 for each failure",
        "checkpoint": str(saved.resolve()),
        "hyperparameters": {
            "training_decisions": bundle["training_decisions"],
            "warmup": WARMUP,
            "batch_size": BATCH_SIZE,
            "epsilon": EPSILON,
            "gamma": GAMMA,
            "episode_decision_limit": DECISION_LIMIT,
        },
        "evaluation_seed_schedule": list(evaluation_seeds),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "llm_prompt": PROMPT if "llm" in arms else None,
        "llm_endpoint": llm_url if "llm" in arms else None,
        "llm_requested_model": llm_model if "llm" in arms else None,
        "sessions": [],
        "aggregate": {},
        "teacher_audit": audits,
    }
    for session in bundle["sessions"]:
        result = {
            "training_seed": session["training_seed"],
            "training": {
                name: policy["training"] for name, policy in session["policies"].items()
            },
            "conditions": {},
        }
        networks = {}
        for name, policy in session["policies"].items():
            network = GoalConditionedQNetwork(
                goal_conditioning=name == "goal_conditioned"
            )
            network.load_state_dict(policy["state_dict"])
            if _hash(network) != policy["parameter_hash"]:
                raise ValueError("Checkpoint parameter hash mismatch.")
            networks[name] = network
        for arm in arms:
            print(
                f"Evaluating {arm}, seed={session['training_seed']}...",
                flush=True,
            )
            env = BasicV1Env()
            try:
                result["conditions"][arm] = evaluate_adviser(
                    networks["no_goal" if arm == "no_goal" else "goal_conditioned"],
                    env,
                    evaluation_seeds,
                    session["training_seed"],
                    teachers.get(arm),
                    audits[arm],
                )
            finally:
                env.close()
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
        report["paired_comparisons"] = _paired_results(report["sessions"], arms)
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--training-decisions", type=int, default=TRAINING_DECISIONS)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoint", help="Reuse saved paired policies without retraining."
    )
    parser.add_argument("--llm-url", default=OPENROUTER_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = run_experiment(
        output_path=args.output,
        arms=tuple(args.arms),
        training_decisions=args.training_decisions,
        evaluation_episodes=args.evaluation_episodes,
        checkpoint_path=args.checkpoint,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        timeout=args.timeout,
    )
    for arm, metrics in result["aggregate"].items():
        print(
            f"{arm}: hits={metrics['hits']}/{metrics['episodes']} "
            f"({metrics['hit_rate']:.1%}), "
            f"decision cost={metrics['decision_cost_mean']:.2f}, "
            f"teacher failures={metrics['teacher_failed_episodes']}"
        )
    return int(any(m["teacher_failed_episodes"] for m in result["aggregate"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
