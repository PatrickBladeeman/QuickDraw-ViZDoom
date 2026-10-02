"""Frozen instruction-to-goal generalization, scored with native Doom events."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from quickdraw_vizdoom.envs.native_basic import NativeBasicV1Env
from quickdraw_vizdoom.goal_conditioned_report import EVALUATION_SEEDS, _hash
from quickdraw_vizdoom.goal_teachers import OPENROUTER_URL, request_json, validate_llm
from quickdraw_vizdoom.learning_components import (
    BasicGoal,
    GoalConditionedQNetwork,
    goal_achieved,
    select_goal_action,
)
from quickdraw_vizdoom.native_evaluation import VERSION as CHECKPOINT_VERSION


VERSION = "quickdraw.instruction-adviser.v1"
ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "artifacts/quickdraw/instruction-development-v1.jsonl"
PLANS = {
    "preserve": (BasicGoal.ALIGN_WITHOUT_FIRE,),
    "hit": (BasicGoal.HIT_TARGET,),
    "align_then_hit": (BasicGoal.ALIGN_WITHOUT_FIRE, BasicGoal.HIT_TARGET),
}
PROMPT = (
    "Translate the user's instruction into a plan for a Doom controller. "
    "ALIGN_WITHOUT_FIRE moves or holds until aligned without firing, then ends. "
    "HIT_TARGET moves and shoots until the target is killed; it may shoot before "
    "alignment. Goals execute in order and the controller stops when the plan ends. "
    "The only plans are [ALIGN_WITHOUT_FIRE], [HIT_TARGET], or "
    "[ALIGN_WITHOUT_FIRE, HIT_TARGET]. Preserve the target if instructed; add "
    "alignment before HIT_TARGET when firing must wait for alignment. "
    "Return only JSON with exactly one key, plan, containing the goal names as "
    'strings, for example {"plan":["HIT_TARGET"]}.'
)
ALIGN_WORDS = (
    r"\b(?:align\w*|aim\w*|center\w*|centre\w*|lin(?:e|ed|ing) up|position|in front)\b"
)
HIT_WORDS = r"\b(?:hit|kill\w*|shoot\w*|fir(?:e|ing)|attack\w*|eliminat\w*|neutraliz\w*|destroy\w*|take (?:it|the target) (?:down|out)|open fire)\b"


def rule_plan(instruction):
    """Development-frozen lexical grammar, including negation and ordering."""
    # ponytail: bounded lexical scope; extend on future development sets, never the test set.
    text = instruction.casefold().replace("\u2019", "'").replace("don't", "do not")
    align, hit = re.search(ALIGN_WORDS, text), re.search(HIT_WORDS, text)
    ordered = (
        align
        and hit
        and (
            re.search(
                ALIGN_WORDS + r".*?\b(?:then|before|followed by)\b.*?" + HIT_WORDS, text
            )
            or re.search(
                HIT_WORDS + r".*?\b(?:after|until|once)\b.*?" + ALIGN_WORDS, text
            )
            or re.search(r"\bfirst\b.*?" + ALIGN_WORDS + r".*?" + HIT_WORDS, text)
        )
    )
    if ordered:
        return PLANS["align_then_hit"]
    protected = re.search(
        r"\b(?:spare|(?:keep|leave).{0,50}alive|(?:do not|never|without|no)\s+(?:ever\s+)?(?:shoot\w*|fir(?:e|ing)|kill\w*|attack\w*)|weapon unused)\b",
        text,
    )
    if protected or (align and not hit):
        return PLANS["preserve"]
    if hit:
        return PLANS["hit"]
    raise ValueError("Instruction is outside the frozen rule grammar.")


def validate_plan(content):
    if not isinstance(content, dict) or set(content) != {"plan"}:
        raise ValueError("Expected exactly one JSON plan field.")
    names = content["plan"]
    if not isinstance(names, list) or not 1 <= len(names) <= 2:
        raise ValueError("Plan must contain one or two goals.")
    if not all(
        isinstance(name, str) and name in BasicGoal.__members__ for name in names
    ):
        raise ValueError("Plan contains an unknown goal.")
    plan = tuple(BasicGoal[name] for name in names)
    if plan not in PLANS.values():
        raise ValueError("Plan is outside the three supported programs.")
    return plan


def read_cases(path):
    cases = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("Instruction cases must be nonempty.")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"id", "family", "instruction", "task"}
            or not all(
                isinstance(value, str) and value.strip() for value in case.values()
            )
            or len(case["instruction"]) > 1000
            or case["task"] not in PLANS
        ):
            raise ValueError("Invalid instruction case schema.")
    for field in ("id", "instruction"):
        if len({case[field].casefold() for case in cases}) != len(cases):
            raise ValueError(f"Duplicate instruction {field}.")
    if Path(path).resolve() != DEVELOPMENT.resolve():
        development = read_cases(DEVELOPMENT)
        families = {case["family"] for case in cases}
        if len(families) < 10 or any(
            {case["task"] for case in cases if case["family"] == family} != set(PLANS)
            for family in families
        ):
            raise ValueError(
                "Held-out evaluation requires ten families covering all tasks."
            )
        for field in ("family", "instruction"):
            if {case[field].casefold() for case in cases} & {
                case[field].casefold() for case in development
            }:
                raise ValueError(f"Test/development {field} overlap.")
    return cases


def rollout(env, seed, plan, network=None):
    """Execute without receiving the task label; retain native constraint evidence."""
    observation, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    stage, total, trace = 0, 0.0, []
    aligned_once = info["aligned"]
    early_shots = 0
    for decision in range(300):
        goal = None if plan is None else plan[stage]
        if network is None:
            error = info["alignment_error"]
            action = (
                1 if error > 8 else 2 if error < -8 else 0,
                int(goal == BasicGoal.HIT_TARGET and info["aligned"]),
            )
        else:
            action = select_goal_action(
                network,
                observation,
                info["next_unavailable_masks"],
                0.0,
                rng,
                goal=goal,
            )
        previous = info
        observation, reward, terminated, truncated, info = env.step(action)
        shots = previous["remaining_ammunition"] - info["remaining_ammunition"]
        early_shots += shots if not aligned_once else 0
        if goal_achieved(BasicGoal.ALIGN_WITHOUT_FIRE, action, info):
            aligned_once = True
        total += reward
        trace.append(
            {
                "decision": decision + 1,
                "goal": None if goal is None else goal.name,
                "action": list(action),
                "aligned_before": previous["aligned"],
                "aligned_after": info.get("aligned", False),
                "shots": shots,
                "alignment_error_before": previous["alignment_error"],
                "native_kill_count": info["native_kill_count"],
            }
        )
        if ("target_hit" in info["events"]) != (info["native_kill_count"] > 0):
            raise RuntimeError("Hit event differs from native kill count.")
        if plan is not None and goal_achieved(goal, action, info):
            stage += 1
            if stage == len(plan):
                break
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "decisions": len(trace),
        "environment_return": total,
        "native_kill_count": info["native_kill_count"],
        "aligned_final": info.get("aligned", False),
        "shots": sum(step["shots"] for step in trace),
        "early_shots": early_shots,
        "infrastructure_invalid": "infrastructure_invalid" in info["events"],
        "plan_completed": plan is not None and stage == len(plan),
        "step_trace": trace,
    }


def task_success(task, episode):
    """Score behavior, allowing different plans that satisfy the same instruction."""
    if episode["infrastructure_invalid"]:
        return False
    killed = episode["native_kill_count"] > 0
    if task == "preserve":
        return episode["aligned_final"] and not killed and episode["shots"] == 0
    return killed and (task == "hit" or episode["early_shots"] == 0)


def paired_interval(rows, first, second):
    """Resample wording families; repeated layouts/policies/calls are not independent."""
    groups = {}
    for row in rows:
        if row["arm"] in (first, second):
            groups.setdefault(row["family"], {}).setdefault(row["arm"], []).append(
                row["success_rate"]
            )
    differences = np.array(
        [np.mean(values[first]) - np.mean(values[second]) for values in groups.values()]
    )
    rng = np.random.default_rng(20261001)
    bootstrap = rng.choice(
        differences, size=(4000, len(differences)), replace=True
    ).mean(axis=1)
    return {
        "first": first,
        "second": second,
        "family_count": len(differences),
        "success_rate_delta": float(differences.mean()),
        "paired_family_bootstrap_97_5_interval": np.quantile(
            bootstrap, [0.0125, 0.9875]
        ).tolist(),
        "per_family_delta": dict(zip(groups, differences.tolist())),
    }


def _load_networks(checkpoint_path, seeds):
    networks = {}
    if checkpoint_path is None:
        return networks
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if bundle.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Expected a native-control checkpoint.")
    for session in bundle["sessions"]:
        for profile in ("alternating", "no_goal"):
            saved = session["policies"][profile]
            if set(seeds) & set(saved["training"]["training_episode_seeds"]):
                raise ValueError("Training/evaluation reset seeds overlap.")
            network = GoalConditionedQNetwork(profile != "no_goal")
            network.load_state_dict(saved["state_dict"])
            network.eval()
            if _hash(network) != saved["parameter_hash"]:
                raise ValueError("Checkpoint parameter hash mismatch.")
            networks[(session["training_seed"], profile)] = network
    return networks


def run_experiment(
    *,
    cases_path,
    output_path,
    checkpoint_path=None,
    llm_model=None,
    repeats=3,
    evaluation_episodes=10,
    workers=3,
    timeout=10.0,
):
    if (
        not 1 <= evaluation_episodes <= 100
        or not 1 <= repeats <= 10
        or not 1 <= workers <= 8
    ):
        raise ValueError("Use 1-100 layouts, 1-10 repeats and 1-8 workers.")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError("Use a new output path to preserve earlier results.")
    cases = read_cases(cases_path)
    if llm_model:
        validate_llm(OPENROUTER_URL, llm_model, timeout)
    seeds = EVALUATION_SEEDS[:evaluation_episodes]
    networks = _load_networks(checkpoint_path, seeds)
    files = [
        Path(__file__),
        Path(__file__).with_name("goal_teachers.py"),
        Path(__file__).with_name("learning_components.py"),
        Path(__file__).parent / "envs/native_basic.py",
        DEVELOPMENT,
        Path(cases_path),
    ]
    if checkpoint_path:
        files.append(Path(checkpoint_path))
    report = {
        "report_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "registered",
        "source_hashes": {
            str(file.resolve()): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in files
        },
        "cases": cases,
        "evaluation_seeds": list(seeds),
        "repeats": repeats,
        "llm_model": llm_model,
        "llm_prompt": PROMPT if llm_model else None,
        "protocol": {
            "primary": "Native task success with all instruction constraints, including provider errors as failures.",
            "comparisons": "LLM versus rule and no-goal on frozen alternating/no-goal policies; rule comparison shares weights. No-goal comparison changes training reward and replay labeling.",
            "uncertainty": "Paired wording-family bootstrap, 4000 draws; 97.5% intervals for two primary comparisons. Synthetic author-curated pilot, not a population claim.",
            "teacher_input": "Instruction only, same skill descriptions; no task labels or expected plans.",
            "constraints": "Preserve: alive, aligned, zero actual shots. Align-then-hit: alignment at reset or completed Idle alignment before first actual ammunition expenditure, then native kill.",
            "cache": "One deterministic native rollout per policy/plan/reset seed; reused task scores are explicitly counterfactual evaluations, not fresh environment interactions.",
            "scripted_executor": "Scripted HIT aligns before shooting; this is a reachability diagnostic, not an unconstrained direct-fire learner.",
            "iteration": "Freeze prompt, parser, cases and source hashes before requests; revisions require fresh held-out families and preserve this report.",
        },
        "predictions": [],
        "rollouts": {},
        "results": [],
        "aggregate": {},
        "paired_comparisons": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save()
    rng = np.random.default_rng(20261001)
    for case in cases:
        for repeat in range(repeats):
            for arm in ("rule", "constant_hit", "random", "oracle", "no_goal"):
                record = {"id": case["id"], "repeat": repeat, "arm": arm}
                try:
                    plan = (
                        rule_plan(case["instruction"])
                        if arm == "rule"
                        else (
                            PLANS[case["task"]]
                            if arm == "oracle"
                            else (
                                tuple(PLANS.values())[int(rng.integers(3))]
                                if arm == "random"
                                else PLANS["hit"]
                            )
                        )
                    )
                    record["plan"] = [goal.name for goal in plan]
                except ValueError as error:
                    record.update(plan=None, error=str(error))
                report["predictions"].append(record)

    def query(case, repeat):
        record = {"id": case["id"], "repeat": repeat, "arm": "llm"}
        started = time.perf_counter()
        try:
            plan = validate_plan(
                request_json(
                    PROMPT,
                    {"instruction": case["instruction"]},
                    url=OPENROUTER_URL,
                    model=llm_model,
                    timeout=timeout,
                    record=record,
                )
            )
            record["plan"] = [goal.name for goal in plan]
        except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
            record.update(plan=None, error=f"{type(error).__name__}: {error}")
        finally:
            record["latency_seconds"] = time.perf_counter() - started
        return record

    if llm_model:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(query, case, repeat)
                for case in cases
                for repeat in range(repeats)
            ]
            for count, future in enumerate(as_completed(futures), 1):
                report["predictions"].append(future.result())
                save()
                if count % 10 == 0:
                    print(f"LLM requests {count}/{len(futures)}", flush=True)
    report["status"] = "plans_collected"
    save()
    before = {
        f"{seed}/{profile}": _hash(network)
        for (seed, profile), network in networks.items()
    }
    executors = [("scripted", 0, None)] + [
        ("learned", seed, network)
        for (seed, profile), network in networks.items()
        if profile == "alternating"
    ]
    for executor, training_seed, network in executors:
        print(
            f"Native execution: {executor}, training seed={training_seed}", flush=True
        )
        with NativeBasicV1Env() as env:
            for name, plan in PLANS.items():
                key = f"{executor}/{training_seed}/{name}"
                report["rollouts"][key] = [
                    rollout(env, seed, plan, network) for seed in seeds
                ]
            if network is not None:
                key = f"{executor}/{training_seed}/no_goal"
                report["rollouts"][key] = [
                    rollout(env, seed, None, networks[(training_seed, "no_goal")])
                    for seed in seeds
                ]
        save()
    for executor, training_seed, _network in executors:
        for prediction in report["predictions"]:
            arm = prediction["arm"]
            if executor == "scripted" and arm == "no_goal":
                continue  # Scripted direct-HIT is already represented by constant_hit.
            case = next(case for case in cases if case["id"] == prediction["id"])
            plan = prediction["plan"]
            name = (
                "no_goal"
                if arm == "no_goal"
                else next(
                    (
                        name
                        for name, goals in PLANS.items()
                        if [goal.name for goal in goals] == plan
                    ),
                    None,
                )
            )
            key = None if name is None else f"{executor}/{training_seed}/{name}"
            episodes = [] if key is None else report["rollouts"][key]
            successes = sum(task_success(case["task"], row) for row in episodes)
            costs = [
                row["decisions"] if task_success(case["task"], row) else 300
                for row in episodes
            ]
            violations = sum(
                (
                    row["shots"] > 0
                    if case["task"] == "preserve"
                    else (
                        row["early_shots"] > 0
                        if case["task"] == "align_then_hit"
                        else False
                    )
                )
                for row in episodes
            )
            report["results"].append(
                {
                    "executor": executor,
                    "training_seed": training_seed,
                    "id": case["id"],
                    "family": case["family"],
                    "task": case["task"],
                    "arm": arm,
                    "repeat": prediction["repeat"],
                    "rollout_key": key,
                    "episodes": len(seeds),
                    "successes": successes,
                    "success_rate": successes / len(seeds),
                    "task_decision_cost_mean": (
                        float(np.mean(costs)) if costs else 300.0
                    ),
                    "constraint_violations": violations,
                    "adviser_error": prediction.get("error"),
                    "canonical_plan_match": (
                        plan == [goal.name for goal in PLANS[case["task"]]]
                        if arm != "no_goal"
                        else None
                    ),
                }
            )
    after = {
        f"{seed}/{profile}": _hash(network)
        for (seed, profile), network in networks.items()
    }
    if before != after:
        raise RuntimeError("Evaluation changed policy parameters.")
    for executor in {item[0] for item in executors}:
        rows = [row for row in report["results"] if row["executor"] == executor]
        report["aggregate"][executor] = {}
        for arm in sorted({row["arm"] for row in rows}):
            selected = [row for row in rows if row["arm"] == arm]
            report["aggregate"][executor][arm] = {
                "success_rate": float(
                    np.mean([row["success_rate"] for row in selected])
                ),
                "successes": sum(row["successes"] for row in selected),
                "counterfactual_task_evaluations": sum(
                    row["episodes"] for row in selected
                ),
                "task_decision_cost_mean": float(
                    np.mean([row["task_decision_cost_mean"] for row in selected])
                ),
                "constraint_violation_rate": sum(
                    row["constraint_violations"] for row in selected
                )
                / sum(row["episodes"] for row in selected),
                "adviser_failed_predictions": sum(
                    prediction["arm"] == arm and "error" in prediction
                    for prediction in report["predictions"]
                ),
                "adviser_failed_unique_instructions": len(
                    {
                        prediction["id"]
                        for prediction in report["predictions"]
                        if prediction["arm"] == arm and "error" in prediction
                    }
                ),
                "per_task_success_rate": {
                    task: float(
                        np.mean(
                            [
                                row["success_rate"]
                                for row in selected
                                if row["task"] == task
                            ]
                        )
                    )
                    for task in sorted({row["task"] for row in selected})
                },
                "per_training_seed_success_rate": {
                    str(seed): float(
                        np.mean(
                            [
                                row["success_rate"]
                                for row in selected
                                if row["training_seed"] == seed
                            ]
                        )
                    )
                    for seed in sorted({row["training_seed"] for row in selected})
                },
            }
        if llm_model:
            report["paired_comparisons"][executor] = {
                arm: paired_interval(rows, "llm", arm)
                for arm in ("rule", "constant_hit", "random", "oracle", "no_goal")
                if arm in report["aggregate"][executor]
            }
    llm_calls = [
        prediction for prediction in report["predictions"] if prediction["arm"] == "llm"
    ]
    costs = [
        record["usage"]["cost"]
        for record in llm_calls
        if isinstance(record.get("usage"), dict)
        and isinstance(record["usage"].get("cost"), (int, float))
    ]
    report["provider"] = {
        "calls": len(llm_calls),
        "failed_calls": sum("error" in record for record in llm_calls),
        "reported_cost": sum(costs),
        "cost_reported_calls": len(costs),
        "latency_seconds_total": sum(record["latency_seconds"] for record in llm_calls),
        "routing": "OpenRouter default; temperature 0, reasoning disabled, max_tokens 64",
        "timeout_seconds": timeout,
    }
    report["validation"] = {
        "evaluation_updates": 0,
        "parameter_hashes_before": before,
        "parameter_hashes_after": after,
        "actual_environment_episodes": sum(
            len(rows) for rows in report["rollouts"].values()
        ),
        "actual_environment_decisions": sum(
            row["decisions"] for rows in report["rollouts"].values() for row in rows
        ),
        "infrastructure_invalid": sum(
            row["infrastructure_invalid"]
            for rows in report["rollouts"].values()
            for row in rows
        ),
        "sources_unchanged": all(
            hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
            for file, digest in report["source_hashes"].items()
        ),
    }
    primary = report["paired_comparisons"].get("learned", {})
    report["supported_advantage_gate"] = (
        bool(primary)
        and all(
            primary[arm]["family_count"] >= 10
            and primary[arm]["paired_family_bootstrap_97_5_interval"][0] > 0
            for arm in ("rule", "no_goal")
        )
        and report["aggregate"]["scripted"]["oracle"]["success_rate"] == 1.0
        and report["validation"]["sources_unchanged"]
        and not report["validation"]["infrastructure_invalid"]
    )
    report["status"] = "completed"
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEVELOPMENT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--llm-model")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    arguments = vars(args)
    arguments["cases_path"] = arguments.pop("cases")
    arguments["output_path"] = arguments.pop("output")
    arguments["checkpoint_path"] = arguments.pop("checkpoint")
    report = run_experiment(**arguments)
    for executor, conditions in report["aggregate"].items():
        for arm, result in conditions.items():
            print(
                f"{executor}/{arm}: task success={result['success_rate']:.1%}, task cost={result['task_decision_cost_mean']:.2f}"
            )
    return int(
        report["validation"]["infrastructure_invalid"]
        or not report["validation"]["sources_unchanged"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
