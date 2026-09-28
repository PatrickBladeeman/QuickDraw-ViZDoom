"""QA runner for the registered goal-conditioned Basic baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

from .results import Outcome, QAReport, ValidationResult
from .validators import VALIDATORS


_STATIC_VALIDATORS = [key for key in VALIDATORS if key != "dev-profile"]


def _load_runtime():
    from quickdraw_vizdoom.goal_conditioning import EnvironmentStartupError, run_session

    return run_session, EnvironmentStartupError


def _report(loaded, results, sessions=()) -> QAReport:
    return QAReport(
        profile="goal-conditioning",
        contract_path=loaded.path,
        contract_id=loaded.data["contract_id"],
        contract_sha256=loaded.sha256,
        results=tuple(results),
        training_sessions=tuple(sessions),
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _research_gate_errors(
    sessions: Mapping[str, list[Mapping[str, Any]]],
) -> list[str]:
    correct = sessions.get("correct_goal_potential", [])
    shuffled = sessions.get("shuffled_goal_potential", [])
    if not correct or not shuffled:
        return ["potential comparison sessions are missing"]

    def mean(values) -> float:
        numbers = [float(value) for value in values]
        return sum(numbers) / len(numbers)

    causal = mean(
        session["counterfactual_goal_action_switch_rate"] for session in correct
    )
    correct_selection = mean(
        rate
        for session in correct
        for rate in session["intended_action_selection_rate_per_goal"].values()
    )
    shuffled_selection = mean(
        rate
        for session in shuffled
        for rate in session["intended_action_selection_rate_per_goal"].values()
    )
    return_advantage = mean(session["held_out_return"] for session in correct) - mean(
        session["held_out_return"] for session in shuffled
    )
    correct_decisions = [
        decision
        for session in correct
        for decision in session.get("evaluation_decisions_to_hit", [])
    ]
    shuffled_decisions = [
        decision
        for session in shuffled
        for decision in session.get("evaluation_decisions_to_hit", [])
    ]
    efficiency_advantage = (
        mean(shuffled_decisions) - mean(correct_decisions)
        if correct_decisions and shuffled_decisions
        else 0.0
    )
    errors = []
    if causal <= 0:
        errors.append("causal gate failed: changing goals never changed an action")
    if correct_selection <= shuffled_selection:
        errors.append(
            "performance gate failed: correct potential goals did not improve "
            "intended-action selection"
        )
    if return_advantage <= 0 and efficiency_advantage <= 0:
        errors.append(
            "performance gate failed: correct potential goals improved neither "
            "held-out environment return nor decisions-to-hit"
        )
    return errors


def _session_summary(
    condition: str, seed: int, result: object
) -> tuple[dict[str, Any], list[str]]:
    values = result if isinstance(result, Mapping) else {}
    summary = {
        "condition": condition,
        "seed": seed,
        "decisions": values.get("decisions"),
        "updates": values.get("updates"),
        "last_loss": values.get("last_loss"),
        "cumulative_episode_return": values.get("cumulative_episode_return"),
        "cumulative_environment_reward": values.get("cumulative_environment_reward"),
        "cumulative_goal_shaping_reward": values.get("cumulative_goal_shaping_reward"),
        "cumulative_training_reward": values.get("cumulative_training_reward"),
        "mean_decisions_to_hit": values.get("mean_decisions_to_hit"),
        "shot_percentage": values.get("shot_percentage"),
        "goal_completion_rate": values.get("goal_completion_rate"),
        "held_out_return": values.get("held_out_return"),
        "held_out_target_hit_rate": values.get("held_out_target_hit_rate"),
        "evaluation_updates": values.get("evaluation_updates"),
        "training_seed_schedule": values.get("training_seed_schedule"),
        "evaluation_seed_schedule": values.get("evaluation_seed_schedule"),
        "goal_selection_counts": values.get("goal_selection_counts"),
        "goal_transition_counts": values.get("goal_transition_counts"),
        "counterfactual_goal_action_switch_rate": values.get(
            "counterfactual_goal_action_switch_rate"
        ),
        "intended_action_selection_rate_per_goal": values.get(
            "intended_action_selection_rate_per_goal"
        ),
        "intended_action_q_margin_per_goal": values.get(
            "intended_action_q_margin_per_goal"
        ),
    }
    errors: list[str] = []
    if not isinstance(result, Mapping):
        errors.append("result is not a mapping")
        return summary, errors
    if values.get("condition") != condition or values.get("seed") != seed:
        errors.append("condition or seed identity drifted")
    if values.get("trainer") != "branching_q_learning":
        errors.append("trainer must remain labeled branching_q_learning")
    if values.get("decisions") != 512:
        errors.append("decisions must equal 512")
    updates = values.get("updates")
    if not isinstance(updates, Integral) or isinstance(updates, bool) or updates < 1:
        errors.append("optimizer updates must be positive")
    for field in (
        "last_loss",
        "cumulative_episode_return",
        "cumulative_environment_reward",
        "cumulative_goal_shaping_reward",
        "cumulative_training_reward",
        "mean_decisions_to_hit",
        "shot_percentage",
        "goal_completion_rate",
        "held_out_return",
        "held_out_target_hit_rate",
        "counterfactual_goal_action_switch_rate",
    ):
        if not _finite(values.get(field)):
            errors.append(f"{field} must be finite")
    for field in (
        "counterfactual_goal_action_switch_rate",
        "held_out_target_hit_rate",
        "goal_completion_rate",
    ):
        value = values.get(field)
        if _finite(value) and not 0 <= float(value) <= 1:
            errors.append(f"{field} must be a rate in [0, 1]")
    if values.get("evaluation_updates") != 0:
        errors.append("evaluation must perform zero optimizer updates")
    if values.get("evaluation_parameters_unchanged") is not True:
        errors.append("evaluation changed policy parameters")
    if values.get("evaluation_episodes") != 100:
        errors.append("evaluation must contain 100 episodes")
    if values.get("process_launches") != 1 or values.get("training_sessions") != 1:
        errors.append("each session must use one process and training session")
    if values.get("external_calls") != 0:
        errors.append("goal baseline must make no external calls")
    if not isinstance(values.get("goal_selection_counts"), Mapping):
        errors.append("goal selection counts are missing")
    if not isinstance(values.get("goal_transition_counts"), Mapping):
        errors.append("goal transition counts are missing")
    goal_names = {"ALIGN_RIGHT", "ALIGN_LEFT", "TAKE_SHOT"}
    for field in (
        "intended_action_selection_rate_per_goal",
        "intended_action_q_margin_per_goal",
    ):
        metrics = values.get(field)
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != goal_names
            or any(not _finite(value) for value in metrics.values())
        ):
            errors.append(f"{field} must contain finite metrics for every goal")
    required = {
        "goal",
        "next_goal",
        "goal_source",
        "goal_success",
        "goal_changed",
        "events",
        "environment_reward",
        "goal_shaping_reward",
        "training_reward",
    }
    for field in ("transition_metrics", "evaluation_transition_metrics"):
        records = values.get(field)
        if not isinstance(records, list) or not records:
            errors.append(f"{field} are missing")
        elif any(
            not isinstance(record, Mapping)
            or not required.issubset(record)
            or record["goal_source"] != condition
            for record in records
        ):
            errors.append(f"{field} goal telemetry is invalid")
        else:
            for record in records:
                environment = record["environment_reward"]
                shaping = record["goal_shaping_reward"]
                training = record["training_reward"]
                if not all(
                    _finite(value) for value in (environment, shaping, training)
                ):
                    errors.append(f"{field} reward components must be finite")
                    break
                if not math.isclose(
                    float(training), float(environment) + float(shaping)
                ):
                    errors.append(f"{field} training reward is inconsistent")
                    break
                if (
                    field == "evaluation_transition_metrics"
                    or not condition.endswith("_potential")
                ) and shaping != 0:
                    errors.append(f"{field} unexpectedly contains shaping reward")
                    break
    return summary, errors


def run_goal_conditioning(loaded) -> QAReport:
    profile = loaded.data["qa"]["profiles"]["goal-conditioning"]
    results = [
        ValidationResult(
            "json-schema", Outcome.PASS, "Contract and local schema are valid."
        ),
        *(VALIDATORS[key](loaded.data) for key in _STATIC_VALIDATORS),
    ]
    results.append(
        ValidationResult(
            "goal-conditioning-profile",
            Outcome.PASS,
            "Goal-conditioning profile is registered.",
        )
    )

    try:
        run_session, startup_error = _load_runtime()
    except ImportError as error:
        results.append(
            ValidationResult(
                "goal-conditioning-sessions",
                Outcome.INFRA_INVALID,
                f"Training dependency unavailable: {error}",
            )
        )
        return _report(loaded, results)

    root = loaded.path.parent.parent
    parameters = {
        key: profile[key]
        for key in (
            "steps",
            "warmup",
            "batch_size",
            "epsilon",
            "gamma",
            "device",
            "evaluation_seed_start",
            "evaluation_episodes",
        )
    }
    sessions: list[dict[str, Any]] = []
    failures: list[str] = []
    infrastructure: list[str] = []
    schedules: dict[int, tuple[object, object]] = {}
    raw_sessions: dict[str, list[Mapping[str, Any]]] = {}
    for condition in profile["conditions"]:
        for seed in profile["seeds"]:
            try:
                raw = run_session(
                    condition=condition,
                    seed=seed,
                    config_path=root
                    / loaded.data["environment"]["scenario"]["config_path"],
                    **parameters,
                )
            except startup_error as error:
                summary, _ = _session_summary(condition, seed, None)
                sessions.append(summary)
                infrastructure.append(f"{condition}/{seed}: {error}")
                continue
            except Exception as error:
                summary, _ = _session_summary(condition, seed, None)
                sessions.append(summary)
                failures.append(f"{condition}/{seed}: {error}")
                continue
            summary, errors = _session_summary(condition, seed, raw)
            sessions.append(summary)
            failures.extend(f"{condition}/{seed}: {error}" for error in errors)
            if isinstance(raw, Mapping):
                raw_sessions.setdefault(condition, []).append(raw)
                pair = (
                    raw.get("training_seed_schedule"),
                    raw.get("evaluation_seed_schedule"),
                )
                previous = schedules.setdefault(seed, pair)
                if pair != previous:
                    failures.append(
                        f"{condition}/{seed}: seed schedules differ by condition"
                    )

    if not failures and not infrastructure:
        failures.extend(_research_gate_errors(raw_sessions))

    if failures:
        outcome, message = Outcome.FAIL, "; ".join(failures)
    elif infrastructure:
        outcome, message = Outcome.INFRA_INVALID, "; ".join(infrastructure)
    else:
        outcome = Outcome.PASS
        message = "All 15 registered sessions and research gates passed."
    results.append(ValidationResult("goal-conditioning-sessions", outcome, message))
    return _report(loaded, results, sessions)
