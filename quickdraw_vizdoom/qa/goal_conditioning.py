"""QA runner for the registered three-condition Basic baseline."""

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
        "mean_decisions_to_hit",
        "shot_percentage",
        "goal_completion_rate",
        "held_out_return",
        "held_out_target_hit_rate",
    ):
        if not _finite(values.get(field)):
            errors.append(f"{field} must be finite")
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
    required = {
        "goal",
        "next_goal",
        "goal_source",
        "goal_success",
        "goal_changed",
        "events",
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
                pair = (
                    raw.get("training_seed_schedule"),
                    raw.get("evaluation_seed_schedule"),
                )
                previous = schedules.setdefault(seed, pair)
                if pair != previous:
                    failures.append(
                        f"{condition}/{seed}: seed schedules differ by condition"
                    )

    if failures:
        outcome, message = Outcome.FAIL, "; ".join(failures)
    elif infrastructure:
        outcome, message = Outcome.INFRA_INVALID, "; ".join(infrastructure)
    else:
        outcome = Outcome.PASS
        message = "All nine registered goal-conditioning sessions passed."
    results.append(ValidationResult("goal-conditioning-sessions", outcome, message))
    return _report(loaded, results, sessions)
