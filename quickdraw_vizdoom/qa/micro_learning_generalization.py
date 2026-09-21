"""QA checks for shuffled-seed Basic-arena training."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from numbers import Integral, Real

from quickdraw_vizdoom.envs.micro_learning_generalization import permuted_episode_seeds

from .results import Outcome, QAReport, ValidationResult


def _load_runtime():
    from quickdraw_vizdoom.micro_learning_generalization import (
        EnvironmentStartupError,
        run_session,
    )

    return run_session, EnvironmentStartupError


def _artifact_result(loaded) -> ValidationResult:
    scenario = loaded.data["environment"]["scenario"]
    root = loaded.path.parent.parent
    try:
        for kind in ("config", "wad"):
            path = root / scenario[f"{kind}_path"]
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != scenario[f"{kind}_sha256"]
            ):
                return ValidationResult(
                    "micro-learning-generalization-artifacts",
                    Outcome.FAIL,
                    f"{kind} hash mismatch.",
                )
    except OSError as error:
        return ValidationResult(
            "micro-learning-generalization-artifacts", Outcome.INFRA_INVALID, str(error)
        )
    return ValidationResult(
        "micro-learning-generalization-artifacts",
        Outcome.PASS,
        "Basic arena scenario hashes match.",
    )


def _int(value) -> int | None:
    return (
        int(value)
        if isinstance(value, Integral) and not isinstance(value, bool)
        else None
    )


def _finite(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _list(values, key, errors, summary):
    value = values.get(key)
    summary[key] = list(value) if isinstance(value, (list, tuple)) else None
    if summary[key] is None:
        errors.append(f"{key} must be recorded.")
    return summary[key]


def _session_summary(seed: int, result: object, profile: Mapping):
    values = result if isinstance(result, Mapping) else {}
    steps, episodes = profile["steps"], profile["evaluation_episodes"]
    summary, errors = {"seed": seed}, []
    expected = {
        "decisions": steps,
        "evaluation_episodes": episodes,
        "evaluation_updates": 0,
        "training_seed_start": seed,
        "training_seed_end": seed + steps - 1,
        "evaluation_seed_start": profile["evaluation_seed_start"],
        "evaluation_seed_end": profile["evaluation_seed_start"] + episodes - 1,
        "process_launches": 1,
        "training_sessions": 1,
        "trace_collections": 0,
        "external_calls": 0,
        "checkpoints": 0,
    }
    for key, wanted in expected.items():
        summary[key] = _int(values.get(key))
        if summary[key] != wanted:
            errors.append(f"Invalid {key}: expected {wanted}.")
    for key in (
        "updates",
        "terminal",
        "truncated",
        "training_episodes",
        "evaluation_decisions",
        "evaluation_terminal",
        "evaluation_truncated",
    ):
        summary[key] = _int(values.get(key))
        if summary[key] is None or summary[key] < 0:
            errors.append(f"{key} must be a nonnegative integer.")
    if summary["updates"] is not None and summary["updates"] < 1:
        errors.append("Optimizer updates must be positive.")
    if summary["terminal"] == 0 and summary["truncated"] == 0:
        errors.append("Training produced no completed episodes.")
    if (
        summary["evaluation_terminal"] is not None
        and summary["evaluation_truncated"] is not None
        and summary["evaluation_terminal"] + summary["evaluation_truncated"] != episodes
    ):
        errors.append("Evaluation episode endings do not match the episode count.")

    for key in ("last_loss", "greedy_success_rate"):
        summary[key] = _finite(values.get(key))
        if summary[key] is None:
            errors.append(f"{key} must be finite.")
    if summary["last_loss"] is not None and summary["last_loss"] < 0:
        errors.append("Loss must be nonnegative.")
    summary["greedy_successes"] = _int(values.get("greedy_successes"))
    if (
        summary["greedy_successes"] is None
        or not 0 <= summary["greedy_successes"] <= episodes
    ):
        errors.append("greedy_successes is outside the evaluation range.")
    if (
        summary["greedy_successes"] is not None
        and summary["greedy_success_rate"] is not None
        and not math.isclose(
            summary["greedy_success_rate"],
            summary["greedy_successes"] / episodes,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        errors.append("greedy_success_rate does not match its success count.")
    if (
        summary["greedy_success_rate"] is not None
        and summary["greedy_success_rate"] < profile["minimum_success_rate"]
    ):
        errors.append("Held-out success is below threshold.")
    summary["parameters_unchanged"] = values.get("parameters_unchanged") is True
    if not summary["parameters_unchanged"]:
        errors.append("Evaluation changed parameters or did not verify them.")

    for key in (
        "training_episode_seeds",
        "training_seed_schedule",
        "evaluation_episode_seeds",
        "training_visual_variants",
        "held_out_visual_variants",
        "training_reset_frame_hashes",
        "held_out_reset_frame_hashes",
        "reset_seed_overlap",
        "frame_hash_overlap",
        "variant_overlap",
    ):
        _list(values, key, errors, summary)
    expected_training = list(permuted_episode_seeds(seed, steps, permutation_seed=seed))
    expected_evaluation = list(
        permuted_episode_seeds(
            profile["evaluation_seed_start"], episodes, permutation_seed=seed
        )
    )
    if summary["training_seed_schedule"] != expected_training:
        errors.append("Training reset seed schedule drifted.")
    if (
        summary["training_episode_seeds"]
        != expected_training[: len(summary["training_episode_seeds"] or ())]
    ):
        errors.append("Training reset seeds are not the schedule prefix.")
    if summary["evaluation_episode_seeds"] != expected_evaluation:
        errors.append("Evaluation reset seed schedule drifted.")
    if summary["reset_seed_overlap"]:
        errors.append("Training and evaluation reset seeds overlap.")
    if (
        not summary["training_reset_frame_hashes"]
        or not summary["held_out_reset_frame_hashes"]
    ):
        errors.append("Reset frame hashes were not recorded.")
    for key in ("training_visual_variants", "held_out_visual_variants"):
        if not summary[key]:
            errors.append(f"{key} were not recorded.")
    return summary, errors


def _report(loaded, results, sessions=()):
    return QAReport(
        profile="micro-learning-generalization",
        contract_path=loaded.path,
        contract_id=loaded.data["contract_id"],
        contract_sha256=loaded.sha256,
        results=tuple(results),
        training_sessions=tuple(sessions),
    )


def run_micro_learning_generalization(loaded):
    profile = loaded.data["qa"]["profiles"]["micro-learning-generalization"]
    results = [
        ValidationResult("json-schema", Outcome.PASS, "Basic-arena contract is valid.")
    ]
    artifact = _artifact_result(loaded)
    results.append(artifact)
    if artifact.outcome is not Outcome.PASS:
        return _report(loaded, results)
    try:
        run_session, startup_error = _load_runtime()
    except ImportError as error:
        results.append(
            ValidationResult(
                "micro-learning-generalization-sessions",
                Outcome.INFRA_INVALID,
                f"Training dependency unavailable: {error}",
            )
        )
        return _report(loaded, results)

    scenario = loaded.data["environment"]["scenario"]
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
    sessions, failures, infrastructure = [], [], []
    for seed in profile["seeds"]:
        try:
            result = run_session(
                config_path=root / scenario["config_path"], seed=seed, **parameters
            )
            summary, session_errors = _session_summary(seed, result, profile)
            failures.extend(f"seed {seed}: {error}" for error in session_errors)
            outcome = Outcome.FAIL if session_errors else Outcome.PASS
            message = (
                "; ".join(session_errors) or "Shuffled-seed Basic-arena session passed."
            )
        except startup_error as error:
            summary, _ = _session_summary(seed, None, profile)
            outcome, message = Outcome.INFRA_INVALID, str(error)
            infrastructure.append(f"seed {seed}: {error}")
        except Exception as error:
            summary, _ = _session_summary(seed, None, profile)
            outcome, message = Outcome.FAIL, str(error)
            failures.append(f"seed {seed}: {error}")
        sessions.append(summary)
        results.append(
            ValidationResult(
                f"micro-learning-generalization-session-{seed}", outcome, message
            )
        )
    if failures:
        outcome, message = Outcome.FAIL, "; ".join(failures)
    elif infrastructure:
        outcome, message = Outcome.INFRA_INVALID, "; ".join(infrastructure)
    else:
        outcome, message = (
            Outcome.PASS,
            "Both shuffled-seed Basic-arena sessions passed.",
        )
    results.append(
        ValidationResult("micro-learning-generalization-sessions", outcome, message)
    )
    return _report(loaded, results, sessions)
