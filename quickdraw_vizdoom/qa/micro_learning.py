"""Micro-learning QA orchestration; importing this module does not import torch."""

import hashlib
import math
from collections.abc import Mapping
from numbers import Integral, Real

from .results import Outcome, QAReport, ValidationResult


def _load_runtime():
    from quickdraw_vizdoom.micro_learning import EnvironmentStartupError, run_session

    return run_session, EnvironmentStartupError


def _session_summary(seed, result):
    values = result if isinstance(result, Mapping) else {}
    summary = {"seed": seed}
    errors = []
    counts = {
        "decisions": 512,
        "terminal": 512,
        "truncated": 0,
        "training_seed_start": seed,
        "training_seed_end": seed + 511,
        "evaluation_seed_start": 36000,
        "evaluation_seed_end": 36099,
        "evaluation_episodes": 100,
        "evaluation_updates": 0,
        "process_launches": 1,
        "training_sessions": 1,
        "trace_collections": 0,
        "external_calls": 0,
        "updates": None,
        "greedy_successes": None,
    }
    for key, expected in counts.items():
        value = values.get(key)
        valid = isinstance(value, Integral) and not isinstance(value, bool)
        summary[key] = int(value) if valid else None
        if not valid or (expected is not None and value != expected):
            errors.append(f"Invalid {key}: expected {expected}")
    if not 1 <= (summary["updates"] or 0) <= 481:
        errors.append("Optimizer updates must be between 1 and 481.")
    successes = summary["greedy_successes"]
    if successes is None or not 90 <= successes <= 100:
        errors.append("Held-out successes must be between 90 and 100.")
    for key in (
        "last_loss",
        "greedy_success_rate",
        "random_baseline_rate",
        "improvement_percentage_points",
    ):
        value = values.get(key)
        valid = (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        summary[key] = float(value) if valid else None
        if not valid:
            errors.append(f"{key} must be finite.")
    if summary["last_loss"] is not None and summary["last_loss"] < 0:
        errors.append("Loss must be nonnegative.")
    rate = summary["greedy_success_rate"]
    improvement = summary["improvement_percentage_points"]
    if rate is None or successes is None or rate != successes / 100:
        errors.append("Success rate must match held-out successes.")
    if summary["random_baseline_rate"] != 0.5:
        errors.append("The exact random baseline is 0.5.")
    if (
        improvement is None
        or rate is None
        or improvement < 30
        or not math.isclose(improvement, (rate - 0.5) * 100)
    ):
        errors.append("Improvement must match the rate and reach 30 percentage points.")
    summary["parameters_unchanged"] = values.get("parameters_unchanged") is True
    if not summary["parameters_unchanged"]:
        errors.append("Evaluation changed parameters or did not verify them.")
    return summary, errors


def run_micro_learning(loaded):
    results = [
        ValidationResult(
            "json-schema", Outcome.PASS, "Fixed micro-learning contract is valid."
        )
    ]
    sessions = []

    def report():
        return QAReport(
            profile="micro-learning",
            contract_path=loaded.path,
            contract_id=loaded.data["contract_id"],
            contract_sha256=loaded.sha256,
            results=tuple(results),
            training_sessions=tuple(sessions),
        )

    scenario = loaded.data["environment"]["scenario"]
    root = loaded.path.parent.parent
    try:
        for kind in ("config", "wad"):
            path = root / scenario[f"{kind}_path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != scenario[f"{kind}_sha256"]:
                results.append(
                    ValidationResult(
                        "micro-learning-artifacts",
                        Outcome.FAIL,
                        f"{kind} hash mismatch.",
                    )
                )
                return report()
    except OSError as error:
        results.append(
            ValidationResult(
                "micro-learning-artifacts", Outcome.INFRA_INVALID, str(error)
            )
        )
        return report()
    results.append(
        ValidationResult(
            "micro-learning-artifacts", Outcome.PASS, "Dedicated scenario hashes match."
        )
    )
    try:
        run_session, startup_error = _load_runtime()
    except ImportError as error:
        results.append(
            ValidationResult(
                "micro-learning-sessions",
                Outcome.INFRA_INVALID,
                f"Training dependency unavailable: {error}",
            )
        )
        return report()

    profile = loaded.data["qa"]["profiles"]["micro-learning"]
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
    for seed in profile["seeds"]:
        try:
            result = run_session(
                config_path=root / scenario["config_path"], seed=seed, **parameters
            )
            summary, errors = _session_summary(seed, result)
            outcome = Outcome.FAIL if errors else Outcome.PASS
            message = (
                "; ".join(errors)
                or "Held-out greedy policy meets the registered thresholds."
            )
        except startup_error as error:
            summary, _ = _session_summary(seed, None)
            outcome, message = Outcome.INFRA_INVALID, str(error)
        except Exception as error:
            summary, _ = _session_summary(seed, None)
            outcome, message = Outcome.FAIL, str(error)
        sessions.append(summary)
        results.append(
            ValidationResult(f"micro-learning-session-{seed}", outcome, message)
        )
    return report()
