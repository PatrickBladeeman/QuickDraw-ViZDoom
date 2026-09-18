"""Profile runner for contract-driven QuickDraw quality gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .contract import ContractError, ContractInfrastructureError, load_contract
from .results import Outcome, QAReport, ValidationResult
from .validators import INTEGRATION_VALIDATORS, VALIDATORS


_STATIC_VALIDATOR_IDS = [
    validator_id for validator_id in VALIDATORS if validator_id != "dev-profile"
]
_TRAINING_SMOKE_PROFILE = {
    "budgets": {
        "process_launches": 2,
        "trace_collections": 0,
        "training_sessions": 2,
        "external_calls": 0,
    },
    "validators": [
        "json-schema",
        *_STATIC_VALIDATOR_IDS,
        "training-smoke-profile",
        "training-smoke-sessions",
    ],
    "seeds": [31001, 31002],
    "steps": 16,
    "warmup": 4,
    "batch_size": 4,
    "device": "cpu",
}


def _report(
    profile: str,
    loaded,
    results: list[ValidationResult],
    training_sessions: list[dict[str, Any]] | None = None,
) -> QAReport:
    return QAReport(
        profile=profile,
        contract_path=loaded.path,
        contract_id=loaded.data["contract_id"],
        contract_sha256=loaded.sha256,
        results=tuple(results),
        training_sessions=(
            tuple(training_sessions) if training_sessions is not None else None
        ),
    )


def _load_training_runtime():
    from quickdraw_vizdoom.learning import EnvironmentStartupError, run_smoke

    return run_smoke, EnvironmentStartupError


def _session_summary(seed: int, result: object) -> tuple[dict[str, Any], list[str]]:
    values = result if isinstance(result, Mapping) else {}
    decisions = values.get("decisions")
    updates = values.get("updates")
    terminal = values.get("terminal")
    truncated = values.get("truncated")
    last_loss = values.get("last_loss")
    integer_values = (decisions, updates, terminal, truncated)
    normalized_integers = [
        (
            int(value)
            if isinstance(value, Integral) and not isinstance(value, bool)
            else None
        )
        for value in integer_values
    ]
    finite_loss = (
        isinstance(last_loss, Real)
        and not isinstance(last_loss, bool)
        and math.isfinite(float(last_loss))
    )
    summary = {
        "seed": seed,
        "decisions": normalized_integers[0],
        "updates": normalized_integers[1],
        "terminal": normalized_integers[2],
        "truncated": normalized_integers[3],
        "last_loss": float(last_loss) if finite_loss else None,
    }
    errors: list[str] = []
    if not isinstance(result, Mapping):
        errors.append("result is not a mapping")
    if summary["decisions"] != 16:
        errors.append("decisions must equal 16")
    if summary["updates"] is None or summary["updates"] < 1:
        errors.append("optimizer updates must be positive")
    counts = (summary["terminal"], summary["truncated"])
    if any(count is None or count < 0 or count > 16 for count in counts):
        errors.append("terminal and truncated counts must be between 0 and 16")
    elif sum(counts) > 16:
        errors.append("terminal and truncated counts exceed decisions")
    if not finite_loss:
        errors.append("last loss must be finite")
    return summary, errors


def _run_training_smoke(loaded) -> QAReport:
    profile = loaded.data["qa"]["profiles"]["training-smoke"]
    results = [
        ValidationResult(
            validator_id="json-schema",
            outcome=Outcome.PASS,
            message="Contract and local schema are valid.",
        ),
        *(
            VALIDATORS[validator_id](loaded.data)
            for validator_id in _STATIC_VALIDATOR_IDS
        ),
    ]
    profile_errors = (
        []
        if profile == _TRAINING_SMOKE_PROFILE
        else ["The training-smoke profile parameters, validators, or budgets drifted."]
    )
    results.append(
        ValidationResult(
            validator_id="training-smoke-profile",
            outcome=Outcome.FAIL if profile_errors else Outcome.PASS,
            message="; ".join(profile_errors) or "Training-smoke profile is valid.",
        )
    )
    if profile_errors:
        return _report("training-smoke", loaded, results, [])

    try:
        run_smoke, environment_startup_error = _load_training_runtime()
    except ImportError as error:
        results.append(
            ValidationResult(
                validator_id="training-smoke-sessions",
                outcome=Outcome.INFRA_INVALID,
                message=f"Training dependency unavailable: {error}",
            )
        )
        return _report("training-smoke", loaded, results, [])

    sessions: list[dict[str, Any]] = []
    failures: list[str] = []
    infrastructure_errors: list[str] = []
    parameters = {
        key: profile[key] for key in ("steps", "warmup", "batch_size", "device")
    }
    for seed in profile["seeds"]:
        try:
            raw_result = run_smoke(seed=seed, **parameters)
        except environment_startup_error as error:
            sessions.append(_session_summary(seed, None)[0])
            infrastructure_errors.append(f"seed {seed}: {error}")
        except Exception as error:
            sessions.append(_session_summary(seed, None)[0])
            failures.append(f"seed {seed}: {error}")
        else:
            summary, session_errors = _session_summary(seed, raw_result)
            sessions.append(summary)
            failures.extend(f"seed {seed}: {error}" for error in session_errors)

    if failures:
        outcome = Outcome.FAIL
        message = "; ".join(failures)
    elif infrastructure_errors:
        outcome = Outcome.INFRA_INVALID
        message = "; ".join(infrastructure_errors)
    else:
        outcome = Outcome.PASS
        message = "Both independent training-smoke sessions completed successfully."
    results.append(
        ValidationResult(
            validator_id="training-smoke-sessions",
            outcome=outcome,
            message=message,
        )
    )
    return _report("training-smoke", loaded, results, sessions)


def run_profile(contract_path: str | Path, profile: str) -> QAReport:
    """Run one QA profile without exceeding the profile's declared budgets."""

    resolved_contract_path = Path(contract_path).resolve()
    try:
        loaded = load_contract(resolved_contract_path)
    except (ContractError, ContractInfrastructureError) as error:
        return QAReport(
            profile=profile,
            contract_path=resolved_contract_path,
            contract_id=None,
            contract_sha256=None,
            results=(
                ValidationResult(
                    validator_id="json-schema",
                    outcome=(
                        Outcome.INFRA_INVALID
                        if isinstance(error, ContractInfrastructureError)
                        else Outcome.FAIL
                    ),
                    message=str(error),
                ),
            ),
        )

    if profile == "training-smoke":
        return _run_training_smoke(loaded)

    if profile == "integration":
        try:
            from .integration import collect_canonical_trace

            trace, _trace_path = collect_canonical_trace(
                loaded.path,
                loaded.data,
            )
        except (ImportError, OSError, RuntimeError) as error:
            return _report(
                profile,
                loaded,
                [
                    ValidationResult(
                        validator_id="integration-runtime",
                        outcome=Outcome.INFRA_INVALID,
                        message=str(error),
                    )
                ],
            )

        static_results = [
            validator(loaded.data)
            for validator_id, validator in VALIDATORS.items()
            if validator_id != "dev-profile"
        ]
        integration_results = [
            INTEGRATION_VALIDATORS["integration-profile"](loaded.data),
            *(
                INTEGRATION_VALIDATORS[validator_id](loaded.data, trace)
                for validator_id in (
                    "integration-runtime",
                    "integration-observation",
                    "integration-actions",
                    "integration-timing",
                    "integration-rewards",
                    "integration-endings",
                )
            ),
        ]
        return _report(
            profile,
            loaded,
            [
                ValidationResult(
                    validator_id="json-schema",
                    outcome=Outcome.PASS,
                    message="Contract and local schema are valid.",
                ),
                *static_results,
                *integration_results,
            ],
        )

    if profile != "dev":
        results = [
            ValidationResult(
                validator_id="profile-selection",
                outcome=Outcome.FAIL,
                message=f"Unsupported QA profile: {profile!r}.",
            )
        ]
    else:
        results = [
            ValidationResult(
                validator_id="json-schema",
                outcome=Outcome.PASS,
                message="Contract and local schema are valid.",
            ),
            *(validator(loaded.data) for validator in VALIDATORS.values()),
        ]

    return _report(profile, loaded, results)
