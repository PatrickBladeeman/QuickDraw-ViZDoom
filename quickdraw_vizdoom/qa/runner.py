"""Profile runner for contract-driven QuickDraw quality gates."""

from __future__ import annotations

from pathlib import Path

from .contract import ContractError, ContractInfrastructureError, load_contract
from .results import Outcome, QAReport, ValidationResult
from .validators import INTEGRATION_VALIDATORS, VALIDATORS


def _report(
    profile: str,
    loaded,
    results: list[ValidationResult],
) -> QAReport:
    return QAReport(
        profile=profile,
        contract_path=loaded.path,
        contract_id=loaded.data["contract_id"],
        contract_sha256=loaded.sha256,
        results=tuple(results),
    )


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
