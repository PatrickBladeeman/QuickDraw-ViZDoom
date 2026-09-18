"""Typed results shared by QuickDraw QA profiles and validators."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Outcome(str, Enum):
    """Top-level outcome vocabulary for every QuickDraw quality gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    INFRA_INVALID = "INFRA_INVALID"

    @property
    def exit_code(self) -> int:
        return {
            Outcome.PASS: 0,
            Outcome.FAIL: 1,
            Outcome.INFRA_INVALID: 2,
        }[self]


@dataclass(frozen=True)
class ValidationResult:
    """One validator's deterministic result."""

    validator_id: str
    outcome: Outcome
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "outcome": self.outcome.value,
            "message": self.message,
        }


def aggregate_outcomes(results: Iterable[ValidationResult]) -> Outcome:
    """Return FAIL before INFRA_INVALID so known defects are never obscured."""

    outcomes = {result.outcome for result in results}
    if Outcome.FAIL in outcomes:
        return Outcome.FAIL
    if Outcome.INFRA_INVALID in outcomes:
        return Outcome.INFRA_INVALID
    return Outcome.PASS


@dataclass(frozen=True)
class QAReport:
    """Complete machine-readable result for one contract/profile invocation."""

    profile: str
    contract_path: Path
    contract_id: str | None
    contract_sha256: str | None
    results: tuple[ValidationResult, ...]

    @property
    def outcome(self) -> Outcome:
        return aggregate_outcomes(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "quickdraw.vizdoom-qa-report.v1",
            "outcome": self.outcome.value,
            "profile": self.profile,
            "contract_path": str(self.contract_path),
            "contract_id": self.contract_id,
            "contract_sha256": self.contract_sha256,
            "results": [result.to_dict() for result in self.results],
        }
