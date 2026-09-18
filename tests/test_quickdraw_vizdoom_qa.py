"""Tests for the static, contract-driven QuickDraw QA development profile."""

from __future__ import annotations

import builtins
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from quickdraw_vizdoom.qa.__main__ import main
from quickdraw_vizdoom.qa.contract import ContractInfrastructureError, canonical_sha256
from quickdraw_vizdoom.qa.results import Outcome, ValidationResult, aggregate_outcomes
from quickdraw_vizdoom.qa.runner import run_profile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASIC_CONTRACT_PATH = REPOSITORY_ROOT / "contracts" / "basic-v1.json"
SCHEMA_PATH = (
    REPOSITORY_ROOT / "contracts" / "schemas" / "environment-contract-v1.schema.json"
)


def _read_basic_contract() -> dict[str, Any]:
    return json.loads(BASIC_CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, contract: dict[str, Any]) -> Path:
    schema_directory = tmp_path / "schemas"
    schema_directory.mkdir()
    (schema_directory / SCHEMA_PATH.name).write_text(
        SCHEMA_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path


def _set_nested_value(
    contract: dict[str, Any], path: tuple[str | int, ...], value: Any
) -> None:
    target: Any = contract
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value


def test_dev_profile_does_not_import_or_launch_vizdoom(monkeypatch) -> None:
    original_import = builtins.__import__

    def reject_vizdoom_import(name, *args, **kwargs):
        if name == "vizdoom" or name.startswith("vizdoom."):
            raise AssertionError("The static dev profile imported ViZDoom.")
        return original_import(name, *args, **kwargs)

    def reject_side_effect(*_args, **_kwargs):
        raise AssertionError(
            "The static dev profile attempted an external side effect."
        )

    monkeypatch.setattr(builtins, "__import__", reject_vizdoom_import)
    monkeypatch.setattr(subprocess, "Popen", reject_side_effect)

    report = run_profile(BASIC_CONTRACT_PATH, "dev")

    assert report.outcome is Outcome.PASS
    assert report.contract_id == "quickdraw.vizdoom-basic.v1"
    assert report.contract_sha256 is not None
    assert all(result.outcome is Outcome.PASS for result in report.results)


def test_canonical_hash_ignores_key_order_and_formatting() -> None:
    first = {"outer": {"alpha": 1, "beta": [2, 3]}, "value": "test"}
    second = {"value": "test", "outer": {"beta": [2, 3], "alpha": 1}}

    assert canonical_sha256(first) == canonical_sha256(second)


def test_duplicate_contract_key_is_rejected(tmp_path: Path) -> None:
    contract_path = tmp_path / "duplicate.json"
    contract_path.write_text('{"contract_id": "one", "contract_id": "two"}')

    report = run_profile(contract_path, "dev")

    assert report.outcome is Outcome.FAIL
    assert "Duplicate JSON object key" in report.results[0].message


def test_unknown_contract_field_is_rejected(tmp_path: Path) -> None:
    contract = _read_basic_contract()
    contract["unexpected"] = True

    report = run_profile(_write_contract(tmp_path, contract), "dev")

    assert report.outcome is Outcome.FAIL
    assert "Additional properties are not allowed" in report.results[0].message


@pytest.mark.parametrize(
    ("path", "value", "validator_id"),
    [
        (
            ("environment", "scenario", "artifact_status"),
            "planned",
            "basic-environment",
        ),
        (
            ("observation", "preprocessing", "crop", "width"),
            161,
            "basic-observation",
        ),
        (
            ("observation", "preprocessing", "normalization", "minimum"),
            2.0,
            "basic-observation",
        ),
        (
            ("actions", "branches", 0, "actions", 1, "index"),
            0,
            "basic-actions",
        ),
        (("actions", "decision_strobe"), "MOVE_LEFT", "basic-actions"),
        (
            ("actions", "branches", 1, "actions", 1, "button"),
            "MOVE_LEFT",
            "basic-actions",
        ),
        (("timing", "episode_tics"), 1049, "basic-timing"),
        (("reward", "terms", 1, "event"), "decision", "basic-reward"),
        (
            ("episode_end", "truncation", 0, "reason"),
            "target_hit",
            "basic-episode",
        ),
        (("reset", "target_slot", "minimum"), 5, "basic-reset"),
        (
            ("state_channel", "variables", 0, "name"),
            "contract_schema",
            "basic-state-channel",
        ),
    ],
)
def test_relational_contract_mutations_fail_their_validator(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: Any,
    validator_id: str,
) -> None:
    contract = _read_basic_contract()
    _set_nested_value(contract, path, value)

    report = run_profile(_write_contract(tmp_path, contract), "dev")

    assert report.results[0].outcome is Outcome.PASS
    result = next(
        result for result in report.results if result.validator_id == validator_id
    )
    assert result.outcome is Outcome.FAIL


def test_nonzero_dev_budget_fails(tmp_path: Path) -> None:
    contract = _read_basic_contract()
    profile = contract["qa"]["profiles"]["dev"]
    profile["budgets"]["process_launches"] = 1

    report = run_profile(_write_contract(tmp_path, contract), "dev")

    assert report.outcome is Outcome.FAIL
    profile_result = next(
        result for result in report.results if result.validator_id == "dev-profile"
    )
    assert profile_result.outcome is Outcome.FAIL
    assert "zero runtime" in profile_result.message


def test_missing_validation_dependency_is_infrastructure_invalid(
    monkeypatch,
    capsys,
) -> None:
    def dependency_failure(*_args, **_kwargs):
        raise ContractInfrastructureError("validation dependency unavailable")

    monkeypatch.setattr(
        "quickdraw_vizdoom.qa.contract._validate_json_schema",
        dependency_failure,
    )

    exit_code = main(
        ["run", "--contract", str(BASIC_CONTRACT_PATH), "--profile", "dev"]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["outcome"] == "INFRA_INVALID"


def test_missing_contract_is_a_failure(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "run",
            "--contract",
            str(tmp_path / "missing.json"),
            "--profile",
            "dev",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["outcome"] == "FAIL"


def test_remote_and_parent_schema_references_are_rejected(tmp_path: Path) -> None:
    for index, schema_reference in enumerate(
        ("https://example.invalid/schema.json", "../schema.json")
    ):
        contract = _read_basic_contract()
        contract["$schema"] = schema_reference
        contract_path = tmp_path / f"invalid-schema-reference-{index}.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")

        report = run_profile(contract_path, "dev")

        assert report.outcome is Outcome.FAIL
        assert "'$schema' must" in report.results[0].message


def test_non_finite_json_number_is_rejected(tmp_path: Path) -> None:
    contract_path = tmp_path / "non-finite.json"
    contract_path.write_text('{"value": NaN}', encoding="utf-8")

    report = run_profile(contract_path, "dev")

    assert report.outcome is Outcome.FAIL
    assert "Non-finite JSON number" in report.results[0].message


def test_non_utf8_contract_is_a_failure(tmp_path: Path) -> None:
    contract_path = tmp_path / "invalid-encoding.json"
    contract_path.write_bytes(b"\xff\xfe")

    report = run_profile(contract_path, "dev")

    assert report.outcome is Outcome.FAIL
    assert "Unable to read contract" in report.results[0].message


def test_unsupported_profile_returns_failure() -> None:
    report = run_profile(BASIC_CONTRACT_PATH, "release")

    assert report.outcome is Outcome.FAIL
    assert report.outcome.exit_code == 1
    assert report.results[-1].validator_id == "profile-selection"


def test_failure_takes_precedence_over_infrastructure_invalid() -> None:
    results = (
        ValidationResult("dependency", Outcome.INFRA_INVALID, "missing"),
        ValidationResult("contract", Outcome.FAIL, "invalid"),
    )

    assert aggregate_outcomes(results) is Outcome.FAIL


def test_cli_prints_machine_readable_report(capsys) -> None:
    exit_code = main(
        ["run", "--contract", str(BASIC_CONTRACT_PATH), "--profile", "dev"]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["outcome"] == "PASS"
    assert output["contract_id"] == "quickdraw.vizdoom-basic.v1"


def test_integration_profile_collects_and_validates_canonical_trace() -> None:
    report = run_profile(BASIC_CONTRACT_PATH, "integration")

    assert report.outcome is Outcome.PASS
    trace_path = REPOSITORY_ROOT / "artifacts" / "quickdraw" / "basic-v1.trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["process_launches"] == 1
    assert [episode["name"] for episode in trace["episodes"]] == [
        "terminal",
        "truncation",
    ]
