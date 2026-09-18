"""Strict loading, JSON Schema validation, and hashing for QA contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ContractError(ValueError):
    """Raised when a local contract or schema is malformed or incompatible."""


class ContractInfrastructureError(RuntimeError):
    """Raised when a required validation dependency is unavailable."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ContractError(f"Non-finite JSON number is forbidden: {value}.")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"Unable to read {label} {path}: {error}.") from error

    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as error:
        raise ContractError(f"Malformed {label} {path}: {error}.") from error

    if not isinstance(value, dict):
        raise ContractError(f"{label.capitalize()} {path} must contain a JSON object.")
    return value


def canonical_sha256(value: Any) -> str:
    """Hash parsed canonical JSON rather than formatting-dependent file bytes."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_schema_path(contract_path: Path, schema_reference: object) -> Path:
    if not isinstance(schema_reference, str) or not schema_reference:
        raise ContractError("Contract field '$schema' must be a non-empty string.")
    if urlparse(schema_reference).scheme:
        raise ContractError("Contract '$schema' must reference a local schema file.")

    contract_directory = contract_path.parent.resolve()
    schema_path = (contract_directory / schema_reference).resolve()
    try:
        schema_path.relative_to(contract_directory)
    except ValueError as error:
        raise ContractError(
            "Contract '$schema' must remain within its contract directory."
        ) from error
    return schema_path


def _validate_json_schema(contract: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        from jsonschema import validate
        from jsonschema.exceptions import SchemaError, ValidationError
    except ModuleNotFoundError as error:
        raise ContractInfrastructureError(
            "JSON Schema validation requires requirements-quickdraw.txt."
        ) from error

    try:
        validate(instance=contract, schema=schema)
    except SchemaError as error:
        raise ContractError(
            f"The contract schema is invalid: {error.message}."
        ) from error
    except ValidationError as error:
        raise ContractError(
            f"JSON Schema validation failed at {error.json_path}: {error.message}"
        ) from error


@dataclass(frozen=True)
class LoadedContract:
    """A schema-validated contract with stable identity."""

    path: Path
    data: dict[str, Any]
    sha256: str


def load_contract(path: str | Path) -> LoadedContract:
    """Load one local contract, validate its schema, and compute its hash."""

    contract_path = Path(path).resolve()
    contract = _load_json_object(contract_path, "contract")
    schema_path = _resolve_schema_path(contract_path, contract.get("$schema"))
    schema = _load_json_object(schema_path, "contract schema")
    _validate_json_schema(contract, schema)
    return LoadedContract(
        path=contract_path,
        data=contract,
        sha256=canonical_sha256(contract),
    )
