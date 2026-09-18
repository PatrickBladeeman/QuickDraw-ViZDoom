"""One-process Basic v1 trace collection for the integration QA profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from quickdraw_vizdoom.envs.basic import BasicV1Env


def _repository_root(contract_path: Path) -> Path:
    return contract_path.resolve().parent.parent


def resolve_contract_path(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return (
        path if path.is_absolute() else _repository_root(contract_path) / path
    ).resolve()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation_summary(observation: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(observation.shape),
        "dtype": str(observation.dtype),
        "minimum": float(observation.min()),
        "maximum": float(observation.max()),
        "nonzero_count": int(np.count_nonzero(observation)),
        "frame_order": "oldest_to_newest",
        "frame_hashes": [
            hashlib.sha256(observation[:, :, index].tobytes()).hexdigest()
            for index in range(observation.shape[2])
        ],
    }


def _step_record(
    observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "observation": _observation_summary(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }


def _episode(
    name: str,
    env: BasicV1Env,
    seed: int,
    actions: list[tuple[int, int]],
    max_steps: int | None = None,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    reset_observation = observation.copy()
    steps: list[dict[str, Any]] = []
    action_iterable = actions if max_steps is None else actions[:max_steps]
    for action in action_iterable:
        observation, reward, terminated, truncated, step_info = env.step(action)
        steps.append(
            _step_record(observation, reward, terminated, truncated, step_info)
        )
        if terminated or truncated:
            break
    return {
        "name": name,
        "seed": seed,
        "reset": {
            "observation": _observation_summary(reset_observation),
            "info": info,
        },
        "steps": steps,
    }


def collect_canonical_trace(
    contract_path: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Collect terminal and 300-decision boundary episodes in one game process."""

    config_path = resolve_contract_path(
        contract_path, contract["environment"]["scenario"]["config_path"]
    )
    wad_path = resolve_contract_path(
        contract_path, contract["environment"]["scenario"]["wad_path"]
    )
    if not config_path.is_file() or not wad_path.is_file():
        raise OSError(f"Scenario artifact missing: {config_path} or {wad_path}.")

    seed = int(contract["reset"]["scenario_seed"])
    env = BasicV1Env(config_path)
    try:
        terminal = _episode(
            "terminal",
            env,
            seed,
            [(2, 0), (0, 1)],
        )
        truncation = _episode(
            "truncation",
            env,
            seed,
            [(0, 0)] * int(contract["timing"]["decision_limit"]),
        )
    finally:
        env.close()

    trace = {
        "schema_version": "quickdraw.vizdoom-basic-trace.v1",
        "collector_version": "quickdraw-basic-collector.v1",
        "contract_id": contract["contract_id"],
        "process_launches": 1,
        "environment_artifacts": {
            "config_sha256": sha256_file(config_path),
            "wad_sha256": sha256_file(wad_path),
        },
        "episodes": [terminal, truncation],
    }
    trace_path = resolve_contract_path(
        contract_path, contract["qa"]["canonical_trace_path"]
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trace, trace_path
