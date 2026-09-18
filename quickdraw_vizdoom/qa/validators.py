"""Composable, side-effect-free validators for the Basic environment contract."""

from __future__ import annotations

import math
from typing import Any

from .results import Outcome, ValidationResult


ContractData = dict[str, Any]


def _result(
    validator_id: str,
    errors: list[str],
) -> ValidationResult:
    return ValidationResult(
        validator_id,
        Outcome.FAIL if errors else Outcome.PASS,
        "; ".join(errors) if errors else "Contract invariant satisfied.",
    )


def validate_basic_environment(contract: ContractData) -> ValidationResult:
    """Validate Basic identity and scenario artifact registration state."""

    environment = contract["environment"]
    scenario = environment["scenario"]
    errors: list[str] = []
    if contract["contract_id"] != "quickdraw.vizdoom-basic.v1":
        errors.append("The Basic v1 contract has an unexpected identifier.")
    if environment["name"] != "QuickDrawVizDoomBasic":
        errors.append("The Basic environment has an unexpected name.")
    if environment["kind"] != "basic":
        errors.append("The Basic environment must declare kind 'basic'.")

    hashes = (scenario["config_sha256"], scenario["wad_sha256"])
    if scenario["artifact_status"] == "available" and any(
        value is None for value in hashes
    ):
        errors.append("Available scenario artifacts require both SHA-256 values.")
    if scenario["artifact_status"] == "planned" and any(
        value is not None for value in hashes
    ):
        errors.append("Planned scenario artifacts must not claim SHA-256 values.")
    return _result("basic-environment", errors)


def validate_basic_observation(contract: ContractData) -> ValidationResult:
    """Validate screen preprocessing and frame-stack relationships."""

    observation = contract["observation"]
    errors: list[str] = []
    source = observation["source"]
    crop = observation["preprocessing"]["crop"]
    if crop["width"] > source["width"] or crop["height"] > source["height"]:
        errors.append("Crop dimensions exceed the source frame.")

    resize = observation["preprocessing"]["resize"]
    stack_count = observation["stack"]["count"]
    expected_shape = [resize["height"], resize["width"], stack_count]
    if observation["output"]["shape"] != expected_shape:
        errors.append("Output shape does not match resize dimensions and stack count.")
    normalization = observation["preprocessing"]["normalization"]
    if normalization["minimum"] >= normalization["maximum"]:
        errors.append("Normalization minimum must be less than maximum.")

    return _result("basic-observation", errors)


def validate_basic_actions(contract: ContractData) -> ValidationResult:
    """Validate branch indices, buttons, and joint-action encoding."""

    actions = contract["actions"]
    errors: list[str] = []
    branch_names = [branch["name"] for branch in actions["branches"]]
    if len(branch_names) != len(set(branch_names)):
        errors.append("Action branch names must be unique.")

    for branch in actions["branches"]:
        indices = [action["index"] for action in branch["actions"]]
        if indices != list(range(len(branch["actions"]))):
            errors.append(f"Branch {branch['name']!r} indices are not contiguous.")

    if actions["decision_strobe"] not in actions["button_order"]:
        errors.append("Decision strobe is absent from button_order.")
    assigned_buttons = [
        action["button"]
        for branch in actions["branches"]
        for action in branch["actions"]
        if action["button"] is not None
    ]
    if set(assigned_buttons) - set(actions["button_order"]):
        errors.append("A policy action uses a button absent from button_order.")
    if len(assigned_buttons) != len(set(assigned_buttons)):
        errors.append("Policy actions must not share a ViZDoom button.")
    if actions["decision_strobe"] in assigned_buttons:
        errors.append("Decision strobe must not also encode a policy action.")

    if len(actions["branches"]) != 2:
        errors.append("The BDQ action contract requires exactly two branches.")

    return _result("basic-actions", errors)


def validate_basic_timing(contract: ContractData) -> ValidationResult:
    """Validate policy frequency and episode tic arithmetic."""

    timing = contract["timing"]
    errors: list[str] = []
    pattern = timing["policy_interval_pattern_tics"]
    calculated_policy_hz = timing["simulation_hz"] / (sum(pattern) / len(pattern))
    cycles, remainder = divmod(timing["decision_limit"], len(pattern))
    calculated_episode_tics = cycles * sum(pattern) + sum(pattern[:remainder])
    if not math.isclose(
        calculated_policy_hz,
        timing["policy_hz_average"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append("Average policy frequency is inconsistent with the tic pattern.")
    if calculated_episode_tics != timing["episode_tics"]:
        errors.append("Episode tic budget is inconsistent with the decision schedule.")
    return _result("basic-timing", errors)


def validate_basic_reward(contract: ContractData) -> ValidationResult:
    """Validate reward-event uniqueness."""

    reward = contract["reward"]
    errors: list[str] = []
    events = [term["event"] for term in reward["terms"]]
    if len(events) != len(set(events)):
        errors.append("Reward events must be unique.")
    return _result("basic-reward", errors)


def validate_basic_episode(contract: ContractData) -> ValidationResult:
    """Validate terminal, truncation, bootstrap, and infrastructure semantics."""

    episode_end = contract["episode_end"]
    errors: list[str] = []
    groups = (
        episode_end["terminal"],
        episode_end["truncation"],
        episode_end["infrastructure_invalid"],
    )
    reasons = [entry["reason"] for group in groups for entry in group]
    codes = [entry["code"] for group in groups for entry in group]
    if len(reasons) != len(set(reasons)):
        errors.append("Episode-end reasons overlap across classifications.")
    if len(codes) != len(set(codes)):
        errors.append("Episode-end codes must be unique.")
    for truncation in episode_end["truncation"]:
        if (
            truncation["reason"] == "decision_limit"
            and truncation["at_decision"] != contract["timing"]["decision_limit"]
        ):
            errors.append(
                "Decision-limit truncation disagrees with the timing contract."
            )
    return _result("basic-episode", errors)


def validate_basic_reset(contract: ContractData) -> ValidationResult:
    """Validate deterministic reset relationships and target sampling bounds."""

    reset = contract["reset"]
    errors: list[str] = []
    target_slot = reset["target_slot"]
    if target_slot["minimum"] > target_slot["maximum"]:
        errors.append("Target-slot minimum exceeds maximum.")
    return _result("basic-reset", errors)


def validate_basic_state_channel(contract: ContractData) -> ValidationResult:
    """Validate the privileged ACS-to-wrapper state-channel allocation."""

    state_channel = contract["state_channel"]
    errors: list[str] = []
    slots = [variable["slot"] for variable in state_channel["variables"]]
    names = [variable["name"] for variable in state_channel["variables"]]
    if len(slots) != len(set(slots)):
        errors.append("State-channel slots must be unique.")
    if len(names) != len(set(names)):
        errors.append("State-channel variable names must be unique.")
    schema_variables = [
        variable
        for variable in state_channel["variables"]
        if variable["name"] == "schema_id"
    ]
    if schema_variables != [{"slot": "USER1", "name": "schema_id"}]:
        errors.append("USER1 must be the unique state-channel schema_id slot.")
    return _result("basic-state-channel", errors)


def validate_dev_profile(contract: ContractData) -> ValidationResult:
    """Prove that the development profile is static and side-effect free."""

    profile = contract["qa"]["profiles"]["dev"]
    errors: list[str] = []
    if any(profile["budgets"].values()):
        errors.append(
            "The dev profile must have zero runtime and external-call budgets."
        )
    if profile["validators"] != ["json-schema", *VALIDATORS]:
        errors.append("The dev profile validator sequence drifted.")
    return _result("dev-profile", errors)


def validate_integration_profile(contract: ContractData) -> ValidationResult:
    """Validate that integration is the only runtime profile in this slice."""

    profile = contract["qa"]["profiles"]["integration"]
    errors: list[str] = []
    if profile["budgets"] != {
        "process_launches": 1,
        "trace_collections": 1,
        "training_sessions": 0,
        "external_calls": 0,
    }:
        errors.append("The integration profile budget drifted.")
    expected = [
        "json-schema",
        *[validator_id for validator_id in VALIDATORS if validator_id != "dev-profile"],
        "integration-profile",
        "integration-runtime",
        "integration-observation",
        "integration-actions",
        "integration-timing",
        "integration-rewards",
        "integration-endings",
    ]
    if profile["validators"] != expected:
        errors.append("The integration validator sequence drifted.")
    return _result("integration-profile", errors)


def _trace_result(
    validator_id: str,
    errors: list[str],
) -> ValidationResult:
    return _result(validator_id, errors)


def _episodes(trace: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = trace.get("episodes")
    return episodes if isinstance(episodes, list) else []


def validate_integration_runtime(
    contract: ContractData, trace: dict[str, Any]
) -> ValidationResult:
    """Validate one process and the two fixed-seed boundary episodes."""

    errors: list[str] = []
    if trace.get("contract_id") != contract["contract_id"]:
        errors.append("Trace contract identity does not match the contract.")
    if trace.get("process_launches") != 1:
        errors.append("Integration must use exactly one ViZDoom process.")
    scenario = contract["environment"]["scenario"]
    artifacts = trace.get("environment_artifacts", {})
    if scenario["artifact_status"] != "available":
        errors.append("Integration requires available scenario artifacts.")
    if artifacts.get("config_sha256") != scenario["config_sha256"]:
        errors.append("Trace/config SHA-256 does not match the contract.")
    if artifacts.get("wad_sha256") != scenario["wad_sha256"]:
        errors.append("Trace/WAD SHA-256 does not match the contract.")
    episodes = _episodes(trace)
    if [episode.get("name") for episode in episodes] != ["terminal", "truncation"]:
        errors.append(
            "The canonical trace must contain terminal then truncation episodes."
        )
    if any(
        episode.get("seed") != contract["reset"]["scenario_seed"]
        for episode in episodes
    ):
        errors.append("Canonical episodes must use the contract scenario seed.")
    return _trace_result("integration-runtime", errors)


def validate_integration_observation(
    contract: ContractData, trace: dict[str, Any]
) -> ValidationResult:
    """Validate frame metadata, stacking order, range, and reset priming."""

    expected_shape = contract["observation"]["output"]["shape"]
    errors: list[str] = []
    for episode in _episodes(trace):
        reset = episode.get("reset", {})
        reset_observation = reset.get("observation", {})
        if reset_observation.get("shape") != expected_shape:
            errors.append("Reset observation shape is incorrect.")
        if reset_observation.get("dtype") != "float32":
            errors.append("Reset observation dtype is incorrect.")
        reset_hashes = reset_observation.get("frame_hashes", [])
        if len(reset_hashes) != 4 or len(set(reset_hashes)) != 1:
            errors.append(
                "Reset must duplicate the first frame across all four channels."
            )
        for step in episode.get("steps", []):
            observation = step.get("observation", {})
            if observation.get("shape") != expected_shape:
                errors.append("Step observation shape is incorrect.")
            if observation.get("dtype") != "float32":
                errors.append("Step observation dtype is incorrect.")
            if observation.get("frame_order") != "oldest_to_newest":
                errors.append("Frame order is not oldest-to-newest.")
            if not 0.0 <= observation.get("minimum", -1.0) <= 1.0:
                errors.append("Observation minimum is outside [0, 1].")
            if not 0.0 <= observation.get("maximum", 2.0) <= 1.0:
                errors.append("Observation maximum is outside [0, 1].")
            if not observation.get("frame_hashes") or not observation.get(
                "nonzero_count"
            ):
                errors.append(
                    "Every canonical step must retain a nonzero real observation."
                )
    return _trace_result("integration-observation", errors)


def validate_integration_actions(
    contract: ContractData, trace: dict[str, Any]
) -> ValidationResult:
    """Validate semantic actions, button encoding, and branch-local masks."""

    errors: list[str] = []
    button_order = contract["actions"]["button_order"]
    for episode in _episodes(trace):
        for step in episode.get("steps", []):
            info = step.get("info", {})
            action = info.get("branch_action", [])
            if len(action) != 2 or info.get("semantic_action") != [
                contract["actions"]["branches"][0]["actions"][action[0]]["semantic"],
                contract["actions"]["branches"][1]["actions"][action[1]]["semantic"],
            ]:
                errors.append("Semantic branch action does not match its indices.")
                continue
            values = info.get("encoded_button_values", [])
            if len(values) != len(button_order) or values[0] != 1:
                errors.append("Each decision must emit a one-tic decision strobe.")
            expected = [
                1,
                int(action[0] == 1),
                int(action[0] == 2),
                int(action[1] == 1),
            ]
            if values != expected:
                errors.append("Branch action encoding drifted.")
            current = info.get("current_unavailable_masks", [])
            next_masks = info.get("next_unavailable_masks", [])
            if [len(mask) for mask in current] != [3, 2] or [
                len(mask) for mask in next_masks
            ] != [3, 2]:
                errors.append("Branch-local mask sizes are incorrect.")
            if current and any(current[index][action[index]] for index in range(2)):
                errors.append("The trace selected an unavailable action.")
    return _trace_result("integration-actions", errors)


def validate_integration_timing(
    contract: ContractData, trace: dict[str, Any]
) -> ValidationResult:
    """Validate alternating intervals and the exact 300-decision tic budget."""

    errors: list[str] = []
    pattern = contract["timing"]["policy_interval_pattern_tics"]
    for episode in _episodes(trace):
        steps = episode.get("steps", [])
        total_tics = 0
        for index, step in enumerate(steps):
            expected = pattern[index % len(pattern)]
            info = step.get("info", {})
            if info.get("interval_tics") != expected:
                errors.append("Policy interval pattern is not alternating 3/4 tics.")
            total_tics += expected
            if info.get("simulation_tic") != total_tics:
                errors.append("Simulation tic identity drifted.")
        if episode.get("name") == "truncation":
            if len(steps) != contract["timing"]["decision_limit"]:
                errors.append("Truncation episode did not reach 300 decisions.")
            if total_tics != contract["timing"]["episode_tics"]:
                errors.append("Truncation episode did not reach 1050 tics.")
    return _trace_result("integration-timing", errors)


def validate_integration_rewards(
    contract: ContractData, trace: dict[str, Any]
) -> ValidationResult:
    """Validate additive decision, hit, and missed-shot rewards."""

    terms = {term["event"]: term["value"] for term in contract["reward"]["terms"]}
    errors: list[str] = []
    for episode in _episodes(trace):
        for step in episode.get("steps", []):
            events = step.get("info", {}).get("events", [])
            expected = sum(terms[event] for event in events if event in terms)
            if not math.isclose(step.get("reward", 0.0), expected, abs_tol=1e-12):
                errors.append("Step reward does not equal its declared event terms.")
            counters = step.get("info", {}).get("event_counters", {})
            if counters.get("decisions") != step.get("info", {}).get("decision"):
                errors.append("Decision event counter is not cumulative.")
    return _trace_result("integration-rewards", errors)


def validate_integration_endings(
    contract: ContractData, trace: dict[str, Any]
) -> ValidationResult:
    """Validate terminal/truncated exclusivity and preserved final frames."""

    errors: list[str] = []
    for episode in _episodes(trace):
        steps = episode.get("steps", [])
        if not steps:
            errors.append("Canonical episode is empty.")
            continue
        terminal = [step for step in steps if step.get("terminated")]
        truncated = [step for step in steps if step.get("truncated")]
        if episode.get("name") == "terminal":
            if (
                len(terminal) != 1
                or truncated
                or terminal[-1]["info"].get("terminal_reason") != "target_hit"
            ):
                errors.append("Terminal episode outcome is incorrect.")
        if episode.get("name") == "truncation":
            if (
                terminal
                or len(truncated) != 1
                or truncated[-1]["info"].get("truncation_reason")
                != "decision_limit"
            ):
                errors.append("Truncation episode outcome is incorrect.")
        final = steps[-1]
        if not final.get("info", {}).get("real_observation"):
            errors.append(
                "Logical termination did not preserve a real final observation."
            )
        if final.get("observation", {}).get("nonzero_count", 0) <= 0:
            errors.append("Logical termination returned a zero final observation.")
    return _trace_result("integration-endings", errors)


VALIDATORS = {
    "basic-environment": validate_basic_environment,
    "basic-observation": validate_basic_observation,
    "basic-actions": validate_basic_actions,
    "basic-timing": validate_basic_timing,
    "basic-reward": validate_basic_reward,
    "basic-episode": validate_basic_episode,
    "basic-reset": validate_basic_reset,
    "basic-state-channel": validate_basic_state_channel,
    "dev-profile": validate_dev_profile,
}


INTEGRATION_VALIDATORS = {
    "integration-profile": validate_integration_profile,
    "integration-runtime": validate_integration_runtime,
    "integration-observation": validate_integration_observation,
    "integration-actions": validate_integration_actions,
    "integration-timing": validate_integration_timing,
    "integration-rewards": validate_integration_rewards,
    "integration-endings": validate_integration_endings,
}
