"""Paired structured-state goal advising in a native resource-ordering arena."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import time
from collections import deque
from pathlib import Path

from quickdraw_vizdoom.envs.resource_arena import TARGET_IDS, ResourceArena
from quickdraw_vizdoom.goal_teachers import (
    DEFAULT_LLM_MODEL,
    OPENROUTER_URL,
    request_json,
    validate_llm,
)


PROMPT = (
    "Select the next goal in a fully observed Doom mission. Return exactly "
    '{"choice":0}, replacing 0 with the zero-based index of the chosen object '
    "in available_goals. The response schema enforces this numeric selection. "
    "Every required target must be eliminated; "
    "optional targets need not be killed. Minimize goals, then movement. The shared "
    "reliable executor aligns with the named target and fires exactly one round. "
    "Each target needs one round. Eliminating cache.requires_eliminated spawns "
    "a native ammo pickup at that target's position. COLLECT_AMMO walks to that "
    "pickup and back to x=0, increasing ammo by ammo_gain. ELIMINATE ends at x=0 "
    "and the selected target's y. Goals persist until completion or failure. "
    "Conserve enough ammunition to unlock the cache when required; spending the "
    "last round on the wrong target can make the mission impossible. Replan after "
    "each goal. FINISH ends the mission and only succeeds if all required targets "
    "are eliminated. No natural-language parsing is needed: use the state table."
)
ARMS = ("rule", "search", "nearest_no_adviser", "random", "llm")


def complete(state):
    return all(
        not t["alive"]
        for t in state["targets"]
        if t["id"] in state["objective"]["required_targets"]
    )


def available_goals(state):
    goals = [
        {"goal": "ELIMINATE", "target_id": t["id"]}
        for t in state["targets"]
        if t["alive"] and state["ammo"] > 0
    ]
    if state["cache"]["available"]:
        goals.append({"goal": "COLLECT_AMMO"})
    return goals + [{"goal": "FINISH"}]


def adviser_state(state):
    result = copy.deepcopy(state)
    result["available_goals"] = available_goals(state)
    return result


def rule_goal(state):
    """Reactive sufficiency/collection/prerequisite rules, with distance ties."""
    remaining = [
        t
        for t in state["targets"]
        if t["alive"] and t["id"] in state["objective"]["required_targets"]
    ]
    if not remaining:
        return {"goal": "FINISH"}
    if state["ammo"] < len(remaining):
        if state["cache"]["available"]:
            return {"goal": "COLLECT_AMMO"}
        gate = next(
            t
            for t in state["targets"]
            if t["id"] == state["cache"]["requires_eliminated"]
        )
        if gate["alive"] and state["ammo"] > 0:
            return {"goal": "ELIMINATE", "target_id": gate["id"]}
    if state["ammo"] <= 0:
        return {"goal": "FINISH"}
    target = min(
        remaining,
        key=lambda t: (abs(t["position"][1] - state["player_position"][1]), t["id"]),
    )
    return {"goal": "ELIMINATE", "target_id": target["id"]}


def reference_plan(state):
    """Enumerate at most three kills and one pickup; no hidden case data."""
    plans = []
    pending = deque([(copy.deepcopy(state), [], 0.0)])
    while pending:
        current, plan, distance = pending.popleft()
        if complete(current):
            plans.append((len(plan), distance, json.dumps(plan, sort_keys=True), plan))
            continue
        for goal in available_goals(current):
            if goal["goal"] == "FINISH":
                continue
            successor = copy.deepcopy(current)
            y = current["player_position"][1]
            if goal["goal"] == "COLLECT_AMMO":
                x2, y2 = current["cache"]["position"]
                travel = abs(y2 - y) + abs(x2 - current["player_position"][0]) + abs(x2)
                successor["ammo"] += current["cache"]["ammo_gain"]
                successor["cache"].update(
                    available=False, collected=True, position=None
                )
            else:
                target = next(
                    t for t in successor["targets"] if t["id"] == goal["target_id"]
                )
                x2, y2 = target["position"]
                travel = abs(y2 - y) + abs(current["player_position"][0])
                target["alive"] = False
                successor["ammo"] -= 1
                if target["id"] == current["cache"]["requires_eliminated"]:
                    successor["cache"].update(available=True, position=[x2, y2])
            successor["player_position"] = [0.0, y2]
            pending.append((successor, plan + [goal], distance + travel))
    if not plans:
        return None
    best = min(plans)
    return {"goals": best[3], "goal_count": best[0], "ideal_movement_units": best[1]}


def generate_cases(split, count):
    """Balanced, predetermined cases; no outcome-based filtering."""
    if (
        split not in ("development", "heldout", "confirmation")
        or not 3 <= count <= 300
        or count % 3
    ):
        raise ValueError(
            "Use development/heldout/confirmation and 3..300 cases divisible by three."
        )
    cases = []
    # Different position grids prevent duplicate development/evaluation layouts.
    grid = {
        "development": (-160, -96, -32, 32, 96, 160),
        "heldout": (-144, -80, -16, 48, 112, 176),
        "confirmation": (-176, -112, -48, 16, 80, 144),
    }[split]
    for i in range(count):
        seed = {"development": 71000, "heldout": 72000, "confirmation": 74000}[
            split
        ] + i
        rng = random.Random(seed)
        ys = rng.sample(grid, 3)
        gate = rng.choice(TARGET_IDS)
        stratum = ("direct", "required_gate", "optional_gate")[i % 3]
        if stratum == "direct":
            required = rng.sample(TARGET_IDS, rng.randint(1, 3))
            ammo = len(required)
        elif stratum == "required_gate":
            required = [gate, rng.choice([t for t in TARGET_IDS if t != gate])]
            ammo = 1
        else:
            required = [t for t in TARGET_IDS if t != gate]
            ammo = 1
        cases.append(
            {
                "seed": seed,
                "split": split,
                "stratum": stratum,
                "initial_ammo": ammo,
                "cache_ammo": rng.choice((2, 3)),
                "cache_requires": gate,
                "required_targets": sorted(required),
                "targets": [{"id": t, "y": y} for t, y in zip(TARGET_IDS, ys)],
            }
        )
    return cases


def select_goal(kind, state, rng, *, model, url, timeout, audit):
    if kind == "rule":
        return rule_goal(state)
    if kind == "search":
        reference = reference_plan(state)
        return (
            reference["goals"][0]
            if reference and reference["goals"]
            else {"goal": "FINISH"}
        )
    if kind == "nearest_no_adviser":
        targets = [t for t in state["targets"] if t["alive"]]
        if not targets or state["ammo"] <= 0:
            return {"goal": "FINISH"}
        return {
            "goal": "ELIMINATE",
            "target_id": min(
                targets,
                key=lambda t: (
                    abs(t["position"][1] - state["player_position"][1]),
                    t["id"],
                ),
            )["id"],
        }
    if kind == "random":
        choices = [g for g in state["available_goals"] if g["goal"] != "FINISH"]
        return rng.choice(choices) if choices else {"goal": "FINISH"}
    record = {"state": state}
    started = time.perf_counter()
    try:
        choices = state["available_goals"]
        content = request_json(
            PROMPT,
            state,
            url=url,
            model=model,
            timeout=timeout,
            record=record,
            response_schema={
                "type": "object",
                "properties": {
                    "choice": {"type": "integer", "enum": list(range(len(choices)))}
                },
                "required": ["choice"],
                "additionalProperties": False,
            },
        )
        if (
            not isinstance(content, dict)
            or set(content) != {"choice"}
            or type(content["choice"]) is not int
            or not 0 <= content["choice"] < len(choices)
        ):
            raise ValueError("LLM output must select one available goal index.")
        goal = choices[content["choice"]]
        record["goal"] = goal
        return goal
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["latency_seconds"] = time.perf_counter() - started
        audit.append(record)


def episode(case, kind, *, model, url, timeout):
    audit, trace = [], []
    result = {
        "seed": case["seed"],
        "stratum": case["stratum"],
        "teacher": kind,
        "success": False,
        "adviser_audit": audit,
        "trace": trace,
    }
    try:
        with ResourceArena(case) as env:
            result.update(
                wad_sha256=env.wad_sha256,
                initial_frame_sha256=env.initial_frame_sha256,
                reference=reference_plan(env.state),
            )
            if result["reference"] is None:
                raise RuntimeError("Preregistered case is not reference-solvable.")
            rng = random.Random(case["seed"])
            for _ in range(8):
                state = adviser_state(env.state)
                if complete(state):
                    result["success"] = True
                    break
                if state["ammo"] <= 0 and not state["cache"]["available"]:
                    result["failure"] = "ammo_deadlock"
                    break
                try:
                    goal = select_goal(
                        kind,
                        state,
                        rng,
                        model=model,
                        url=url,
                        timeout=timeout,
                        audit=audit,
                    )
                except Exception as error:
                    result.update(
                        failure="adviser_failure",
                        error=f"{type(error).__name__}: {error}",
                    )
                    break
                start = env.decisions
                event_start = len(env.events)
                status = env.execute(goal)
                trace.append(
                    {
                        "state": state,
                        "goal": goal,
                        "status": status,
                        "decisions": env.decisions - start,
                        "native_events": env.events[event_start:],
                        "next_state": adviser_state(env.state),
                    }
                )
                if status == "finish":
                    result["failure"] = "premature_finish"
                    break
                if status != "achieved":
                    result["failure"] = "executor_" + status
                    break
            result["success"] = complete(env.state)
            if not result["success"] and "failure" not in result:
                result["failure"] = "goal_limit"
            result.update(
                decisions=env.decisions,
                goal_count=len(trace),
                native_events=env.events,
                final_state=env.state,
                final_frame_sha256=env.frame_hash(),
            )
    except Exception as error:
        result.update(
            failure="infrastructure_failure", error=f"{type(error).__name__}: {error}"
        )
    return result


def paired_interval(a, b):
    """Paired case bootstrap, including errors as failures; descriptive pilot CI."""
    deltas = [int(x["success"]) - int(y["success"]) for x, y in zip(a, b)]
    rng = random.Random(73000)
    draws = sorted(
        statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(5000)
    )
    return {
        "difference": statistics.mean(deltas),
        "paired_case_bootstrap_95_percent": [draws[125], draws[4874]],
        "llm_only_successes": deltas.count(1),
        "baseline_only_successes": deltas.count(-1),
    }


def summarize(rows):
    audits = [item for row in rows for item in row["adviser_audit"]]
    costs = [
        a.get("usage", {}).get("cost")
        for a in audits
        if isinstance(a.get("usage"), dict)
    ]
    shots = sum(
        max(0, -event["ammo_delta"])
        for row in rows
        for event in row.get("native_events", [])
    )
    kill_goals = [
        step
        for row in rows
        for step in row["trace"]
        if step["goal"]["goal"] == "ELIMINATE"
    ]
    return {
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_rate": statistics.mean(r["success"] for r in rows),
        "mean_decisions_all_episodes": statistics.mean(
            r.get("decisions", 0) for r in rows
        ),
        "mean_goals": statistics.mean(r.get("goal_count", 0) for r in rows),
        "shots_fired": shots,
        "selected_target_kill_successes": sum(
            step["status"] == "achieved" for step in kill_goals
        ),
        "selected_target_kill_attempts": len(kill_goals),
        "wrong_target_kills": sum(
            target != step["goal"]["target_id"]
            for step in kill_goals
            for event in step["native_events"]
            for target in event["killed"]
        ),
        "native_pickups": sum(
            event["ammo_delta"] > 0
            for row in rows
            for event in row.get("native_events", [])
        ),
        "ammo_deadlocks": sum(r.get("failure") == "ammo_deadlock" for r in rows),
        "adviser_failures": sum(r.get("failure") == "adviser_failure" for r in rows),
        "executor_failures": sum(
            r.get("failure", "").startswith("executor_") for r in rows
        ),
        "infrastructure_failures": sum(
            r.get("failure") == "infrastructure_failure" for r in rows
        ),
        "api_calls": len(audits),
        "api_latency_seconds": sum(a["latency_seconds"] for a in audits),
        "reported_cost_usd": sum(c for c in costs if isinstance(c, (int, float))),
        "cost_reporting_calls": sum(isinstance(c, (int, float)) for c in costs),
    }


def run(
    *,
    split="heldout",
    count=60,
    arms=("rule", "search", "nearest_no_adviser", "random"),
    model=DEFAULT_LLM_MODEL,
    url=OPENROUTER_URL,
    timeout=10.0,
    output,
):
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("Choose unique, supported adviser arms.")
    if "llm" in arms:
        validate_llm(url, model, timeout)
    path = Path(output)
    if path.exists():
        raise FileExistsError("Preserve previous results: choose a new output path.")
    cases = generate_cases(split, count)
    sources = [
        Path(__file__),
        Path(__file__).parent / "envs/resource_arena.py",
        Path(__file__).parent / "goal_teachers.py",
    ]
    report = {
        "report_version": "quickdraw.resource-adviser.v3",
        "split": split,
        "cases": cases,
        "case_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "scope": "Adviser-only native resource ordering; shared scripted skills, no policy learning or language parsing.",
        "reference_cost": "Minimum goal count, then ideal axis-aligned movement units; not an exact native decision optimum.",
        "llm_model": model if "llm" in arms else None,
        "llm_endpoint": url if "llm" in arms else None,
        "prompt": PROMPT if "llm" in arms else None,
        "timeout_seconds": timeout,
        "goal_limit": 8,
        "skill_decision_limit": 600,
        "decision_tics": 1,
        "conditions": {},
        "status": "running",
    }
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save()  # Design, cases and hashes are recorded before any provider call.
    for arm in arms:
        rows = []
        report["conditions"][arm] = {"episodes": rows}
        for i, case in enumerate(cases):
            rows.append(episode(case, arm, model=model, url=url, timeout=timeout))
            if (i + 1) % 10 == 0 or i + 1 == len(cases):
                print(
                    f"{arm}: {i + 1}/{len(cases)}, {sum(r['success'] for r in rows)} successes",
                    flush=True,
                )
                save()
        report["conditions"][arm]["aggregate"] = summarize(rows)
        report["conditions"][arm]["strata"] = {
            s: summarize([r for r in rows if r["stratum"] == s])
            for s in ("direct", "required_gate", "optional_gate")
        }
        save()
    if "llm" in arms:
        report["paired_success_comparisons"] = {
            a: paired_interval(
                report["conditions"]["llm"]["episodes"],
                report["conditions"][a]["episodes"],
            )
            for a in arms
            if a != "llm"
        }
    # Failure rates accompany costs: early deadlocks must not look like efficient completion.
    if "search" in arms:
        reference_rows = report["conditions"]["search"]["episodes"]
        for arm in arms:
            paired = [
                (a, b)
                for a, b in zip(report["conditions"][arm]["episodes"], reference_rows)
                if a["success"] and b["success"]
            ]
            report["conditions"][arm]["successful_pair_cost"] = {
                "pairs": len(paired),
                "mean_extra_native_decisions_vs_search": (
                    statistics.mean(a["decisions"] - b["decisions"] for a, b in paired)
                    if paired
                    else None
                ),
            }
    report["source_unchanged"] = all(
        hashlib.sha256(p.read_bytes()).hexdigest()
        == report["source_sha256"][str(p.relative_to(Path(__file__).parents[1]))]
        for p in sources
    )
    report["status"] = (
        "completed" if report["source_unchanged"] else "invalid_source_changed"
    )
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("development", "heldout", "confirmation"), default="heldout"
    )
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:-1]))
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-url", default=OPENROUTER_URL)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        split=args.split,
        count=args.cases,
        arms=tuple(args.arms),
        model=args.llm_model,
        url=args.llm_url,
        timeout=args.timeout,
        output=args.output,
    )
    return int(
        report["status"] != "completed"
        or any(
            c["aggregate"]["infrastructure_failures"]
            or c["aggregate"]["executor_failures"]
            for c in report["conditions"].values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
