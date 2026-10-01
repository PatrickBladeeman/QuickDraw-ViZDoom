"""Exploratory rule/random/LLM collection teachers for two-goal Basic."""

from __future__ import annotations

import argparse
import json
import math
import operator
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

from quickdraw_vizdoom.goal_conditioned_report import (
    EVALUATION_SEEDS,
    run_goal_conditioned_report,
)
from quickdraw_vizdoom.learning import BasicGoal


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "qwen/qwen3.5-9b"
PROMPT = (
    "Choose a desired outcome for a low-level FPS policy. The episode objective "
    "is to hit the target with few decisions. ALIGN_WITHOUT_FIRE moves or holds "
    "position until position_slot equals target_slot without shooting. "
    "HIT_TARGET permits movement and shooting to hit the target. Goals persist "
    "until achieved or the episode ends. Return only JSON with exactly one key: "
    '{"goal":"ALIGN_WITHOUT_FIRE"} or {"goal":"HIT_TARGET"}.'
)


def teacher_summary(info):
    """The identical privileged symbolic input used by every teacher."""
    state = {
        "position_slot": operator.index(info["position_slot"]),
        "target_slot": operator.index(info["target_slot"]),
        "decision": operator.index(info.get("decision", 0)),
        "remaining_ammunition": operator.index(info.get("remaining_ammunition", 300)),
    }
    if (
        not -4 <= state["position_slot"] <= 4
        or not -4 <= state["target_slot"] <= 4
        or not 0 <= state["decision"] <= 300
        or not 0 <= state["remaining_ammunition"] <= 300
    ):
        raise ValueError("Teacher state is outside the Basic contract.")
    return state


def make_teacher(kind, *, url=None, model=None, timeout=10.0, audit=None):
    """Return a goal selector; failed LLM requests never fall back to a rule."""
    if kind not in ("rule", "random", "llm"):
        raise ValueError("Teacher must be rule, random, or llm.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Teacher timeout must be finite and positive.")
    if kind == "llm":
        parsed = urllib.parse.urlsplit(url or "")
        local = parsed.scheme in ("http", "https") and parsed.hostname in (
            "localhost",
            "127.0.0.1",
            "::1",
        )
        if (
            not (local or url == OPENROUTER_URL)
            or parsed.username is not None
            or parsed.password is not None
            or not model
        ):
            raise ValueError(
                "LLM teacher requires a loopback or OpenRouter URL and model ID."
            )
        if url == OPENROUTER_URL:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ValueError(
                    "Set OPENROUTER_API_KEY in the environment before using OpenRouter."
                )
            if not all(33 <= ord(character) <= 126 for character in key):
                raise ValueError(
                    "OPENROUTER_API_KEY must be visible ASCII without spaces or "
                    "line breaks; paste only the API key."
                )

    def select(info, rng):
        state = teacher_summary(info)
        started = time.perf_counter()
        record = {"teacher": kind, "state": state}
        try:
            if kind == "rule":
                goal = (
                    BasicGoal.ALIGN_WITHOUT_FIRE
                    if state["position_slot"] != state["target_slot"]
                    else BasicGoal.HIT_TARGET
                )
            elif kind == "random":
                goal = BasicGoal(int(rng.integers(len(BasicGoal))))
            else:
                request = {
                    "model": model,
                    "temperature": 0,
                    "max_tokens": 64,
                    "messages": [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": json.dumps(state, sort_keys=True)},
                    ],
                }
                if url == OPENROUTER_URL:
                    request["reasoning"] = {"enabled": False}
                    request["response_format"] = {"type": "json_object"}
                record["request"] = request
                headers = {"Content-Type": "application/json"}
                if url == OPENROUTER_URL:
                    headers["Authorization"] = (
                        "Bearer " + os.environ["OPENROUTER_API_KEY"]
                    )
                boundary = urllib.request.Request(
                    url,
                    data=json.dumps(request).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(boundary, timeout=timeout) as response:
                    raw = response.read(65537)
                if len(raw) > 65536:
                    raise ValueError("LLM response exceeds 64 KiB.")
                record["raw_response"] = raw.decode("utf-8")
                payload = json.loads(record["raw_response"])
                record["response_model"] = payload.get("model")
                record["usage"] = payload.get("usage")
                content = json.loads(payload["choices"][0]["message"]["content"])
                if not isinstance(content, dict) or set(content) != {"goal"}:
                    raise ValueError("LLM must return exactly one JSON goal field.")
                goal = BasicGoal[content["goal"]]
            record["goal"] = goal.name
            return goal
        except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise ValueError(f"{kind} teacher failed: {error}") from error
        finally:
            record["latency_seconds"] = time.perf_counter() - started
            if audit is not None:
                audit.append(record)

    return select


def run_comparison(
    *,
    teachers=("rule", "random"),
    evaluation_episodes=100,
    llm_url=None,
    llm_model=None,
    timeout=10.0,
    output_path=None,
):
    """Train paired seeds per teacher, then evaluate both fixed requested goals."""
    if not 1 <= evaluation_episodes <= len(EVALUATION_SEEDS):
        raise ValueError("Evaluation episodes must be between 1 and 100.")
    if not teachers or len(set(teachers)) != len(teachers):
        raise ValueError("Teachers must be nonempty and unique.")
    audits = {kind: [] for kind in teachers}
    selectors = {
        kind: make_teacher(
            kind, url=llm_url, model=llm_model, timeout=timeout, audit=audits[kind]
        )
        for kind in teachers
    }
    path = None if output_path is None else Path(output_path)
    if path is not None:
        if path.exists():
            raise FileExistsError(
                f"Preserve the earlier result; use a new output path: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": "quickdraw.goal-teacher-comparison.v1",
        "scope": "exploratory two-goal Basic collection-teacher comparison",
        "design": "Same initial seeds, training budget, rewards and exhaustive two-goal "
        "relabeling; teacher selects behavior goals only at boundaries. "
        "Frozen evaluation requests each goal directly, without a teacher.",
        "teacher_input": "privileged symbolic state, identical fields for all teachers",
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "llm_prompt": PROMPT if "llm" in teachers else None,
        "llm_endpoint": llm_url if "llm" in teachers else None,
        "conditions": {},
        "teacher_audit": audits,
    }
    for kind, selector in selectors.items():
        print(f"Training/evaluating {kind} teacher...", flush=True)
        try:
            result = run_goal_conditioned_report(
                evaluation_seeds=EVALUATION_SEEDS[:evaluation_episodes],
                behavior_goal_selector=selector,
            )
            report["conditions"][kind] = {"status": "completed", "result": result}
        except Exception as error:
            report["conditions"][kind] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        if path is not None:
            path.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teachers",
        nargs="+",
        choices=("rule", "random", "llm"),
        default=["rule", "random"],
    )
    parser.add_argument("--llm-url", default=OPENROUTER_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = run_comparison(
        teachers=tuple(args.teachers),
        evaluation_episodes=args.evaluation_episodes,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        timeout=args.timeout,
        output_path=args.output,
    )
    for kind, condition in report["conditions"].items():
        if condition["status"] == "failed":
            print(f"{kind}: FAILED {condition['error']}")
        else:
            aggregate = condition["result"]["aggregate"]
            print(
                f"{kind}: correct-goal success={aggregate['correct_goal_success_rate']:.1%}, "
                f"swapped-goal success={aggregate['swapped_goal_success_rate']:.1%}"
            )
    return int(any(c["status"] == "failed" for c in report["conditions"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
