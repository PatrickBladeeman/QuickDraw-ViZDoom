"""Basic-arena training with deterministic shuffled reset seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .basic import DEFAULT_CONFIG_PATH as BASIC_CONFIG_PATH
from .basic import BasicV1Env


DEFAULT_CONFIG_PATH = BASIC_CONFIG_PATH


def permuted_episode_seeds(
    start: int, count: int, *, permutation_seed: int | None = None
) -> tuple[int, ...]:
    """Return a deterministic shuffled permutation of a contiguous seed range."""

    if count < 1:
        raise ValueError("count must be positive")
    result = [
        int(seed)
        for seed in np.random.default_rng(
            start if permutation_seed is None else permutation_seed
        ).permutation(np.arange(start, start + count, dtype=np.int64))
    ]
    if count > 2 and all(
        result[index] % 2 != result[index - 1] % 2 for index in range(1, count)
    ):
        result[1], result[2] = result[2], result[1]
    return tuple(result)


def validate_seed_schedule(
    seeds: tuple[int, ...], start: int, *, require_balanced: bool = True
) -> None:
    expected = set(range(start, start + len(seeds)))
    if set(seeds) != expected or len(seeds) != len(expected):
        raise ValueError("Episode seed schedule is not a permutation of its range.")
    if require_balanced and sum(seed % 2 for seed in seeds) != len(seeds) // 2:
        raise ValueError("Episode seed schedule is not class balanced.")
    if len(seeds) > 2 and all(
        seeds[index] % 2 != seeds[index - 1] % 2 for index in range(1, len(seeds))
    ):
        raise ValueError("Episode seed schedule must not alternate classes.")


class MicroLearningGeneralizationV1Env(BasicV1Env):
    """Use the Basic arena and lifecycle, adding split/reset metadata only."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        window_visible: bool = False,
    ) -> None:
        super().__init__(config_path, window_visible=window_visible)
        self.visual_split = "training"
        self.visual_variant = ""
        self._reset_records: dict[str, list[dict[str, Any]]] = {
            "training": [],
            "held-out": [],
        }

    def set_visual_split(self, visual_split: str) -> None:
        if visual_split not in self._reset_records:
            raise ValueError(f"Unknown visual split: {visual_split!r}.")
        self.visual_split = visual_split

    @property
    def reset_records(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {split: tuple(records) for split, records in self._reset_records.items()}

    def reset_frame_hashes(self, visual_split: str | None = None) -> tuple[str, ...]:
        split = self.visual_split if visual_split is None else visual_split
        return tuple(
            record["reset_frame_hash"] for record in self._reset_records[split]
        )

    def reset(
        self, seed: int | None = None, *, visual_split: str | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if visual_split is not None:
            self.set_visual_split(visual_split)
        observation, info = super().reset(seed)
        self.visual_variant = f"target-slot-{self._target_slot}"
        reset_frame_hash = self._frame_hash(observation[:, :, 0])
        self._reset_records[self.visual_split].append(
            {
                "seed": self._seed,
                "visual_split": self.visual_split,
                "visual_variant": self.visual_variant,
                "target_slot": self._target_slot,
                "reset_frame_hash": reset_frame_hash,
            }
        )
        info.update(
            {
                "visual_split": self.visual_split,
                "visual_variant": self.visual_variant,
                "reset_frame_hash": reset_frame_hash,
            }
        )
        return observation, info

    def step(
        self, branch_action: tuple[int, int] | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(branch_action)
        info.update(
            {
                "visual_split": self.visual_split,
                "visual_variant": self.visual_variant,
                "real_final_observation": True,
            }
        )
        return observation, reward, terminated, truncated, info


MicroLearningGeneralizationEnv = MicroLearningGeneralizationV1Env
