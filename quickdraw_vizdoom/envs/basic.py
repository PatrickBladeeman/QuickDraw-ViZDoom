"""The minimal Basic v1 ViZDoom environment."""

from __future__ import annotations

import hashlib
import operator
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "scenarios" / "quickdraw" / "basic-v1.cfg"


def _area_weights(source_size: int, target_size: int) -> np.ndarray:
    weights = np.zeros((target_size, source_size), dtype=np.float32)
    edges = np.linspace(0.0, source_size, target_size + 1)
    for target, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        first = int(np.floor(start))
        last = int(np.ceil(end))
        source = np.arange(first, last)
        overlap = np.minimum(source + 1.0, end) - np.maximum(source, start)
        weights[target, source] = np.maximum(overlap, 0.0) / (end - start)
    return weights


class BasicV1Env:
    """A thin ``DoomGame`` boundary with the Basic v1 contract semantics."""

    interval_pattern = (3, 4)
    branch_sizes = (3, 2)
    semantic_actions = (("Stay", "Left", "Right"), ("Idle", "Shoot"))

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path).resolve()
        self._vzd: Any = None
        self._game: Any = None
        self._raw_frame: np.ndarray | None = None
        self._frames: list[np.ndarray] = []
        self._episode_index = -1
        self._seed = 0
        self._target_slot = 0
        self._position_slot = 0
        self._remaining_ammunition = 300
        self._shots_fired = 0
        self._misses = 0
        self._target_hits = 0
        self._decision_count = 0
        self._simulation_tic = 0
        self._terminated = False
        self._truncated = False

        self._vertical_weights = _area_weights(120, 84)
        self._horizontal_weights = _area_weights(120, 84)

        try:
            from gymnasium import spaces

            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(84, 84, 4),
                dtype=np.float32,
            )
            self.action_space = spaces.MultiDiscrete(self.branch_sizes)
        except ImportError:  # pragma: no cover - ViZDoom installs Gymnasium.
            self.observation_space = None
            self.action_space = None

    def _load_game(self, seed: int) -> None:
        try:
            import vizdoom as vzd
        except ImportError as error:  # pragma: no cover - exercised by QA.
            raise RuntimeError(
                "ViZDoom is required for the integration profile."
            ) from error

        game = vzd.DoomGame()
        game.load_config(str(self.config_path))
        game.set_window_visible(False)
        game.set_seed(seed)
        game.init()
        self._vzd = vzd
        self._game = game

    @staticmethod
    def _copy_screen(state: Any) -> np.ndarray:
        return np.asarray(state.screen_buffer, dtype=np.uint8).copy()

    def _capture_frame(self) -> np.ndarray:
        state = self._game.get_state()
        if state is not None:
            self._raw_frame = self._copy_screen(state)
        if self._raw_frame is None:
            raise RuntimeError("ViZDoom returned no real frame.")
        return self._raw_frame.copy()

    def _preprocess(self, raw_frame: np.ndarray) -> np.ndarray:
        expected_height, expected_width = 120, 160
        if raw_frame.shape != (expected_height, expected_width):
            raise RuntimeError(
                f"Expected GRAY8 frame {(expected_height, expected_width)}, "
                f"got {raw_frame.shape}."
            )
        cropped = raw_frame[:, 20:140]
        resized = self._vertical_weights @ cropped @ self._horizontal_weights.T
        return (resized / 255.0).astype(np.float32)

    @staticmethod
    def _frame_hash(frame: np.ndarray) -> str:
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def _current_masks(self) -> tuple[np.ndarray, np.ndarray]:
        movement = np.array(
            [False, self._position_slot <= -4, self._position_slot >= 4],
            dtype=bool,
        )
        combat = np.array(
            [False, self._remaining_ammunition <= 0],
            dtype=bool,
        )
        return movement, combat

    @staticmethod
    def _mask_dict(masks: tuple[np.ndarray, np.ndarray]) -> list[list[bool]]:
        return [mask.tolist() for mask in masks]

    def _state_channel(self, terminal_reason: int = 0) -> dict[str, int]:
        movement, combat = self._current_masks()
        return {
            "USER1": 1001,
            "USER2": self._episode_index,
            "USER3": self._decision_count,
            "USER4": self._position_slot,
            "USER5": self._target_slot,
            "USER6": sum(int(value) << index for index, value in enumerate(movement)),
            "USER7": sum(int(value) << index for index, value in enumerate(combat)),
            "USER8": terminal_reason,
            "USER9": self._shots_fired,
            "USER10": self._misses,
            "USER11": self._remaining_ammunition,
            "USER12": self._decision_count,
        }

    def action_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Return branch-local masks where ``True`` means unavailable."""

        return self._current_masks()

    def _reset_info(self, observation: np.ndarray) -> dict[str, Any]:
        masks = self._current_masks()
        return {
            "episode_index": self._episode_index,
            "seed": self._seed,
            "decision": 0,
            "simulation_tic": 0,
            "target_slot": self._target_slot,
            "position_slot": self._position_slot,
            "remaining_ammunition": self._remaining_ammunition,
            "current_unavailable_masks": self._mask_dict(masks),
            "next_unavailable_masks": self._mask_dict(masks),
            "mask_polarity": "true_means_unavailable",
            "reset_primed": True,
            "reset_frame_hashes": [self._frame_hash(observation)] * 4,
            "frame_order": "oldest_to_newest",
            "real_observation": True,
            "state_channel": self._state_channel(),
        }

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode and duplicate its first post-reset frame four times."""

        self._seed = int(31001 if seed is None else seed)
        if self._game is None:
            self._load_game(self._seed)
        else:
            self._game.set_seed(self._seed)
            self._game.new_episode()

        self._episode_index += 1
        self._target_slot = self._seed % 9 - 4
        self._position_slot = 0
        self._remaining_ammunition = 300
        self._shots_fired = 0
        self._misses = 0
        self._target_hits = 0
        self._decision_count = 0
        self._simulation_tic = 0
        self._terminated = False
        self._truncated = False

        frame = self._preprocess(self._capture_frame())
        self._frames = [frame.copy() for _ in range(4)]
        observation = np.stack(self._frames, axis=2).astype(np.float32, copy=False)
        return observation, self._reset_info(observation[:, :, 0])

    def _validate_action(self, branch_action: Any) -> tuple[int, int]:
        try:
            if len(branch_action) != 2:
                raise ValueError("branch_action must contain movement and combat.")
            movement = operator.index(branch_action[0])
            combat = operator.index(branch_action[1])
        except (TypeError, IndexError) as error:
            raise ValueError(
                "branch_action must be a two-item integer sequence."
            ) from error

        values = (movement, combat)
        for value, size in zip(values, self.branch_sizes):
            if value < 0 or value >= size:
                raise ValueError(f"Action value {value} is outside branch size {size}.")

        masks = self._current_masks()
        if masks[0][movement] or masks[1][combat]:
            raise ValueError("branch_action selects an unavailable action.")
        return values

    def _encoded_buttons(self, movement: int, combat: int, strobe: bool) -> list[int]:
        return [
            int(strobe),
            int(movement == 1),
            int(movement == 2),
            int(combat == 1),
        ]

    def _advance(self, action: list[int], interval: int) -> None:
        self._game.set_action(action)
        self._game.advance_action(1)
        state = self._game.get_state()
        if state is not None:
            self._raw_frame = self._copy_screen(state)

        if interval > 1 and not self._game.is_episode_finished():
            self._game.set_action([0, *action[1:]])
            self._game.advance_action(interval - 1)
            state = self._game.get_state()
            if state is not None:
                self._raw_frame = self._copy_screen(state)

    def _game_variable(self, variable_name: str) -> float:
        return float(
            self._game.get_game_variable(
                getattr(self._vzd.GameVariable, variable_name)
            )
        )

    def step(
        self, branch_action: tuple[int, int] | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one branch action using the alternating 3/4-tic schedule."""

        if self._game is None:
            raise RuntimeError("reset(seed) must be called before step().")
        if self._terminated or self._truncated:
            raise RuntimeError("reset() must be called after an episode ends.")

        movement, combat = self._validate_action(branch_action)
        current_masks = self._current_masks()
        interval = self.interval_pattern[
            self._decision_count % len(self.interval_pattern)
        ]

        if movement == 1:
            self._position_slot = max(-4, self._position_slot - 1)
        elif movement == 2:
            self._position_slot = min(4, self._position_slot + 1)

        target_hit = combat == 1 and self._position_slot == self._target_slot
        encoded = self._encoded_buttons(movement, combat, strobe=True)
        self._advance(encoded, interval)

        engine_kills = self._game_variable("KILLCOUNT")
        target_hit = target_hit or engine_kills > 0
        events = ["decision"]
        reward = -0.01
        if combat == 1:
            self._remaining_ammunition -= 1
            self._shots_fired += 1
            if target_hit:
                reward += 1.0
                events.append("target_hit")
                self._target_hits += 1
            else:
                reward -= 0.02
                events.append("missed_shot")
                self._misses += 1
        elif target_hit:
            reward += 1.0
            events.append("target_hit")
            self._target_hits += 1

        self._decision_count += 1
        self._simulation_tic += interval
        terminated = bool(target_hit)
        truncated = not terminated and self._decision_count >= 300
        self._terminated = terminated
        self._truncated = truncated

        if terminated:
            terminal_reason = "target_hit"
            truncation_reason = None
        elif truncated:
            terminal_reason = None
            truncation_reason = "decision_limit"
        else:
            terminal_reason = None
            truncation_reason = None

        final_frame = self._capture_frame()
        processed = self._preprocess(final_frame)
        self._frames = [*self._frames[1:], processed]
        observation = np.stack(self._frames, axis=2).astype(np.float32, copy=False)
        next_masks = self._current_masks()
        info = {
            "episode_index": self._episode_index,
            "seed": self._seed,
            "decision": self._decision_count,
            "simulation_tic": self._simulation_tic,
            "interval_tics": interval,
            "branch_action": [movement, combat],
            "semantic_action": [
                self.semantic_actions[0][movement],
                self.semantic_actions[1][combat],
            ],
            "encoded_buttons": {
                "ALTATTACK": bool(encoded[0]),
                "MOVE_LEFT": bool(encoded[1]),
                "MOVE_RIGHT": bool(encoded[2]),
                "ATTACK": bool(encoded[3]),
            },
            "encoded_button_values": encoded,
            "current_unavailable_masks": self._mask_dict(current_masks),
            "next_unavailable_masks": self._mask_dict(next_masks),
            "mask_polarity": "true_means_unavailable",
            "position_slot": self._position_slot,
            "target_slot": self._target_slot,
            "remaining_ammunition": self._remaining_ammunition,
            "shots_fired": 1 if combat == 1 else 0,
            "events": events,
            "event_counters": {
                "decisions": self._decision_count,
                "shots_fired": self._shots_fired,
                "misses": self._misses,
                "target_hits": self._target_hits,
            },
            "reward_terms": {
                "decision": -0.01,
                "target_hit": 1.0 if "target_hit" in events else 0.0,
                "missed_shot": -0.02 if "missed_shot" in events else 0.0,
            },
            "terminal_reason": terminal_reason,
            "truncation_reason": truncation_reason,
            "real_observation": True,
            "final_frame_source": "vizdoom_state",
            "frame_order": "oldest_to_newest",
            "frame_hashes": [
                self._frame_hash(self._frames[index]) for index in range(4)
            ],
            "state_channel": self._state_channel(
                1 if terminated else 2 if truncated else 0
            ),
        }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._game is not None:
            self._game.close()
            self._game = None

    def __enter__(self) -> "BasicV1Env":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


QuickDrawBasicEnv = BasicV1Env
