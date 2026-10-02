"""A native-geometry Basic environment for goal-conditioning experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .basic import BasicV1Env


class NativeBasicV1Env(BasicV1Env):
    """Ground alignment and hits in ViZDoom's object state, not wrapper slots."""

    interval_pattern = (4,)
    alignment_tolerance = 8.0
    native_kill_hit_fallback = False

    def _configure_episode(self, game: Any) -> None:
        game.set_objects_info_enabled(True)
        game.set_labels_buffer_enabled(True)

    def _current_masks(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([False, False, False]), np.array(
            [False, self._remaining_ammunition <= 0]
        )

    def _native_geometry(self) -> dict[str, Any]:
        state = self._game.get_state()
        objects = () if state is None or state.objects is None else state.objects
        player = next((item for item in objects if item.name == "DoomPlayer"), None)
        target = next((item for item in objects if item.name == "Cacodemon"), None)
        if player is None or target is None:
            raise RuntimeError("Native Basic state is missing its player or target.")
        labels = () if state.labels is None else state.labels
        visible = any(item.object_name == "Cacodemon" for item in labels)
        error = float(target.position_y - player.position_y)
        return {
            "aligned": abs(error) <= self.alignment_tolerance,
            "alignment_error": error,
            "player_position": [float(player.position_x), float(player.position_y)],
            "target_position": [float(target.position_x), float(target.position_y)],
            "target_visible": visible,
        }

    def _native_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info.pop("position_slot", None)
        info.pop("target_slot", None)
        info["remaining_ammunition"] = self._remaining_ammunition
        info["native_geometry"] = self._native_geometry()
        info["native_geometry_available"] = True
        info["aligned"] = info["native_geometry"]["aligned"]
        info["alignment_error"] = info["native_geometry"]["alignment_error"]
        info["geometry_definition"] = "target_y - player_y; positive means move left"
        info["state_channel"] = {"schema_id": "quickdraw.native-basic.v1"}
        return info

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed)
        self._remaining_ammunition = int(self._game_variable("AMMO2"))
        self._native_kill_count = int(self._game_variable("KILLCOUNT"))
        info = self._native_info(info)
        if not info["native_geometry"]["target_visible"]:
            raise RuntimeError(
                "Native Basic reset target is not visible to the policy."
            )
        return observation, info

    def _advance(self, action: list[int], interval: int) -> None:
        self._game.make_action(action, interval)
        state = self._game.get_state()
        if state is not None:
            self._raw_frame = self._copy_screen(state)

    def step(
        self, branch_action: tuple[int, int] | list[int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._game is None:
            raise RuntimeError("reset(seed) must be called before step().")
        if self._terminated or self._truncated:
            raise RuntimeError("reset() must be called after an episode ends.")

        movement, combat = self._validate_action(branch_action)
        current_masks = self._current_masks()
        interval = self.interval_pattern[0]
        previous_ammo = self._remaining_ammunition
        encoded = self._encoded_buttons(movement, combat, strobe=False)
        self._advance(encoded, interval)
        engine_kills = int(self._game_variable("KILLCOUNT"))
        target_hit = engine_kills > self._native_kill_count
        self._native_kill_count = engine_kills
        self._remaining_ammunition = int(self._game_variable("AMMO2"))
        shot_fired = self._remaining_ammunition < previous_ammo
        self._decision_count += 1
        self._simulation_tic += interval
        self._shots_fired += int(shot_fired)
        self._target_hits += int(target_hit)
        self._misses += int(shot_fired and not target_hit)
        terminated = target_hit
        engine_finished = bool(self._game.is_episode_finished())
        truncated = not terminated and (engine_finished or self._decision_count >= 300)
        self._terminated, self._truncated = terminated, truncated

        reward = -0.01 + (1.0 if target_hit else -0.02 if shot_fired else 0.0)
        events = ["decision"]
        if target_hit:
            events.append("target_hit")
        elif shot_fired:
            events.append("missed_shot")
        if engine_finished and not target_hit:
            events.append("infrastructure_invalid")

        processed = self._preprocess(self._capture_frame())
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
            "current_unavailable_masks": self._mask_dict(current_masks),
            "next_unavailable_masks": self._mask_dict(next_masks),
            "mask_polarity": "true_means_unavailable",
            "remaining_ammunition": self._remaining_ammunition,
            "events": events,
            "event_counters": {
                "decisions": self._decision_count,
                "shots_fired": self._shots_fired,
                "misses": self._misses,
                "target_hits": self._target_hits,
            },
            "reward_terms": {
                "decision": -0.01,
                "target_hit": 1.0 if target_hit else 0.0,
                "missed_shot": -0.02 if shot_fired and not target_hit else 0.0,
            },
            "terminal_reason": "target_hit" if terminated else None,
            "truncation_reason": (
                "infrastructure_invalid"
                if engine_finished and not target_hit
                else "decision_limit" if truncated else None
            ),
            "real_observation": True,
            "final_frame_source": "vizdoom_state",
            "frame_order": "oldest_to_newest",
            "frame_hashes": [self._frame_hash(frame) for frame in self._frames],
            "native_kill_count": engine_kills,
            "native_target_killed": target_hit,
            "engine_episode_finished": engine_finished,
            "native_geometry_available": False,
        }
        if not (terminated or engine_finished):
            self._native_info(info)
        return observation, float(reward), terminated, truncated, info
