"""One-decision, policy-visible target classification; independent of Basic rewards."""

from pathlib import Path

import numpy as np

from .basic import REPOSITORY_ROOT, BasicV1Env


DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "scenarios/quickdraw/micro-learning-v1.cfg"


class MicroLearningV1Env(BasicV1Env):
    """Reuse only Basic's image, action-validation, and game lifecycle helpers."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        window_visible: bool = False,
    ) -> None:
        super().__init__(config_path, window_visible=window_visible)
        self._class_frames: dict[bool, np.ndarray] = {}

    def _current_masks(self):
        return np.array([False, True, True]), np.array([False, False])

    def reset(self, seed=None):
        self._seed = int(32001 if seed is None else seed)
        self.target_present = bool(self._seed % 2)
        if self._game is None:
            self._load_game(self._seed)
        self._game.set_seed(self._seed)
        self._game.set_doom_map("map01" if self.target_present else "map02")
        self._game.new_episode()
        self._terminated = False
        state = self._game.get_state()
        if state is None:
            raise ValueError("Micro-learning reset has no real observation.")
        frame = self._preprocess(self._copy_screen(state))
        if not np.isfinite(frame).all() or not frame.any():
            raise ValueError("Micro-learning reset must be finite and nonzero.")
        self._class_frames[self.target_present] = frame.copy()
        if len(self._class_frames) == 2 and np.array_equal(
            self._class_frames[False], self._class_frames[True]
        ):
            raise ValueError("Target classes must have distinct policy observations.")
        self._frames = [frame.copy() for _ in range(4)]
        return np.stack(self._frames, axis=2), {
            "seed": self._seed,
            "target_present": self.target_present,
            "real_observation": True,
            "next_unavailable_masks": self._mask_dict(self._current_masks()),
        }

    def step(self, branch_action):
        if self._game is None or self._terminated:
            raise RuntimeError("reset() must precede each one-decision episode.")
        movement, combat = self._validate_action(branch_action)
        self._game.make_action(self._encoded_buttons(movement, combat, False), 1)
        state = self._game.get_state()
        if state is None:
            raise ValueError("Micro-learning decision has no real final observation.")
        self._frames = [*self._frames[1:], self._preprocess(self._copy_screen(state))]
        correct = combat == int(self.target_present)
        self._terminated = True
        return (
            np.stack(self._frames, axis=2),
            1.0 if correct else -1.0,
            True,
            False,
            {
                "seed": self._seed,
                "target_present": self.target_present,
                "branch_action": [movement, combat],
                "semantic_action": ["Stay", self.semantic_actions[1][combat]],
                "correct": correct,
                "bootstrap": False,
                "terminal_reason": "correct" if correct else "incorrect",
                "real_observation": True,
                "final_frame_source": "vizdoom_state",
                "next_unavailable_masks": self._mask_dict(self._current_masks()),
            },
        )
