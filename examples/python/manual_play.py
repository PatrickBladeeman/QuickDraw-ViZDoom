#!/usr/bin/env python3

"""Play a ViZDoom scenario manually with keyboard and mouse controls."""

import argparse
import os
import re
import tempfile
from pathlib import Path

import vizdoom as vzd


def _create_manual_config() -> Path:
    """Create a temporary engine config with WASD movement bindings."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (repo_root / "_vizdoom.ini", Path.cwd() / "_vizdoom.ini")
    source = next((path for path in candidates if path.is_file()), None)

    if source is None:
        config = "[Doom.Bindings]\n"
    else:
        config = source.read_text(encoding="utf-8-sig")

    bindings = {
        "w": "+forward",
        "s": "+back",
        "a": "+moveleft",
        "d": "+moveright",
    }

    section = re.search(
        r"(?ms)^\[Doom\.Bindings\]\s*\n(?P<body>.*?)(?=^\[|\Z)",
        config,
    )
    if section is None:
        config = config.rstrip() + "\n\n[Doom.Bindings]\n"
        section = re.search(
            r"(?ms)^\[Doom\.Bindings\]\s*\n(?P<body>.*?)(?=^\[|\Z)",
            config,
        )
        assert section is not None

    body = section.group("body")
    for key, command in bindings.items():
        line = f"{key}={command}"
        pattern = rf"(?im)^{re.escape(key)}\s*=.*$"
        if re.search(pattern, body):
            body = re.sub(pattern, line, body, count=1)
        else:
            body = body.rstrip() + "\n" + line + "\n"

    config = config[: section.start("body")] + body + config[section.end("body") :]
    file_descriptor, file_name = tempfile.mkstemp(prefix="vizdoom_manual_", suffix=".ini")
    os.close(file_descriptor)
    manual_config = Path(file_name)
    manual_config.write_text(config, encoding="utf-8")
    return manual_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(vzd.scenarios_path, "deathmatch.cfg"),
        help="Path to a ViZDoom scenario config file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to play (default: 1).",
    )
    args = parser.parse_args()

    manual_config = _create_manual_config()
    game = vzd.DoomGame()
    game.load_config(args.config)
    game.set_doom_config_path(str(manual_config))
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_window_visible(True)
    game.set_sound_enabled(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    try:
        game.init()
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}; focus the game window to play.")
            game.new_episode()

            while not game.is_episode_finished():
                # In SPECTATOR mode, human input controls the player.
                # This only advances the engine; it sends no agent action.
                game.advance_action()
    finally:
        game.close()
        manual_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
