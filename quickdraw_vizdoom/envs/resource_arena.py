"""Three native Doom targets and a kill-unlocked ammunition pickup."""

from __future__ import annotations

import hashlib
import struct
import tempfile
from pathlib import Path

import vizdoom as vzd


TARGET_IDS = ("A", "B", "C")
BUTTONS = ("MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD", "ATTACK")


def arena_wad(case):
    """Build UDMF/DECORATE assets; all kills and ammo changes happen in Doom."""
    decorate = f"""
    actor ResourcePistol : Pistol {{
      Weapon.AmmoGive 0
      States {{ Fire: PISG A 1 A_FireBullets(0, 0, 1, 5, "BulletPuff")
                       PISG A 18
                       Goto Ready }}
    }}
    actor ResourcePlayer : DoomPlayer {{
      Speed 0.15
      Player.StartItem "ResourcePistol"
      Player.StartItem "Clip", {case["initial_ammo"]}
    }}
    actor ResourceCache : CustomInventory {{
      +INVENTORY.ALWAYSPICKUP
      States {{ Spawn: CLIP A -1
                       Stop
                Pickup: TNT1 A 0 A_GiveInventory("Clip", {case["cache_ammo"]})
                        Stop }}
    }}
    """
    room = """namespace = "ZDoom";
    vertex { x = -128; y = -256; }
    vertex { x = -128; y = 256; }
    vertex { x = 384; y = 256; }
    vertex { x = 384; y = -256; }
    sector { heightfloor = 0; heightceiling = 160;
             texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1";
             lightlevel = 255; }
    thing { x = 0; y = 0; angle = 0; type = 1;
            skill1 = true; skill2 = true; skill3 = true;
            skill4 = true; skill5 = true; single = true; }
    """
    for i in range(4):
        room += f"""
        sidedef {{ sector = 0; texturemiddle = "STARTAN3"; }}
        linedef {{ v1 = {i}; v2 = {(i + 1) % 4}; sidefront = {i}; blocking = true; }}
        """
    for i, target in enumerate(case["targets"]):
        drop = (
            'TNT1 A 0 A_SpawnItemEx("ResourceCache", 0, 0, 0, 0, 0, 0)'
            if target["id"] == case["cache_requires"]
            else ""
        )
        decorate += f"""
        actor Resource{target["id"]} : Actor {15000 + i} {{
          Health 1
          Radius 16
          Height 56
          Monster
          States {{ Spawn: POSS A -1
                           Stop
                    Death: {drop}
                           TNT1 A 0 A_NoBlocking
                           Stop }}
        }}
        """
        room += f"""
        thing {{ x = 160; y = {target["y"]}; angle = 180; type = {15000 + i};
                 skill1 = true; skill2 = true; skill3 = true;
                 skill4 = true; skill5 = true; single = true; }}
        """
    lumps = [
        ("DECORATE", decorate.encode("ascii")),
        ("KEYCONF", b"clearplayerclasses\naddplayerclass ResourcePlayer\n"),
        ("MAP01", b""),
        ("TEXTMAP", room.encode("ascii")),
        ("ENDMAP", b""),
    ]
    payload, directory = bytearray(), bytearray()
    for name, data in lumps:
        directory.extend(
            struct.pack("<II8s", 12 + len(payload), len(data), name.encode())
        )
        payload.extend(data)
    return (
        struct.pack("<4sII", b"PWAD", len(lumps), 12 + len(payload))
        + payload
        + directory
    )


class ResourceArena:
    """Reliable common skills, grounded in native actor identities and AMMO2."""

    def __init__(self, case):
        self.case = case
        self._directory = tempfile.TemporaryDirectory(prefix="quickdraw-resource-")
        self.game = vzd.DoomGame()
        self.decisions = 0
        self.events = []
        self.collected = False
        try:
            wad = arena_wad(case)
            path = Path(self._directory.name) / "arena.wad"
            path.write_bytes(wad)
            self.wad_sha256 = hashlib.sha256(wad).hexdigest()
            self.game.set_doom_scenario_path(str(path))
            self.game.set_doom_map("map01")
            self.game.set_seed(case["seed"])
            self.game.set_doom_skill(3)  # Easy skill doubles native ammo pickups.
            self.game.set_window_visible(False)
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
            self.game.set_objects_info_enabled(True)
            self.game.set_available_buttons(
                [getattr(vzd.Button, name) for name in BUTTONS]
            )
            self.game.set_available_game_variables(
                [vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT]
            )
            self.game.set_episode_start_time(40)
            self.game.set_episode_timeout(0)
            self.game.init()
            self.state = self._read_state()
            if self.state["ammo"] != case["initial_ammo"] or not all(
                t["alive"] for t in self.state["targets"]
            ):
                raise RuntimeError("Native reset disagrees with the arena case.")
            self.initial_frame_sha256 = self.frame_hash()
        except Exception:
            self.close()
            raise

    def _read_state(self):
        native = self.game.get_state()
        if native is None or self.game.is_episode_finished():
            raise RuntimeError("Native arena unexpectedly ended.")
        objects = {item.name: item for item in native.objects}
        player = objects["ResourcePlayer"]
        targets = []
        for target in self.case["targets"]:
            actor = objects.get("Resource" + target["id"])
            targets.append(
                {
                    "id": target["id"],
                    "alive": actor is not None,
                    "position": (
                        [160.0, float(target["y"])]
                        if actor is None
                        else [actor.position_x, actor.position_y]
                    ),
                }
            )
        cache = objects.get("ResourceCache")
        return {
            "player_position": [player.position_x, player.position_y],
            "ammo": int(self.game.get_game_variable(vzd.GameVariable.AMMO2)),
            "native_kills": int(
                self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            ),
            "targets": targets,
            "objective": {"required_targets": list(self.case["required_targets"])},
            "cache": {
                "requires_eliminated": self.case["cache_requires"],
                "ammo_gain": self.case["cache_ammo"],
                "available": cache is not None,
                "collected": self.collected,
                "position": (
                    None if cache is None else [cache.position_x, cache.position_y]
                ),
            },
        }

    def frame_hash(self):
        return hashlib.sha256(self.game.get_state().screen_buffer.tobytes()).hexdigest()

    def _step(self, button=None):
        previous = self.state
        self.game.make_action([int(name == button) for name in BUTTONS], 1)
        self.decisions += 1
        current = self._read_state()
        killed = [
            a["id"]
            for a, b in zip(previous["targets"], current["targets"])
            if a["alive"] and not b["alive"]
        ]
        delta = current["ammo"] - previous["ammo"]
        if killed or delta:
            if len(killed) != current["native_kills"] - previous["native_kills"]:
                raise RuntimeError(
                    "Native kill count disagrees with target identities."
                )
            if delta > 0:
                if (
                    not previous["cache"]["available"]
                    or current["cache"]["available"]
                    or delta != self.case["cache_ammo"]
                ):
                    raise RuntimeError(
                        "Native ammunition increase lacks its expected pickup."
                    )
                self.collected = True
                current["cache"]["collected"] = True
            self.events.append(
                {"decision": self.decisions, "killed": killed, "ammo_delta": delta}
            )
        self.state = current

    def _move(self, x, y, deadline):
        while self.decisions < deadline:
            px, py = self.state["player_position"]
            button = (
                "MOVE_BACKWARD"
                if px > x + 4
                else (
                    "MOVE_FORWARD"
                    if px < x - 4
                    else (
                        "MOVE_LEFT"
                        if py < y - 4
                        else "MOVE_RIGHT" if py > y + 4 else None
                    )
                )
            )
            if button is None:
                for _ in range(8):
                    self._step()
                px, py = self.state["player_position"]
                if abs(px - x) <= 4 and abs(py - y) <= 4:
                    return True
            self._step(button)
        return False

    def execute(self, goal, limit=600):
        """Goal stays fixed until native success, impossibility or timeout."""
        deadline = self.decisions + limit
        if goal["goal"] == "FINISH":
            return "finish"
        if goal["goal"] == "COLLECT_AMMO":
            cache = self.state["cache"]
            if not cache["available"]:
                return "unavailable"
            if not self._move(*cache["position"], deadline):
                return "timeout"
            if not self.collected:
                return "pickup_failed"
            return (
                "achieved"
                if self._move(0, self.state["player_position"][1], deadline)
                else "timeout"
            )
        target = next(t for t in self.state["targets"] if t["id"] == goal["target_id"])
        if not target["alive"] or self.state["ammo"] <= 0:
            return "unavailable"
        if not self._move(0, target["position"][1], deadline):
            return "timeout"
        ammo = self.state["ammo"]
        while self.decisions < deadline and self.state["ammo"] == ammo:
            self._step("ATTACK")
        # Release the trigger through cooldown; never spend a second round on one goal.
        for _ in range(20):
            self._step()
        alive = next(
            t["alive"] for t in self.state["targets"] if t["id"] == goal["target_id"]
        )
        return "achieved" if not alive else "shot_failed"

    def close(self):
        self.game.close()
        self._directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
