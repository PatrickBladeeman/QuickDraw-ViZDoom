"""Build the dedicated two-map UDMF WAD using only the standard library."""

import struct
from pathlib import Path


def build_wad() -> bytes:
    room = """namespace = "ZDoom";
    vertex { x = -128; y = -192; }
    vertex { x = -128; y = 192; }
    vertex { x = 384; y = 192; }
    vertex { x = 384; y = -192; }
    sector { heightfloor = 0; heightceiling = 160;
             texturefloor = "FLOOR0_1"; textureceiling = "CEIL1_1";
             lightlevel = 255; }
    thing { x = 0; y = 0; angle = 0; type = 1;
            skill1 = true; skill2 = true; skill3 = true;
            skill4 = true; skill5 = true; single = true; }
    """
    for index in range(4):
        room += f"""
        sidedef {{ sector = 0; texturemiddle = "STARTAN3"; }}
        linedef {{ v1 = {index}; v2 = {(index + 1) % 4};
                   sidefront = {index}; blocking = true; }}
        """
    target = """thing { x = 112; y = 0; angle = 180; type = 15000;
                        skill1 = true; skill2 = true; skill3 = true;
                        skill4 = true; skill5 = true; single = true; }"""
    decorate = """actor MicroTarget 15000 {
        Radius 20
        Height 112
        Scale 2.0
        States { Spawn: POSS A -1
                 Stop }
    }"""
    lumps = [("DECORATE", decorate.encode("ascii"))]
    for name, things in (("MAP01", target), ("MAP02", "")):
        lumps.extend(
            [
                (name, b""),
                ("TEXTMAP", (room + things).encode("ascii")),
                ("ENDMAP", b""),
            ]
        )
    payload = bytearray()
    directory = bytearray()
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


if __name__ == "__main__":
    Path(__file__).with_name("micro-learning-v1.wad").write_bytes(build_wad())
