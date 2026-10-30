"""Arena-driven robot base offset.

Mirrors LIBERO's `MountedPanda.base_xpos_offset` table (libero/envs/robots/
mounted_panda.py:41-48): each scene archetype implies a deterministic shift of
the robot base so that the arms reach the workspace consistently. In LIBERO
those values are derived from arena geometry; here we keep the same intent —
"changing the scene auto-repositions the base" — but tuned for our dual
FR3v2 + Sharpa rig sitting at table height z=0.8m with a 45-deg yaw.
"""

from __future__ import annotations

from typing import Optional, Tuple


# pos = (x, y, z) in world frame, quat = (w, x, y, z)
# These defaults match the working setting in scripts/config.yaml for the
# kitchen tabletop scene; per-arena tweaks stay close to that baseline so a
# task that switches arenas does not need to re-author the whole pose.
ARENA_BASE_OFFSETS: dict[str, dict] = {
    "kitchen_tabletop": {
        # Kitchen / tabletop: robot in front of counter at 0.6m height
        "pos": (0.5, -0.45, 0.6),
        "quat": (0.382683, 0.0, 0.0, 0.923880),
    },
    "robocasa_kitchen": {
        # Full RoboCasa procedural kitchen: default near main counter.
        # Task YAMLs override per-fixture (stove, sink, etc.) if needed.
        "pos": (1.5, -1.4, 0.6),
        "quat": (0.7071068, 0.0, 0.0, 0.3535534),
    },
    "robocasa_kitchen_stove": {
        # Near stove in robocasa kitchen
        "pos": (1.5, -1.4, 0.6),
        "quat": (0.7071068, 0.0, 0.0, 0.3535534),
    },
    "robocasa_kitchen_sink": {
        # Near sink in robocasa kitchen
        "pos": (0.8, -2.0, 0.6),
        "quat": (0.7071068, 0.0, 0.0, 1.5),
    },
    "living_room_tabletop": {
        # Living room: lower table height
        "pos": (0.5, -0.45, 0.3),
        "quat": (0.382683, 0.0, 0.0, 0.923880),
    },
    "study": {
        # Study / desk: higher desk
        "pos": (0.8, -0.45, 0.7),
        "quat": (0.382683, 0.0, 0.0, 0.923880),
    },
    "tabletop": {
        # Generic tabletop
        "pos": (0.5, -0.45, 0.6),
        "quat": (0.382683, 0.0, 0.0, 0.923880),
    },
    "dexjoco_table": {
        # Dexjoco-style table arena (table_top at z=0.75m)
        "pos": (0.8, 0.0, 0.0),
        "quat": (0.0, 0.0, 0.0, 1.0),
    },
    "floor": {
        # floor scenes: robot stands on the floor, no z lift
        "pos": (0.0, -0.4, 0.0),
        "quat": (0.382683, 0.0, 0.0, 0.923880),
    },
}


def arena_for_scene(scene_name: str) -> str:
    """Map a scene xml stem (e.g. 'libero_kitchen_tabletop_base_style') to an arena key."""
    name = scene_name.lower()
    # RoboCasa kitchen sub-arenas: allow per-fixture robot placement
    if name.startswith("robocasa_kitchen_stove"):
        return "robocasa_kitchen_stove"
    if name.startswith("robocasa_kitchen_sink"):
        return "robocasa_kitchen_sink"
    if name == "robocasa_kitchen" or name.startswith("robocasa_kitchen"):
        return "robocasa_kitchen"
    if "dexjoco_table" in name:
        return "dexjoco_table"
    if "kitchen" in name:
        return "kitchen_tabletop"
    if "living_room" in name:
        return "living_room_tabletop"
    if "study" in name:
        return "study"
    if "floor" in name:
        return "floor"
    if "tabletop" in name:
        return "tabletop"
    return "tabletop"  # safe default


def base_offset_for(
    scene_name: str,
    explicit: Optional[dict] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Return (pos, quat) for the robot base.

    `explicit` (if provided) wins over the arena default. Pass `{"pos": [...],
    "quat": [...]}` to fully override; any missing key falls back to the arena
    default. This lets task YAMLs say "use the kitchen default" by omitting
    `robot_pos`/`robot_quat`, while still allowing per-task overrides.
    """
    arena_key = arena_for_scene(scene_name)
    base = ARENA_BASE_OFFSETS.get(arena_key, ARENA_BASE_OFFSETS["tabletop"])
    pos = tuple(base["pos"])
    quat = tuple(base["quat"])
    if explicit:
        if explicit.get("pos") is not None:
            pos = tuple(explicit["pos"])
        if explicit.get("quat") is not None:
            quat = tuple(explicit["quat"])
    return pos, quat
