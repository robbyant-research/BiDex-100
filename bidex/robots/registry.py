"""Robot assembly XML registry.

Central lookup table that maps a ``robot_type`` string to its MJCF
assembly file. New robots (different arms or different hands) are added
as additional rows of ``ROBOT_REGISTRY`` plus a sibling directory under
``robots/assembly/``.

Resolution priority for callers (see env.py, replay_episode.py):
    1. ``--robot`` CLI override
    2. ``Task.robot_type`` from the active task YAML
    3. ``config.robot_config.robot_type`` from scripts/config.yaml

For inverse-kinematics, ``resolve_ik_profile(robot_type, side)`` returns
a ready-to-construct ``RobotProfile`` so AVP teleop and any future motion
planner can drive a robot's arm without hardcoding which XML / joint
names belong to that robot.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PKG_ROOT)  # one up from bidex/bidex/

ROBOT_REGISTRY: Dict[str, str] = {
    "dual_fr3v2_sharpa": "robots/assembly/fr3v2_sharpa/dual_fr3v2_sharpa.xml",
    "astribot_s1_fixed_sharpa": "robots/assembly/astribot_s1/astribot_s1_fixed_sharpa.xml",
    "astribot_s1_mobile_sharpa": "robots/assembly/astribot_s1/astribot_s1_mobile_sharpa.xml",
}

DEFAULT_ROBOT_TYPE: str = "dual_fr3v2_sharpa"

_IK_PROFILES_DIR = os.path.join(_PKG_ROOT, "robots", "ik_profiles")
_INIT_QPOS_DIR = os.path.join(_PKG_ROOT, "robots", "init_qpos")


def resolve_robot_xml(robot_type: str) -> str:
    """Return the absolute path to the assembly XML for ``robot_type``.

    Raises KeyError on unknown robot_type so config typos fail loudly.
    """
    rel = ROBOT_REGISTRY[robot_type]
    return os.path.join(_PKG_ROOT, rel)


def available_robot_types() -> List[str]:
    return sorted(ROBOT_REGISTRY.keys())


def _abs_xml_path(p: str) -> str:
    """Profile YAMLs store ``xml_path`` relative to the bidex package
    root (e.g. ``bidex/robots/...``). Absolute paths pass through."""
    if os.path.isabs(p):
        return p
    # Profile paths are written relative to repo root for readability; resolve
    # against _REPO_ROOT first, then fall back to _PKG_ROOT for tolerance.
    cand1 = os.path.join(_REPO_ROOT, p)
    if os.path.isfile(cand1):
        return cand1
    return os.path.join(_PKG_ROOT, p)


def resolve_ik_profile(robot_type: str, side: str) -> Dict[str, Any]:
    """Return the IK ``RobotProfile`` kwargs (+ chosen solver) for one arm
    of ``robot_type``. ``side`` is ``"left"`` or ``"right"``.

    Output shape::

        {
          "solver": "mink",
          "target_frame": "world",        # or "base_local" for humanoid_arm
          "profile_kwargs": {
            "xml_path": "assets/robots/example.xml",
            "end_site_name": "arm_left_tool",
            "arm_joint_names": [...],
            "posture_freeze_joints": [...],
            "base_body_name": "chassis_base",   # humanoid_arm only
          }
        }

    Designed to feed straight into ``RobotProfile(**out["profile_kwargs"])``
    and ``FrankaKinematicsSolver(profile=..., solver=out["solver"])``.

    ``target_frame`` is consumed by env's _eef_cmd_handler — see env.py
    for the dispatch. ``world`` (default) keeps the legacy FACTR /
    replay pipeline: incoming KINEMATICS_END_CMD is rebased from sim
    world to sim arm-base local, then mink solves in IK world via the
    chassis post-mult. ``base_local`` skips both rebases — the AVP
    teleop config matrices already produce per-arm-base local targets
    matching the humanoid-arm-retarget reference convention, and the
    humanoid_arm solver consumes them directly.

    Raises KeyError on unknown robot_type, FileNotFoundError on missing
    profile, ValueError on malformed YAML / unknown side — all loud
    failures so typos surface immediately.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if robot_type not in ROBOT_REGISTRY:
        raise KeyError(f"Unknown robot_type {robot_type!r}; "
                       f"available: {available_robot_types()}")
    path = os.path.join(_IK_PROFILES_DIR, f"{robot_type}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"IK profile not found for {robot_type!r} at {path}. "
            f"Add the YAML to bidex/robots/ik_profiles/.")
    with open(path, "r") as f:
        blob = yaml.safe_load(f) or {}
    side_block = blob.get(side)
    if side_block is None:
        raise ValueError(f"IK profile {path} missing {side!r} block")
    solver = blob.get("solver", "mink")
    target_frame = blob.get("target_frame", "world")
    if target_frame not in ("world", "base_local"):
        raise ValueError(
            f"IK profile {path} has invalid target_frame "
            f"{target_frame!r}; expected 'world' or 'base_local'")
    return {
        "solver": solver,
        "target_frame": target_frame,
        "profile_kwargs": {
            "xml_path": _abs_xml_path(side_block["xml_path"]),
            "end_site_name": side_block.get("end_site_name", "end"),
            "arm_joint_names": side_block.get("arm_joint_names"),
            "posture_freeze_joints": side_block.get("posture_freeze_joints") or [],
            "base_body_name": side_block.get("base_body_name"),
        },
    }


def resolve_init_qpos(robot_type: str) -> Dict[str, float]:
    """Return ``{joint_name: qpos}`` for the robot's reset pose.

    Resolution: ``bidex/robots/init_qpos/<robot_type>.yaml``.
    Returns an empty dict (no override) when the file is absent — env then
    leaves the model at MuJoCo's defaults (typically all zeros). Joints
    referenced in the YAML that don't exist in the loaded MJCF are
    silently skipped by the env-side applier (avoids hard fails when an
    init pose file lists joints for a robot variant that doesn't have
    them, e.g. legs on gr1_arms_only).
    """
    if robot_type not in ROBOT_REGISTRY:
        return {}
    path = os.path.join(_INIT_QPOS_DIR, f"{robot_type}.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        blob = yaml.safe_load(f) or {}
    joints = blob.get("joints") or {}
    return {str(k): float(v) for k, v in joints.items()}
