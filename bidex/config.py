"""Default configuration for BiDexEnv.

A ``BiDexConfig`` mirrors the keys the original ``scripts/config.yaml``
carried, but the primary API is the ``BiDexEnv`` constructor kwargs —
``config=`` is a back-compat escape hatch for callers that already have a
YAML dict. Use ``default_config()`` for sane defaults or ``load_config()``
to read a legacy YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class SimConfig:
    step_freq_hz: int = 500
    control_freq_hz: int = 25   # 25 Hz control (matches the rebuilt 25 fps bidex_qc3 dataset)
    camera_render_scale: float = 0.75   # 0.75× of 480×640 = 360×480, matching the bidex_qc3 dataset storage res (what all models trained on)
    add_freejoint: bool = True


@dataclass
class ControlConfig:
    control_mode: str = "joint"   # "joint" | "ee"
    ik_solver: str = "mink"       # "mink" | "dls" | "opspace"


@dataclass
class RobotConfig:
    robot_type: str = "dual_fr3v2_sharpa"


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 25   # 25 Hz, unified with the rebuilt 25 fps dataset
    # cameras[] lives in robots/camera_profiles/<robot>.yaml now.


@dataclass
class BiDexConfig:
    sim_config: SimConfig = field(default_factory=SimConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    robot_config: RobotConfig = field(default_factory=RobotConfig)
    camera_config: CameraConfig = field(default_factory=CameraConfig)

    def to_dict(self) -> dict:
        return {
            "sim_config": self.sim_config.__dict__,
            "control_config": self.control_config.__dict__,
            "robot_config": self.robot_config.__dict__,
            "camera_config": self.camera_config.__dict__,
        }


def default_config() -> BiDexConfig:
    """Sane defaults for a joint-mode franka rollout."""
    return BiDexConfig()


def load_config(path: str) -> BiDexConfig:
    """Load a legacy YAML config file into a BiDexConfig.

    Unknown keys are ignored; missing keys fall back to dataclass defaults.
    """
    with open(path, "r") as f:
        blob = yaml.safe_load(f) or {}

    sim = SimConfig(**{k: v for k, v in (blob.get("sim_config") or {}).items()
                       if k in SimConfig.__dataclass_fields__})
    ctrl = ControlConfig(**{k: v for k, v in (blob.get("control_config") or {}).items()
                            if k in ControlConfig.__dataclass_fields__})
    robot = RobotConfig(**{k: v for k, v in (blob.get("robot_config") or {}).items()
                           if k in RobotConfig.__dataclass_fields__})
    cam = CameraConfig(**{k: v for k, v in (blob.get("camera_config") or {}).items()
                          if k in CameraConfig.__dataclass_fields__})
    return BiDexConfig(sim, ctrl, robot, cam)
