"""BiDex-100 — gym-style MuJoCo simulation environment.

Synchronous, single-threaded environment for imitation-learning evaluation.
Replaces the original LCM-message-driven ``SimEnv`` service with a clean
Python API:

    env = BiDexEnv(task, robot_type="dual_fr3v2_sharpa",
                    control_mode="joint", cameras=["head"])
    obs = env.reset(seed=0)
    for t in range(max_steps):
        action = policy(obs)                 # {"arm_joint_position": [14], ...}
        obs, reward, terminated, info = env.step(action)
        if terminated or info["timed_out"]:
            break
    print(info["success"])

Observations (right-arm-first packing, matching the dataset schema):
    obs["arm_joint_position"]  float32[14]  # [right_q(7), left_q(7)]
    obs["arm_joint_velocity"]  float32[14]
    obs["dex_joint_position"]  float32[44]  # [right_hand(22), left_hand(22)]
    obs["dex_joint_velocity"]  float32[44]
    obs["abs_end_effector"]    float32[14]  # [right_xyz(3),right_quat_wxyz(4), left_...]
    obs["images"]              dict[str, uint8[H,W,3]]  # + "<cam>_depth" uint16[H,W] if depth on
    obs["language"]            str

Actions:
    joint mode: {"arm_joint_position": [14], "dex_joint_position": [44]}
    ee mode:    {"abs_end_effector":   [14], "dex_joint_position": [44]}

EE quaternions are wxyz (MuJoCo native) throughout the live API. The replay
module converts xyzw↔wxyz at the parquet boundary.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import pyexpat  # noqa: F401 -- load Python's ABI-matched expat before MuJoCo/EGL
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import mujoco

from bidex.scene_builder import (
    ObjectSpec,
    _parse_object_spec,
    _prepare_object_xml,
    resolve_object_xml,
    resolve_scene_xml,
)
from bidex.robots.registry import (
    DEFAULT_ROBOT_TYPE,
    resolve_init_qpos,
    resolve_ik_profile,
    resolve_robot_xml,
)
from bidex.ik import FrankaKinematicsSolver, RobotProfile
from bidex.tasks import (
    SubtaskTracker,
    Task,
    apply_task_initial_state,
    base_offset_for,
    evaluate_goal,
    init_states_path,
    load_init_states,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LEFT_SHARPA_ACTUATORS = [
    "left_thumb_CMC_FE_ctrl", "left_thumb_CMC_AA_ctrl",
    "left_thumb_MCP_FE_ctrl", "left_thumb_MCP_AA_ctrl", "left_thumb_IP_ctrl",
    "left_index_MCP_FE_ctrl", "left_index_MCP_AA_ctrl",
    "left_index_PIP_ctrl", "left_index_DIP_ctrl",
    "left_middle_MCP_FE_ctrl", "left_middle_MCP_AA_ctrl",
    "left_middle_PIP_ctrl", "left_middle_DIP_ctrl",
    "left_ring_MCP_FE_ctrl", "left_ring_MCP_AA_ctrl",
    "left_ring_PIP_ctrl", "left_ring_DIP_ctrl",
    "left_pinky_CMC_ctrl", "left_pinky_MCP_FE_ctrl", "left_pinky_MCP_AA_ctrl",
    "left_pinky_PIP_ctrl", "left_pinky_DIP_ctrl",
]

RIGHT_SHARPA_ACTUATORS = [
    "right_thumb_CMC_FE_ctrl", "right_thumb_CMC_AA_ctrl",
    "right_thumb_MCP_FE_ctrl", "right_thumb_MCP_AA_ctrl", "right_thumb_IP_ctrl",
    "right_index_MCP_FE_ctrl", "right_index_MCP_AA_ctrl",
    "right_index_PIP_ctrl", "right_index_DIP_ctrl",
    "right_middle_MCP_FE_ctrl", "right_middle_MCP_AA_ctrl",
    "right_middle_PIP_ctrl", "right_middle_DIP_ctrl",
    "right_ring_MCP_FE_ctrl", "right_ring_MCP_AA_ctrl",
    "right_ring_PIP_ctrl", "right_ring_DIP_ctrl",
    "right_pinky_CMC_ctrl", "right_pinky_MCP_FE_ctrl", "right_pinky_MCP_AA_ctrl",
    "right_pinky_PIP_ctrl", "right_pinky_DIP_ctrl",
]

LEFT_SHARPA_JOINTS = [n.replace("_ctrl", "") for n in LEFT_SHARPA_ACTUATORS]
RIGHT_SHARPA_JOINTS = [n.replace("_ctrl", "") for n in RIGHT_SHARPA_ACTUATORS]


def _euler_to_quat(euler):
    """Convert euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    r, p, y = euler
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y2 = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y2, z]


# Per-robot actuator/joint/body name dispatch for IK control and obs packing.
_ARM_ID_PROFILES: Dict[str, dict] = {
    "dual_fr3v2_sharpa": {
        "left_joints":  [f"left_fr3v2_joint{i}" for i in range(1, 8)],
        "right_joints": [f"right_fr3v2_joint{i}" for i in range(1, 8)],
        "left_eef":  "left_fr3v2_link8",
        "right_eef": "right_fr3v2_link8",
        "left_base":  "left_fr3v2_link0",
        "right_base": "right_fr3v2_link0",
        "skip_link8_correction": False,
        "default_left_q":  [0.7854, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.7854],
        "default_right_q": [-0.7854, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.7854],
    },
    "astribot_s1_fixed_sharpa": {
        "left_joints":  [f"astribot_arm_left_joint_{i}"  for i in range(1, 8)],
        "right_joints": [f"astribot_arm_right_joint_{i}" for i in range(1, 8)],
        "left_eef":  "astribot_arm_left_tool_link",
        "right_eef": "astribot_arm_right_tool_link",
        "left_base":  "chassis_base",
        "right_base": "chassis_base",
        "skip_link8_correction": True,
        "default_left_q":  [0.0, 0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0],
        "default_right_q": [0.0, 0.0,  1.5708, 1.5708, 0.0, 0.0, 0.0],
    },
    "astribot_s1_mobile_sharpa": {
        "left_joints":  [f"astribot_arm_left_joint_{i}"  for i in range(1, 8)],
        "right_joints": [f"astribot_arm_right_joint_{i}" for i in range(1, 8)],
        "left_eef":  "astribot_arm_left_tool_link",
        "right_eef": "astribot_arm_right_tool_link",
        "left_base":  "chassis_base",
        "right_base": "chassis_base",
        "skip_link8_correction": True,
        "default_left_q":  [0.0, 0.0, -1.5708, 1.5708, 0.0, 0.0, 0.0],
        "default_right_q": [0.0, 0.0,  1.5708, 1.5708, 0.0, 0.0, 0.0],
    },
}

DEFAULT_LEFT_FRANKA_QPOS = [0.7854, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.7854]
DEFAULT_RIGHT_FRANKA_QPOS = [-0.7854, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.7854]

# Local quat baked into each assembly link8 in dual_fr3v2_sharpa.xml (~45°
# around Z). The standalone fr3v2.xml's link8 has no such quat, so we peel K
# off when handing IK targets to the standalone solver.
_LINK8_CORRECTION = np.array(
    [0.9238795325112867, 0.0, 0.0, 0.3826834323650898])
_LINK8_CORRECTION_INV = np.array(
    [0.9238795325112867, 0.0, 0.0, -0.3826834323650898])


# ---------------------------------------------------------------------------
# Module-level scene/robot assembly helpers
# ---------------------------------------------------------------------------

def _resolve_robot_type(task: Optional[Task] = None,
                        override: Optional[str] = None,
                        config_robot_type: Optional[str] = None) -> str:
    """Priority: explicit override > Task.robot_type > config default > built-in."""
    if override:
        return override
    if task is not None and getattr(task, "robot_type", None):
        return task.robot_type
    return config_robot_type or DEFAULT_ROBOT_TYPE


def _load_camera_profile(robot_type: str, fallback: Optional[dict] = None) -> dict:
    """Load bidex/robots/camera_profiles/<robot_type>.yaml, else fallback."""
    profiles_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "robots", "camera_profiles"
    )
    path = os.path.join(profiles_dir, f"{robot_type}.yaml")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return fallback or {}


def _resolve_task_scene_xml(task: Task) -> Tuple[str, List[str]]:
    """Resolve a task's scene to an MJCF path, materializing it if needed."""
    if task.scene.startswith("robocasa_kitchen"):
        if task.layout_id is None or task.style_id is None:
            raise ValueError(
                f"task '{task.name}' uses scene={task.scene} but has no "
                f"layout_id/style_id"
            )
        from bidex.robocasa import build_robocasa_scene_xml
        return build_robocasa_scene_xml(
            task.layout_id,
            task.style_id,
            enable_fixtures=task.enable_fixtures or None,
            clutter_mode=task.clutter_mode,
            seed=task.scene_seed,
        )
    scene_xml = resolve_scene_xml(task.scene)
    generated_ood = (os.path.basename(scene_xml).startswith("ood_scene_")
                     and scene_xml.endswith(".xml"))
    return scene_xml, [scene_xml] if generated_ood else []


def _resolve_scene_objects_pose(
    task: Optional[Task] = None,
    add_freejoint: bool = True,
) -> Tuple[str, Tuple[float, float, float], Tuple[float, float, float, float],
           List[ObjectSpec], List[str]]:
    """Resolve (scene_xml, robot_pos, robot_quat, object_specs, scene_temp_files).

    Task wins if provided: scene + objects + robot pose come from the YAML.
    Otherwise baked-in defaults are used (standalone-env case).
    """
    if task is not None:
        robot_pos, robot_quat = base_offset_for(task.scene, task.robot_pose)
        specs = task.to_object_specs()
        scene_xml, extra = _resolve_task_scene_xml(task)
        return scene_xml, robot_pos, robot_quat, specs, extra

    DEFAULT_SCENE = "libero_kitchen_tabletop_base_style"
    DEFAULT_POS = (0.0, -0.4, 0.8)
    DEFAULT_QUAT = (0.382683, 0.0, 0.0, 0.92388)
    DEFAULT_OBJECTS = [
        "akita_black_bowl@-0.15,0.0,0.85",
        "porcelain_mug@0.10,0.10,0.85",
    ]
    specs = [_parse_object_spec(s) for s in DEFAULT_OBJECTS]
    return resolve_scene_xml(DEFAULT_SCENE), DEFAULT_POS, DEFAULT_QUAT, specs, []


def _inline_mjcf_includes(parent_path: str, root: ET.Element) -> None:
    """Recursively replace every <include file="..."> with the referenced
    file's <mujoco> children, in place. Paths resolve relative to the file
    that contained the include (MuJoCo's own convention).

    Needed before camera injection: the per-robot camera profile's attach
    body (e.g. astribot_arm_left_tool_link) lives in an <include>d subtree
    that ElementTree doesn't follow natively.
    """
    parent_dir = os.path.dirname(os.path.abspath(parent_path))
    changed = True
    while changed:
        changed = False
        parents = {c: p for p in root.iter() for c in p}
        for elem in list(root.iter()):
            if elem.tag != "include":
                continue
            fpath = elem.get("file")
            if not fpath:
                continue
            inc_path = fpath if os.path.isabs(fpath) else os.path.join(parent_dir, fpath)
            if not os.path.isfile(inc_path):
                continue
            inc_tree = ET.parse(inc_path)
            inc_root = inc_tree.getroot()
            _inline_mjcf_includes(inc_path, inc_root)
            parent = parents.get(elem)
            if parent is None:
                continue
            idx = list(parent).index(elem)
            parent.remove(elem)
            for offset, child in enumerate(list(inc_root)):
                parent.insert(idx + offset, child)
            changed = True
            break


def _build_combined_xml_with_cameras(
    config: dict,
    task: Optional[Task] = None,
    robot_type: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Compat shim for fork-derived helper scripts that call the legacy
    ``_build_combined_xml_with_cameras(config, task=...)`` scene assembler.

    The target refactored this into ``build_combined_xml(task, robot_type, ...)``.
    The fork's contract ignored ``config["sim_config"]`` whenever a ``task`` was
    provided (scene/objects/robot-pose came from the Task), and every caller in
    ``bidex/scripts/`` passes ``task=``, so ``config`` is unused here and we
    simply delegate. Kept so the probe/validate scripts import unchanged.
    """
    _ = config  # accepted for signature compat; ignored when task is given (fork semantics)
    return build_combined_xml(task=task, robot_type=robot_type)


def build_combined_xml(
    task: Optional[Task] = None,
    robot_type: Optional[str] = None,
    add_freejoint: bool = True,
    camera_profile: Optional[dict] = None,
) -> Tuple[str, List[str]]:
    """Build combined MJCF (scene + robot + objects) with cameras injected.

    Returns (combined_xml_path, temp_files) — caller must clean up temp_files.
    """
    resolved_robot_type = _resolve_robot_type(task=task, override=robot_type)
    cam_cfg = camera_profile or _load_camera_profile(resolved_robot_type)
    robot_xml = resolve_robot_xml(resolved_robot_type)

    scene_xml, robot_pos, robot_quat, object_specs, scene_temp_files = (
        _resolve_scene_objects_pose(task=task, add_freejoint=add_freejoint)
    )

    if task is not None:
        per_obj_freejoint = [obj.freejoint for obj in task.all_object_inits()]
    else:
        per_obj_freejoint = [add_freejoint] * len(object_specs)

    robot_tree = ET.parse(robot_xml)
    robot_root = robot_tree.getroot()
    _inline_mjcf_includes(robot_xml, robot_root)

    wb = robot_root.find("worldbody")
    base = None
    for b in wb.findall("body"):
        if b.get("name") == "base":
            base = b
            break
    if base is None:
        base = wb.find("body")

    base.set("pos", f"{robot_pos[0]} {robot_pos[1]} {robot_pos[2]}")
    # Set the base orientation to the task-specified `robot_quat` (default: identity).
    # By default we OVERWRITE the chassis body's built-in euler (e.g. astribot's
    # `euler="0 0 -1.57"`) rather than composing with it, matching how the
    # collection-time environment placed
    # the robot — so eval and collection share the same base orientation. Setting
    # env BIDEX_COMPOSE_BASE_QUAT=1 restores the prior "compose task quat on top
    # of the chassis euler" behaviour (used to fix a ~0.1m ee-mode IK offset).
    import numpy as _np
    tq = _np.array([robot_quat[0], robot_quat[1], robot_quat[2], robot_quat[3]],
                   dtype=np.float64)
    if os.environ.get("BIDEX_COMPOSE_BASE_QUAT") == "1":
        orig_quat = _np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        if "quat" in base.attrib:
            orig_quat = _np.array([float(x) for x in base.get("quat").split()])
        elif "euler" in base.attrib:
            from scipy.spatial.transform import Rotation as _Rot
            e = [float(x) for x in base.get("euler").split()]
            xyzw = _Rot.from_euler('XYZ', e).as_quat()
            orig_quat = _np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        final_quat = _np.zeros(4)
        mujoco.mju_mulQuat(final_quat, tq, orig_quat)
    else:
        final_quat = tq
    for _orient_attr in ("euler", "axisangle", "xyaxes", "zaxis"):
        if _orient_attr in base.attrib:
            del base.attrib[_orient_attr]
    base.set("quat", f"{final_quat[0]} {final_quat[1]} {final_quat[2]} {final_quat[3]}")

    for cam in cam_cfg.get("cameras", []):
        attach = cam.get("attach", "world")
        pos_str = " ".join(str(v) for v in cam["pos"])
        fovy = cam.get("fovy", 60)
        cam_attrs = {
            "name": cam["name"],
            "pos": pos_str,
            "fovy": str(fovy),
            "mode": "fixed",
        }
        if "quat" in cam:
            cam_attrs["quat"] = " ".join(str(v) for v in cam["quat"])
        elif "euler" in cam:
            cam_attrs["euler"] = " ".join(str(v) for v in cam["euler"])
        else:
            raise ValueError(
                f"Camera profile entry '{cam.get('name')}' has neither quat nor euler"
            )
        cam_elem = ET.Element("camera", cam_attrs)

        if attach == "world":
            wb.append(cam_elem)
        else:
            target_body = None
            for body in robot_root.iter("body"):
                if body.get("name") == attach:
                    target_body = body
                    break
            if target_body is not None:
                target_body.append(cam_elem)
            else:
                print(f"[BiDexEnv] Warning: camera attach body '{attach}' not found")
                wb.append(cam_elem)

    robot_dir = os.path.dirname(os.path.abspath(robot_xml))
    rfd, posed_robot = tempfile.mkstemp(
        prefix="sim_robot_posed_", suffix=".xml", dir=robot_dir
    )
    os.close(rfd)
    # The robot XML ships a robot-only ``home`` keyframe whose qpos spans only the
    # robot's own joints. Once the assembled scene adds free-jointed task objects
    # (each free object contributes +7 qpos, e.g. basket+fruits -> many extra), the
    # combined model's nq outgrows that keyframe's qpos length. mujoco builds that
    # strictly validate keyframe sizes (3.3.x galaxea) raise at MjModel.from_xml_path
    # ("keyframe 0: invalid qpos size, expected length 79"), while tolerant builds
    # (3.8.x openpi) silently zero-pad it — so the SAME task/robot builds for one
    # model and crashes for another. The eval never reads the ``home`` keyframe
    # (qpos is set via arm_init_qpos + restored collection init_states), so strip
    # keyframes from the posed-robot temp copy so the combined XML compiles
    # identically on every mujoco version. Only the temp copy is touched, not the
    # source robot XML on disk.
    _robot_root = robot_tree.getroot()
    for _kf in list(_robot_root.findall("keyframe")):
        _robot_root.remove(_kf)
    robot_tree.write(posed_robot, encoding="utf-8", xml_declaration=True)

    temp_files = [posed_robot]
    temp_files.extend(scene_temp_files)

    object_temp_paths: List[str] = []
    for i, spec in enumerate(object_specs, start=1):
        if len(spec) == 5:
            name, pos, quat, obj_scale, obj_mass_scale = spec
        else:
            name, pos, quat = spec
            obj_scale, obj_mass_scale = 1.0, 1.0
        obj_xml, _cat = resolve_object_xml(name)
        obj_dir = os.path.dirname(obj_xml)
        ofd, op = tempfile.mkstemp(
            prefix=f"obj{i}_{name.replace('/', '_')}_", suffix=".xml", dir=obj_dir
        )
        os.close(ofd)
        fj = per_obj_freejoint[i - 1] if i - 1 < len(per_obj_freejoint) else add_freejoint
        _prepare_object_xml(
            obj_xml, i, pos, quat, fj, op,
            scale=float(obj_scale),
            mass_scale=float(obj_mass_scale) if obj_mass_scale is not None else 1.0,
        )
        object_temp_paths.append(op)
        temp_files.append(op)

    includes = [f'  <include file="{os.path.abspath(scene_xml)}"/>',
                f'  <include file="{posed_robot}"/>']
    for op in object_temp_paths:
        includes.append(f'  <include file="{op}"/>')

    cfd, combined_path = tempfile.mkstemp(prefix="sim_combined_", suffix=".xml")
    os.close(cfd)
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write('<mujoco model="bidex_sim_env">\n')
        f.write("\n".join(includes))
        f.write("\n</mujoco>\n")
    temp_files.append(combined_path)

    return combined_path, temp_files


# ===========================================================================
# BiDexEnv
# ===========================================================================

class BiDexEnv:
    """Synchronous gym-style MuJoCo environment for imitation-learning eval.

    Single-threaded: ``step()`` advances physics in the caller's thread.
    The renderer is created lazily on the first ``render()``/``get_obs()``
    call (one GL context owned by the caller's thread). A re-entrant lock
    is kept for safety but is never contended in normal single-threaded use.
    """

    # IK solver budget / tolerances (see _solve_eef_ik).
    _IK_MAX_ITERS = 200
    _IK_POS_TOL = 1e-3
    _IK_ROT_TOL = 1e-2

    def __init__(
        self,
        task: Optional[Task] = None,
        *,
        robot_type: Optional[str] = None,
        control_mode: str = "joint",
        ik_solver: str = "mink",
        cameras: Optional[List[str]] = None,
        render_depth: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
        camera_render_scale: float = 0.75,   # eval renders 0.75x (360x480) = the bidex_qc3 train res; config.py SimConfig.camera_render_scale mirrors this for callers that pass config=.
        step_freq_hz: int = 500,
        control_freq_hz: int = 25,   # 25 Hz control (rebuilt 25 fps bidex_qc3 dataset)
        add_freejoint: bool = True,
        seed: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        if control_mode not in ("joint", "ee"):
            raise ValueError(f"control_mode must be 'joint' or 'ee', got {control_mode!r}")

        self.task = task
        self._config = config or {}
        sim_cfg = self._config.get("sim_config", {}) if self._config else {}
        cfg_robot_type = (self._config.get("robot_config", {}) or {}).get("robot_type") if self._config else None

        self.robot_type = _resolve_robot_type(
            task=task, override=robot_type, config_robot_type=cfg_robot_type)
        self.control_mode = control_mode
        self.ik_solver_name = ik_solver
        self._render_depth = render_depth
        self._requested_cameras = cameras  # None = all profile cameras

        print(f"[BiDexEnv] Task: {task.suite+'/'+task.name if task else '<none>'}"
              f"  Robot: {self.robot_type}  mode: {control_mode}")

        # ---- Build combined MJCF + load model ----
        cam_cfg = _load_camera_profile(self.robot_type)
        self.cam_cfg = cam_cfg
        self.combined_path, self.temp_files = build_combined_xml(
            task=task, robot_type=self.robot_type,
            add_freejoint=sim_cfg.get("add_freejoint", add_freejoint),
            camera_profile=cam_cfg,
        )
        print(f"[BiDexEnv] Loading model from: {self.combined_path}")
        try:
            # MuJoCo occasionally reports a transient
            # ``unknown error in attribute 'mass'/'rgba'`` while many evaluator
            # processes compile include-heavy MJCFs from the shared NAS. The same
            # immutable combined XML succeeds immediately on retry. Retry only
            # this narrow parser signature so deterministic malformed MJCF still
            # fails fast and remains visible.
            for attempt in range(3):
                try:
                    self.model = mujoco.MjModel.from_xml_path(self.combined_path)
                    break
                except ValueError as exc:
                    transient = "XML Error: unknown error in attribute" in str(exc)
                    if not transient or attempt == 2:
                        raise
                    delay = 0.2 * (attempt + 1)
                    print(f"[BiDexEnv] transient MuJoCo XML parse failure; "
                          f"retry {attempt + 2}/3 after {delay:.1f}s: {exc}",
                          flush=True)
                    time.sleep(delay)
        finally:
            # MuJoCo has fully compiled meshes, textures and included MJCF into
            # MjModel at this point. Keeping the generated files until close()
            # only makes interrupted evaluations leak temporary XML files.
            for path in self.temp_files:
                try:
                    os.remove(path)
                except OSError:
                    pass
            self.temp_files = []
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Deep-copy the task so each env instance owns its ObjectInit runtime
        # state (place_seq, remove_seq, press_count, flyover_count, color_*).
        # In the vectorized eval path, 50 BiDexEnv instances are constructed
        # from the SAME loaded Task; without this copy they share ObjectInit
        # instances and pollute each other's runtime state (e.g. env A's
        # basket place_seq gets entries from env B's objects, making
        # `placed_n` latch prematurely).
        # Safe: MJCF is already built from the original task; deepcopy
        # preserves instance_ids so mj_name2id lookups still resolve.
        import copy as _copy
        self.task = _copy.deepcopy(task)

        self.step_freq = int(sim_cfg.get("step_freq_hz", step_freq_hz))
        self.control_freq = int(sim_cfg.get("control_freq_hz", control_freq_hz))
        self.model.opt.timestep = 1.0 / self.step_freq
        self._n_substeps = max(1, round(self.step_freq / self.control_freq))

        self._build_actuator_maps()
        self._build_robot_jnt_ids()
        self._build_camera_ids()
        self.camera_pose_metadata = self._apply_camera_pose_plan(
            getattr(task, "_ood_camera_plan", None)
        )
        self._build_body_ids()

        self.lock = threading.RLock()

        # Camera resolution.
        render_scale = float(sim_cfg.get("camera_render_scale", camera_render_scale))
        self.img_width = int(cam_cfg.get("width", 640) * render_scale)
        self.img_height = int(cam_cfg.get("height", 480) * render_scale)
        if image_size is not None:
            self.img_height, self.img_width = int(image_size[0]), int(image_size[1])
        self.img_width = self.img_width & ~1
        self.img_height = self.img_height & ~1

        # Renderer created lazily on first render in the caller's thread.
        self.renderer = None
        self._scene_option = None
        self._render_available = True
        # Shared-renderer support (vectorized path): when ``_render_hook`` is
        # set (a callable ``(cam_name, need_depth) -> (color, depth)`` owned by
        # a single dedicated render thread), ``_render_camera`` routes through
        # it instead of building/using this env's own GL context. This keeps
        # exactly ONE EGL context per worker (=> no NVRM Xid-31 multi-context
        # fault). ``_owns_renderer`` tells ``close()`` whether it may free the
        # renderer (False when the hook supplies a shared one owned elsewhere).
        self._render_hook = None
        self._owns_renderer = True

        # Resolve which cameras to actually render.
        all_cams = [c["name"] for c in cam_cfg.get("cameras", [])]
        if self._requested_cameras is None:
            self._active_cameras = [n for n in all_cams if n in self.camera_ids]
        else:
            self._active_cameras = [n for n in self._requested_cameras
                                    if n in self.camera_ids]

        # ---- IK solver (per-arm) ----
        self._build_ik_solvers()

        # ---- Episode bookkeeping ----
        self._step_count = 0
        self._subtask_tracker: Optional[SubtaskTracker] = None
        self._max_seconds = float(getattr(task, "max_seconds", 0) or 0)
        self._last_sampled: dict = {}

        # Apply initial state once so the model is in a sane pose.
        self._apply_initial_state(seed=seed)

        # dexjoco per-feature indices (no-op for non-dexjoco tasks).
        self._build_hammer_nail_indices()
        self._build_spray_plant_indices()
        self._build_button_indices()
        self._build_basket_indices()
        self._build_sink_water_indices()

    # ------------------------------------------------------------------
    # Build maps / solvers
    # ------------------------------------------------------------------

    def _build_actuator_maps(self):
        def _name2id_or_skip(obj_kind, names):
            ids = []
            for n in names:
                i = mujoco.mj_name2id(self.model, obj_kind, n)
                if i < 0:
                    return []
                ids.append(i)
            return ids

        prof = _ARM_ID_PROFILES.get(self.robot_type)
        if prof is not None:
            left_names = prof["left_joints"]
            right_names = prof["right_joints"]
        else:
            left_names = [f"left_fr3v2_joint{i}" for i in range(1, 8)]
            right_names = [f"right_fr3v2_joint{i}" for i in range(1, 8)]

        self.left_franka_act_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_ACTUATOR, left_names)
        self.right_franka_act_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_ACTUATOR, right_names)
        self.left_sharpa_act_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_SHARPA_ACTUATORS)
        self.right_sharpa_act_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_SHARPA_ACTUATORS)

        self.left_franka_jnt_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_JOINT, left_names)
        self.right_franka_jnt_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_JOINT, right_names)
        self.left_sharpa_jnt_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_JOINT, LEFT_SHARPA_JOINTS)
        self.right_sharpa_jnt_ids = _name2id_or_skip(mujoco.mjtObj.mjOBJ_JOINT, RIGHT_SHARPA_JOINTS)

    def _build_robot_jnt_ids(self):
        """Cache ``self._robot_jnt_ids`` — the sorted list of ROBOT joint ids
        (arm+dex+torso+head): the contiguous qpos prefix that precedes any object
        joint. Used by :meth:`settle` to rigidly pin the robot while only
        freejoint objects relax.

        Mirrors the module-level ``_robot_jnt_ids(env)`` helper in
        run_leaderboard_eval (kept locally here to avoid a circular import).
        Computed once at build — the MJCF/model is immutable across resets, so
        the id set never changes."""
        ids = []
        for attr in ("right_franka_jnt_ids", "left_franka_jnt_ids",
                     "right_sharpa_jnt_ids", "left_sharpa_jnt_ids"):
            v = getattr(self, attr, None)
            if v:
                ids.extend(j for j in v if j >= 0)
        # astribot (+ any robot with torso/head joints BEFORE arm/dex in the
        # MJCF): include all joints at qposadr < min(arm/dex qposadr) that are
        # NOT object joints — object free joints come AFTER the robot prefix.
        if ids:
            m = self.model
            min_arm_adr = min(m.jnt_qposadr[j] for j in ids)
            ids_seen = set(ids)
            for j in range(m.njnt):
                if j in ids_seen:
                    continue
                adr = m.jnt_qposadr[j]
                if adr < min_arm_adr:
                    name = m.jnt(j).name
                    if name and not name.startswith("obj_") and "free" not in name.lower():
                        ids.append(j)
        self._robot_jnt_ids = sorted(ids, key=lambda j: self.model.jnt_qposadr[j])

    def _build_camera_ids(self):
        self.camera_ids = {}
        for cam in self.cam_cfg.get("cameras", []):
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam["name"])
            self.camera_ids[cam["name"]] = cid

    def _apply_camera_pose_plan(self, plan: Optional[dict]) -> dict:
        """Apply committed OOD extrinsic deltas in attachment-body coordinates.

        ``model.cam_pos`` and ``model.cam_quat`` are local to the camera's
        attachment body.  Position deltas therefore model mount translation;
        right-composed intrinsic XYZ rotations model mount-angle error around
        the calibrated camera axes.  Intrinsics remain unchanged.
        """
        if not plan:
            return {}
        from bidex.eval.ood import CAMERA_RANDOMIZATION_SCHEMA
        if plan.get("schema") != CAMERA_RANDOMIZATION_SCHEMA:
            raise ValueError(f"unsupported OOD camera plan: {plan.get('schema')!r}")
        if plan.get("robot_type") != self.robot_type:
            raise ValueError(
                f"camera plan robot {plan.get('robot_type')!r} does not match "
                f"environment robot {self.robot_type!r}"
            )
        from scipy.spatial.transform import Rotation

        applied = {
            "schema": plan["schema"],
            "manifest_fingerprint": plan["manifest_fingerprint"],
            "cameras": {},
        }
        for camera_name, delta in plan.get("cameras", {}).items():
            cid = self.camera_ids.get(camera_name, -1)
            if cid is None or cid < 0:
                raise ValueError(f"OOD camera not found in model: {camera_name}")
            nominal_pos = np.asarray(self.model.cam_pos[cid], dtype=np.float64).copy()
            nominal_wxyz = np.asarray(self.model.cam_quat[cid], dtype=np.float64).copy()
            delta_pos = np.asarray(delta["delta_pos"], dtype=np.float64)
            delta_euler = np.asarray(delta["delta_euler_deg"], dtype=np.float64)
            if delta_pos.shape != (3,) or delta_euler.shape != (3,):
                raise ValueError(f"invalid OOD camera delta for {camera_name}")

            nominal_xyzw = np.asarray([
                nominal_wxyz[1], nominal_wxyz[2], nominal_wxyz[3], nominal_wxyz[0]
            ])
            nominal_rot = Rotation.from_quat(nominal_xyzw)
            local_delta = Rotation.from_euler("XYZ", delta_euler, degrees=True)
            final_xyzw = (nominal_rot * local_delta).as_quat()
            final_wxyz = np.asarray([
                final_xyzw[3], final_xyzw[0], final_xyzw[1], final_xyzw[2]
            ])
            final_wxyz /= np.linalg.norm(final_wxyz)
            final_pos = nominal_pos + delta_pos
            self.model.cam_pos[cid] = final_pos
            self.model.cam_quat[cid] = final_wxyz
            applied["cameras"][camera_name] = {
                "pose_id": delta["pose_id"],
                "nominal_pos": nominal_pos.tolist(),
                "nominal_quat_wxyz": nominal_wxyz.tolist(),
                "delta_pos": delta_pos.tolist(),
                "delta_euler_deg": delta_euler.tolist(),
                "final_pos": final_pos.tolist(),
                "final_quat_wxyz": final_wxyz.tolist(),
            }
        mujoco.mj_forward(self.model, self.data)
        return applied

    def _build_body_ids(self):
        m = self.model
        prof = _ARM_ID_PROFILES.get(self.robot_type, {
            "left_eef": "left_fr3v2_link8",
            "right_eef": "right_fr3v2_link8",
            "left_base": "left_fr3v2_link0",
            "right_base": "right_fr3v2_link0",
        })
        self.left_eef_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prof["left_eef"])
        self.right_eef_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prof["right_eef"])
        self.left_base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prof["left_base"])
        self.right_base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prof["right_base"])

    def _build_ik_solvers(self):
        """Construct per-arm IK solvers from the robot's IK profile.

        Franka uses the vendored mink QP solver (FrankaKinematicsSolver).
        Astribot uses the SLSQP ArmRetarget from the humanoid-arm-retarget
        pipeline (mink never converged on astribot — see the original
        start_sim_robot_avp.sh). The retarget solves both arms jointly
        (10-DOF SLSQP) and its FK lives in the chassis_base local frame.
        """
        _DEFAULT_FRANKA_XML = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "robots", "arms", "franka_fr3_v2", "fr3v2.xml")
        ik_backend = self.ik_solver_name

        self._left_eef_target_frame = "world"
        self._right_eef_target_frame = "world"
        self._use_retarget = self.robot_type.startswith("astribot")

        if self._use_retarget:
            from bidex.ik.retarget.astribot.arms_retarget import (
                ArmRetarget as _AstribotRetarget)
            self._astribot_retarget = _AstribotRetarget(
                config_file="config_astribot.yaml")
            # mink solvers still constructed for FK-only introspection, but
            # the ee control path uses the retarget.
            self.left_ik_solver = None
            self.right_ik_solver = None
            self._left_ik_chassis = None
            self._right_ik_chassis = None
            print(f"[BiDexEnv] IK solver: retarget(SLSQP)  robot={self.robot_type}")
            return

        def _build_solver(side: str):
            try:
                entry = resolve_ik_profile(self.robot_type, side)
                tf = entry.get("target_frame", "world")
                if side == "left":
                    self._left_eef_target_frame = tf
                else:
                    self._right_eef_target_frame = tf
                return FrankaKinematicsSolver(
                    profile=RobotProfile(**entry["profile_kwargs"]),
                    solver=entry["solver"],
                    orientation_cost=1.0,
                )
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"[BiDexEnv] no IK profile for {self.robot_type!r}/{side} "
                      f"({e}); falling back to franka_model_path")
                return FrankaKinematicsSolver(_DEFAULT_FRANKA_XML, solver=ik_backend)

        try:
            self.left_ik_solver = _build_solver("left")
            self.right_ik_solver = _build_solver("right")
            print(f"[BiDexEnv] IK solver: {ik_backend}  robot={self.robot_type}  "
                  f"target_frame L={self._left_eef_target_frame} R={self._right_eef_target_frame}")
        except Exception as e:
            print(f"[BiDexEnv] WARNING: failed to init '{ik_backend}' solver "
                  f"({e}); falling back to dls")
            self.left_ik_solver = FrankaKinematicsSolver(_DEFAULT_FRANKA_XML, solver="dls")
            self.right_ik_solver = FrankaKinematicsSolver(_DEFAULT_FRANKA_XML, solver="dls")

        self._left_ik_chassis = self._cache_ik_chassis_pose(self.left_ik_solver, "left_base")
        self._right_ik_chassis = self._cache_ik_chassis_pose(self.right_ik_solver, "right_base")

    def _cache_ik_chassis_pose(self, solver, prof_key: str):
        """Read the IK solver's per-arm base body STATIC pose for the chassis
        post-mult in _solve_eef_ik. Returns (xyz, quat_wxyz) or None."""
        try:
            prof = _ARM_ID_PROFILES.get(self.robot_type, {})
            body_name = prof.get(prof_key)
            if not body_name:
                return None
            ik_model = getattr(solver, "model", None)
            if ik_model is None:
                return None
            bid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                bid = mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_BODY, "base")
                if bid < 0:
                    return None
            xyz = np.asarray(ik_model.body_pos[bid], dtype=np.float64).copy()
            quat = np.asarray(ik_model.body_quat[bid], dtype=np.float64).copy()
            if (np.linalg.norm(xyz) < 1e-9
                    and abs(quat[0] - 1.0) < 1e-9
                    and np.linalg.norm(quat[1:]) < 1e-9):
                return None
            print(f"[BiDexEnv] IK chassis post-mult for {prof_key}: "
                  f"body={body_name!r} pos={xyz.tolist()} quat={quat.tolist()}")
            return xyz, quat
        except Exception as e:
            print(f"[BiDexEnv] _cache_ik_chassis_pose({prof_key}) failed: {e}")
            return None

    # ------------------------------------------------------------------
    # State save/restore
    # ------------------------------------------------------------------

    def _flatten_state(self) -> np.ndarray:
        return np.concatenate(
            [[self.data.time], self.data.qpos.copy(), self.data.qvel.copy()]
        )

    def _set_state_from_flattened(self, state: np.ndarray):
        nq, nv = self.model.nq, self.model.nv
        if state.size != 1 + nq + nv:
            raise ValueError(
                f"init state size {state.size} != 1 + nq({nq}) + nv({nv})"
            )
        self.data.time = float(state[0])
        self.data.qpos[:] = state[1:1 + nq]
        self.data.qvel[:] = state[1 + nq:1 + nq + nv]
        mujoco.mj_forward(self.model, self.data)

    def _repose_welded_after_restore(self, seed: Optional[int] = None) -> None:
        """After restoring a saved qpos state, re-pose welded+region objects.

        A welded (freejoint: false) object has no free joint, so its pose is NOT
        in the saved qpos -- the restore leaves it at its baked placeholder pose,
        ignoring its `region:`. This re-runs the region sampler for welded
        objects only (free-jointed bodies keep their restored qpos) and writes
        the fresh pose via model.body_pos/quat, then mj_forward. Mirrors the
        welded loop in domain.apply_task_initial_state (Path 3) for the restore
        paths (Path 1/2), which skip apply_task_initial_state.
        """
        if self.task is None:
            return
        try:
            from bidex.tasks.domain import repose_welded_objects, apply_articulations
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            with self.lock:
                repose_welded_objects(self.model, self.data, self.task, rng)
                # Also reset articulated joints (drawer/door/cap) to their
                # declared init qpos -- a restored saved state may have the
                # drawer open from the previous episode, and without this the
                # reset/refresh leaves it open.
                apply_articulations(self.model, self.data, self.task)
                mujoco.mj_forward(self.model, self.data)
        except Exception as e:  # noqa: BLE001 -- never break reset over a pose fix
            print(f"[BiDexEnv] _repose_welded_after_restore failed: {type(e).__name__}: {e}")

    def _apply_initial_state(self, init_state_idx: Optional[int] = None,
                             seed: Optional[int] = None):
        """Set robot init qpos/ctrl, optionally restore a saved scene state."""
        # Path 1: task init-states library
        if self.task is not None and init_state_idx is not None:
            path = init_states_path(
                self.task.suite, self.task.name, self.task.init_states_file)
            # Staleness guard: if the task YAML changed AFTER the init-state
            # library was saved, the saved qpos is from an older scene (different
            # object scales / poses / regions / spawn z). Free-jointed objects
            # would be restored at outdated spawn poses -- e.g. sugar cubes set
            # to the old z=0.87 scatter around a since-resized bowl, so they
            # fall/fly out at reset. Skip the restore and fall through to Path 3
            # (fresh region sampling against the current YAML).
            yaml_newer = (
                getattr(self.task, "yaml_path", None)
                and os.path.exists(self.task.yaml_path)
                and os.path.exists(path)
                and os.path.getmtime(self.task.yaml_path) > os.path.getmtime(path)
            )
            if not yaml_newer:
                states = load_init_states(path)
            else:
                states = np.empty(0)
                print("[BiDexEnv] init state library stale (task YAML newer than "
                      "saved states); fresh init")
            if states.size and 0 <= init_state_idx < states.shape[0]:
                try:
                    self._set_state_from_flattened(states[init_state_idx])
                    self._repose_welded_after_restore(seed)
                    self._park_reveal_objects(recompute=True)
                    print(f"[BiDexEnv] Restored init state #{init_state_idx} from {path}")
                    return
                except ValueError as e:
                    # Stale library (e.g. saved against an older MJCF whose qpos
                    # size differs -- a freejoint removed/added, an object scale
                    # change). Fall through to Path 3 for a fresh init rather
                    # than crashing reset.
                    print(f"[BiDexEnv] init state #{init_state_idx} incompatible "
                          f"({e}); falling through to fresh init")
            elif not yaml_newer:
                print(f"[BiDexEnv] init_state_idx={init_state_idx} unavailable "
                      f"(have {states.shape[0]} rows); falling through to defaults")

        # Path 2: defaults + domain randomization
        prof = _ARM_ID_PROFILES.get(self.robot_type, {})
        default_left = prof.get("default_left_q", DEFAULT_LEFT_FRANKA_QPOS)
        default_right = prof.get("default_right_q", DEFAULT_RIGHT_FRANKA_QPOS)
        if self.task is not None and self.task.arm_init_qpos:
            left_q = self.task.arm_init_qpos.get("left", default_left)
            right_q = self.task.arm_init_qpos.get("right", default_right)
        else:
            left_q = default_left
            right_q = default_right

        for jid, q in zip(self.left_franka_jnt_ids, left_q):
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = q
        for jid, q in zip(self.right_franka_jnt_ids, right_q):
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = q
        for aid, q in zip(self.left_franka_act_ids, left_q):
            if aid >= 0:
                self.data.ctrl[aid] = q
        for aid, q in zip(self.right_franka_act_ids, right_q):
            if aid >= 0:
                self.data.ctrl[aid] = q

        # Per-robot init qpos from robots/init_qpos/<robot_type>.yaml.
        init_qpos = resolve_init_qpos(self.robot_type)
        if init_qpos:
            applied = 0
            for jname, qval in init_qpos.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                self.data.qpos[self.model.jnt_qposadr[jid]] = qval
                for aid in range(self.model.nu):
                    if self.model.actuator_trnid[aid, 0] == jid:
                        self.data.ctrl[aid] = qval
                applied += 1
            if applied:
                print(f"[BiDexEnv] init_qpos: applied {applied}/{len(init_qpos)} "
                      f"joints for robot={self.robot_type!r}")

        mujoco.mj_forward(self.model, self.data)

        if self.task is not None:
            rng = np.random.default_rng(seed)
            self._last_sampled = apply_task_initial_state(
                self.model, self.data, self.task, rng=rng)
            self._park_reveal_objects(recompute=False)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def _world_to_base_target(self, target_xyz_world, target_quat_world,
                              base_body_id) -> Tuple[np.ndarray, np.ndarray]:
        """Express a world-frame EE target in the arm-base frame."""
        base_xyz = self.data.xpos[base_body_id]
        base_quat = self.data.xquat[base_body_id]
        inv_q = np.zeros(4)
        mujoco.mju_negQuat(inv_q, base_quat)
        delta = target_xyz_world - base_xyz
        delta_rotated = np.zeros(3)
        mujoco.mju_rotVecQuat(delta_rotated, delta, inv_q)
        out_quat = np.zeros(4)
        mujoco.mju_mulQuat(out_quat, inv_q, target_quat_world)
        return delta_rotated, out_quat

    def _solve_eef_ik(self, target_xyz, target_quat, jnt_ids, body_id) -> np.ndarray:
        """Dispatch IK for one 7-DOF arm. Inputs are world-frame xyz + [w,x,y,z].
        Returns target qpos (length 7). Caller holds self.lock."""
        qadrs = [self.model.jnt_qposadr[j] for j in jnt_ids if j >= 0]
        if not qadrs:
            return np.zeros(len(jnt_ids))
        q_sim = np.array([self.data.qpos[a] for a in qadrs], dtype=np.float64)

        target_quat = np.asarray(target_quat, dtype=np.float64)
        if np.linalg.norm(target_quat) < 1e-9:
            return q_sim

        is_left = (body_id == self.left_eef_body_id)
        base_body_id = (self.left_base_body_id if is_left else self.right_base_body_id)
        solver = (self.left_ik_solver if is_left else self.right_ik_solver)

        prof_default = _ARM_ID_PROFILES.get(self.robot_type, {})
        default_q_key = "default_left_q" if is_left else "default_right_q"
        default_q = np.asarray(
            prof_default.get(default_q_key, [0.0] * 7), dtype=np.float64)
        # Seed IK from the current sim qpos (not a fixed default). In the
        # gym API there's no teleop leader-q to seed from, so the live arm
        # pose is the right basin: a no-op command (target = current EE)
        # converges back to the current q, and small deltas stay on the
        # same kinematic branch. Falling back to default_q only when the
        # sim qpos is degenerate (all zeros, e.g. before reset).
        q_current = q_sim.copy() if np.any(np.abs(q_sim) > 1e-6) else default_q.copy()

        target_frame = (self._left_eef_target_frame if is_left
                        else self._right_eef_target_frame)
        if target_frame == "base_local":
            xyz_base = np.asarray(target_xyz, dtype=np.float64)
            quat_base_asm = np.asarray(target_quat, dtype=np.float64)
        else:
            xyz_base, quat_base_asm = self._world_to_base_target(
                np.asarray(target_xyz, dtype=np.float64), target_quat, base_body_id)
            ik_chassis = (self._left_ik_chassis if is_left else self._right_ik_chassis)
            if ik_chassis is not None:
                ik_xyz, ik_quat = ik_chassis
                rotated = np.zeros(3)
                mujoco.mju_rotVecQuat(rotated, xyz_base, ik_quat)
                xyz_base = rotated + ik_xyz
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, ik_quat, quat_base_asm)
                quat_base_asm = new_quat

        prof = _ARM_ID_PROFILES.get(self.robot_type, {})
        if prof.get("skip_link8_correction"):
            quat_base = quat_base_asm
        else:
            quat_base = np.zeros(4)
            mujoco.mju_mulQuat(quat_base, quat_base_asm, _LINK8_CORRECTION_INV)

        try:
            # Retarget the posture task to the current arm pose so it stops
            # biasing the solution toward zeros. Use the solver's arm qpos
            # slots (_qpos_adr), NOT [:7] — on astribot the arm lives at
            # qpos[35:42] and [:7] would clobber the torso joints.
            solver.configuration.q[solver._qpos_adr] = q_current
            solver.posture_task.set_target_from_configuration(solver.configuration)
        except Exception:
            pass
        q_target, _ok = solver.inverse_kinematics(
            xyz_base, quat_base, q_init=q_current,
            max_iterations=self._IK_MAX_ITERS,
            pos_threshold=self._IK_POS_TOL,
            ori_threshold=self._IK_ROT_TOL,
        )
        return q_target

    def _solve_eef_ik_astribot(self, right_xyz, right_quat,
                                left_xyz, left_quat) -> np.ndarray:
        """Astribot dual-arm EE IK via a full-7-DoF SLSQP against the SIM
        model's own FK (``data.xpos``/``xquat`` of the tool_link bodies).

        Inputs are WORLD-frame xyz + wxyz quat per arm (right-first caller,
        but we solve both arms jointly). Returns q_14d ``[left(7), right(7)]``.

        Why not the ArmRetarget solver: the retarget's FK (analytical or the
        standalone-model ``fk_functions``) lives in the chassis-euler (-90 deg)
        local frame, and the qc collection recorded `abs_end_effector` in yet
        another (dual-base / banana) frame -- so a target expressed in sim
        WORLD frame does not round-trip through the retarget FK without an
        error-prone multi-frame rebase (the original 5-DoF retarget also folds
        wrist joints 6/7 at zero, so it can't represent the recorded EE at all,
        ~0.5 m off). Solving directly against the sim model's FK is
        frame-exact: a no-op target (current EE) returns the current arm q
        exactly, and the closed loop tracks commanded absolute EE poses
        tightly. Caller holds self.lock.
        """
        import scipy.optimize as _opt

        m, d = self.model, self.data
        lqa = [m.jnt_qposadr[j] for j in self.left_franka_jnt_ids if j >= 0]
        rqa = [m.jnt_qposadr[j] for j in self.right_franka_jnt_ids if j >= 0]
        l_bid = self.left_eef_body_id
        r_bid = self.right_eef_body_id
        # joint ranges for bounds (left 7 then right 7)
        lb = np.array([m.jnt_range[j][0] for j in self.left_franka_jnt_ids if j >= 0]
                      + [m.jnt_range[j][0] for j in self.right_franka_jnt_ids if j >= 0])
        ub = np.array([m.jnt_range[j][1] for j in self.left_franka_jnt_ids if j >= 0]
                      + [m.jnt_range[j][1] for j in self.right_franka_jnt_ids if j >= 0])
        bounds = list(zip(lb, ub))

        # q vector layout [L7, R7]; warm-start from current sim arm qpos.
        q0 = np.zeros(14, dtype=np.float64)
        q0[0:7] = [d.qpos[a] for a in lqa]
        q0[7:14] = [d.qpos[a] for a in rqa]

        tgt_l = (np.asarray(left_xyz, dtype=np.float64),
                 np.asarray(left_quat, dtype=np.float64))
        tgt_r = (np.asarray(right_xyz, dtype=np.float64),
                 np.asarray(right_quat, dtype=np.float64))

        # Save/restore the caller's qpos so IK probing does not mutate sim state.
        qpos_save = d.qpos.copy()

        def _fk(q):
            for a, v in zip(lqa, q[0:7]):
                d.qpos[a] = v
            for a, v in zip(rqa, q[7:14]):
                d.qpos[a] = v
            mujoco.mj_kinematics(m, d)  # xpos/xquat only -- cheap, no dynamics
            return (d.xpos[l_bid].copy(), d.xquat[l_bid].copy(),
                    d.xpos[r_bid].copy(), d.xquat[r_bid].copy())

        def _qang(a, b):
            dot = abs(float(np.dot(a, b)))
            return 2.0 * np.arccos(min(1.0, max(-1.0, dot)))

        def _cost(q):
            lp, lq, rp, rq = _fk(q)
            pc = (np.sum((lp - tgt_l[0]) ** 2) + np.sum((rp - tgt_r[0]) ** 2))
            oc = (_qang(lq, tgt_l[1]) ** 2 + _qang(rq, tgt_r[1]) ** 2)
            vc = float(np.sum((q - q0) ** 2))
            return 5000.0 * pc + 200.0 * oc + 0.5 * vc

        try:
            res = _opt.minimize(_cost, q0, method="SLSQP", bounds=bounds,
                                tol=1e-6,
                                options={"maxiter": 300, "ftol": 1e-6})
            q_14d = np.asarray(res.x, dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            print(f"[BiDexEnv] astribot ee IK failed ({e}); holding pose")
            q_14d = q0
        finally:
            d.qpos[:] = qpos_save
            mujoco.mj_kinematics(m, d)
        return q_14d  # [L7, R7]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _init_renderer(self):
        """Lazily construct the Renderer (and its GL context) in the caller's thread.

        We do NOT create a separate ``mujoco.GLContext`` here: ``mujoco.Renderer``
        already creates and owns its own GL context internally and re-makes it
        current on every ``render()`` call. A second, dangling context on the
        same EGL display only inflates the per-process context count (≈2× across
        all vectorized envs) — under that pressure the EGL driver can transiently
        swap/evict an offscreen framebuffer, so ``glReadPixels`` returns a
        row-reversed buffer for a frame and the (deterministic) ``flipud`` in
        ``Renderer.render()`` then produces an upside-down image. One context per
        env removes that pressure at the source."""
        if self.renderer is not None:
            return
        try:
            self.renderer = mujoco.Renderer(
                self.model, height=self.img_height, width=self.img_width)
            scene_opt = mujoco.MjvOption()
            scene_opt.geomgroup[0] = 0  # hide collision proxies
            scene_opt.geomgroup[1] = 1
            scene_opt.geomgroup[2] = 1
            self._scene_option = scene_opt
            print(f"[BiDexEnv] Renderer initialized ({self.model.ngeom} geoms, "
                  f"{self.img_width}x{self.img_height})", flush=True)
        except Exception as e:
            print(f"[BiDexEnv] Renderer init failed ({e}); rendering disabled.", flush=True)
            self._render_available = False

    def _render_camera(self, cam_name: str, need_depth: bool = True):
        """Render one camera's color (+ optional depth). Returns (color_rgb,
        depth_uint16) or (None, None) if unavailable."""
        if self._render_hook is not None:
            # Route through the shared renderer owned by a dedicated render
            # thread (vectorized path): never touch this env's own GL context.
            return self._render_hook(cam_name, need_depth)
        if self.renderer is None:
            self._init_renderer()
        if self.renderer is None or cam_name not in self.camera_ids:
            return None, None
        cam_id = self.camera_ids[cam_name]
        self.renderer.update_scene(self.data, camera=cam_id,
                                   scene_option=self._scene_option)
        color = self.renderer.render()
        depth_uint16 = None
        if need_depth:
            self.renderer.enable_depth_rendering()
            try:
                depth_raw = self.renderer.render()
            finally:
                self.renderer.disable_depth_rendering()
            depth_uint16 = (depth_raw * 1000).astype(np.uint16)
        return color, depth_uint16

    def render(self, camera: Optional[str] = None,
               depth: Optional[bool] = None) -> np.ndarray | tuple:
        """Render a single camera. If ``camera`` is None, renders the first
        active camera. Returns RGB uint8 (H,W,3), or (color, depth_uint16)
        when depth=True."""
        with self.lock:
            cam = camera or (self._active_cameras[0] if self._active_cameras else None)
            if cam is None:
                raise RuntimeError("no active camera to render")
            need_depth = self._render_depth if depth is None else depth
            color, d = self._render_camera(cam, need_depth=need_depth)
            if need_depth:
                return color, d
            return color

    # ------------------------------------------------------------------
    # Observation / action packing
    # ------------------------------------------------------------------

    @staticmethod
    def _get_joint_state(data, model, jnt_ids):
        q = np.array([data.qpos[model.jnt_qposadr[j]] for j in jnt_ids])
        dq = np.array([data.qvel[model.jnt_dofadr[j]] for j in jnt_ids])
        return q, dq

    def get_obs(self) -> dict:
        """Build the observation dict (right-arm-first packing)."""
        with self.lock:
            m, d = self.model, self.data

            # Arm joints: right-first [right_q(7), left_q(7)].
            if self.right_franka_jnt_ids and self.left_franka_jnt_ids:
                rq, rdq = self._get_joint_state(d, m, self.right_franka_jnt_ids)
                lq, ldq = self._get_joint_state(d, m, self.left_franka_jnt_ids)
                arm_q = np.concatenate([rq, lq]).astype(np.float32)
                arm_dq = np.concatenate([rdq, ldq]).astype(np.float32)
            else:
                arm_q = np.zeros(14, dtype=np.float32)
                arm_dq = np.zeros(14, dtype=np.float32)

            # Sharpa hands: right-first [right_hand(22), left_hand(22)].
            if self.right_sharpa_jnt_ids and self.left_sharpa_jnt_ids:
                rh, rhd = self._get_joint_state(d, m, self.right_sharpa_jnt_ids)
                lh, lhd = self._get_joint_state(d, m, self.left_sharpa_jnt_ids)
                dex_q = np.concatenate([rh, lh]).astype(np.float32)
                dex_dq = np.concatenate([rhd, lhd]).astype(np.float32)
            else:
                dex_q = np.zeros(44, dtype=np.float32)
                dex_dq = np.zeros(44, dtype=np.float32)

            # EE poses: right-first [right_xyz(3), right_quat_wxyz(4), left_...].
            ee = np.zeros(14, dtype=np.float32)
            if self.right_eef_body_id >= 0:
                ee[0:3] = d.xpos[self.right_eef_body_id]
                ee[3:7] = d.xquat[self.right_eef_body_id]
            if self.left_eef_body_id >= 0:
                ee[7:10] = d.xpos[self.left_eef_body_id]
                ee[10:14] = d.xquat[self.left_eef_body_id]

            # Images.
            images: Dict[str, np.ndarray] = {}
            if self._active_cameras and self._render_available:
                for cam_name in self._active_cameras:
                    color, depth = self._render_camera(
                        cam_name, need_depth=self._render_depth)
                    if color is not None:
                        images[cam_name] = color
                        if self._render_depth and depth is not None:
                            images[f"{cam_name}_depth"] = depth

        obs = {
            "arm_joint_position": arm_q,
            "arm_joint_velocity": arm_dq,
            "dex_joint_position": dex_q,
            "dex_joint_velocity": dex_dq,
            "abs_end_effector": ee,
            "images": images,
            "language": self._current_language(),
            "robot_type": self.robot_type,
            "control_mode": self.control_mode,
        }
        return obs

    def _current_language(self) -> str:
        """Instruction served to the policy at the current control step.

        Default: the task's static ``language``. When the task declares a
        ``language_schedule`` (Instruction_Following "switch" tasks), serve the
        text of the first segment whose ``until_step`` exceeds the current
        control step; the last segment (no until_step) is the final
        instruction. ``until_step`` counts CONTROL steps; ``until_fraction``
        is relative to the task's step budget (max_steps, else
        max_seconds x control freq)."""
        task = self.task
        if task is None:
            return ""
        sched = getattr(task, "language_schedule", None) or []
        if not sched:
            return task.language
        # resolve the task's step budget for until_fraction
        budget = getattr(task, "max_steps", None)
        if not budget:
            budget = int((getattr(task, "max_seconds", None) or 60.0) * self.control_freq)
        t = int(self._step_count)
        for seg in sched:
            until = seg.get("until_step")
            if until is None and seg.get("until_fraction") is not None:
                until = int(float(seg["until_fraction"]) * budget)
            if until is None:
                return seg.get("text", task.language)  # final segment
            if t < int(until):
                return seg.get("text", task.language)
        return sched[-1].get("text", task.language)

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------

    def reset(self, init_state_idx: Optional[int] = None,
              seed: Optional[int] = None) -> dict:
        """Reset to a fresh task initial state. Returns the first observation."""
        with self.lock:
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            self.data.time = 0.0
            self._apply_initial_state(init_state_idx=init_state_idx, seed=seed)
            mujoco.mj_forward(self.model, self.data)
            # dexjoco: re-pose mocap nails / spray / buttons / order for a fresh episode.
            self._reset_hammer_nail_state()
            self._reset_spray_state(seed=seed)
            self._reset_button_state(seed=seed)
            self._reset_order_state(seed=seed)
            self._reset_flyover_state(seed=seed)
            self._reset_color_swap_state(seed=seed)
            self._reset_sink_water_state(seed=seed)
        self._step_count = 0
        self._subtask_tracker = SubtaskTracker(self.task) if self.task is not None else None
        return self.get_obs()

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """Apply ``action`` and advance physics. Returns (obs, reward, terminated, info).

        Action (right-arm-first packing):
            joint mode: {"arm_joint_position": [14], "dex_joint_position": [44]}
            ee mode:    {"abs_end_effector":   [14], "dex_joint_position": [44]}
        """
        with self.lock:
            reward, terminated, info = self._physics_and_track(action)
        obs = self.get_obs()
        return obs, reward, terminated, info

    def step_physics_only(self, action: dict) -> Tuple[float, bool, dict]:
        """Advance physics + subtask/success tracking WITHOUT rendering (no
        ``get_obs``). Returns ``(reward, terminated, info)`` — no observation,
        since render-skip steps do not feed a policy query (the cached action
        chunk is executed; obs images are unused those steps).

        Physics + tracking order is byte-identical to ``step`` (both delegate
        to ``_physics_and_track``), so success/score/timed_out are unchanged
        vs a full ``step`` — only the rendered images are skipped.
        """
        with self.lock:
            reward, terminated, info = self._physics_and_track(action)
        return reward, terminated, info

    def settle(self, duration_s: float = 1.0) -> None:
        """Step physics ~``duration_s`` seconds WITHOUT rendering/recording so
        freshly placed freejoint objects settle under gravity, while the robot
        is RIGIDLY PINNED at its current (post-reset/restore) pose.

        Why pin: the robot arms are ``<position kp kv>`` servos. After a reset or
        in-domain collection-state restore, ``ctrl`` can mismatch the restored
        joint pose (restore sets qpos but leaves ctrl at the home pose) -> an
        unpinned settle would let the servos drive the arm off its target pose.
        Pinning (re-write robot qpos + zero robot qvel every tick) holds the arm
        exactly, independent of actuator type, while only freejoint objects relax.

        The eval rollout opens the mp4 writer AFTER this returns, so the settle
        period is excluded from the saved video, and the policy's first action is
        only dispatched once settle finishes. ``data.time`` is reset to 0 so the
        episode wall-clock budget (``max_seconds``) is not consumed, and the
        SubtaskTracker is re-created so the scoring baseline matches the
        stabilized frame-0 the policy sees. Settle is NOT a scored control step
        (``_step_count`` is unchanged). No-op if ``duration_s <= 0`` or the env
        has no robot joints.
        """
        if duration_s <= 0:
            return
        jids = getattr(self, "_robot_jnt_ids", None) or []
        if not jids:
            return  # nothing to pin -> nothing to settle
        m, d = self.model, self.data
        ts = max(float(m.opt.timestep), 1e-9)
        n_ticks = max(1, int(round(duration_s / ts)))
        qadr = np.array(sorted(int(m.jnt_qposadr[j]) for j in jids), dtype=int)
        vadr = np.array(sorted(int(m.jnt_dofadr[j]) for j in jids), dtype=int)
        jid_set = set(int(j) for j in jids)
        with self.lock:
            q0 = d.qpos[qadr].copy()
            # Quiesce robot position servos to the pinned pose so the per-tick
            # servo force is ~0 (avoids buzzing held contacts). Done once: settle
            # never calls _apply_action, so ctrl is unchanged between ticks.
            for aid in range(m.nu):
                jid = int(m.actuator_trnid[aid, 0])
                if jid in jid_set:
                    d.ctrl[aid] = d.qpos[m.jnt_qposadr[jid]]
            for _ in range(n_ticks):
                mujoco.mj_step(m, d)         # advance all bodies incl free objects
                d.qpos[qadr] = q0           # freeze robot joints at reset pose
                d.qvel[vadr] = 0.0          # zero robot joint velocities
            # Reflect the final pinned qpos in xpos/xquat so the next get_obs/
            # render shows the settled scene. (mj_step does not refresh derived
            # quantities from a manually-overwritten qpos, so mj_forward them.)
            mujoco.mj_forward(m, d)
            d.time = 0.0                    # don't charge settle to max_seconds
        # Re-create the tracker from the stabilized state (mirrors reset();
        # outside the lock, same as reset does). Guarded: settle must never break.
        try:
            self._subtask_tracker = SubtaskTracker(self.task) if self.task is not None else None
        except Exception:
            pass

    def _physics_and_track(self, action: dict) -> Tuple[float, bool, dict]:
        """Apply ``action``, advance physics, update subtask/success tracking.
        Caller MUST hold ``self.lock`` (``_step_physics`` re-acquires it as an
        RLock — same owner, re-entrant). Factored out of ``step`` so
        ``step`` and ``step_physics_only`` share identical ordering;
        returning ``info`` built here keeps the block a pure code-motion of
        the original ``step`` body.
        """
        self._apply_action(action)
        for _ in range(self._n_substeps):
            self._step_physics()
        self._step_count += 1

        # Subtask / success tracking.
        success = False
        predicates: List[dict] = []
        if self._subtask_tracker is not None:
            self._subtask_tracker.update(self.model, self.data)
            status = self._subtask_tracker.status()
            success = bool(status.get("success", False))
            predicates = status.get("subtasks", [])
        terminated = success

        timed_out = False
        if self._max_seconds > 0 and self.data.time >= self._max_seconds:
            timed_out = True

        reward = 1.0 if success else 0.0
        info = {
            "success": success,
            "subtasks": predicates,
            "predicates": predicates,
            "step_count": self._step_count,
            "time": float(self.data.time),
            "timed_out": timed_out,
        }
        return reward, terminated, info

    def _apply_action(self, action: dict) -> None:
        """Write action into data.ctrl. Caller holds self.lock."""
        d = self.data

        if self.control_mode == "joint":
            arm = action.get("arm_joint_position")
            if arm is not None:
                arm = np.asarray(arm, dtype=np.float64)
                # right-first: [0:7]=right, [7:14]=left
                if self.right_franka_act_ids and arm.shape[0] >= 7:
                    for i, aid in enumerate(self.right_franka_act_ids):
                        if aid >= 0:
                            d.ctrl[aid] = arm[i]
                if self.left_franka_act_ids and arm.shape[0] >= 14:
                    for i, aid in enumerate(self.left_franka_act_ids):
                        if aid >= 0:
                            d.ctrl[aid] = arm[7 + i]
        elif self.control_mode == "ee":
            ee = action.get("abs_end_effector")
            if ee is not None:
                ee = np.asarray(ee, dtype=np.float64)
                # right-first: [0:3]=right_xyz, [3:7]=right_quat_wxyz,
                #               [7:10]=left_xyz, [10:14]=left_quat_wxyz
                if self._use_retarget:
                    # Astribot: solve both arms jointly via the SLSQP retarget.
                    if ee.shape[0] >= 14:
                        q_lr = self._solve_eef_ik_astribot(
                            ee[0:3], ee[3:7], ee[7:10], ee[10:14])
                        # q_lr is [left(7), right(7)].
                        for i, aid in enumerate(self.left_franka_act_ids):
                            if aid >= 0:
                                d.ctrl[aid] = float(q_lr[i])
                        for i, aid in enumerate(self.right_franka_act_ids):
                            if aid >= 0:
                                d.ctrl[aid] = float(q_lr[7 + i])
                else:
                    # Franka: per-arm mink IK.
                    if ee.shape[0] >= 7 and self.right_eef_body_id >= 0:
                        q = self._solve_eef_ik(ee[0:3], ee[3:7],
                                               self.right_franka_jnt_ids,
                                               self.right_eef_body_id)
                        for i, aid in enumerate(self.right_franka_act_ids):
                            if aid >= 0:
                                d.ctrl[aid] = float(q[i])
                    if ee.shape[0] >= 14 and self.left_eef_body_id >= 0:
                        q = self._solve_eef_ik(ee[7:10], ee[10:14],
                                               self.left_franka_jnt_ids,
                                               self.left_eef_body_id)
                        for i, aid in enumerate(self.left_franka_act_ids):
                            if aid >= 0:
                                d.ctrl[aid] = float(q[i])

        # Hands (same in both modes): right-first [0:22]=right, [22:44]=left.
        dex = action.get("dex_joint_position")
        if dex is not None:
            dex = np.asarray(dex, dtype=np.float64)
            if self.right_sharpa_act_ids and dex.shape[0] >= 22:
                for i, aid in enumerate(self.right_sharpa_act_ids):
                    if aid >= 0:
                        d.ctrl[aid] = dex[i]
            if self.left_sharpa_act_ids and dex.shape[0] >= 44:
                for i, aid in enumerate(self.left_sharpa_act_ids):
                    if aid >= 0:
                        d.ctrl[aid] = dex[22 + i]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def check_success(self) -> dict:
        """Evaluate the bound task's goal predicates. Returns
        ``{"success": bool, "predicates": [...]}`` (empty with no task)."""
        if self.task is None:
            return {"success": False, "predicates": []}
        with self.lock:
            return evaluate_goal(self.model, self.data, self.task)

    def set_control_mode(self, mode: str) -> None:
        if mode not in ("joint", "ee"):
            raise ValueError(f"control_mode must be 'joint' or 'ee', got {mode!r}")
        self.control_mode = mode

    @property
    def action_space(self) -> dict:
        if self.control_mode == "joint":
            return {"arm_joint_position": (14,), "dex_joint_position": (44,)}
        return {"abs_end_effector": (14,), "dex_joint_position": (44,)}

    @property
    def observation_space(self) -> dict:
        space = {
            "arm_joint_position": (14,),
            "arm_joint_velocity": (14,),
            "dex_joint_position": (44,),
            "dex_joint_velocity": (44,),
            "abs_end_effector": (14,),
            "images": {n: (self.img_height, self.img_width, 3) for n in self._active_cameras},
        }
        if self._render_depth:
            space["images"] = dict(space["images"])
            for n in self._active_cameras:
                space["images"][f"{n}_depth"] = (self.img_height, self.img_width)
        return space

    def close(self) -> None:
        """Release the renderer + GL context and clean up temp MJCF files.

        Must be called from the thread that owns the GLContext (mujoco GLContext is
        thread-affine). VectorRolloutManager closes each env in its own env-thread on
        stop; otherwise the GLContext leaks and EGL accumulates across the many
        per-task envs the vectorized path builds -> silent segfault after a few tasks."""
        try:
            if self._owns_renderer and self.renderer is not None:
                self.renderer.close()  # also frees the Renderer's own GL context
                self.renderer = None
            # else: this env renders via a shared renderer owned by the
            # vectorized render thread — do NOT close it here (would kill the
            # one shared context). When _render_hook is set, self.renderer is
            # None anyway (the hook short-circuits _init_renderer).
        except Exception:
            pass
        for p in self.temp_files:
            try:
                os.remove(p)
            except OSError:
                pass
        self.temp_files = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_hammer_nail_indices(self):
        """Index mocap nails + free-body hammers for the dexjoco hammer-nail
        task family. Populated only when the bound task declares objects with
        ``nail: true`` / ``hammer: true``; for every other task this builds an
        empty index so the per-step contact script is a no-op (zero overhead).

        For each nail we cache: body id, mocap id, the declared initial z
        (``z0``, from ObjectInit.pos), and the current driven depth. For each
        hammer we cache the set of geom ids that belong to it (the face/head/
        neck/claw/handle geoms) so the contact script can recognise
        hammer↔nail strikes on data.contact.

        Body/geom names are prefixed ``obj{instance_id}_`` by
        scripts/visualize_scene._prepare_object_xml, so we resolve via the
        instance-id prefix rather than the catalog name.
        """
        self._nail_index = []      # list of dicts: {bid, mocapid, z0, depth, geom_ids}
        self._hammer_geom_ids = set()
        self._nail_geom_ids = set()
        self._hammer_index = []    # list of dicts: {obj, bid, qposadr, pos, quat}
        if self.task is None:
            return
        m = self.model
        for obj in self.task.all_object_inits():
            prefix = f"obj{obj.instance_id}_"
            if obj.is_nail:
                bid = -1
                for cand in (prefix + "nail", prefix + obj.name, prefix + "root"):
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                    if bid >= 0:
                        break
                if bid < 0:
                    print(f"[SimEnv] hammer-nail: nail object {obj.name!r} "
                          f"({prefix}*) not found in model; skipping")
                    continue
                mocapid = int(m.body_mocapid[bid])
                if mocapid < 0:
                    print(f"[SimEnv] hammer-nail: {obj.name!r} is marked nail "
                          f"but its body is not mocap; contact script won't drive it")
                z0 = float(obj.pos[2]) if obj.pos is not None else None
                # Collect this nail body's own geoms (nail_head/nail_shaft).
                nail_geoms = set()
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == bid:
                        nail_geoms.add(g)
                self._nail_geom_ids |= nail_geoms
                self._nail_index.append({
                    "obj": obj, "bid": bid, "mocapid": mocapid,
                    "z0": z0, "depth": 0.0, "geom_ids": nail_geoms,
                })
            elif obj.is_hammer:
                bid = -1
                for cand in (prefix + "hammer_body", prefix + obj.name, prefix + "root"):
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                    if bid >= 0:
                        break
                if bid < 0:
                    print(f"[SimEnv] hammer-nail: hammer object {obj.name!r} "
                          f"({prefix}*) not found in model; skipping")
                    continue
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == bid:
                        self._hammer_geom_ids.add(g)
                # Cache the free joint's qpos address + declared pose so reset
                # can put a dropped hammer back where the task declared it.
                # Hammers have no region (region sampling would overwrite the
                # tool's non-yaw orientation), so apply_task_initial_state does
                # NOT re-pose them — we do it here.
                qposadr = -1
                for jid in range(m.njnt):
                    if (m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
                            and m.jnt_bodyid[jid] == bid):
                        qposadr = int(m.jnt_qposadr[jid])
                        break
                # Striking geoms: the head + face are the parts that press a
                # nail. We drive the nail based on THESE geom world positions
                # (not the hammer body origin, which sits mid-handle and may be
                # far from the nail even when the head is touching it).
                strike_geoms = []
                face_gid = -1
                for gname in ("head", "face"):
                    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, prefix + gname)
                    if g >= 0:
                        strike_geoms.append(g)
                        if gname == "face":
                            face_gid = g
                self._hammer_index.append({
                    "obj": obj, "bid": bid, "qposadr": qposadr,
                    "pos": obj.pos, "quat": obj.quat,
                    "strike_geoms": strike_geoms,
                    "face_gid": face_gid,
                    # Mirrors dexjoco's per-face vz buffer used to gate the
                    # nail insertion on a real downward swing. Reset in
                    # _reset_hammer_nail_state.
                    "prev_face_z": None,
                    "vz_buf": [],
                })
        if self._nail_index:
            print(f"[SimEnv] hammer-nail: indexed {len(self._nail_index)} nail(s) "
                  f"and {len(self._hammer_geom_ids)} hammer geom(s)")

    def _step_physics(self):
        """One physics tick + (optional) hammer-nail contact driving.

        Single entry point for mj_step so the per-step contact script can't
        be missed by a second call site. Caller must NOT hold self.lock; this
        method acquires it so mocap-pose writes race neither mj_step nor the
        snapshot/camera loops.
        """
        with self.lock:
            mujoco.mj_step(self.model, self.data)
            self._apply_hammer_nail_interaction()
            self._apply_spray_interaction()
            self._apply_button_interaction()
            self._apply_reveal_on_press()
            self._apply_order_tracking()
            self._apply_flyover_tracking()
            self._apply_color_swap_tick()
            self._apply_sink_water_viz()

    def _apply_hammer_nail_interaction(self):
        """Drive mocap nails downward when a hammer STRIKES them.

        Faithful port of dexjoco's PandaHammerNailGymEnv._apply_hammer_nail_interaction:

        1. Each tick, update a rolling buffer of the hammer face's world-z
           velocity (12-tick window). This captures the pre-impact downward
           speed of the swing.
        2. Scan data.contact for any hammer-geom ↔ nail-geom pair. No real
           contact => nothing happens (proximity alone does NOT sink the nail).
        3. A contact only sinks the nail if the hammer was actually swinging
           down: ``preimpact_vz < -impact_vel_threshold`` (0.02 m/s). A resting
           / gently-touching hammer returns without advancing depth.
        4. On a genuine strike, ``scale = min(3.0, |preimpact_vz| / threshold)``
           and ``delta = impact_insert_step (0.008) * scale``; the harder the
           swing the more the nail advances. Capped at max_insert_depth (0.0726).

        Success (checked separately via the `sunk` predicate) is
        ``nail_depth >= 0.04``.
        """
        if not self._nail_index or not self._hammer_geom_ids:
            return
        m, d = self.model, self.data
        control_dt = max(self.model.opt.timestep, 1e-9)

        # Downsample the contact-scan + mocap-write to every Nth physics tick.
        # The face-z velocity buffer still updates every tick (so the swing
        # window stays accurate), but scanning data.contact and writing
        # mocap_pos only every N ticks cuts the per-tick solver disturbance
        # from the sinking nail N-fold — with two hammers driving two nails
        # at once this keeps the 500 Hz loop ahead of real time. The per-shot
        # delta is scaled by N so the per-second sink rate is unchanged.
        _N = 5
        self._hammer_nail_tick = getattr(self, "_hammer_nail_tick", 0) + 1
        do_drive = (self._hammer_nail_tick % _N == 0)

        # 1) Update each hammer's face-z velocity buffer (mirrors dexjoco's
        #    vz_buf updated every substep in step()).
        for ham in self._hammer_index:
            fg = ham.get("face_gid", -1)
            if fg < 0:
                continue
            face_z = float(d.geom_xpos[fg][2])
            prev = ham.get("prev_face_z")
            if prev is None:
                ham["prev_face_z"] = face_z
                continue
            vz = (face_z - prev) / control_dt
            ham["prev_face_z"] = face_z
            buf = ham["vz_buf"]
            buf.append(vz)
            if len(buf) > 12:
                buf.pop(0)

        # 2) Scan contacts for hammer↔nail geom pairs — only on drive ticks.
        if not do_drive:
            return
        hammer_set = self._hammer_geom_ids
        impacted = {}  # id(nail) -> striking hammer entry
        ncon = int(d.ncon)
        for i in range(ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            g1h = g1 in hammer_set
            g2h = g2 in hammer_set
            if not (g1h or g2h):
                continue
            nail_g = g2 if g1h else g1
            hammer_g = g1 if g1h else g2
            for nail in self._nail_index:
                if nail_g in nail["geom_ids"]:
                    # find which hammer owns hammer_g
                    owning = None
                    for ham in self._hammer_index:
                        if m.geom_bodyid[hammer_g] == ham["bid"]:
                            owning = ham
                            break
                    impacted[id(nail)] = owning
                    break

        if not impacted:
            return

        # 3+4) Velocity-gated, velocity-scaled insertion (dexjoco params).
        impact_vel_threshold = 0.02   # m/s downward swing required to sink
        impact_insert_step = 0.008    # dexjoco base depth advance per CONTROL step
        max_insert_depth = 0.0726     # dexjoco max_insert_depth
        max_scale = 2.0
        # dexjoco applies impact_insert_step once per control step (0.02 s).
        # We run this every physics tick (control_dt = opt.timestep, 0.002 s @
        # 500 Hz), so scale the per-tick delta by physics_dt/dexjoco_control_dt
        # to match dexjoco's per-second insertion rate. The raw ratio (0.1) is
        # further halved here (0.05): a smaller per-tick mocap jump keeps the
        # contact solver cooler when two hammers drive two nails at once while
        # the arms run IK — the sink rate halves but stays well above the 0.04 m
        # success depth given the 500 Hz tick rate (~0.2 m/s even at scale 1).
        # We then multiply by _N because we only drive every Nth tick (the
        # per-second rate is preserved while the solver sees the mocap jump
        # only 1/N as often).
        dt_scale = (control_dt / 0.02) * 0.5 * _N
        for nail in self._nail_index:
            key = id(nail)
            if key not in impacted or nail["mocapid"] < 0 or nail["z0"] is None:
                continue
            ham = impacted[key]
            buf = ham["vz_buf"] if ham else []
            preimpact_vz = min(buf) if buf else 0.0
            # Not swinging down hard enough => no insertion this tick.
            if preimpact_vz >= -impact_vel_threshold:
                continue
            scale = min(max_scale, abs(preimpact_vz) / max(impact_vel_threshold, 1e-6))
            delta = impact_insert_step * scale * dt_scale
            nail["depth"] = min(max_insert_depth, nail["depth"] + delta)
            d.mocap_pos[nail["mocapid"]][2] = nail["z0"] - nail["depth"]
            # xy unchanged: the nail stays put horizontally.
        # No mj_forward here: the next mj_step in _step_physics re-derives
        # xpos/xmat from the updated mocap_pos. A redundant mj_forward every
        # physics tick made the contact solver twice as hot and, with two
        # hammers driving two nails at once while the arms run IK, pushed the
        # 500 Hz loop below real time (visual stall). The 10 Hz web poll and
        # the sunk predicate (which reads mocap-driven xpos) both tolerate the
        # sub-tick lag until the next mj_step.

    def _reset_hammer_nail_state(self):
        """Restore every indexed mocap nail to its declared pose and zero the
        driven depth, and put every indexed hammer back at its declared pose.
        Called from reset() so a fresh episode starts with the nails upright
        and the hammers back where the task declared them — even if a hammer
        was knocked off the table last episode. Hammers are freejoint bodies
        with no region (region sampling would overwrite the tool's non-yaw
        orientation), so apply_task_initial_state does not re-pose them;
        we write the free joint qpos [x,y,z, w,qx,qy,qz] directly here.
        Caller (reset) already holds self.lock.
        """
        if not self._nail_index and not self._hammer_index:
            # Indices can be empty if env was constructed before a hammer-nail
            # task was bound (e.g. --collect-cycle boots with task=None). The
            # scene rebuild on first set_task rebuilds them, but a same-task
            # refresh takes the scene_changed=False path and never rebuilds.
            # Rebuild defensively here so reset always restores hammers/nails.
            if self.task is not None and any(
                    o.is_nail or o.is_hammer for o in self.task.all_object_inits()):
                print("[SimEnv] _reset_hammer_nail_state: indices empty, rebuilding")
                self._build_hammer_nail_indices()
            if not self._nail_index and not self._hammer_index:
                return
        print(f"[SimEnv] _reset_hammer_nail_state: nails={len(self._nail_index)} hammers={len(self._hammer_index)}")
        m, d = self.model, self.data
        for nail in self._nail_index:
            nail["depth"] = 0.0
            obj = nail["obj"]
            mid = nail["mocapid"]
            if mid < 0 or obj.pos is None:
                continue
            d.mocap_pos[mid][0] = float(obj.pos[0])
            d.mocap_pos[mid][1] = float(obj.pos[1])
            d.mocap_pos[mid][2] = float(obj.pos[2])
            if obj.quat is not None:
                d.mocap_quat[mid][0] = float(obj.quat[0])
                d.mocap_quat[mid][1] = float(obj.quat[1])
                d.mocap_quat[mid][2] = float(obj.quat[2])
                d.mocap_quat[mid][3] = float(obj.quat[3])
        for ham in self._hammer_index:
            obj = ham["obj"]
            qa = ham["qposadr"]
            if qa < 0 or obj.pos is None:
                continue
            # free joint qpos layout: [x, y, z, w, qx, qy, qz]
            d.qpos[qa + 0] = float(obj.pos[0])
            d.qpos[qa + 1] = float(obj.pos[1])
            d.qpos[qa + 2] = float(obj.pos[2])
            q = obj.quat if obj.quat is not None else (1.0, 0.0, 0.0, 0.0)
            d.qpos[qa + 3] = float(q[0])
            d.qpos[qa + 4] = float(q[1])
            d.qpos[qa + 5] = float(q[2])
            d.qpos[qa + 6] = float(q[3])
            # zero this hammer's velocity so it doesn't carry momentum from
            # the previous episode into the freshly reset pose.
            if ham["bid"] >= 0:
                did = m.body_dofadr[ham["bid"]]
                if did >= 0 and d.qvel is not None:
                    for k in range(6):
                        idx = did + k
                        if 0 <= idx < d.qvel.shape[0]:
                            d.qvel[idx] = 0.0
            # Clear the face-velocity buffer so the next episode's first
            # strike is gated on its own swing, not leftover from last time.
            ham["prev_face_z"] = None
            ham["vz_buf"] = []
        mujoco.mj_forward(m, d)

    # ---- dexjoco water-plant (spray + plant) indexing / interaction ----------
    # Mirrors the hammer-nail machinery above: index spray free-bodies + their
    # trigger hinge + cone geom on scene build; drive the water-cone visibility
    # from the trigger each physics tick; restore declared spray poses on reset.
    # The `watered` predicate in tasks/predicates.py reads the trigger joint
    # qpos directly (stateless), so success = "trigger pressed AND nozzle aimed
    # at a plant" at any frame (latched by SubtaskTracker).

    _SPRAY_TRIGGER_PULL_THRESHOLD = 0.05   # light press triggers water (was dexjoco 0.34)
    _SPRAY_TRIGGER_RELEASE_THRESHOLD = 0.02
    _SPRAY_CONE_ALPHA = 0.5                # dexjoco cone alpha when spraying

    # RoboCasa sink faucet: the `water` site alpha (0..1) scales with how open the
    # flow handle is. _SINK_HANDLE_FULL_OPEN is the handle_joint qpos at fully
    # open (the Sink001 hinge range is 0..0.5; qpos 0 = closed).
    _SINK_WATER_MAX_ALPHA = 0.6
    _SINK_HANDLE_FULL_OPEN = 0.5

    # Touch/press button family: a press counts once the cap's slide joint
    # descends past _BUTTON_PRESS_QPOS, and does NOT count again until it
    # rises back below _BUTTON_RELEASE_QPOS (hysteresis, mirroring the spray
    # trigger pull/release). A faster press moves the cap further/faster but a
    # single down-up stroke is exactly one count regardless of depth.
    _BUTTON_PRESS_QPOS = 0.0015   # m of cap deflection (axis -z) that counts as a press
    _BUTTON_RELEASE_QPOS = 0.0006  # m the cap must rise back under before the next press counts
    # Software debounce: a finite press force resting right on the threshold
    # (common when the operator presses lightly and the spring is stiff, or
    # when a teleop fingertip bounces on the cap) makes the cap oscillate
    # across the press/release thresholds, so a single physical press would
    # edge-trigger many counts in a few ms. Ignore any rising edge that fires
    # within this many sim-seconds of the last counted press for the SAME
    # button. 0.05 s caps the count rate at 20 Hz -- far above any human tap
    # rate (~5 Hz) -- while collapsing a chattering burst to one count. Use
    # simulation time (data.time) so the refractory is deterministic and not
    # skewed by the real-time viewer loop.
    _BUTTON_PRESS_REFRACTORY = 0.05  # s
    # Per-episode button placement: jitter each button's xy around its declared
    # pos (the task author's anchor), rejection-sampled so no two buttons land
    # within _BUTTON_MIN_SEP (they must not overlap). Button bases are mocap, so
    # they can be re-posed at reset without rolling/tipping, mirroring the nail.
    _BUTTON_JITTER_X = 0.06       # +/- m around declared x
    _BUTTON_JITTER_Y = 0.05       # +/- m around declared y
    _BUTTON_MIN_SEP = 0.13        # min centre-to-centre xy distance between buttons

    def _build_spray_plant_indices(self):
        """Index spray free-bodies (freejoint qposadr, trigger joint qposadr,
        cone geom id, ref_point site id) for the dexjoco water-plant task
        family. Populated only when the bound task declares objects with
        ``spray: true``; empty otherwise so the per-step script is a no-op.

        Body/joint/site names are prefixed ``obj{instance_id}_`` by
        visualize_scene._prepare_object_xml; the spray XML's root body is
        ``spray_body``, the trigger hinge is ``joint_0`` (on child body
        ``link_0``), the nozzle site is ``ref_point``, and the water cone geom
        is ``cone_visual``.
        """
        self._spray_index = []
        if self.task is None:
            return
        m = self.model
        for obj in self.task.all_object_inits():
            if not obj.is_spray:
                continue
            prefix = f"obj{obj.instance_id}_"
            bid = -1
            for cand in (prefix + "spray_body", prefix + "root", prefix + obj.name):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                if bid >= 0:
                    break
            if bid < 0:
                print(f"[SimEnv] water-plant: spray object {obj.name!r} "
                      f"({prefix}*) not found in model; skipping")
                continue
            # freejoint qpos address (free body: [x,y,z, w,qx,qy,qz]).
            qposadr = -1
            for jid in range(m.njnt):
                if (m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
                        and m.jnt_bodyid[jid] == bid):
                    qposadr = int(m.jnt_qposadr[jid])
                    break
            # trigger hinge joint_0 (lives on a child body, so search by name).
            trigger_jid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "joint_0")
            trigger_qposadr = int(m.jnt_qposadr[trigger_jid]) if trigger_jid >= 0 else -1
            # cone geom (visibility driven by trigger).
            cone_gid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_GEOM, prefix + "cone_visual")
            # nozzle site.
            ref_sid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_SITE, prefix + "ref_point")
            if trigger_qposadr < 0:
                print(f"[SimEnv] water-plant: spray {obj.name!r} has no "
                      f"trigger joint ({prefix}joint_0); watering disabled")
            self._spray_index.append({
                "obj": obj, "bid": bid, "qposadr": qposadr,
                "trigger_qposadr": trigger_qposadr,
                "cone_gid": cone_gid, "ref_sid": ref_sid,
                # latched trigger state with hysteresis (dexjoco semantics):
                # once pulled past 0.34 it stays "pressed" until released
                # below 0.25. Keeps the cone from flickering at the threshold.
                "trigger_pulled": False,
                # last alpha written to geom_rgba; we only touch model.geom_rgba
                # on an edge (alpha actually changes) -- writing it every physics
                # tick (1000 Hz, two cones) races the render thread and stalls
                # the sim (same class of bug as the hammer-nail redundant
                # mj_forward). See _apply_spray_interaction.
                "last_alpha": 0.0,
            })
        if self._spray_index:
            # Start every cone invisible until a trigger is pressed.
            for sp in self._spray_index:
                if sp["cone_gid"] >= 0:
                    m.geom_rgba[sp["cone_gid"]][3] = 0.0
            print(f"[SimEnv] water-plant: indexed {len(self._spray_index)} spray can(s)")

    def _apply_spray_interaction(self):
        """Each physics tick: read every spray's trigger hinge qpos, apply
        hysteresis (pull >= 0.34 latches pressed, release <= 0.25 unlatches),
        and drive the water-cone geom alpha accordingly.

        The cone is purely visual (contype/conaffinity 0, density 0), so
        toggling its rgba alpha has no physics effect. ``trigger_pulled`` is
        cached per spray but the `watered` predicate re-reads the live joint
        qpos itself (stateless) -- the cache here only owns the cone.

        PERFORMANCE: model.geom_rgba is written ONLY when the alpha actually
        changes (edge-triggered), not every tick. Writing it every physics tick
        (1000 Hz, two cones at once) races the concurrent render thread reading
        the same model and stalls the sim -- the same class of bug the
        hammer-nail path fixed by dropping a redundant mj_forward. With
        edge-triggering there are at most a handful of model writes per episode
        instead of tens of thousands.
        """
        if not self._spray_index:
            return
        m, d = self.model, self.data
        for sp in self._spray_index:
            ta = sp["trigger_qposadr"]
            if ta < 0 or sp["cone_gid"] < 0:
                continue
            tq = float(d.qpos[ta])
            if tq >= self._SPRAY_TRIGGER_PULL_THRESHOLD:
                sp["trigger_pulled"] = True
            elif tq <= self._SPRAY_TRIGGER_RELEASE_THRESHOLD:
                sp["trigger_pulled"] = False
            alpha = self._SPRAY_CONE_ALPHA if sp["trigger_pulled"] else 0.0
            # Only touch model memory when the visible alpha actually flips.
            if alpha != sp["last_alpha"]:
                m.geom_rgba[sp["cone_gid"]][3] = alpha
                sp["last_alpha"] = alpha

    def _build_sink_water_indices(self):
        """Index the RoboCasa sink faucet's flow-handle joint + the ``water``
        site so _apply_sink_water_viz can show the water stream only while the
        faucet is open. Populated only when the scene contains a sink (a site
        whose name ends in ``_water`` paired with a ``{prefix}_handle_joint``);
        left ``None`` otherwise so the per-step script is a cheap no-op for every
        non-sink task.

        RoboCasa fixtures are baked into the kitchen MJCF by
        build_robocasa_scene_xml (NOT BiDex-100 ObjectInits, so no
        ``obj{i}_`` prefix / instance_id), and upstream names them with a nested
        sub-group, e.g. layout 11's sink flow handle is
        ``sink_left_group_handle_joint`` and the stream site is
        ``sink_left_group_water``. We scan by suffix (``_water``) rather than
        hardcoding the prefix, so this works for any sink layout (``sink_``,
        ``sink_1_``, ``sink_left_group_``, ...).
        """
        self._sink_water_index = None
        m = self.model
        water_sid = -1
        water_prefix = None
        for sid in range(m.nsite):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid)
            if name and name.endswith("_water"):
                water_sid = sid
                water_prefix = name[: -len("_water")]
                break
        if water_sid < 0:
            return  # no sink water site in this scene -> per-step viz is a no-op
        jid = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{water_prefix}_handle_joint")
        if jid < 0:
            print(f"[SimEnv] sink-water: found site {water_prefix}_water but no "
                  f"{water_prefix}_handle_joint; water viz disabled")
            return
        self._sink_water_index = {
            "qposadr": int(m.jnt_qposadr[jid]),
            "sid": water_sid,
            # last alpha written to site_rgba; we only touch model.site_rgba on
            # an edge (alpha actually changes) -- writing it every physics tick
            # races the concurrent render thread reading the same model, the same
            # class of bug _apply_spray_interaction avoids (see its note).
            "last_alpha": m.site_rgba[water_sid][3],
        }
        # The sink ships the `water` site at a constant alpha 0.5 (always wet);
        # start it hidden until the operator actually opens the faucet.
        m.site_rgba[water_sid][3] = 0.0
        self._sink_water_index["last_alpha"] = 0.0
        print(f"[SimEnv] sink-water: indexed faucet {water_prefix}_handle_joint / "
              f"{water_prefix}_water")

    def _apply_sink_water_viz(self):
        """Each physics tick: drive the sink ``water`` site alpha from the faucet
        handle qpos (alpha proportional to openness, 0 when closed). Purely
        visual -- a site has no physics. Edge-triggered: model.site_rgba is
        written ONLY when the alpha actually changes, to avoid racing the
        concurrent render thread (same rationale as _apply_spray_interaction).
        """
        sw = getattr(self, "_sink_water_index", None)
        if sw is None:
            return
        qpos = float(self.data.qpos[sw["qposadr"]])
        frac = max(0.0, min(1.0, qpos / self._SINK_HANDLE_FULL_OPEN))
        alpha = frac * self._SINK_WATER_MAX_ALPHA
        if alpha != sw["last_alpha"]:
            self.model.site_rgba[sw["sid"]][3] = alpha
            sw["last_alpha"] = alpha

    def _reset_sink_water_state(self, seed: Optional[int] = None):
        """Zero the sink faucet's flow handle (closed) and hide the water stream
        at the start of every episode. The sink is a RoboCasa scene fixture (no
        BiDex-100 instance_id), so apply_task_initial_state does NOT touch it;
        without this, a faucet left open at the end of one episode would start
        the next already open -- latching the first 'open' milestone instantly
        and showing water at frame 0. Caller (reset) already holds self.lock.
        """
        sw = getattr(self, "_sink_water_index", None)
        if sw is None:
            return
        m, d = self.model, self.data
        # Restore the flow handle to its model default (closed, qpos 0).
        d.qpos[sw["qposadr"]] = float(m.qpos0[sw["qposadr"]])
        # Hide the water stream until the faucet is opened again.
        m.site_rgba[sw["sid"]][3] = 0.0
        sw["last_alpha"] = 0.0

    def _reset_spray_state(self, seed: Optional[int] = None):
        """Restore every indexed spray can to its declared pose (freejoint
        qpos [x,y,z, w,qx,qy,qz]), zero its trigger, and hide the cone.
        Mirrors _reset_hammer_nail_state: hammers/sprays are freejoint bodies
        with no region (region sampling would overwrite the tool's non-yaw
        orientation), so apply_task_initial_state does NOT re-pose them; we
        write the free joint qpos directly here so a fresh episode starts with
        the cans back where the task declared them -- even if one was knocked
        off the table last episode. Caller (reset) already holds self.lock.
        """
        if not self._spray_index:
            if self.task is not None and any(
                    o.is_spray for o in self.task.all_object_inits()):
                print("[SimEnv] _reset_spray_state: indices empty, rebuilding")
                self._build_spray_plant_indices()
            if not self._spray_index:
                return
        print(f"[SimEnv] _reset_spray_state: sprays={len(self._spray_index)}")
        m, d = self.model, self.data
        # Per-episode randomization of the resting pose: jitter both x and y,
        # add a +/-30 deg yaw on top of each spray's declared (compound)
        # orientation, and keep any two cans far enough apart that they don't
        # overlap. Unlike region sampling (which emits a pure-yaw quat and would
        # erase the can's flip/pitch), we keep the declared quat and only
        # compose a small world-z yaw on top -- so the can still faces the right
        # way, just not at a dead-fixed spot/heading every episode. Seeded for
        # reproducible resets.
        from scipy.spatial.transform import Rotation as _R
        rng = np.random.default_rng(seed)
        x_jitter = 0.08           # +/-8 cm along world x
        y_jitter = 0.05           # +/-5 cm along world y
        min_sep = 0.15            # min centre-to-centre distance between cans
        yaw_half = np.deg2rad(30) # +/-30 deg

        # Pre-sample (dx, dy) for every spray with rejection sampling so no two
        # cans end up overlapping. The declared y's are +/-0.18 (36 cm apart),
        # so with +/-5 cm jitter they rarely collide; rejection just guarantees
        # it across the rare unlucky draws.
        dxdy = []
        attempts = 0
        while len(dxdy) < len(self._spray_index) and attempts < 200:
            attempts += 1
            cand_dx = float(rng.uniform(-x_jitter, x_jitter))
            cand_dy = float(rng.uniform(-y_jitter, y_jitter))
            cand_idx = len(dxdy)
            cand_obj = self._spray_index[cand_idx]["obj"]
            cx = float(cand_obj.pos[0]) + cand_dx
            cy = float(cand_obj.pos[1]) + cand_dy
            ok = True
            for (px, py) in dxdy:
                if (cx - px) ** 2 + (cy - py) ** 2 < min_sep ** 2:
                    ok = False
                    break
            if ok:
                dxdy.append((cand_dx, cand_dy))

        for idx, sp in enumerate(self._spray_index):
            obj = sp["obj"]
            qa = sp["qposadr"]
            if qa < 0 or obj.pos is None:
                continue
            dx, dy = dxdy[idx] if idx < len(dxdy) else (0.0, 0.0)
            yaw = float(rng.uniform(-yaw_half, yaw_half))
            d.qpos[qa + 0] = float(obj.pos[0]) + dx
            d.qpos[qa + 1] = float(obj.pos[1]) + dy
            d.qpos[qa + 2] = float(obj.pos[2])
            q = obj.quat if obj.quat is not None else (1.0, 0.0, 0.0, 0.0)
            # compose world-z yaw ON TOP of declared quat: q_new = Rz(yaw) * q_decl
            qd = _R.from_quat([q[1], q[2], q[3], q[0]])
            qn = (_R.from_euler('z', yaw) * qd).as_quat()
            d.qpos[qa + 3] = float(qn[3])
            d.qpos[qa + 4] = float(qn[0])
            d.qpos[qa + 5] = float(qn[1])
            d.qpos[qa + 6] = float(qn[2])
            # zero the free body's velocity so it doesn't carry momentum across
            # the episode boundary.
            did = m.body_dofadr[sp["bid"]]
            if did >= 0 and d.qvel is not None:
                for k in range(6):
                    idx = did + k
                    if 0 <= idx < d.qvel.shape[0]:
                        d.qvel[idx] = 0.0
            # zero the trigger hinge + hide the cone.
            if sp["trigger_qposadr"] >= 0:
                d.qpos[sp["trigger_qposadr"]] = 0.0
            sp["trigger_pulled"] = False
            if sp["cone_gid"] >= 0:
                m.geom_rgba[sp["cone_gid"]][3] = 0.0
            sp["last_alpha"] = 0.0
        mujoco.mj_forward(m, d)

    def _build_button_indices(self):
        """Index mocap-base spring buttons for the dexjoco touch/press family.

        Populated only when the bound task declares objects with ``button:
        true``; empty otherwise so the per-step script is a no-op. For each
        button we cache the cap's slide-joint qpos/dof addresses (the joint
        lives on the child ``cap_body`` and is named ``cap_slide`` -> prefixed
        ``obj{i}_cap_slide`` by visualize_scene._prepare_object_xml) AND the
        mocap base body (``button_body`` -> ``obj{i}_button_body``) so the
        reset hook can re-pose the button each episode. The cap is the only
        moving part; the mocap base is kinematic so it never rolls, tips, or
        is knocked away.

        ``prev_pressed`` holds the hysteresis state across ticks; the runtime
        press counter lives on ``ObjectInit.press_count`` (read by the
        `pressed` predicate), zeroed here via _reset_button_state.
        """
        self._button_index = []   # list of dicts: {obj, qposadr, dofadr, base_mocapid, pos, prev_pressed}
        if self.task is None:
            return
        m = self.model
        for obj in self.task.all_object_inits():
            if not obj.is_button:
                continue
            prefix = f"obj{obj.instance_id}_"
            jid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "cap_slide")
            if jid < 0:
                print(f"[SimEnv] touch-button: {obj.name!r} has no cap_slide "
                      f"joint ({prefix}cap_slide); press counting disabled")
                continue
            # Mocap base body (re-posable at reset). Fall back through likely
            # names; the XML declares button_body.
            base_bid = -1
            for cand in (prefix + "button_body", prefix + "root", prefix + obj.name):
                base_bid = mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_BODY, cand)
                if base_bid >= 0:
                    break
            base_mocapid = int(m.body_mocapid[base_bid]) if base_bid >= 0 else -1
            if base_mocapid < 0:
                print(f"[SimEnv] touch-button: {obj.name!r} base is not mocap; "
                      f"position will not be randomized at reset")
            self._button_index.append({
                "obj": obj,
                # qpos address (cap deflection) and dof address (cap velocity)
                # DIFFER because free joints contribute 7 qpos / 6 dofs, so we
                # cache both instead of deriving one from the other.
                "qposadr": int(m.jnt_qposadr[jid]),
                "dofadr": int(m.jnt_dofadr[jid]),
                "base_mocapid": base_mocapid,
                "pos": obj.pos,           # declared anchor for per-episode jitter
                "prev_pressed": False,
                "last_press_time": -1e9,  # sim-time of last counted press (debounce)
            })
        if self._button_index:
            print(f"[SimEnv] touch-button: indexed {len(self._button_index)} button(s)")

    def _apply_button_interaction(self):
        """Each physics tick: hysteresis edge-detect every button's cap slide
        qpos and count distinct presses.

        A press is registered on the rising edge -- the cap was NOT pressed
        last tick and IS pressed now (qpos >= _BUTTON_PRESS_QPOS). The cap
        must then rise back below _BUTTON_RELEASE_QPOS before the next press
        counts, so holding a finger down registers exactly one press, not a
        run of ticks. The counter is stored on ``ObjectInit.press_count`` so
        the stateless `pressed` predicate and the SubtaskTracker see it
        without a handle to the env (same pattern as `sunk` reading the
        env-driven nail depth).

        Runs every physics tick (1000 Hz) so a quick tap is caught; the
        per-tick work is one qpos read + compare per button, negligible vs
        mj_step.
        """
        if not self._button_index:
            return
        d = self.data
        for btn in self._button_index:
            qa = btn["qposadr"]
            if qa < 0:
                continue
            qpos = float(d.qpos[qa])
            if btn["prev_pressed"]:
                # Cap is down; wait for it to rise back under the release
                # threshold before arming the next press.
                if qpos <= self._BUTTON_RELEASE_QPOS:
                    btn["prev_pressed"] = False
            else:
                if qpos >= self._BUTTON_PRESS_QPOS:
                    btn["prev_pressed"] = True
                    # Debounce: discard a rising edge that fires within the
                    # refractory window of the last counted press -- this is a
                    # contact-bounce / threshold-chatter artifact, not a new
                    # distinct press (the cap cannot have been released and
                    # re-pressed by a human in <50 ms). prev_pressed stays True
                    # so the release-edge re-arm below is still required before
                    # the next genuine press counts. The print + press_log
                    # snapshot only run on a counted press, so a chattering
                    # burst yields exactly one count, one log line, one
                    # snapshot (keeping pressed_after's log-length == count
                    # invariant intact).
                    now_t = float(d.time)
                    if now_t - btn["last_press_time"] >= self._BUTTON_PRESS_REFRACTORY:
                        btn["last_press_time"] = now_t
                        btn["obj"].press_count = int(getattr(btn["obj"], "press_count", 0)) + 1
                        print(f"[button] {btn['obj'].name} pressed "
                              f"(count={btn['obj'].press_count}, cap_qpos={qpos:.4f})")
                        if self.task is not None:
                            snap = {}
                            for other in self.task.all_object_inits():
                                if other is btn["obj"] or not other.is_button:
                                    continue
                                snap[other.name] = int(getattr(other, "press_count", 0))
                            btn["obj"].press_log.append(snap)

    def _reset_button_state(self, seed: Optional[int] = None):
        """Zero every indexed button's press counter + per-press log, arm the
        hysteresis state, return the cap to rest, and re-pose each mocap base.

        The cap slide joint is NOT a free joint, so apply_task_initial_state
        (which handles freejoint bodies only) does not touch it; we reset the
        qpos/qvel directly here. The mocap base is re-posed each episode for
        per-episode placement variety, in one of two modes:

          * Region mode (button declares a `region`): the base xy (+ yaw) is
            drawn from that region via the SAME RegionSampler that places
            free bodies like the bowls in swap_fruits_between_bowls, so a button
            gets the identical xy_range / yaw_range jitter scheme as a bowl.
          * Legacy jitter mode (no region): flat +/- jitter around the declared
            pos, rejection-sampled so no two buttons land within
            _BUTTON_MIN_SEP.

        Both modes keep the buttons non-rolling and non-overlapping. Opt out
        entirely via Task.randomize_buttons=False: every button stays fixed at
        its declared pos (no jitter, no sampling) -- needed when a task lays
        many buttons out in a known row (e.g. the "numbers 1-5" row, where
        value is encoded by position). Caller (reset) already holds self.lock.
        """
        if not self._button_index:
            if self.task is not None and any(
                    o.is_button for o in self.task.all_object_inits()):
                print("[SimEnv] _reset_button_state: indices empty, rebuilding")
                self._build_button_indices()
            if not self._button_index:
                return
        m, d = self.model, self.data

        randomize = getattr(self.task, "randomize_buttons", True)
        # Resolve each button's per-episode base pose. `poses[idx]` is a
        # (xyz, quat) for the mocap base (quat may be None -> keep identity).
        if randomize:
            rng = np.random.default_rng(seed)
            from bidex.tasks.region_sampler import RegionSampler
            sampler = RegionSampler(rng)
            poses = []
            placed_xy: list = []   # for _BUTTON_MIN_SEP rejection
            for btn in self._button_index:
                obj = btn["obj"]
                region = (self.task.regions.get(obj.region)
                          if (self.task is not None and obj.region) else None)
                if region is not None:
                    # Bowl-style placement: sample xy + yaw from the region,
                    # rejection-sampled against earlier buttons so the two
                    # never overlap (a no-op when zones sit in opposite
                    # hemispheres far apart, as in swap_fruits).
                    chosen = None
                    pos = quat = None
                    for _ in range(256):
                        pos, quat = sampler.sample(region, model=m, data=d)
                        if all(((pos[0] - px) ** 2 + (pos[1] - py) ** 2) ** 0.5
                               >= self._BUTTON_MIN_SEP for px, py in placed_xy):
                            chosen = (pos, quat)
                            break
                    if chosen is None:
                        chosen = (pos, quat)
                    poses.append((tuple(chosen[0]), chosen[1]))
                    placed_xy.append((chosen[0][0], chosen[0][1]))
                else:
                    # Legacy flat jitter around the declared pos (rejection
                    # sampled for min separation), no yaw.
                    anchor = btn.get("pos")
                    chosen_dxdy = (0.0, 0.0)
                    for _ in range(256):
                        cand = (float(rng.uniform(-self._BUTTON_JITTER_X,
                                                  self._BUTTON_JITTER_X)),
                                float(rng.uniform(-self._BUTTON_JITTER_Y,
                                                  self._BUTTON_JITTER_Y)))
                        if all(((cand[0] - p[0]) ** 2 + (cand[1] - p[1]) ** 2) ** 0.5
                               >= self._BUTTON_MIN_SEP for p in placed_xy):
                            chosen_dxdy = cand
                            break
                    if anchor is not None:
                        poses.append(((float(anchor[0]) + chosen_dxdy[0],
                                       float(anchor[1]) + chosen_dxdy[1],
                                       float(anchor[2])), None))
                    else:
                        poses.append(((0.0, 0.0, 0.85), None))
                    placed_xy.append((poses[-1][0][0], poses[-1][0][1]))
        else:
            # Fixed at declared pos, no jitter -- paper_number-style rows.
            poses = []
            for btn in self._button_index:
                anchor = btn.get("pos")
                poses.append((tuple(anchor) if anchor is not None
                              else (0.0, 0.0, 0.85), None))

        for idx, btn in enumerate(self._button_index):
            btn["obj"].press_count = 0
            btn["obj"].press_log = []   # clear the per-press snapshot history
            btn["prev_pressed"] = False
            btn["last_press_time"] = -1e9   # arm the debounce window fresh
            qa = btn["qposadr"]
            if qa >= 0:
                d.qpos[qa] = 0.0
            da = btn.get("dofadr", -1)
            if da >= 0 and d.qvel is not None and 0 <= da < d.qvel.shape[0]:
                d.qvel[da] = 0.0
            # Re-pose the mocap base to its per-episode pose. The cap is a
            # child of the mocap body, so it rides along at slide qpos=0.
            mid = btn.get("base_mocapid", -1)
            if mid >= 0 and idx < len(poses):
                xyz, quat = poses[idx]
                d.mocap_pos[mid][0] = float(xyz[0])
                d.mocap_pos[mid][1] = float(xyz[1])
                d.mocap_pos[mid][2] = float(xyz[2])
                if quat is not None:
                    d.mocap_quat[mid][0] = float(quat[0])
                    d.mocap_quat[mid][1] = float(quat[1])
                    d.mocap_quat[mid][2] = float(quat[2])
                    d.mocap_quat[mid][3] = float(quat[3])
        mujoco.mj_forward(m, d)

    # ---- "place into a basket in order, then remove in the same order" -------
    # A per-tick observer for the dexjoco-style ordered-transfer task family.
    # Marks: `basket: true` on the open container, `orderable: true` on each small
    # object. Each tick we test every orderable object against the basket's
    # bounding-box interior (in the basket geom's live world frame, so it tracks
    # the heavy-but-freejoint basket wherever it rests / however it yaws) and
    # record, on the basket ObjectInit, the ordered catalog-name lists
    # `place_seq` (objects in the order they FIRST entered) and `remove_seq`
    # (order they FIRST left after entering). The stateless `placed_n` /
    # `order_match` predicates read those lists -- the env owns the history, the
    # predicate stays stateless, exactly as `pressed` reads `press_count`.

    def _build_basket_indices(self):
        """Index the task's basket (its interior bbox geom) and every orderable
        object (free body to track in/out of the basket).

        Populated only when the bound task declares a `basket: true` object;
        empty otherwise so the per-step tracking script is a no-op (zero
        overhead) for every other task. The basket's interior half-extents are
        read from its `obj{i}_reg_bbox` box geom (already size-scaled by
        _prepare_object_xml), stored once here. Per-object transition flags
        (prev_inside / place_done / remove_done) live in the index, not on
        ObjectInit, mirroring the button index's `prev_pressed`.
        """
        self._basket_index = None      # {obj, bbox_gid, s, margin_xy, margin_z}
        self._order_index = []         # [{obj, bid, name, prev_inside, place_done, remove_done}]
        self._basket_obj = None
        if self.task is None:
            return
        m = self.model

        # The basket: find its `obj{i}_reg_bbox` box geom.
        basket_obj = next((o for o in self.task.all_object_inits() if o.is_basket), None)
        if basket_obj is None:
            return
        prefix = f"obj{basket_obj.instance_id}_"
        bbox_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, prefix + "reg_bbox")
        if bbox_gid < 0:
            # Fallback: scan the basket's free body (and its subtree) for the
            # largest box geom -- its bounding region. Resolve the body via its
            # freejoint, like the hammer/spray indexers.
            b_bid = -1
            fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "freejoint")
            if fj >= 0:
                b_bid = m.jnt_bodyid[fj]
            if b_bid < 0:
                b_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "root")
            best_gid, best_vol = -1, -1.0
            for g in (self._body_geom_ids(b_bid) if b_bid >= 0 else []):
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    sz = m.geom_size[g]
                    vol = float(sz[0]) * float(sz[1]) * float(sz[2])
                    if vol > best_vol:
                        best_vol, best_gid = vol, g
            bbox_gid = best_gid
            if bbox_gid < 0:
                print(f"[SimEnv] basket-order: {basket_obj.name!r} has no reg_bbox "
                      f"geom ({prefix}reg_bbox); order tracking disabled")
                return
        sx, sy, sz = (float(v) for v in m.geom_size[bbox_gid][:3])
        self._basket_index = {
            "obj": basket_obj,
            "bbox_gid": bbox_gid,
            "s": (sx, sy, sz),
            "margin_xy": 0.015,   # interior is the bbox minus wall thickness
            "margin_z": 0.012,    # generous cavity band; xy discriminates in/out
        }
        self._basket_obj = basket_obj

        # Orderable contents: each tracked free body.
        for obj in self.task.all_object_inits():
            if not obj.orderable:
                continue
            prefix = f"obj{obj.instance_id}_"
            bid = -1
            for cand in (prefix + "object", prefix + obj.name, prefix + "root", prefix + "main"):
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cand)
                if bid >= 0:
                    break
            if bid < 0:
                # Fallback: first body whose name starts with the instance prefix.
                for b in range(m.nbody):
                    if m.body(b).name.startswith(prefix):
                        bid = b
                        break
            if bid < 0:
                print(f"[SimEnv] basket-order: orderable {obj.name!r} ({prefix}*) "
                      f"not found in model; skipping")
                continue
            self._order_index.append({
                "obj": obj, "bid": bid, "name": obj.name,
                "prev_inside": False, "place_done": False, "remove_done": False,
            })
        if self._order_index:
            # Start each episode with empty sequences (also cleared in
            # _reset_order_state, but clear here too so the index is consistent
            # right after a scene build before the first reset).
            basket_obj.place_seq = []
            basket_obj.remove_seq = []
            print(f"[SimEnv] basket-order: tracking {len(self._order_index)} "
                  f"orderable object(s) into basket {basket_obj.name!r}")

    def _body_geom_ids(self, bid: int) -> list:
        """Return geom ids belonging to body `bid` (and its descendant bodies).

        The basket's bbox may live on a child body of the free body, so we walk
        the body subtree rather than using the flat geomadr range (which only
        covers the body's OWN geoms).
        """
        m = self.model
        out = []
        # BFS over the body subtree rooted at bid.
        stack = [bid]
        children = {b: [] for b in range(m.nbody)}
        for b in range(m.nbody):
            p = m.body_parentid[b]
            if p >= 0:
                children.setdefault(p, []).append(b)
        seen = set()
        while stack:
            b = stack.pop()
            if b in seen or b < 0:
                continue
            seen.add(b)
            adr = m.body_geomadr[b]
            cnt = m.body_geomnum[b]
            if adr >= 0 and cnt > 0:
                out.extend(range(adr, adr + cnt))
            stack.extend(children.get(b, []))
        return out

    def _apply_order_tracking(self):
        """Each physics tick: test every orderable object against the basket's
        interior bbox and record first-entry / first-exit transitions.

        Entry (outside->inside) appends the object's catalog name to the basket
        ObjectInit's ``place_seq``; exit (inside->outside, after an entry)
        appends to ``remove_seq``. Each object is recorded at most once per
        direction (place_done / remove_done gates), so re-entry or a re-grab does
        not mutate the recorded order. The test is in the basket bbox geom's
        live world frame (geom_xpos / geom_xmat), so it follows the basket
        wherever it rests and however it yaws.

        Runs every physics tick (1000 Hz); work is one point-in-box test per
        orderable object, negligible vs mj_step. No-op when no basket is bound.
        """
        if not self._basket_index or not self._order_index:
            return
        d = self.data
        idx = self._basket_index
        gid = idx["bbox_gid"]
        sx, sy, sz = idx["s"]
        mxy = idx["margin_xy"]
        mz = idx["margin_z"]
        # Basket bbox geom live world frame.
        gx = np.asarray(d.geom_xpos[gid], dtype=np.float64)
        gmat = np.asarray(d.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        basket_obj = idx["obj"]
        place_seq = basket_obj.place_seq
        remove_seq = basket_obj.remove_seq
        for e in self._order_index:
            ox = np.asarray(d.xpos[e["bid"]], dtype=np.float64)
            dw = ox - gx                       # world-frame delta from bbox centre
            dl = gmat.T @ dw                   # into the bbox local frame (box is axis-aligned here)
            lx, ly, lz = float(dl[0]), float(dl[1]), float(dl[2])
            inside = (abs(lx) < sx - mxy
                      and abs(ly) < sy - mxy
                      and (-sz - mz) < lz < (sz + mz))
            if inside and not e["prev_inside"] and not e["place_done"]:
                place_seq.append(e["name"])
                e["place_done"] = True
            elif (not inside and e["prev_inside"] and e["place_done"]
                  and not e["remove_done"]):
                remove_seq.append(e["name"])
                e["remove_done"] = True
            e["prev_inside"] = inside
        # No mj_forward: data.xpos / geom_xpos / geom_xmat are already current
        # after mj_step; a redundant forward every tick (1000 Hz) would race the
        # render thread and stall the sim (same lesson as hammer-nail / spray).

    def _reset_order_state(self, seed: Optional[int] = None):
        """Zero the basket's place/remove sequences and re-arm every orderable
        object's transition flags at the start of each episode.

        Only bookkeeping is reset (no pose writes), so -- unlike the
        hammer/spray/button reset hooks -- there is no mj_forward here. Caller
        (reset) already holds self.lock.
        """
        # The basket index holds references to ObjectInit instances. When the
        # same task is re-selected, set_task parses a FRESH Task (new ObjectInit
        # instances) but -- because the scene did not change -- takes the
        # `else: self.task = new_task` branch and does NOT rebuild the indices.
        # That left _basket_obj pointing at the PREVIOUS parse's basket while
        # the SubtaskTracker / `placed_n` predicate read the new parse's basket,
        # so the env wrote place_seq to one object and the predicate read another
        # -> placed_n never latched. Rebuild whenever the bound basket instance
        # is not the current task's basket (also clears a stale index when the
        # task is no longer a basket task).
        cur_basket = None
        if self.task is not None:
            cur_basket = next((o for o in self.task.all_object_inits() if o.is_basket), None)
        if self._basket_obj is not cur_basket:
            print("[SimEnv] _reset_order_state: basket instance out of sync, "
                  "rebuilding order-tracker index")
            self._build_basket_indices()
        if not self._basket_index:
            return
        print(f"[SimEnv] _reset_order_state: orderables={len(self._order_index)} "
              f"placed_so_far={len(self._basket_obj.place_seq)}")
        self._basket_obj.place_seq = []
        self._basket_obj.remove_seq = []
        for e in self._order_index:
            e["prev_inside"] = False
            e["place_done"] = False
            e["remove_done"] = False

    # ---- "caps appear on button press" (reveal objects) ----------------------
    def _park_reveal_objects(self, recompute: bool = False):
        """Park every `reveal` object at its declared pose and stash its
        region-sampled spawn pose as the teleport target for the first button
        press.

        A `reveal: true` object (e.g. a cap that must hide a cube) is not
        supposed to exist in the scene until the robot presses the button. Its
        `region` IS sampled like any other (so the reveal pose follows the
        episode's jittered layout), but at reset the env writes the object's
        explicit `pos`/`quat` (a hidden park pose, e.g. below the floor) over
        its free joint instead. `_apply_reveal_on_press` teleports it to the
        stashed pose on the first counted press.

        `recompute=False` (fresh Path-3 resets): targets come from the
        `_last_sampled` dict written by apply_task_initial_state.
        `recompute=True` (Path-1 init-state restores, which skip the sampler):
        targets are re-derived live through RegionSampler against the restored
        model/data, so an anchor-relative pinned region still resolves against
        the restored anchor body.
        """
        self._reveal_targets = {}
        self._reveal_done = False
        if self.task is None:
            return
        from bidex.tasks.domain import _set_freejoint_pose
        from bidex.tasks.region_sampler import RegionSampler
        n_parked = 0
        for obj in self.task.all_object_inits():
            if not getattr(obj, "reveal", False):
                continue
            joint_name = f"obj{obj.instance_id}_freejoint"
            if obj.pos is not None and _set_freejoint_pose(
                self.model, self.data, joint_name, tuple(obj.pos),
                tuple(obj.quat) if obj.quat is not None else None,
            ):
                n_parked += 1
            target = None
            if not recompute:
                target = (getattr(self, "_last_sampled", None) or {}).get(obj.name)
            elif obj.region and obj.region in self.task.regions:
                sampler = RegionSampler(np.random.default_rng(0))
                target = sampler.sample(
                    self.task.regions[obj.region], model=self.model, data=self.data)
            if target is not None:
                self._reveal_targets[obj.name] = (obj.instance_id, target[0], target[1])
        if n_parked:
            mujoco.mj_forward(self.model, self.data)
            print(f"[SimEnv] reveal: {n_parked} object(s) parked out of sight; "
                  f"they appear on the first button press")

    def _apply_reveal_on_press(self):
        """Each physics tick (after the button counter): the first time any
        button's press_count crosses 1, teleport every parked `reveal` object
        to its stashed spawn pose with zero velocity -- the "cups appear and
        cover the cubes" moment. After the teleport the objects are normal
        free bodies (the robot can lift them). No-op for tasks without
        `reveal` objects and after the reveal has fired."""
        if getattr(self, "_reveal_done", True) or not getattr(self, "_reveal_targets", None):
            return
        if self.task is None:
            return
        if not any(int(getattr(o, "press_count", 0)) >= 1
                   for o in self.task.all_object_inits() if o.is_button):
            return
        from bidex.tasks.domain import _set_freejoint_pose
        n = 0
        for name, (iid, pos, quat) in list(self._reveal_targets.items()):
            joint_name = f"obj{iid}_freejoint"
            if _set_freejoint_pose(self.model, self.data, joint_name, pos, quat):
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid >= 0:
                    vadr = self.model.jnt_dofadr[jid]
                    self.data.qvel[vadr:vadr + 6] = 0.0
                n += 1
        mujoco.mj_forward(self.model, self.data)
        self._reveal_done = True
        print(f"[SimEnv] reveal: {n} object(s) appeared (button pressed)")

    # ---- flyover counter ("over the other cup N times") ----------------------
    _FLYOVER_RADIUS = 0.06   # xy radius around the zone body's live xpos

    def _reset_flyover_state(self, seed: Optional[int] = None):
        """(Re)build the flyover tracker and zero every counter at reset.

        Indexes the task's `flyover_zone` container (its live body) and every
        `flyover` object (the moving item), mirroring the basket-order index
        build. No-op for tasks without the flags. Only bookkeeping is reset
        (no pose writes), so there is no mj_forward here (caller holds
        self.lock).
        """
        self._flyover_index = []
        self._flyover_zone_bid = -1
        if self.task is None:
            return
        zone_obj = next((o for o in self.task.all_object_inits()
                         if getattr(o, "flyover_zone", False)), None)
        if zone_obj is None:
            return
        prefix = f"obj{zone_obj.instance_id}_"
        zbid = -1
        for cand in (prefix + "object", prefix + "root"):
            zbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if zbid >= 0:
                break
        if zbid < 0:
            print(f"[SimEnv] flyover: zone body for {zone_obj.name!r} not found; "
                  f"tracking disabled")
            return
        self._flyover_zone_bid = zbid
        for obj in self.task.all_object_inits():
            if not getattr(obj, "flyover", False):
                continue
            prefix = f"obj{obj.instance_id}_"
            bid = -1
            for cand in (prefix + "object", prefix + "root"):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cand)
                if bid >= 0:
                    break
            if bid < 0:
                print(f"[SimEnv] flyover: item body for {obj.name!r} not found; skipping")
                continue
            obj.flyover_count = 0
            self._flyover_index.append({"obj": obj, "bid": bid, "prev": False})
        if self._flyover_index:
            print(f"[SimEnv] flyover: tracking {len(self._flyover_index)} object(s) "
                  f"over zone {zone_obj.name!r}")

    def _apply_flyover_tracking(self):
        """Each physics tick: count every outside->inside transition of each
        `flyover` object into the flyover zone's live xy radius + z window.

        The zone test mirrors `in_bottle` (xy radius around the zone body's
        live xpos; object z within origin-0.25 .. origin+0.20), so a HELD item
        passing over the container's mouth counts exactly as an item resting
        inside would -- "passed over the cup" and "placed into the cup" are the
        same zone, and the caller picks the choreography accordingly. The
        counter lives on the item's ObjectInit (`flyover_count`) so the
        stateless `flyover_n` predicate reads it without an env handle (same
        pattern as `pressed` reading press_count). No mj_forward: xpos is
        already current after mj_step.
        """
        if not getattr(self, "_flyover_index", None):
            return
        d = self.data
        zx, zy, zz = (float(v) for v in d.xpos[self._flyover_zone_bid][:3])
        for e in self._flyover_index:
            ox, oy, oz = (float(v) for v in d.xpos[e["bid"]][:3])
            dxy = ((ox - zx) ** 2 + (oy - zy) ** 2) ** 0.5
            in_zone = dxy < self._FLYOVER_RADIUS and (zz - 0.25) < oz < (zz + 0.20)
            if in_zone and not e["prev"]:
                e["obj"].flyover_count = int(getattr(e["obj"], "flyover_count", 0)) + 1
                print(f"[flyover] {e['obj'].name} entered zone "
                      f"(count={e['obj'].flyover_count}, dxy={dxy:.3f}, dz={oz - zz:+.3f})")
            e["prev"] = in_zone

    # ---- color-memory cube/cups (color swap on flyover) ----------------------
    _CUP_RGBA = {
        "red": (0.80, 0.25, 0.25, 1.0),
        "blue": (0.25, 0.45, 0.85, 1.0),
    }
    _CUBE_RGBA = {
        "red": (0.82, 0.28, 0.28, 1.0),
        "blue": (0.25, 0.40, 0.85, 1.0),
        "grey": (0.55, 0.55, 0.58, 1.0),
    }
    _COLOR_SWAP_MAX = 5

    def _color_visual_gids(self, iid: int) -> list:
        """Group-2 (visual) geom ids of object `iid` -- the geoms whose rgba
        the color hooks rewrite (collision geoms stay untouched)."""
        prefix = f"obj{iid}_"
        return [g for g in range(self.model.ngeom)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                .startswith(prefix) and self.model.geom_group[g] == 2]

    def _set_visual_rgba(self, iid: int, rgba) -> None:
        for g in self._color_visual_gids(iid):
            self.model.geom_rgba[g] = rgba

    def _reset_color_swap_state(self, seed: Optional[int] = None):
        """Randomize the color-cube's spawn color, pin the cups' start colors,
        and zero the swap/restore bookkeeping at the start of each episode.

        The cube's initial color is drawn from the reset seed (per-episode
        random, "red" or "blue") and written to its visual geoms; the FIRST
        `color_cup` in declaration order starts red, the second blue. All
        state lives on the ObjectInits (`cube_initial_color` / `cup_color` /
        `color_swap_count` / flags) so the predicates stay stateless. Only
        bookkeeping + visual rgba are written here; no mj_forward (the caller
        holds self.lock and forwards right after).
        """
        self._color_cube_obj = None
        self._color_cup_objs = []
        self._color_prev_zone = -1   # cup-zone index; -1 = outside both zones
        if self.task is None:
            return
        cube = next((o for o in self.task.all_object_inits()
                     if getattr(o, "color_cube", False)), None)
        cups = [o for o in self.task.all_object_inits() if getattr(o, "color_cup", False)]
        if cube is None or len(cups) < 2:
            return
        rng = np.random.default_rng(seed)
        cube.cube_initial_color = str(rng.choice(["red", "blue"]))
        cube.color_greyed = False
        cube.color_restored = False
        cube.color_matched = False
        cube.color_swap_count = 0
        self._set_visual_rgba(cube.instance_id, self._CUBE_RGBA[cube.cube_initial_color])
        for k, cup in enumerate(cups[:2]):
            cup.cup_color = "red" if k == 0 else "blue"
            self._set_visual_rgba(cup.instance_id, self._CUP_RGBA[cup.cup_color])
        self._color_cube_obj = cube
        self._color_cup_objs = cups[:2]
        print(f"[SimEnv] color-swap: cube starts {cube.cube_initial_color}; "
              f"cups {[c.cup_color for c in cups[:2]]}")

    def _color_cup_body_pos(self, cup) -> tuple:
        prefix = f"obj{cup.instance_id}_"
        for cand in (prefix + "object", prefix + "root"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if bid >= 0:
                p = self.data.xpos[bid]
                return float(p[0]), float(p[1]), float(p[2])
        return (float("nan"),) * 3

    def _apply_color_swap_tick(self):
        """Each physics tick: drive the color-memory state machine.

        1. GREY-OUT: once any button's press_count reaches 3, the cube's visual
           rgba turns grey ("its color disappears") -- the model must remember
           the spawn color it saw at frame 0.
        2. SWAP: while fewer than 5 swaps have fired, a CUP-TO-CUP TRANSITION
           of the cube (leaving one cup's flyover zone and entering the
           OTHER's; xy radius 0.06 m, z window origin-0.25 .. origin+0.22)
           swaps the two cups' colors (visual rgba + `cup_color` fields).
           Arriving at the FIRST cup from outside (e.g. straight from the
           table after the grasp) does NOT swap -- the count only tracks
           transitions BETWEEN the cups, matching the choreographed "carry it
           back and forth over the two cups, five times".
        3. RESTORE+JUDGE: once greyed, the first frame the cube rests INSIDE a
           cup (tight zone: dxy < 0.06, dz < +0.04 -- placed cubes sit at dz
           ~0.026) AND is in CONTACT with the cup body (a released cube settles
           onto the cup floor; a held flyover -- even one cruising below the
           old dz < +0.04-only bound -- never touches, so it can no longer
           trigger a premature restore) restores its rgba to the initial color
           and records whether that cup's CURRENT color matches
           (`color_matched`). BOTH must hold for a match: matching color AND
           all 5 swaps already fired -- an EARLY placement records
           restored=True / matched=False, a permanent failure, exactly like a
           wrong-color placement. The `color_match_place` predicate reads the
           two flags.
        """
        cube = getattr(self, "_color_cube_obj", None)
        cups = getattr(self, "_color_cup_objs", None)
        if cube is None or not cups:
            return
        d = self.data
        prefix = f"obj{cube.instance_id}_"
        cbid = -1
        for cand in (prefix + "object", prefix + "root"):
            cbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if cbid >= 0:
                break
        if cbid < 0:
            return
        ox, oy, oz = (float(v) for v in d.xpos[cbid][:3])

        # 1) grey-out after the 3rd press.
        if not cube.color_greyed and any(
                int(getattr(o, "press_count", 0)) >= 3
                for o in self.task.all_object_inits() if o.is_button):
            cube.color_greyed = True
            self._set_visual_rgba(cube.instance_id, self._CUBE_RGBA["grey"])
            print("[color-swap] 3 presses: cube color hidden")

        # 2) cup-color swap ON ENTERING THE MATCH-COLORED cup's zone.
        # zone_id: -1 = outside both zones, else the index of the cup whose
        # zone holds the cube. A swap fires the tick the cube ARRIVES at the
        # cup whose CURRENT color matches its initial color -- including the
        # very first arrival from outside ("move over the same-color cup and
        # the colors change immediately"). Entering the different-colored cup
        # never swaps. After each swap the cube sits over the non-matching
        # cup, so the alternating shuttle fires exactly one swap per cup
        # visit, up to 5; after the cap the colors are frozen no matter where
        # the cube goes.
        zone_id = -1
        for ci, c in enumerate(cups):
            zx, zy, zz = self._color_cup_body_pos(c)
            dxy = ((ox - zx) ** 2 + (oy - zy) ** 2) ** 0.5
            if dxy < 0.06 and (zz - 0.25) < oz < (zz + 0.22):
                zone_id = ci
                break
        prev_zone = getattr(self, "_color_prev_zone", -1)
        if (zone_id >= 0 and zone_id != prev_zone
                and cube.color_swap_count < self._COLOR_SWAP_MAX
                and cups[zone_id].cup_color == cube.cube_initial_color):
            a, b = cups
            a.cup_color, b.cup_color = b.cup_color, a.cup_color
            for c in cups:
                self._set_visual_rgba(c.instance_id, self._CUP_RGBA[c.cup_color])
            cube.color_swap_count += 1
            print(f"[color-swap] swap {cube.color_swap_count}/"
                  f"{self._COLOR_SWAP_MAX} (arrive cup {zone_id}, match): "
                  f"cups now {[c.cup_color for c in cups]}")
        # Sticky zone memory: only record CUP zones, never the outside state.
        # A shuttle crossing the tangent boundary between the two zones reads
        # -1 (outside) for a tick or two; writing that into prev would turn
        # the next entry into a fresh "arrival" and DOUBLE-count. Keeping the
        # last CUP zone makes each cup visit fire exactly once, however slow
        # the crossing.
        if zone_id >= 0:
            self._color_prev_zone = zone_id

        # 3) restore on placement + judge the color match.
        if cube.color_greyed and not cube.color_restored:
            for c in cups:
                zx, zy, zz = self._color_cup_body_pos(c)
                dxy = ((ox - zx) ** 2 + (oy - zy) ** 2) ** 0.5
                placed = dxy < 0.06 and (zz - 0.25) < oz < (zz + 0.04)
                if not placed:
                    continue
                # Contact gate: the cube must be TOUCHING the cup (resting on
                # its interior floor/wall). The pure volume test above fires
                # while the cube is still HELD low over the cup (a gripper
                # cruising at dz < 0.04 during the shuttle reads as "placed"),
                # so the cube's color snapped back before it was ever dropped
                # in. A released cube settles onto the cup floor -> contact;
                # a held/falling one does not touch -> no restore yet.
                cup_bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY,
                    f"obj{c.instance_id}_object")
                if cup_bid < 0:
                    cup_bid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY,
                        f"obj{c.instance_id}_root")
                if cup_bid < 0:
                    continue
                touching = False
                for con_i in range(d.ncon):
                    con = d.contact[con_i]
                    b1 = self.model.geom_bodyid[con.geom1]
                    b2 = self.model.geom_bodyid[con.geom2]
                    if (b1 == cbid and b2 == cup_bid) or (b2 == cbid and b1 == cup_bid):
                        touching = True
                        break
                if not touching:
                    continue
                cube.color_restored = True
                self._set_visual_rgba(
                    cube.instance_id, self._CUBE_RGBA[cube.cube_initial_color])
                # Early placement before all swaps fired is a permanent
                # failure, exactly like a wrong-color placement.
                cube.color_matched = (
                    c.cup_color == cube.cube_initial_color
                    and cube.color_swap_count >= self._COLOR_SWAP_MAX)
                print(f"[color-swap] cube restored {cube.cube_initial_color} in a "
                      f"{c.cup_color} cup after {cube.color_swap_count}/"
                      f"{self._COLOR_SWAP_MAX} swaps -> match={cube.color_matched}")
                break
