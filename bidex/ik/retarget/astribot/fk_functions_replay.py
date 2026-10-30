"""Forward kinematics for the astribot S1 left/right 7-DOF arms.

Mirror of gr1_avp_teleop/fk_functions.py for astribot. The
humanoid-arm-retarget pipeline needs `left_fk(q)` / `right_fk(q)` that take
7 joint angles and return (pos_xyz, quat_wxyz) of the arm end-effector
(astribot_arm_{left,right}_tool_link), computed from the first 5 joints
(joint_1..5); joint_6/7 are the analytic wrist DOF held at zero in the fixed
tool segment.

Astribot arm DOF split is 3-1-3 (shoulder 3, elbow 1, wrist 3). The retarget
optimizes joints 1..5 (shoulder3 + elbow + wrist_1) for end-effector position
and solves the last 2 wrist DOF geometrically — same 5+2 convention as GR1.

Chain numbers come straight from the astribot_s1_fixed_sharpa MJCF. The base
is welded to world (root chassis_base), and torso(4)+head(2) joints are held
at zero, so the static prefix chassis_base->torso->arm_link_1 collapses into
the world-frame static pose of arm_link_1 (pre-computed below). `selftest()`
cross-validates against MuJoCo's mj_forward over random poses.

Pose frame: returned (pos, quat) is in world == robot-base frame (base welded
to world), matching the config left_base/right_base convention.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ----------------------------- quaternion utils -----------------------------

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


# ----------------------------- chain definitions -----------------------------
# Each entry: (body name, parent->child translation in parent frame,
#              parent->child rotation as wxyz quaternion, joint axis in
#              child-local frame, joint slot — index into q, or -1 if fixed).
#
# First entry = arm_link_1 expressed in WORLD (static prefix chassis->torso
# folded in, torso held at zero). Joints 1..5 are slots 0..4; joint_6/7 are
# held at zero inside the fixed tool segment. Last entry = link_7 -> tool_link
# static offset. All numbers from the astribot_s1_fixed_sharpa MJCF.

_RIGHT_CHAIN: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = [
    # arm_right_link_1 static world pose (torso=0), then joint_1 about z.
    ("astribot_arm_right_link_1",
     np.array([-0.150004, -0.000123, 1.386353]),
     np.array([0.819151, 0.000230, -0.573578, 0.000328]),
     np.array([0.0, 0.0, 1.0]), 0),  # joint_1
    ("astribot_arm_right_link_2",
     np.array([0.0, 0.0, 0.0]),
     np.array([0.70711, 0.70711, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0]), 1),  # joint_2
    ("astribot_arm_right_link_3",
     np.array([0.0, 0.05, 0.0]),
     np.array([0.5, 0.5, -0.5, -0.5]),
     np.array([0.0, 0.0, 1.0]), 2),  # joint_3
    ("astribot_arm_right_link_4",
     np.array([0.03, 0.0, 0.309]),
     np.array([0.70711, -0.70711, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0]), 3),  # joint_4
    ("astribot_arm_right_link_5",
     np.array([0.0, 0.0, 0.0]),
     np.array([0.70711, 0.70711, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0]), 4),  # joint_5 (wrist_1)
    # joint_6/7 held at zero — fold link_5->link_6->link_7->tool into one
    # static segment. link_6 pos=[0,0,0.277] quat=[.7071,-.7071,0,0];
    # link_7 pos=0 quat=[.7071,0,.7071,0]; tool pos=[0,-0.048,0]
    # quat=[.7071,.7071,0,0]. Composed below (all at zero joint angle).
    ("astribot_arm_right_link_6",
     np.array([0.0, 0.0, 0.277]),
     np.array([0.70711, -0.70711, 0.0, 0.0]),
     np.zeros(3), -1),
    ("astribot_arm_right_link_7",
     np.array([0.0, 0.0, 0.0]),
     np.array([0.70711, 0.0, 0.70711, 0.0]),
     np.zeros(3), -1),
    ("astribot_arm_right_tool_link",
     np.array([0.0, -0.048, 0.0]),
     np.array([0.70711, 0.70711, 0.0, 0.0]),
     np.zeros(3), -1),
]

_LEFT_CHAIN: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = [
    ("astribot_arm_left_link_1",
     np.array([0.150000, 0.000117, 1.386355]),
     np.array([-0.000328, 0.573575, 0.000230, 0.819153]),
     np.array([0.0, 0.0, 1.0]), 0),
    ("astribot_arm_left_link_2",
     np.array([0.0, 0.0, 0.0]),
     np.array([0.70711, 0.70711, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0]), 1),
    ("astribot_arm_left_link_3",
     np.array([0.0, 0.05, 0.0]),
     np.array([0.5, 0.5, -0.5, -0.5]),
     np.array([0.0, 0.0, 1.0]), 2),
    ("astribot_arm_left_link_4",
     np.array([0.03, 0.0, 0.309]),
     np.array([0.70711, -0.70711, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0]), 3),
    ("astribot_arm_left_link_5",
     np.array([0.0, 0.0, 0.0]),
     np.array([0.70711, 0.70711, 0.0, 0.0]),
     np.array([0.0, 0.0, 1.0]), 4),
    ("astribot_arm_left_link_6",
     np.array([0.0, 0.0, 0.277]),
     np.array([0.70711, -0.70711, 0.0, 0.0]),
     np.zeros(3), -1),
    ("astribot_arm_left_link_7",
     np.array([0.0, 0.0, 0.0]),
     np.array([0.70711, 0.0, 0.70711, 0.0]),
     np.zeros(3), -1),
    ("astribot_arm_left_tool_link",
     np.array([0.0, -0.048, 0.0]),
     np.array([0.70711, 0.70711, 0.0, 0.0]),
     np.zeros(3), -1),
]


def _fk_chain(q: np.ndarray, chain) -> Tuple[np.ndarray, np.ndarray]:
    """Walk the chain; return (pos, quat) of the last body in world frame."""
    pos = np.zeros(3)
    rot = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz

    for _name, t_parent_child, q_parent_child, axis, slot in chain:
        pos = pos + _quat_rotate(rot, t_parent_child)
        rot = _quat_mul(rot, q_parent_child)
        if slot >= 0:
            rot = _quat_mul(rot, _quat_from_axis_angle(axis, float(q[slot])))
        rot = _quat_normalize(rot)

    return pos, rot


def right_fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FK of right arm to astribot_arm_right_tool_link, in world/base coords.
    Uses joints 0..4; joint_6/7 held at zero in the fixed tool segment."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return _fk_chain(q, _RIGHT_CHAIN)


def left_fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FK of left arm to astribot_arm_left_tool_link, in world/base coords."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    return _fk_chain(q, _LEFT_CHAIN)


# ----------------------------- selftest -----------------------------

def _mujoco_body_pose(model, data, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.array(data.xpos[bid], dtype=np.float64), \
        np.array(data.xquat[bid], dtype=np.float64)


def _quat_angle_dist(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * np.arccos(dot)


ARM_JOINT_NAMES = [
    "astribot_arm_left_joint_1", "astribot_arm_left_joint_2",
    "astribot_arm_left_joint_3", "astribot_arm_left_joint_4",
    "astribot_arm_left_joint_5", "astribot_arm_left_joint_6",
    "astribot_arm_left_joint_7",
    "astribot_arm_right_joint_1", "astribot_arm_right_joint_2",
    "astribot_arm_right_joint_3", "astribot_arm_right_joint_4",
    "astribot_arm_right_joint_5", "astribot_arm_right_joint_6",
    "astribot_arm_right_joint_7",
]


def selftest(seed: int = 0, n: int = 10, verbose: bool = False) -> bool:
    """Cross-validate analytical FK vs MuJoCo's xpos/xquat. Returns True on pass."""
    import mujoco
    from bidex.robots.registry import resolve_robot_xml
    xml_path = resolve_robot_xml("astribot_s1_fixed_sharpa")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    joint_qadr = []
    for name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_qadr.append(model.jnt_qposadr[jid])

    rng = np.random.default_rng(seed)
    ok = True
    for trial in range(n):
        data.qpos[:] = 0.0
        # left = q_full[0:7], right = q_full[7:14]; only slots 0..4 feed FK.
        q_full = np.zeros(14)
        for slot in (0, 1, 2, 3, 4):
            q_full[slot] = rng.uniform(-1.0, 1.0)
            q_full[7 + slot] = rng.uniform(-1.0, 1.0)
        for j in range(14):
            data.qpos[joint_qadr[j]] = q_full[j]

        mujoco.mj_forward(model, data)

        mj_pos_r, mj_quat_r = _mujoco_body_pose(
            model, data, "astribot_arm_right_tool_link")
        mj_pos_l, mj_quat_l = _mujoco_body_pose(
            model, data, "astribot_arm_left_tool_link")

        fk_pos_r, fk_quat_r = right_fk(q_full[7:14])
        fk_pos_l, fk_quat_l = left_fk(q_full[0:7])

        dp_r = np.linalg.norm(fk_pos_r - mj_pos_r)
        dp_l = np.linalg.norm(fk_pos_l - mj_pos_l)
        dq_r = _quat_angle_dist(fk_quat_r, mj_quat_r)
        dq_l = _quat_angle_dist(fk_quat_l, mj_quat_l)

        if verbose:
            print(f"trial {trial:2d}  R dp={dp_r:.2e} dq={dq_r:.2e}  "
                  f"L dp={dp_l:.2e} dq={dq_l:.2e}")
        if dp_r > 1e-4 or dp_l > 1e-4 or dq_r > 1e-4 or dq_l > 1e-4:
            ok = False
            if not verbose:
                print(f"FK MISMATCH trial {trial}: R dp={dp_r:.3e} dq={dq_r:.3e}  "
                      f"L dp={dp_l:.3e} dq={dq_l:.3e}")
    return ok


if __name__ == "__main__":
    import sys
    passed = selftest(verbose=True)
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)
