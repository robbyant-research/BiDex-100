"""Franka (dual_fr3v2_sharpa) head-camera frame + the deterministic B<->world
transforms, numpy-only (no mujoco), shared by the ee-replay lib (which needs
mujoco) and the matplotlib GIF (which does not).

Frames
------
* W : eval sim world (where the env renders; link8 `xpos`, base body, head cam).
* B : the parquet `abs_end_effector` frame -- the EVAL ARENA dual-arm-base frame
      (= FRANKA_BASE_WORLD, Rz(+135deg)+[0.5,-0.45,0.6]). Since the plan-B rebase, parquet ee_B is stored in this deterministic arena frame,
      NOT the legacy collection `_calibrate_arm_base` frame (see M_CALIB_*).
* Cam (optical) : head camera, OpenCV optical convention (+Z toward scene,
      +X right, +Y down). The MuJoCo camera looks along -Z; optical = MuJoCo
      body axes with R @ diag(1,-1,-1) (origin unchanged).

The B -> W lift is the DETERMINISTIC arena pose (no empirical fit):

    ee_W = M_WORLD_FROM_B @ ee_B = FRANKA_BASE_WORLD @ ee_B   (ee_B: 14-D xyzw right-first)
    head-cam-in-W = FRANKA_HEAD_WORLD (MuJoCo body) / FRANKA_HEAD_OPTICAL for optical
    head-cam-in-B = M_B_FROM_WORLD @ FRANKA_HEAD_OPTICAL = T_CAM_B_OPTICAL
    camera_ee = inv(T_CAM_B_OPTICAL) @ ee_B = inv(head_optical) @ ee_W   (base-frame invariant)

Legacy (M_CALIB_*, kept for the one-time rebase only): before plan B the parquet
ee lived in the collection `_calibrate_arm_base` frame, which deviated from the
arena pose by a fixed FK-model offset (standalone FK vs assembly link8 FK). The
offline `examples/franka_rebase_arena.py` applies T_REBASE_CALIB_TO_ARENA to
convert legacy `data_fixed/` -> arena-frame `data_arena/`. Replay IK is unaffected:
T_coll is re-fit per episode from (state_arm, state_ee) so it adapts to whatever
base frame the ee is stored in, and the env is driven in JOINT mode.

Constants below are derived from FRANKA_BASE_WORLD / FRANKA_HEAD_WORLD (do not
hand-edit M_* / T_CAM_B_OPTICAL). M_CALIB_R / M_CALIB_t are the fitted legacy
offset; regenerate only if the collection FK model changes.
"""
import numpy as np

# Eval arena base pose (robot base body in W) -- the collection fork tabletop
# task yaml uses the SAME robot_pose. This is the table-default
# `base_offset_for(scene, task.robot_pose)` (tabletop yamls carry no
# robot_pose, so the arena default applies): Rz(+135deg) + [0.5,-0.45,0.6].
FRANKA_BASE_WORLD = np.array([
    [-0.70710678, -0.70710678,  0.0,  0.5],
    [ 0.70710678, -0.70710678,  0.0, -0.45],
    [ 0.0,         0.0,         1.0,  0.6],
    [ 0.0,         0.0,         0.0,  1.0],
])
# Head cam MuJoCo body pose in W (== env d.cam_xpos / d.cam_xmat at reset).
FRANKA_HEAD_WORLD = np.array([
    [ 0.011288, -0.704399,  0.709714,  0.450855],
    [ 0.999915,  0.012608, -0.003391,  0.033281],
    [-0.006559,  0.709692,  0.704481,  1.419674],
    [ 0.0,       0.0,       0.0,       1.0],
])
# Head cam in OpenCV OPTICAL convention (+Z toward scene, +X right, +Y down):
# MuJoCo body axes post-multiplied by diag(1,-1,-1) (origin unchanged).
FRANKA_HEAD_OPTICAL = FRANKA_HEAD_WORLD.copy()
FRANKA_HEAD_OPTICAL[:3, :3] = FRANKA_HEAD_WORLD[:3, :3] @ np.diag([1.0, -1.0, -1.0])

# --------------------------------------------------------------------------- #
# Convention (plan-B rebase):
#   The parquet `abs_end_effector` (robot_ee) is stored in the EVAL ARENA base
#   frame B (== FRANKA_BASE_WORLD), NOT the collection `_calibrate_arm_base`
#   frame. Therefore the B->world lift is the DETERMINISTIC arena pose -- no
#   empirical fit is needed:
#       ee_W = M_WORLD_FROM_B @ ee_B = FRANKA_BASE_WORLD @ ee_B
M_WORLD_FROM_B = FRANKA_BASE_WORLD.copy()
M_R = M_WORLD_FROM_B[:3, :3].copy()
M_t = M_WORLD_FROM_B[:3, 3].copy()
M_B_FROM_WORLD = np.linalg.inv(M_WORLD_FROM_B)

# Head camera OPTICAL pose in the arena B frame:
#   head_optical_in_B = inv(arena) @ head_optical_in_W.
# camera_ee is base-frame INVARIANT (it only depends on ee_W + the head_world
# pose): camera_ee = inv(head_optical) @ ee_W = inv(T_CAM_B_OPTICAL) @ ee_B.
T_CAM_B_OPTICAL = M_B_FROM_WORLD @ FRANKA_HEAD_OPTICAL
CAM_R = T_CAM_B_OPTICAL[:3, :3].copy()
CAM_t = T_CAM_B_OPTICAL[:3, 3].copy()

# --------------------------------------------------------------------------- #
# Legacy: the COLLECTION FK-calibrated base -> world fit (pre-rebase).
# Before plan B the parquet ee lived in the collection `_calibrate_arm_base`
# frame (data_collect_sim.py:1186/1308), which deviated from the arena pose by
# a FIXED FK-model offset (standalone FK vs assembly link8 FK, beyond the
# link8 +45deg-Z flange). M_CALIB_* recovers that offset (fitted once, residual
# ~0.3 mm, validated episode-independent). Kept ONLY for the one-time offline
# rebase (franka_rebase_arena.py) + historical reference; the live pipeline uses
# the arena M_* / T_CAM_B_OPTICAL above.
M_CALIB_R = np.array([
    [ 0.939545192267,  0.000054603494,  0.342424924189],
    [ 0.242248025536, -0.706870810206, -0.664567191338],
    [ 0.242013895905,  0.707342671339, -0.664150298872],
])
M_CALIB_t = np.array([-0.655236797705, 0.435005439987, 2.195140688594])
M_CALIB_WORLD_FROM_B = np.eye(4)
M_CALIB_WORLD_FROM_B[:3, :3] = M_CALIB_R
M_CALIB_WORLD_FROM_B[:3, 3] = M_CALIB_t
# T_rebase maps the legacy calibrated-B frame -> arena-B frame:
#   ee_arena = inv(FRANKA_BASE) @ M_CALIB @ ee_calib
T_REBASE_CALIB_TO_ARENA = M_B_FROM_WORLD @ M_CALIB_WORLD_FROM_B


# --------------------------------------------------------------------------- #
# quaternion helpers (wxyz internally; parquet uses xyzw)                     #
# --------------------------------------------------------------------------- #
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qinv(a):
    w, x, y, z = a
    n = w*w + x*x + y*y + z*z
    return np.array([w, -x, -y, -z]) / n


def rotmat_to_quat_wxyz(R):
    """Rotation matrix -> quaternion (w, x, y, z) via the numerically robust
    Shepperd method (handles the 180deg / negative-trace branches). Self-consistent
    with :func:`qmul` / the xyzw<->wxyz helpers (verified vs scipy to ~1e-9)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0          # S = 4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


_CAM_qR = None
_CAM_qR_inv = None


def _cam_qR():
    global _CAM_qR, _CAM_qR_inv
    if _CAM_qR is None:
        _CAM_qR = rotmat_to_quat_wxyz(CAM_R)
        _CAM_qR_inv = qinv(_CAM_qR)
    return _CAM_qR, _CAM_qR_inv


# --------------------------------------------------------------------------- #
# ee transforms (per-arm 7-D xyzw, right-first 14-D rows)                     #
# --------------------------------------------------------------------------- #
def ee_B_to_camera(ee_B_xyzw):
    """Rebase one arm's ee from the parquet B frame (xyzw, w/ link8 flange) into
    the head-cam OPTICAL frame (xyzw, w/ flange; rigid rebase leaves the flange
    untouched).

        p_cam = R^T (p_B - t);  q_cam = qR^{-1} (x) q_B_wxyz
    """
    qR, qR_inv = _cam_qR()
    pB = np.asarray(ee_B_xyzw[:3], dtype=np.float64)
    qB = xyzw_to_wxyz(ee_B_xyzw[3:7])
    p_cam = CAM_R.T @ (pB - CAM_t)
    q_cam = qmul(qR_inv, qB)
    return np.concatenate([p_cam, wxyz_to_xyzw(q_cam)])


def ee_camera_to_B(ee_cam_xyzw):
    """Inverse of ee_B_to_camera: optical-camera-frame ee (xyzw) -> parquet B
    frame ee (xyzw, w/ flange).

        p_B = R p_cam + t;  q_B = qR (x) q_cam_wxyz
    """
    qR, _ = _cam_qR()
    pc = np.asarray(ee_cam_xyzw[:3], dtype=np.float64)
    qc = xyzw_to_wxyz(ee_cam_xyzw[3:7])
    p_B = CAM_R @ pc + CAM_t
    q_B = qmul(qR, qc)
    return np.concatenate([p_B, wxyz_to_xyzw(q_B)])


def ee_row_B_to_camera_row(ee_B_row14):
    """Apply ee_B_to_camera to both arms of a 14-D right-first parquet row."""
    out = np.empty(14, dtype=np.float64)
    out[0:7]  = ee_B_to_camera(np.asarray(ee_B_row14[0:7],  dtype=np.float64))
    out[7:14] = ee_B_to_camera(np.asarray(ee_B_row14[7:14], dtype=np.float64))
    return out


def ee_B_to_world(ee_B_xyzw):
    """Lift one arm's parquet ee (B frame, xyzw) into the eval sim WORLD frame
    (xyzw) via the fixed M. For the GIF / world-frame plots."""
    pB = np.asarray(ee_B_xyzw[:3], dtype=np.float64)
    qB = xyzw_to_wxyz(ee_B_xyzw[3:7])
    pW = M_R @ pB + M_t
    return np.concatenate([pW, wxyz_to_xyzw(qB)])
