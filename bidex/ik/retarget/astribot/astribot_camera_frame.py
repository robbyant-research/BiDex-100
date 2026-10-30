"""Astribot S1 head-camera frame + the deterministic chassis-local (L) <cam/world>
transforms, numpy-only (no mujoco), shared by the ee-replay IK (which needs
mujoco) and the matplotlib GIF (which does not).

Frames
------
* W : eval sim world (where the env renders; env `data.xpos`/`xquat`).
* L : chassis_base LOCAL frame -- the robot base frame. The parquet
      `abs_end_effector` (robot_ee) is stored in L (recomputed from arm joints via
      the assembly FK with torso=0/head_joint_2=1.0, the collection's fixed
      springref config). L is placement-independent; the only thing that depends
      on the world placement is the L -> W lift (the chassis_base world pose).
* Cam (optical) : head camera, OpenCV optical convention (+Z toward scene).
      The head cam is mounted behind the actuated torso+head joints; since the
      collection holds torso=~0 / head_joint_2=1.0 (springref) CONSTANT across
      every episode/task (verified), the head-cam-in-L is a single deterministic
      kinematic constant (no per-episode / no fit).

Convention (post rebuild):
    ee_W        = ASTRIBOT_BASE_WORLD @ ee_L                      (deterministic;
                   chassis_base world pose = (0.8,0,0) identity, constant for all
                   kitchen astribot tasks)
    head-cam-in-W (optical) = ASTRIBOT_BASE_WORLD @ ASTRIBOT_HEAD_OPTICAL_IN_L
    camera_ee   = inv(ASTRIBOT_HEAD_OPTICAL_IN_L) @ ee_L           (= inv(head_opt_w) @ ee_W)

No data fitting: ASTRIBOT_BASE_WORLD is the arena `robot_pose`; ASTRIBOT_HEAD_OPTICAL_IN_L
is the head-link-2->chassis_base relative transform (assembly MJCF) composed with the
head-cam mount offset (camera_profiles/astribot_s1_fixed_sharpa.yaml) and the MuJoCo->OpenCV
diag(1,-1,-1), all evaluated at the fixed nominal torso/head config.
"""
import numpy as np

# Eval arena base pose (chassis_base world) for kitchen astribot tasks: task yaml
# `robot_pose` pos=(0.8,0,0) quat=(0,0,0,1) wxyz = Rz(+180deg); BIDEX_COMPOSE_BASE_QUAT
# unset so the env writes this quat onto chassis_base (overwriting the MJCF euler
# "0 0 -1.57"). Constant across all 50 kitchen tasks. (NOTE: quat [0,0,0,1] in wxyz is
# Rz180, NOT identity -- identity is [1,0,0,0].)
ASTRIBOT_BASE_WORLD = np.array([
    [-1.0, 0.0, 0.0, 0.8],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
M_WORLD_FROM_L = ASTRIBOT_BASE_WORLD
M_L_FROM_WORLD = np.linalg.inv(M_WORLD_FROM_L)

# Nominal torso/head joint config the collection holds (springref; torso springs to 0,
# head_joint_2 springs to 1.0). Verified constant across episodes/tasks. Used by the FK
# build so the recomputed tool matches the sim truth to ~0.2 mm.
NOMINAL_TORSO_HEAD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # [torso1..4, head1, head2]

# Head camera OPTICAL pose in the chassis_base LOCAL frame L, at the nominal torso/head
# config (computed from assembly `astribot_s1_fixed_sharpa.xml` + head-cam profile
# pos=[0.0603,-0.2139,-0.0135] quat(wxyz)=[0.0135,-0.7105,0.0295,0.7030] @ head_link_2,
# then MuJoCo->OpenCV via R@diag(1,-1,-1)). This is the camera-ee <-> robot_ee rigid
# constant (base-frame invariant).
ASTRIBOT_HEAD_OPTICAL_IN_L = np.array([
    [ 0.02462347, -0.80691567,  0.59035567,  0.21257401],
    [-0.99975940, -0.02229687,  0.01136317, -0.01350100],
    [ 0.00411263, -0.59048850, -0.80715314,  1.42247997],
    [ 0.0,         0.0,         0.0,         1.0       ],
])
T_CAM_L_OPTICAL = ASTRIBOT_HEAD_OPTICAL_IN_L
CAM_R = T_CAM_L_OPTICAL[:3, :3].copy()
CAM_t = T_CAM_L_OPTICAL[:3, 3].copy()

ASTRIBOT_HEAD_OPTICAL_WORLD = M_WORLD_FROM_L @ ASTRIBOT_HEAD_OPTICAL_IN_L

# --------------------------------------------------------------------------- #
# quaternion helpers (wxyz internally; parquet uses xyzw)                      #
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
    """Rotation matrix -> quaternion (w, x, y, z), Shepperd robust method."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0; qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S; qy = (R[0, 2] - R[2, 0]) / S; qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0; qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S; qy = (R[0, 1] + R[1, 0]) / S; qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0; qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S; qy = 0.25 * S; qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0; qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S; qy = (R[1, 2] + R[2, 1]) / S; qz = 0.25 * S
    q = np.array([qw, qx, qy, qz]); return q / np.linalg.norm(q)


def xyzw_to_wxyz(q): return np.array([q[3], q[0], q[1], q[2]])
def wxyz_to_xyzw(q): return np.array([q[1], q[2], q[3], q[0]])

_CAM_qR = None


def _cam_qR():
    global _CAM_qR
    if _CAM_qR is None:
        _CAM_qR = rotmat_to_quat_wxyz(CAM_R)
    return _CAM_qR


# --------------------------------------------------------------------------- #
# ee transforms (per-arm 7-D xyzw, right-first 14-D rows)                     #
# --------------------------------------------------------------------------- #
def ee_L_to_camera(ee_L_xyzw):
    """Rebase one arm's ee from chassis-local L (xyzw, w/ tool flange) into the
    head-cam OPTICAL frame (xyzw). p_cam = R^T (p_L - t); q_cam = qR^{-1} (x) q_L_wxyz."""
    qR = _cam_qR(); qR_inv = qinv(qR)
    pL = np.asarray(ee_L_xyzw[:3], dtype=np.float64)
    qL = xyzw_to_wxyz(ee_L_xyzw[3:7])
    p_cam = CAM_R.T @ (pL - CAM_t)
    q_cam = qmul(qR_inv, qL)
    return np.concatenate([p_cam, wxyz_to_xyzw(q_cam)])


def ee_camera_to_L(ee_cam_xyzw):
    """Inverse: optical-camera-frame ee (xyzw) -> chassis-local L (xyzw)."""
    qR = _cam_qR()
    pc = np.asarray(ee_cam_xyzw[:3], dtype=np.float64)
    qc = xyzw_to_wxyz(ee_cam_xyzw[3:7])
    pL = CAM_R @ pc + CAM_t
    qL = qmul(qR, qc)
    return np.concatenate([pL, wxyz_to_xyzw(qL)])


def ee_row_L_to_camera_row(ee_L_row14):
    out = np.empty(14, dtype=np.float64)
    out[0:7] = ee_L_to_camera(np.asarray(ee_L_row14[0:7], dtype=np.float64))
    out[7:14] = ee_L_to_camera(np.asarray(ee_L_row14[7:14], dtype=np.float64))
    return out


def ee_L_to_world(ee_L_xyzw):
    """Lift chassis-local ee (xyzw) into the eval sim WORLD frame (xyzw) via ASTRIBOT_BASE_WORLD."""
    pL = np.asarray(ee_L_xyzw[:3], dtype=np.float64)
    qL = xyzw_to_wxyz(ee_L_xyzw[3:7])
    pW = M_WORLD_FROM_L[:3, :3] @ pL + M_WORLD_FROM_L[:3, 3]
    qWR = rotmat_to_quat_wxyz(M_WORLD_FROM_L[:3, :3])
    qW = qmul(qWR, qL)
    return np.concatenate([pW, wxyz_to_xyzw(qW)])