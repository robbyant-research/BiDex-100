#!/usr/bin/env python3
"""Shared helpers for Franka (dual_fr3v2_sharpa) ee replay / action_ee fix.

Coordinate convention (see examples/FRANKA_COORDINATE_TRANSFORM.md):
  * parquet abs_end_effector is in the COLLECTION dual-arm-base frame B
    (xyzw, right-first 14-D, includes the link8 +45deg-Z flange).
  * state_ee is FK-consistent with state_arm:  state_ee = T_coll * FK_standalone(state_arm)
    where FK_standalone is the standalone fr3v2.xml `end`-site pose in the
    per-arm link0 frame (NO flange). T_coll = (R, t) is the per-arm rigid
    transform from link0-frame to collection-B-frame, empirically recovered
    per episode via Kabsch on positions (residual ~0.3 mm).
  * action_ee in the released dataset comes from a DIFFERENT (per-episode-
    drifting) FK/base path, so it is NOT FK-consistent with action_arm and
    sits in a frame that does not match B (root cause of the large
    action_ee-vs-state_ee error). The fix rewrites action_ee in the SAME
    frame/state_ee convention: action_ee_fixed = T_coll * FK_standalone(action_arm)
    (with the link8 flange re-added).

ee -> joint IK pipeline (for ee replay): convert a parquet ee (B, xyzw, w/ flange)
back to standalone-link8 in link0 frame, then run the vendored mink IK
(FrankaKinematicsSolver) with chained warm-start:
    p_link0   = R^T (p_B - t)
    q_link0   = qR^{-1} * q_B_wxyz                 # undo frame rotation
    q_std     = q_link0 * K^{-1}                   # peel flange -> standalone link8
    q7        = solver.inverse_kinematics(p_link0, q_std, q_init=q_prev)
The env is driven in JOINT mode (not env ee-mode) to avoid the sagged
warm-start drift documented for astribot.
"""
import numpy as np
import os

# Head-camera frame + the deterministic B<->world transforms live in a
# numpy-only module shared with the matplotlib GIF (which cannot import mujoco).
from .franka_camera_frame import (
    ee_B_to_camera, ee_camera_to_B, T_CAM_B_OPTICAL, CAM_t, M_R, M_t,
)

# fr3v2 standalone-arm MJCF for the vendored mink IK (FrankaKinematicsSolver).
# This module lives at <repo>/bidex/ik/retarget/franka/franka_ee_replay_lib.py,
# so 4 dirnames up is <repo>/bidex (the package root holding robots/...).
# Overridable via $BIDEX_FR3V2_XML.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
XML_PATH = os.environ.get("BIDEX_FR3V2_XML") or os.path.join(
    _PKG_ROOT, "robots", "arms", "franka_fr3_v2", "fr3v2.xml")

# link8 flange baked into the assembly dual_fr3v2_sharpa.xml (+45deg Z, wxyz).
# standalone fr3v2.xml's link8 has no such quat, so peel K before standalone IK
# and re-add K when emitting assembly-link8 poses into parquet.
K_FLANGE    = np.array([0.9238795325112867, 0.0, 0.0,  0.3826834323650898])
K_FLANGE_INV= np.array([0.9238795325112867, 0.0, 0.0, -0.3826834323650898])


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
    Shepperd method. NOTE: the previous trace-only formula was buggy
    (qw = sqrt((1+tr)/2) instead of 0.5*sqrt(1+tr) -> non-unit -> renormalized
    to a wrong rotation), which silently corrupted every T_coll orientation
    and the camera-column quaternions. This matches franka_camera_frame's copy."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
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


def fit_T_coll(fk_pos, ee_pos):
    """Kabsch: R, t s.t. ee_pos ~= R @ fk_pos + t. Returns (R, t, resid_per_row)."""
    P = np.asarray(fk_pos, dtype=np.float64)
    Q = np.asarray(ee_pos, dtype=np.float64)
    cp = P.mean(0); cq = Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    resid = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    return R, t, resid


def ee_B_to_standalone_link0(ee_xyzw, R, t, qR):
    """parquet ee (B frame, xyzw, w/ flange) -> (pos, quat_wxyz) of standalone
    link8 (`end` site, no flange) in the per-arm link0 frame, ready for the
    standalone IK solver."""
    pB = ee_xyzw[:3]
    qB = xyzw_to_wxyz(ee_xyzw[3:7])
    p_lk = R.T @ (pB - t)
    q_lk = qmul(qinv(qR), qB)            # undo frame rotation
    q_std = qmul(q_lk, K_FLANGE_INV)     # peel link8 flange
    return p_lk, q_std


def fk_to_ee_B(fk_pos, fk_quat_wxyz, R, t, qR):
    """Inverse of ee_B_to_standalone_link0: standalone link8 (link0 frame, no
    flange) FK pose -> parquet ee in collection-B frame (xyzw, w/ flange).
    Used to WRITE the fixed action_ee."""
    q_asm = qmul(fk_quat_wxyz, K_FLANGE)        # add flange -> assembly link8
    q_B = qmul(qR, q_asm)                         # rotate into B frame
    p_B = R @ fk_pos + t
    return np.concatenate([p_B, wxyz_to_xyzw(q_B)])


class FrankaEEPipeline:
    """Per-arm T_coll fit + ee->joint IK over an episode, shared by fix & replay."""

    def __init__(self):
        from bidex.ik.franka_kinematics_solver import FrankaKinematicsSolver
        self.solver = FrankaKinematicsSolver(XML_PATH)

    def fk(self, q7):
        return self.solver.forward_kinematics(q7)   # (xyz, quat_wxyz) link8 std, link0 frame

    def fit(self, state_arm_rows, state_ee_rows, arm_slice):
        """Fit per-arm T_coll from state_arm/state_ee. Returns dict(R,t,qR,resid).

        R,t : from Kabsch on POSITIONS (optimal translation + rotation when the
              arm moves; under-determined rotation when the arm is stationary).
        qR  : from the ORIENTATION relationship q_ee = qR ⊗ q_fk ⊗ K_flange, so
              qR = q_ee[0] ⊗ Kinv ⊗ qinv(q_fk[0]). This is well-defined from a
              SINGLE frame -- unlike Kabsch R, which is arbitrary for a stationary
              (degenerate-point-cloud) arm and would give a WRONG ee->joint IK
              orientation target (=> the stationary arm's wrist flips in replay)."""
        sl = arm_slice
        fk = [self.fk(state_arm_rows[i, sl[0]:sl[1]]) for i in range(len(state_arm_rows))]
        fk_pos = np.array([f[0] for f in fk])
        fk_quat = np.array([f[1] for f in fk])            # wxyz, standalone link8, link0 frame
        s0 = 0 if sl[0] == 0 else 7
        R, t, resid = fit_T_coll(fk_pos, state_ee_rows[:, s0:s0 + 3])
        q_ee0 = xyzw_to_wxyz(np.asarray(state_ee_rows[0, s0 + 3:s0 + 7]))  # B frame, WITH flange
        qR = qmul(qmul(q_ee0, K_FLANGE_INV), qinv(fk_quat[0]))
        qR = qR / np.linalg.norm(qR)
        return {"R": R, "t": t, "qR": qR, "resid": resid, "s0": s0, "sl": sl}

    def ee_to_joints(self, ee_xyzw, T, q_init):
        """ee (B, xyzw) -> q7 via standalone mink IK (chained warm-start)."""
        p_lk, q_std = ee_B_to_standalone_link0(ee_xyzw, T["R"], T["t"], T["qR"])
        q7, ok = self.solver.inverse_kinematics(p_lk, q_std, q_init=q_init,
                                                max_iterations=60)
        return q7, ok


if __name__ == "__main__":
    # Smoke test: FK a random arm, round-trip ee_B<->camera, confirm the
    # head-cam frame module is wired in.
    np.set_printoptions(precision=6, suppress=True)
    print("M (B->world) R det=", np.linalg.det(M_R), " t=", M_t)
    print("T_CAM_B_OPTICAL (head cam in B, OpenCV optical)=\n", T_CAM_B_OPTICAL)
    pipe = FrankaEEPipeline()
    q7 = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.2, 0.0])
    p, qw = pipe.fk(q7)
    eeB = fk_to_ee_B(p, qw, np.eye(3), np.zeros(3), np.array([1, 0, 0, 0]))
    rt = np.linalg.norm(ee_camera_to_B(ee_B_to_camera(eeB)) - eeB)
    cam_origin_B = np.concatenate([CAM_t, [1, 0, 0, 0]])
    print(f"ee_B->camera->B round-trip err {rt*1e6:.2f} um; "
          f"cam origin in B {CAM_t} -> in camera {ee_B_to_camera(cam_origin_B)[:3]}")