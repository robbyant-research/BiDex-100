"""Full 7-DoF end-effector IK for the astribot S1 dual arm, solved against the
SIM model's own FK (``data.xpos``/``xquat`` of the tool_link bodies), in the
sim WORLD frame.

This is the frame-exact inverse of the full-7-DoF FK
(`bidex/ik/retarget/astribot/fk_functions.py:right_fk_full/left_fk_full`)
used to (re)compute the parquet ``abs_end_effector`` in `data/`. It is the IK to
use for ee replay / ee policy eval that must round-trip the recorded full-FK ee.

Why NOT the env ee-mode IK (`BiDexEnv._solve_eef_ik_astribot`) or a
current-qpos-seeded solve: with the weak astribot shoulder actuators
(kp=100, ~10 cm gravity sag), the *achieved* qpos is ~1 rad below the commanded
pose. Warm-starting SLSQP from that sagged qpos makes it fail to climb back and
return ~the seed, so the arm never tracks (runaway drift, ~30 mm). The fix used
here is a **chained warm-start**: seed frame 0 from the restored init arm q,
then carry each solved q14 forward as the next seed. That keeps the seed on the
*commanded* manifold (within one step of the target ee), so SLSQP converges
(verified ~1.4 mm right / ~3.6 mm left vs the action-joint replay).

Callers own the env + its lock; this function only reads ``env.model`` /
``env.data`` and restores qpos after probing (it never steps dynamics).
"""
from __future__ import annotations

import numpy as np
import mujoco
import scipy.optimize as _opt


def _qang(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * np.arccos(min(1.0, max(-1.0, abs(float(np.dot(a, b))))))


def make_full_fk_ik(env):
    """Build a solve(ee_world, q_init_lr) -> q14[L7,R7] against the env's sim FK.

    Args:
        env: a BiDexEnv (any control_mode; only model/data/joint/body ids are
            read). Caller must hold ``env.lock`` if it also steps the env.
        ee_world: right-first 14-vector
            [r_xyz(3), r_quat_wxyz(4), l_xyz(3), l_quat_wxyz(4)] in the sim
            WORLD frame (i.e. already passed through ``policy_ee_to_our``).
        q_init_lr: [L7, R7] warm-start seed. Pass the *previous solved* q14
            (chained), NOT the sagged current qpos.

    Returns:
        q14 = [left(7), right(7)] joint angles.
    """
    m, d = env.model, env.data
    lqa = [m.jnt_qposadr[j] for j in env.left_franka_jnt_ids if j >= 0]
    rqa = [m.jnt_qposadr[j] for j in env.right_franka_jnt_ids if j >= 0]
    lb = env.left_eef_body_id
    rb = env.right_eef_body_id
    lo = ([m.jnt_range[j][0] for j in env.left_franka_jnt_ids if j >= 0]
          + [m.jnt_range[j][0] for j in env.right_franka_jnt_ids if j >= 0])
    hi = ([m.jnt_range[j][1] for j in env.left_franka_jnt_ids if j >= 0]
          + [m.jnt_range[j][1] for j in env.right_franka_jnt_ids if j >= 0])
    bounds = list(zip(lo, hi))

    def solve(ee_world, q_init_lr):
        ee_world = np.asarray(ee_world, dtype=np.float64).reshape(-1)
        tgt_l = (ee_world[7:10].copy(), ee_world[10:14].copy())
        tgt_r = (ee_world[0:3].copy(), ee_world[3:7].copy())
        q0 = np.asarray(q_init_lr, dtype=np.float64).copy()
        qpos_save = d.qpos.copy()

        def _fk(q):
            for a, v in zip(lqa, q[0:7]):
                d.qpos[a] = v
            for a, v in zip(rqa, q[7:14]):
                d.qpos[a] = v
            mujoco.mj_kinematics(m, d)  # xpos/xquat only -- cheap, no dynamics
            return (d.xpos[lb].copy(), d.xquat[lb].copy(),
                    d.xpos[rb].copy(), d.xquat[rb].copy())

        def _cost(q):
            lp, lq, rp, rq = _fk(q)
            pc = float(np.sum((lp - tgt_l[0]) ** 2) + np.sum((rp - tgt_r[0]) ** 2))
            oc = _qang(lq, tgt_l[1]) ** 2 + _qang(rq, tgt_r[1]) ** 2
            vc = float(np.sum((q - q0) ** 2))
            return 5000.0 * pc + 200.0 * oc + 0.5 * vc

        try:
            res = _opt.minimize(_cost, q0, method="SLSQP", bounds=bounds,
                                tol=1e-6, options={"maxiter": 300, "ftol": 1e-6})
            q = np.asarray(res.x, dtype=np.float64)
        except Exception:
            q = q0
        finally:
            d.qpos[:] = qpos_save
            mujoco.mj_kinematics(m, d)
        return q  # [L7, R7]

    return solve