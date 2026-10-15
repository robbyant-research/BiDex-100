"""
FK/IK solver with pluggable IK backends.

Three backends share one ``(xyz, quat[w,x,y,z]) -> q[N]`` IK interface and
one ``q[N] -> (xyz, quat[w,x,y,z])`` FK interface:

* ``mink``    — mink's QP-based IK with a FrameTask + PostureTask.
* ``dls``     — damped least-squares (Levenberg-Marquardt), ported from
                ``bidex.bidex.env.SimEnv._solve_eef_ik``.
* ``opspace`` — wraps ``dexjoco.sim.controllers.opspace.opspace`` by
                running its torque output into a scratch ``MjData`` via
                ``qfrc_applied`` + ``mj_step`` until pose error settles.

The solver was originally Franka-only (qpos[:7] hardcoded). It now accepts
an optional ``RobotProfile`` that names the arm joints by string, so a
generalised robot whose arm lives deep inside a chassis/torso chain
(e.g. astribot_s1_fixed_sharpa: chassis → 4 torso joints → 7 arm joints)
can use the same IK plumbing. Old call sites that pass only ``model_path``
fall back to the legacy 7-DoF arange(7) behaviour byte-identical to the
pre-profile version.

Backend imports are lazy: only the chosen backend's third-party module
needs to be installed. ``mujoco`` is always required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def _xmat_to_quat_wxyz(xmat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a mujoco-style [w, x, y, z] quat."""
    quat_xyzw = R.from_matrix(xmat).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return R.from_quat(quat_xyzw).as_matrix()


@dataclass
class RobotProfile:
    """Per-robot kinematics descriptor consumed by ``FrankaKinematicsSolver``.

    The legacy code assumed a standalone Franka MJCF where the first seven
    hinge joints ARE the arm — so ``qpos[:7]`` indexed the arm subspace
    directly. For a robot whose arm joints live inside a larger chain
    (chassis / torso / head also use qpos slots) this is wrong; we instead
    resolve qpos / qvel addresses from named joints.

    Attributes
    ----------
    xml_path
        MJCF to load. May be a standalone arm (legacy) or a full assembly.
    end_site_name
        Site whose pose the IK targets. For Franka the standalone XML names
        it ``end``; an assembly might use ``arm_left_tool``.
    arm_joint_names
        Ordered list of N joint names the IK actuates. When None, falls back
        to ``arange(7)`` (legacy Franka behaviour).
    posture_freeze_joints
        Mink-only: joint names whose PostureTask cost is boosted high so
        mink doesn't accidentally drive them. Used for non-arm DoFs in an
        assembly XML (torso, head, chassis) that you want to pin to their
        initial pose during arm IK. Other backends ignore this.
    base_body_name
        ``humanoid_arm`` solver only. Name of the body whose LOCAL frame
        the IK target lives in (and which FK rebases against). For
        standalone single-arm MJCFs this is typically ``"base"`` /
        ``"link0"``; for a full-assembly XML this is the chassis the arms
        branch off of (``"chassis_base"`` for astribot). Other backends
        ignore this — they consume targets in the IK model's WORLD frame.
    """

    xml_path: str
    end_site_name: str = "end"
    arm_joint_names: Optional[List[str]] = None
    posture_freeze_joints: Optional[List[str]] = field(default_factory=list)
    base_body_name: Optional[str] = None


class FrankaKinematicsSolver:
    """FK/IK wrapper around an MJCF, with multi-backend IK.

    The solver owns one ``MjModel`` / ``MjData`` pair. FK and IK both
    operate on that scratch data, so consecutive IK calls warm-start
    against the previous solution unless ``q_init`` is passed explicitly.

    Accepts either the legacy ``(model_path, end_site_name="end")`` constructor
    args (which synthesise a 7-DoF arange profile — byte-identical to the
    pre-profile behaviour) OR a keyword-only ``profile=RobotProfile(...)`` for
    robots whose arm lives inside a larger chain. Both forms coexist so the
    existing FR3v2 call sites in BiDex-100 need no edits.
    """

    SUPPORTED = ("mink", "dls", "opspace", "humanoid_arm")

    def __init__(
        self,
        model_path: Optional[str] = None,
        end_site_name: str = "end",
        solver: str = "mink",
        *,
        profile: Optional[RobotProfile] = None,
        **solver_kwargs,
    ):
        if solver not in self.SUPPORTED:
            raise ValueError(
                f"Unknown solver '{solver}'. Supported: {self.SUPPORTED}"
            )

        # Synthesise a legacy 7-DoF profile from positional args when no
        # profile was supplied. This is the back-compat shim — all three
        # historical BiDex-100 call sites (env.py:566, data_collect_sim.py:131,
        # replay_episode.py:284) take this branch.
        if profile is None:
            if model_path is None:
                raise ValueError(
                    "Either model_path (legacy) or profile=RobotProfile(...) "
                    "must be supplied"
                )
            profile = RobotProfile(xml_path=model_path,
                                   end_site_name=end_site_name)
        self.profile = profile
        self.model_path = profile.xml_path
        self.end_site_name = profile.end_site_name
        self.solver = solver

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.end_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.end_site_name
        )
        if self.end_site_id < 0:
            raise ValueError(
                f"Site '{self.end_site_name}' not found in {self.model_path}"
            )
        self.end_body_id = int(self.model.site_bodyid[self.end_site_id])

        # Arm qpos / qvel slots. Legacy: first 7 hinges. Profile: derived
        # from joint names so an arm buried under torso/chassis joints
        # still indexes correctly.
        if profile.arm_joint_names:
            jids = []
            for n in profile.arm_joint_names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if jid < 0:
                    raise ValueError(
                        f"Joint '{n}' not found in {self.model_path}"
                    )
                jids.append(jid)
            self._qpos_adr = np.array(
                [int(self.model.jnt_qposadr[j]) for j in jids], dtype=int)
            self._qvel_adr = np.array(
                [int(self.model.jnt_dofadr[j]) for j in jids], dtype=int)
            self._joint_ids = np.asarray(jids, dtype=int)
        else:
            self._qpos_adr = np.arange(7, dtype=int)
            self._qvel_adr = np.arange(7, dtype=int)
            self._joint_ids = np.arange(7, dtype=int)
        self._n_arm = int(self._qpos_adr.size)

        # Posture-freeze joints (mink only). Pre-resolve ids so the mink
        # init can bias the posture task accordingly.
        self._freeze_joint_ids: List[int] = []
        for n in (profile.posture_freeze_joints or []):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                self._freeze_joint_ids.append(jid)

        # Backend-specific setup
        if solver == "mink":
            self._init_mink(**solver_kwargs)
        elif solver == "dls":
            self._init_dls(**solver_kwargs)
        elif solver == "opspace":
            self._init_opspace(**solver_kwargs)
        elif solver == "humanoid_arm":
            self._init_humanoid_arm(**solver_kwargs)

    # ------------------------------------------------------------------
    # Backend init
    # ------------------------------------------------------------------

    def _init_mink(self, position_cost: float = 1.0,
                   orientation_cost: float = 0.01,
                   posture_cost: float = 1e-8,
                   lm_damping: float = 0.5,
                   ik_solver: str = "daqp",
                   damping: float = 1e-12,
                   **_unused):
        import mink  # lazy
        self._mink = mink
        self.configuration = mink.Configuration(self.model)
        self.end_task = mink.FrameTask(
            frame_name=self.end_site_name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=posture_cost)
        mujoco.mj_resetData(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self._mink_ik_solver = ik_solver
        self._mink_damping = damping

    def _init_dls(self, damping: float = 0.05,
                  pos_step_max: float = 0.05,
                  rot_step_max: float = 0.5,
                  **_unused):
        # Ported defaults from env.SimEnv._solve_eef_ik. These constants
        # keep the LM linearization valid even when the target is far
        # from the current pose.
        self._dls_damping = damping
        self._dls_pos_step_max = pos_step_max
        self._dls_rot_step_max = rot_step_max

    def _init_opspace(self,
                      pos_gains=(200.0, 200.0, 200.0),
                      ori_gains=(200.0, 200.0, 200.0),
                      damping_ratio: float = 1.0,
                      nullspace_stiffness: float = 0.5,
                      **_unused):
        # No external opspace import — the controller logic is inlined in
        # `_opspace_torque` below using mujoco's built-in quat ops, so the
        # solver has no third-party dep beyond mujoco/numpy/scipy.
        self._ops_pos_gains = np.asarray(pos_gains, dtype=np.float64)
        self._ops_ori_gains = np.asarray(ori_gains, dtype=np.float64)
        self._ops_damping_ratio = damping_ratio
        self._ops_nullspace_stiffness = nullspace_stiffness

    # ------------------------------------------------------------------
    # Forward kinematics (shared)
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """End-site pose as ``(xyz, quat=[w, x, y, z])`` for arm joints ``q``."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self.data.qpos[self._qpos_adr] = q[:self._n_arm]
        mujoco.mj_forward(self.model, self.data)
        xyz = self.data.site_xpos[self.end_site_id].copy()
        xmat = self.data.site_xmat[self.end_site_id].reshape(3, 3).copy()
        quat = _xmat_to_quat_wxyz(xmat)
        return xyz, quat

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        # Resolve range from the actual arm joints (not jnt_range[:7]) so
        # an assembly XML where the arm starts at jnt_index>0 returns the
        # right limits.
        lo = self.model.jnt_range[self._joint_ids, 0].copy()
        hi = self.model.jnt_range[self._joint_ids, 1].copy()
        return lo, hi

    # ------------------------------------------------------------------
    # Inverse kinematics (dispatch)
    # ------------------------------------------------------------------

    def inverse_kinematics(
        self,
        xyz: np.ndarray,
        quat: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        dt: float = 0.01,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-2,
    ) -> Tuple[np.ndarray, bool]:
        """Solve for joint angles reaching pose ``(xyz, quat)``.

        ``quat`` is mujoco-style ``[w, x, y, z]``. ``q_init`` (optional)
        seeds the iterative solver; if None, the solver continues from
        its scratch state (warm start).

        Returns ``(q[7], success)``. ``success`` reflects convergence
        within the position and orientation thresholds.
        """
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if np.linalg.norm(quat) < 1e-9:
            return self.data.qpos[self._qpos_adr].copy(), False

        if self.solver == "mink":
            return self._ik_mink(xyz, quat, q_init, max_iterations, dt,
                                 pos_threshold, ori_threshold)
        if self.solver == "dls":
            return self._ik_dls(xyz, quat, q_init, max_iterations,
                                pos_threshold, ori_threshold)
        if self.solver == "opspace":
            return self._ik_opspace(xyz, quat, q_init, max_iterations, dt,
                                    pos_threshold, ori_threshold)
        if self.solver == "humanoid_arm":
            return self._ik_humanoid_arm(xyz, quat, q_init, max_iterations,
                                         pos_threshold, ori_threshold)
        raise AssertionError("unreachable")  # pragma: no cover

    # --- mink ---------------------------------------------------------

    def _ik_mink(self, xyz, quat, q_init, max_iters, dt,
                 pos_tol, ori_tol):
        if q_init is not None:
            # mink's Configuration.update(q) only syncs its internal MjData
            # when handed the FULL qpos vector — mutating configuration.q
            # in place leaves configuration.data at the prior pose, so
            # solve_ik would compute the Jacobian/error against a stale
            # state and a no-op target (FK of the seed) diverges. Build the
            # full qpos, set the arm slots, and update with it.
            #
            # Non-arm joints (torso/head on a humanoid) must be re-pinned to
            # self.data's values here too: mink's integrate_inplace drifts
            # them every call (the PostureTask doesn't hard-freeze them), so
            # a stale non-zero torso from a prior IK call would desync the
            # solver's FK from the sim and bias the arm solution by ~0.1 m.
            full_q = self.data.qpos.copy()
            full_q[self._qpos_adr] = np.asarray(q_init, dtype=np.float64)[:self._n_arm]
            self.configuration.update(full_q)

        rot_mat = _quat_wxyz_to_matrix(quat)
        target = self._mink.SE3.from_rotation_and_translation(
            self._mink.SO3.from_matrix(rot_mat), xyz)
        self.end_task.set_target(target)

        tasks = (self.end_task, self.posture_task)
        for _ in range(max_iters):
            vel = self._mink.solve_ik(
                self.configuration, tasks, dt,
                self._mink_ik_solver, damping=self._mink_damping)
            self.configuration.integrate_inplace(vel, dt)
            err = self.end_task.compute_error(self.configuration)
            if (np.linalg.norm(err[:3]) <= pos_tol
                    and np.linalg.norm(err[3:]) <= ori_tol):
                return self.configuration.q[self._qpos_adr].copy(), True
        return self.configuration.q[self._qpos_adr].copy(), False

    # --- damped least-squares (env.py port) --------------------------

    def _ik_dls(self, xyz, quat, q_init, max_iters, pos_tol, ori_tol):
        # Levenberg-Marquardt with a damped pseudo-inverse:
        #   dq = J^T (J J^T + λ²I)^-1 e
        # Per-iter step clamps keep the linearization valid even when the
        # target is far from the current pose.
        if q_init is not None:
            q = np.asarray(q_init, dtype=np.float64)[:self._n_arm].copy()
        else:
            q = self.data.qpos[self._qpos_adr].copy()

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        err = np.zeros(6)

        for _ in range(max_iters):
            self.data.qpos[self._qpos_adr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            cur_xyz = self.data.xpos[self.end_body_id].copy()
            cur_quat = self.data.xquat[self.end_body_id].copy()
            err[:3] = xyz - cur_xyz
            inv_cur = np.zeros(4)
            mujoco.mju_negQuat(inv_cur, cur_quat)
            err_q = np.zeros(4)
            mujoco.mju_mulQuat(err_q, quat, inv_cur)
            mujoco.mju_quat2Vel(err[3:], err_q, 1.0)

            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < pos_tol and rot_err < ori_tol:
                return q, True

            if pos_err > self._dls_pos_step_max:
                err[:3] *= self._dls_pos_step_max / pos_err
            if rot_err > self._dls_rot_step_max:
                err[3:] *= self._dls_rot_step_max / rot_err

            mujoco.mj_jacBody(self.model, self.data, jacp, jacr,
                              self.end_body_id)
            J = np.vstack([jacp[:, self._qvel_adr],
                           jacr[:, self._qvel_adr]])
            JJt = J @ J.T + (self._dls_damping ** 2) * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(JJt, err)
            except np.linalg.LinAlgError:
                break
            q = q + dq

        # Final FK to settle data.qpos
        self.data.qpos[self._qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return q, False

    # --- opspace wrapper ---------------------------------------------

    def _ik_opspace(self, xyz, quat, q_init, max_iters, dt,
                    pos_tol, ori_tol):
        # opspace returns torques. Bypass the model's actuators by
        # writing torque into qfrc_applied, then mj_step advances
        # dynamics. We track pose error per step and break on
        # convergence. Velocity is reset between calls so opspace
        # doesn't pick up leftover momentum from a previous IK.
        if q_init is not None:
            self.data.qpos[self._qpos_adr] = np.asarray(q_init,
                                                        dtype=np.float64)[:self._n_arm]
        self.data.qvel[self._qvel_adr] = 0.0
        # Also zero out any leftover applied force / actuator ctrl so
        # only our opspace torque drives the dynamics.
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0

        dof_ids = self._qvel_adr
        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)

            cur_xyz = self.data.site_xpos[self.end_site_id].copy()
            cur_xmat = self.data.site_xmat[self.end_site_id].reshape(3, 3)
            cur_quat = _xmat_to_quat_wxyz(cur_xmat)

            pos_err = np.linalg.norm(xyz - cur_xyz)
            # Orientation error via mujoco quat math (matches DLS path).
            err_q = np.zeros(4)
            inv_cur = np.zeros(4)
            mujoco.mju_negQuat(inv_cur, cur_quat)
            mujoco.mju_mulQuat(err_q, quat, inv_cur)
            err_rot = np.zeros(3)
            mujoco.mju_quat2Vel(err_rot, err_q, 1.0)
            rot_err = np.linalg.norm(err_rot)
            if pos_err < pos_tol and rot_err < ori_tol:
                return self.data.qpos[self._qpos_adr].copy(), True

            tau = self._opspace_torque(
                self.end_site_id, dof_ids,
                pos_des=xyz, quat_des=quat,
                pos_gains=self._ops_pos_gains,
                ori_gains=self._ops_ori_gains,
                damping_ratio=self._ops_damping_ratio,
                nullspace_stiffness=self._ops_nullspace_stiffness,
                gravity_comp=True,
            )
            self.data.qfrc_applied[dof_ids] = tau
            mujoco.mj_step(self.model, self.data)

        return self.data.qpos[self._qpos_adr].copy(), False

    # ------------------------------------------------------------------
    # Inlined opspace torque controller (port of dexjoco/sim/controllers/
    # opspace.opspace, with dm_robotics.transformations swapped out for
    # mujoco's built-in quat ops). Returns generalized force τ for the
    # arm DOFs that drives the site pose toward (pos_des, quat_des) via
    # task-space PD with a nullspace posture term.
    # ------------------------------------------------------------------

    def _opspace_torque(
        self,
        site_id: int,
        dof_ids: np.ndarray,
        pos_des: np.ndarray,
        quat_des: np.ndarray,
        joint_des: Optional[np.ndarray] = None,
        pos_gains=(200.0, 200.0, 200.0),
        ori_gains=(200.0, 200.0, 200.0),
        damping_ratio: float = 1.0,
        nullspace_stiffness: float = 0.5,
        max_pos_acceleration: Optional[float] = None,
        max_ori_acceleration: Optional[float] = None,
        gravity_comp: bool = True,
    ) -> np.ndarray:
        model, data = self.model, self.data

        x_des = np.asarray(pos_des, dtype=np.float64).reshape(3)
        q_des_quat = np.asarray(quat_des, dtype=np.float64).reshape(4)

        if joint_des is None:
            q_des = data.qpos[dof_ids].copy()
        else:
            q_des = np.asarray(joint_des, dtype=np.float64)

        kp_pos = np.asarray(pos_gains, dtype=np.float64)
        kd_pos = damping_ratio * 2.0 * np.sqrt(kp_pos)
        kp_ori = np.asarray(ori_gains, dtype=np.float64)
        kd_ori = damping_ratio * 2.0 * np.sqrt(kp_ori)
        kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
        kd_joint = damping_ratio * 2.0 * np.sqrt(kp_joint)

        ddx_max = (max_pos_acceleration
                   if max_pos_acceleration is not None else 0.0)
        dw_max = (max_ori_acceleration
                  if max_ori_acceleration is not None else 0.0)

        q = data.qpos[dof_ids]
        dq = data.qvel[dof_ids]

        # Jacobian of the eef site (translational + rotational), restricted
        # to the arm DOFs.
        J_v = np.zeros((3, model.nv), dtype=np.float64)
        J_w = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, J_v, J_w, site_id)
        J_v = J_v[:, dof_ids]
        J_w = J_w[:, dof_ids]
        J = np.concatenate([J_v, J_w], axis=0)

        # ---- position PD ------------------------------------------------
        x = data.site_xpos[site_id]
        dx = J_v @ dq
        x_err = x - x_des
        if ddx_max > 0.0:
            x_err_sq = np.sum(x_err ** 2)
            if x_err_sq > ddx_max ** 2:
                x_err *= ddx_max / np.sqrt(x_err_sq)
        ddx = -kp_pos * x_err - kd_pos * dx

        # ---- orientation PD --------------------------------------------
        # current quat from site xmat (mujoco scalar-first convention)
        xmat = data.site_xmat[site_id].reshape(3, 3)
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, xmat.reshape(9))
        # Double-cover handling: pick the shortest-path representative.
        if np.dot(cur_quat, q_des_quat) < 0.0:
            cur_quat = -cur_quat
        # quat_err = cur * inv(des)  (active difference: rotation from des to cur)
        inv_des = np.zeros(4)
        mujoco.mju_negQuat(inv_des, q_des_quat)
        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, cur_quat, inv_des)
        ori_err = np.zeros(3)
        mujoco.mju_quat2Vel(ori_err, quat_err, 1.0)  # axis-angle vec
        if dw_max > 0.0:
            ori_err_sq = np.sum(ori_err ** 2)
            if ori_err_sq > dw_max ** 2:
                ori_err *= dw_max / np.sqrt(ori_err_sq)
        w = J_w @ dq
        dw = -kp_ori * ori_err - kd_ori * w

        # ---- task-space inertia ----------------------------------------
        M = np.zeros((model.nv, model.nv), dtype=np.float64)
        mujoco.mj_fullM(model, M, data.qM)
        M = M[dof_ids, :][:, dof_ids]
        M_inv = np.linalg.inv(M)
        Mx_inv = J @ M_inv @ J.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        ddx_dw = np.concatenate([ddx, dw], axis=0)
        tau = J.T @ Mx @ ddx_dw

        # ---- nullspace joint task --------------------------------------
        ddq = -kp_joint * (q - q_des) - kd_joint * dq
        Jnull = M_inv @ J.T @ Mx
        tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq

        if gravity_comp:
            tau += data.qfrc_bias[dof_ids]
        return tau

    # ------------------------------------------------------------------
    # humanoid_arm — SLSQP port of humanoid-arm-retarget/arms_retarget.py
    # ------------------------------------------------------------------
    #
    # Differs from mink/dls/opspace in two important ways:
    #
    #   1. **Target frame is the BASE BODY LOCAL frame**, not the IK
    #      model's world. The reference repo (config_fftai_gr1.yaml) and
    #      its `_objective_function` both operate in per-arm-base
    #      coordinates: VR-world wrist is first transformed via
    #      `left_base @ wrist @ left_wrist` into the robot's left arm
    #      base frame, then handed to the optimizer; the FK function
    #      `left_fk(q)` also returns the EE in that same base frame, so
    #      the comparison is apples-to-apples.
    #
    #      To replicate without robot-specific analytic FK we let
    #      MuJoCo do the FK on the loaded MJCF and then rebase from IK
    #      world to the body named in `profile.base_body_name`. The
    #      body's pose is STATIC in the IK model (it has no joints
    #      between itself and worldbody), so we cache it once at init.
    #
    #   2. **Cost function is the reference's `_fi`-shaped scalar
    #      objective**: `60 * pos_cost + 6 * ori_cost + 2 * vel_cost`
    #      where `_fi(n=1, s=0, c=0.2, r=5)(x) = -exp(-x^2/0.08) +
    #      5 x^4`. The exp pulls hard toward x=0; the x^4 keeps SLSQP
    #      from running away into unreachable joint configurations.
    #      We collapse the reference's 5-up + 2-wrist split into a
    #      single 7-DoF SLSQP because (a) MuJoCo FK already handles
    #      the wrist axes naturally and (b) without a per-robot
    #      analytic forearm-vector function the analytic split would
    #      need an AVP-side hint we don't have on the IK channel.

    def _init_humanoid_arm(self,
                           pos_weight: float = 60.0,
                           ori_weight: float = 6.0,
                           vel_weight: float = 2.0,
                           fi_c: float = 0.2,
                           fi_r: float = 5.0,
                           shoulder_weight: float = 2.0,
                           tol: float = 1e-3,
                           **_unused):
        # Resolve base body whose LOCAL frame holds the IK target.
        # Fall back to the first non-world body (the assembly's root) so
        # standalone arm MJCFs without an explicit `base` body still work.
        bname = self.profile.base_body_name
        bid = -1
        if bname:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            # nbody=1 is worldbody itself; bid=1 is the first real body.
            if self.model.nbody >= 2:
                bid = 1
        if bid < 0:
            raise ValueError(
                f"humanoid_arm solver could not locate a base body "
                f"(profile.base_body_name={bname!r}) in {self.model_path}"
            )
        self._ha_base_body_id = bid
        # Cache the body's static pose in IK world. body_pos / body_quat
        # are the XML-declared offsets relative to the parent (worldbody
        # for our case), so for a worldbody child these ARE the IK-world
        # pose.
        self._ha_base_xyz = np.asarray(self.model.body_pos[bid],
                                       dtype=np.float64).copy()
        self._ha_base_quat = np.asarray(self.model.body_quat[bid],
                                        dtype=np.float64).copy()
        # Pre-build the rotation matrix to skip repeated quat→mat conv.
        self._ha_base_R = _quat_wxyz_to_matrix(self._ha_base_quat)

        # Optimizer state — bounds from MJCF joint ranges, warm-start.
        lo, hi = self.get_joint_limits()
        self._ha_bounds = list(zip(lo.tolist(), hi.tolist()))
        self._ha_last_q = None  # warm-start across calls
        self._ha_pos_w = float(pos_weight)
        self._ha_ori_w = float(ori_weight)
        self._ha_vel_w = float(vel_weight)
        self._ha_fi_c = float(fi_c)
        self._ha_fi_r = float(fi_r)
        self._ha_shoulder_w = float(shoulder_weight)
        self._ha_tol = float(tol)

    @staticmethod
    def _ha_fi(c: float, r: float, x: float) -> float:
        # Reference `_fi(n=1, s=0, c, r)`: minus-gaussian + quartic, so
        # the global minimum sits at x=0 and the quartic wall keeps the
        # optimizer from straying into unreachable regions.
        return -np.exp(-(x * x) / (2.0 * c * c)) + r * (x ** 4)

    def _ha_fk_in_base_local(self, q_arm: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """MuJoCo FK → end-site pose in base body LOCAL frame."""
        self.data.qpos[self._qpos_adr] = q_arm
        mujoco.mj_kinematics(self.model, self.data)
        site_xyz_world = self.data.site_xpos[self.end_site_id]
        site_xmat_world = self.data.site_xmat[self.end_site_id].reshape(3, 3)
        # base^T * (site_world - base_pos)
        xyz_local = self._ha_base_R.T @ (site_xyz_world - self._ha_base_xyz)
        mat_local = self._ha_base_R.T @ site_xmat_world
        return xyz_local, mat_local

    def _ha_objective(self, q_arm: np.ndarray,
                      target_xyz: np.ndarray,
                      target_mat: np.ndarray,
                      q_prev: np.ndarray) -> float:
        actual_xyz, actual_mat = self._ha_fk_in_base_local(q_arm)
        pos_err = float(np.linalg.norm(target_xyz - actual_xyz))
        # angular distance via trace identity:
        # angle = acos( (trace(R_target^T R_actual) - 1) / 2 )
        R_rel = target_mat.T @ actual_mat
        cos_ang = (np.trace(R_rel) - 1.0) * 0.5
        cos_ang = max(-1.0, min(1.0, cos_ang))
        ori_err = float(np.arccos(cos_ang))
        # Joint-velocity cost: shoulder joints weighted higher so SLSQP
        # doesn't spin them gratuitously when a wrist-only rotation
        # would suffice — same `shoulder_weight=2.0` boost the
        # reference applies to its q[0:3].
        dq = q_arm - q_prev
        if len(dq) >= 3:
            dq = dq.copy()
            dq[:3] *= self._ha_shoulder_w
        vel_err = float(np.linalg.norm(dq))
        c, r = self._ha_fi_c, self._ha_fi_r
        return (self._ha_pos_w * self._ha_fi(c, r, pos_err)
                + self._ha_ori_w * self._ha_fi(c, r, ori_err)
                + self._ha_vel_w * self._ha_fi(c, r, vel_err))

    def _ik_humanoid_arm(self, xyz, quat, q_init, max_iters,
                         pos_tol, ori_tol):
        # Target is in BASE LOCAL frame (caller's responsibility — see
        # the class docstring for why this differs from mink/dls/opspace).
        import scipy.optimize as opt  # lazy import: scipy needed only here

        target_xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        target_mat = _quat_wxyz_to_matrix(np.asarray(quat, dtype=np.float64))

        # Warm start: prefer caller-supplied q_init, then the previous
        # call's solution, then the current scratch qpos.
        if q_init is not None:
            q0 = np.asarray(q_init, dtype=np.float64)[:self._n_arm].copy()
        elif self._ha_last_q is not None:
            q0 = self._ha_last_q.copy()
        else:
            q0 = self.data.qpos[self._qpos_adr].copy()
        # Clip seed into bounds — SLSQP otherwise complains on the first
        # iteration if the warm-start sits outside the box.
        for i, (lo, hi) in enumerate(self._ha_bounds):
            q0[i] = min(max(q0[i], lo), hi)

        q_prev = (self._ha_last_q.copy() if self._ha_last_q is not None
                  else q0.copy())

        result = opt.minimize(
            self._ha_objective,
            q0,
            args=(target_xyz, target_mat, q_prev),
            method="SLSQP",
            tol=self._ha_tol,
            bounds=self._ha_bounds,
            options={"maxiter": int(max_iters)},
        )
        q_sol = np.asarray(result.x, dtype=np.float64)
        # Convergence check via actual residuals — SLSQP's success flag
        # reports gradient/step-size convergence which is too lax for
        # our pose tolerance contract.
        actual_xyz, actual_mat = self._ha_fk_in_base_local(q_sol)
        pos_err = float(np.linalg.norm(target_xyz - actual_xyz))
        R_rel = target_mat.T @ actual_mat
        cos_ang = max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) * 0.5))
        ori_err = float(np.arccos(cos_ang))
        ok = (pos_err <= pos_tol and ori_err <= ori_tol)
        # Cache for warm start regardless of convergence — even a
        # near-miss solution is a useful seed for the next AVP frame.
        self._ha_last_q = q_sol
        return q_sol, ok


# Neutral alias so callers writing new code can import a robot-agnostic name.
# Keeps existing `from franka_kinematics_solver import FrankaKinematicsSolver`
# imports working.
KinematicsSolver = FrankaKinematicsSolver
