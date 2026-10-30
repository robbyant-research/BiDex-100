"""Shared SLSQP arm-retarget base used by the Astribot IK (``arms_retarget.py``).

Ported from humanoid-arm-retarget/arms_retarget.py. The math is the upstream
algorithm verbatim: SLSQP over 10 DOF (5 per arm) plus the analytic
wrist-roll/pitch solver; position/orientation/velocity costs weighted 60/6/2.

This module was previously located at ``bidex/ik/retarget/gr1/arms_retarget.py``
and shared with a GR1 retarget. GR1 is no longer supported, so the base class is
vendored here next to the Astribot subclass that uses it. ``load_config`` resolves
relative config paths against THIS package dir (where ``config_astribot.yaml`` lives).
"""

import importlib
import os

import numpy as np
import quaternionic
import scipy.optimize as opt
import yaml


def calculate_angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ArmRetarget:
    def __init__(self, config_file="config_astribot.yaml"):
        self.config = self.load_config(config_file)

        self.left_fk = self.load_fk_function('left_fk_function')
        self.right_fk = self.load_fk_function('right_fk_function')

        self.scale_factor = self.config['scale_factor']
        self.num_joints = self.config['num_joints']

        self.lower_bounds = np.array(self.config['lower_bounds']) * np.pi / 180
        self.upper_bounds = np.array(self.config['upper_bounds']) * np.pi / 180
        self.bounds = [(l, u) for l, u in zip(self.lower_bounds, self.upper_bounds)]

        self.last_optimized_q_10d = np.array(self.config['init_q']) * np.pi / 180

        self.left_base = np.array(self.config['left_base'])
        self.right_base = np.array(self.config['right_base'])
        self.left_wrist = np.array(self.config['left_wrist'])
        self.right_wrist = np.array(self.config['right_wrist'])

        self.last_valid_r = None

    def load_config(self, config_file):
        if not os.path.isabs(config_file):
            config_file = os.path.join(_THIS_DIR, config_file)
        with open(config_file, 'r') as fh:
            return yaml.safe_load(fh)

    def load_fk_function(self, fk_key):
        module_name, function_name = self.config[fk_key].rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)

    def _fi(self, n, s, c, r, x):
        return (-1) ** n * np.exp(-((x - s) ** 2) / (2 * c**2)) + r * (x - s) ** 4

    def _position_cost(self, target_pos, actual_pos):
        n, s, c, r = 1, 0, 0.2, 5.0
        xl = np.linalg.norm(target_pos[0] - actual_pos[0])
        xr = np.linalg.norm(target_pos[1] - actual_pos[1])
        return self._fi(n, s, c, r, xl) + self._fi(n, s, c, r, xr)

    def _orientation_cost(self, target_quat, actual_quat):
        n, s, c, r = 1, 0, 0.2, 5.0
        q1 = quaternionic.array(target_quat[0])
        q2 = quaternionic.array(actual_quat[0])
        xl = quaternionic.distance.rotation.intrinsic(q1, q2)
        q1 = quaternionic.array(target_quat[1])
        q2 = quaternionic.array(actual_quat[1])
        xr = quaternionic.distance.rotation.intrinsic(q1, q2)
        return self._fi(n, s, c, r, xl) + self._fi(n, s, c, r, xr)

    def _joint_velocity_cost(self, q0_10d, q_10d):
        n, s, c, r = 1, 0, 0.2, 5.0
        xll = q0_10d[0:5] - q_10d[0:5]
        xll[0:3] = xll[0:3] * 2
        xl = np.linalg.norm(xll)
        xrr = q0_10d[5:10] - q_10d[5:10]
        xrr[0:3] = xrr[0:3] * 2
        xr = np.linalg.norm(xrr)
        return self._fi(n, s, c, r, xl) + self._fi(n, s, c, r, xr)

    def _objective_function(self, q_10d):
        q0_14d = np.zeros(14)
        q0_14d[0:5] = q_10d[0:5]
        q0_14d[7:12] = q_10d[5:10]

        r_left = self.left_fk(q0_14d[0:7])
        r_right = self.right_fk(q0_14d[7:14])

        actual_pos = np.vstack((r_left[0], r_right[0]))
        actual_quat = np.vstack((r_left[1], r_right[1]))

        pos_cost = self._position_cost(self.target_pos, actual_pos)
        ori_cost = self._orientation_cost(self.target_quat, actual_quat)
        vel_cost = self._joint_velocity_cost(self.last_optimized_q_10d, q_10d)

        return 60 * pos_cost + 6 * ori_cost + 2 * vel_cost

    def _solve_wrist_angles(self, T_wrist, T_forearmWrist, chirality=None):
        wrist_y = T_wrist[:3, 1]
        arm_y = T_forearmWrist[:3, 1]
        angle6 = calculate_angle_between_vectors(wrist_y, arm_y)
        arm_x = T_forearmWrist[:3, 0]
        if np.dot(arm_x, wrist_y) > 0:
            q6 = -angle6 if chirality == "left" else angle6
        else:
            q6 = angle6 if chirality == "left" else -angle6

        wrist_z = T_wrist[:3, 2]
        arm_z = T_forearmWrist[:3, 2]
        angle7 = calculate_angle_between_vectors(wrist_z, -arm_z)
        if np.dot(arm_x, wrist_z) > 0:
            q7 = angle7
        else:
            q7 = -angle7

        q7 = np.clip(q7, -np.pi * 35 / 180, np.pi * 35 / 180)
        q6 = np.clip(q6, -np.pi * 50 / 180, np.pi * 55 / 180)

        return q7, q6  # reverse 6 and 7 — upstream rcs interface quirk

    def _solve_uparm_angles(self, q0, target_pos, target_quat):
        self.target_pos = target_pos
        self.target_quat = target_quat

        q0_10d = np.concatenate((q0[0:5], q0[7:12]))
        result = opt.minimize(
            self._objective_function, q0_10d,
            method="SLSQP", tol=1e-3, bounds=self.bounds,
            options={"maxiter": 100},
        )

        if result.success:
            self.last_optimized_q_10d = result.x
            return result.x
        raise RuntimeError(
            f"SLSQP failed after {result.nit} iterations: {result.message}")

    def solve_ik(self, target_pos, target_quat, q_init=None):
        """Direct EE-target IK (no AVP frame needed).

        Replay (and any callers that already have the target EE pose in the
        robot base frame) use this in place of the AVP-driven
        solve_arm_angles. Skips _solve_wrist_angles — the wrist roll/pitch
        DoF are kept at whatever q_init has (or zero), since there's no
        forearm pose to derive them from. Front 5 joints per side are
        solved by the same SLSQP cost as live retarget.

        Args:
            target_pos:  (2, 3) — [left_xyz, right_xyz] in robot base frame
            target_quat: (2, 4) — [left_wxyz, right_wxyz] in robot base frame
            q_init:      optional (14,) seed; wrist_roll/pitch (slots 5,6
                         and 12,13) come from here. Defaults to zeros.
        Returns:
            q_14d (14,) — [L 7-DoF, R 7-DoF] in the same order
            solve_arm_angles emits.
        """
        if q_init is None:
            q_init = np.zeros(14, dtype=np.float64)
        q_init = np.asarray(q_init, dtype=np.float64).reshape(14)
        q_10d = self._solve_uparm_angles(q_init, target_pos, target_quat)
        q_14d = np.zeros(14, dtype=np.float64)
        q_14d[0:5]  = q_10d[0:5]      # L shoulder/elbow/wrist_yaw
        q_14d[5:7]  = q_init[5:7]     # L wrist roll/pitch — pass through
        q_14d[7:12] = q_10d[5:10]     # R shoulder/elbow/wrist_yaw
        q_14d[12:14] = q_init[12:14]  # R wrist roll/pitch — pass through
        return q_14d

    def solve_arm_angles(self, q0, r):
        if self.last_valid_r is None:
            self.last_valid_r = r

        is_left_missing = np.all(r["left_fingers"][1][0, :3] == 0)
        is_right_missing = np.all(r["right_fingers"][1][0, :3] == 0)
        if is_left_missing or is_right_missing:
            r = self.last_valid_r
        else:
            self.last_valid_r = r

        left_wrist = r["left_wrist"][0]
        right_wrist = r["right_wrist"][0]
        left_forearmWrist = left_wrist @ r["left_fingers"][-2]
        right_forearmWrist = right_wrist @ r["right_fingers"][-2]

        q_2d_l = self._solve_wrist_angles(left_wrist, left_forearmWrist, chirality="left")
        q_2d_r = self._solve_wrist_angles(right_wrist, right_forearmWrist, chirality="right")

        left_forearmWrist = self.left_base @ left_forearmWrist @ self.left_wrist
        right_forearmWrist = self.right_base @ right_forearmWrist @ self.right_wrist
        pos = np.vstack((left_forearmWrist[:3, 3], right_forearmWrist[:3, 3]))
        pos = pos * self.scale_factor
        quat1 = quaternionic.array.from_rotation_matrix(left_forearmWrist)
        quat2 = quaternionic.array.from_rotation_matrix(right_forearmWrist)
        quat = np.vstack((quat1, quat2))

        q_10d = self._solve_uparm_angles(q0, pos, quat)
        q_14d = np.concatenate((q_10d[0:5], q_2d_l, q_10d[5:10], q_2d_r))
        return q_14d
