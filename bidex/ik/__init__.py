"""Vendored inverse-kinematics solver for EE-mode control.

Wrapped around ``FrankaKinematicsSolver`` (mink QP backend) and the
``RobotProfile`` dataclass, vendored from force_cbf's
``easy_kinematic_solver`` (origin/main, commit cb6d253) so this eval repo
is self-contained — no ``FORCE_CBF`` env var or external checkout needed.

The solver is constructed per-arm by ``BiDexEnv`` via
``robots.registry.resolve_ik_profile`` which yields ``RobotProfile`` kwargs
plus the chosen backend (``mink`` for both franka and astribot).
"""

from bidex.ik.franka_kinematics_solver import (
    FrankaKinematicsSolver,
    RobotProfile,
)

__all__ = ["FrankaKinematicsSolver", "RobotProfile"]
