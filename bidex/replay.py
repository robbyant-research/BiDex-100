"""Replay mode for BiDex-100.

Feed a recorded episode's actions back through ``BiDexEnv.step()`` frame by
frame to visualize / verify a policy rollout — the in-process equivalent of
the original web "replay mode".

Two entry points:

  - ``replay_episode(env, episode, mode=...)`` — drive an existing env with a
    recorded action sequence (dict of arrays or a parquet path). The IK for
    ee-mode runs *inside* ``env.step()`` (unlike the old LCM replay which had
    to run IK client-side), so this module carries no IK logic of its own.

  - ``record_episode(env, policy_fn, max_steps)`` — run a policy and stack the
    per-frame obs/action into the same dict schema, so a rollout can be saved
    and later replayed.

Quaternion convention boundary: the live API uses wxyz (MuJoCo native); the
on-disk parquet (egodex) uses xyzw. ``replay_episode`` converts at the parquet
boundary when ``mode="eef"``.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Union

import numpy as np


def _eef_xyzw_to_wxyz(pose7: np.ndarray) -> np.ndarray:
    """Convert a single [x, y, z, qx, qy, qz, qw] (xyzw) pose to
    [x, y, z, qw, qx, qy, qz] (wxyz)."""
    out = np.asarray(pose7, dtype=np.float64).copy()
    out[3:7] = np.array([pose7[6], pose7[3], pose7[4], pose7[5]])
    return out


def _load_episode(episode: Union[dict, str]) -> dict:
    """Accept a dict of arrays or a parquet path; return a dict of arrays."""
    if isinstance(episode, dict):
        return episode
    import pandas as pd
    df = pd.read_parquet(episode)
    out = {}
    for col in ("actions.arm_joint_position",
                "actions.dex_joint_position",
                "actions.abs_end_effector"):
        if col in df.columns:
            out[col] = np.stack(df[col].values).astype(np.float64)
    return out


def replay_episode(
    env,
    episode: Union[dict, str],
    *,
    mode: str = "joint",
    warmup_steps: int = 0,
    fps: float = 25.0,   # 25 Hz, unified with the rebuilt 25 fps dataset
    real_time: bool = False,
    max_steps: Optional[int] = None,
) -> dict:
    """Feed a recorded episode through ``env.step()`` frame by frame.

    Parameters
    ----------
    env : BiDexEnv
        An already-constructed env (call ``env.reset()`` first, or this will
        replay against whatever state the env is in).
    episode : dict | str
        Dict with action arrays (keys ``actions.arm_joint_position``,
        ``actions.dex_joint_position``, and ``actions.abs_end_effector`` for ee
        mode), or a path to a parquet with those columns.
    mode : "joint" | "ee"
        Which action key drives the arms. Hands always come from
        ``actions.dex_joint_position``.
    warmup_steps : int
        Number of times to repeat frame-0's action before the real loop, to
        settle the arm onto the recorded init pose.
    fps : float
        Only used when ``real_time=True``.
    real_time : bool
        If True, sleep between frames to match ``fps``; else run as fast as
        physics allows.
    max_steps : int | None
        Cap on replayed frames (defaults to the episode length).

    Returns
    -------
    dict
        ``{"success": bool, "subtasks": [...], "frames": int, "timed_out": bool}``
    """
    if mode not in ("joint", "ee"):
        raise ValueError(f"mode must be 'joint' or 'ee', got {mode!r}")

    data = _load_episode(episode)
    n = _episode_len(data, mode)
    if n == 0:
        return {"success": False, "subtasks": [], "frames": 0, "timed_out": False}

    dex = data.get("actions.dex_joint_position")
    arm = data.get("actions.arm_joint_position")
    eef = data.get("actions.abs_end_effector")

    if max_steps is None:
        max_steps = n
    frame_dt = 1.0 / fps if fps > 0 else 0.0

    def _build_action(i: int) -> dict:
        action: Dict[str, np.ndarray] = {}
        if mode == "joint":
            if arm is not None:
                action["arm_joint_position"] = arm[i]
        else:  # ee
            if eef is not None:
                # parquet stores xyzw; live API wants wxyz.
                action["abs_end_effector"] = _eef_xyzw_to_wxyz(eef[i])
        if dex is not None:
            action["dex_joint_position"] = dex[i]
        return action

    # Warmup: repeat frame-0 action to settle onto the recorded init pose.
    for _ in range(warmup_steps):
        obs, _r, done, info = env.step(_build_action(0))
        if done or info.get("timed_out"):
            return _summarize(info, 0, False)

    timed_out = False
    last_info = {"success": False, "subtasks": [], "timed_out": False}
    frames = 0
    for i in range(min(n, max_steps)):
        t0 = time.time()
        obs, _r, done, info = env.step(_build_action(i))
        last_info = info
        frames = i + 1
        if done or info.get("timed_out"):
            timed_out = info.get("timed_out", False)
            break
        if real_time and frame_dt > 0:
            sleep = frame_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    return _summarize(last_info, frames, timed_out)


def _episode_len(data: dict, mode: str) -> int:
    for key in ("actions.arm_joint_position",
                "actions.dex_joint_position",
                "actions.abs_end_effector"):
        if key in data and data[key] is not None:
            return int(data[key].shape[0])
    return 0


def _summarize(info: dict, frames: int, timed_out: bool) -> dict:
    return {
        "success": bool(info.get("success", False)),
        "subtasks": info.get("subtasks", []),
        "frames": frames,
        "timed_out": timed_out,
    }


def record_episode(env, policy_fn: Callable[[dict], dict],
                    max_steps: int = 300) -> dict:
    """Run ``policy_fn`` against ``env`` and stack per-frame obs/action.

    Returns a dict with the same schema ``replay_episode`` accepts, so a
    recorded rollout can be saved and later replayed:

        {
          "actions.arm_joint_position": (N, 14),   # joint mode
          "actions.abs_end_effector":   (N, 14),   # ee mode (wxyz)
          "actions.dex_joint_position": (N, 44),
          "observations...":            (N, ...),  # per-frame obs, optional
        }
    """
    obs = env.reset()
    arm_buf, eef_buf, dex_buf = [], [], []
    obs_buf = []
    last_info = {"success": False, "subtasks": [], "timed_out": False}
    for t in range(max_steps):
        action = policy_fn(obs)
        arm_buf.append(np.asarray(action.get("arm_joint_position",
                                              obs["arm_joint_position"]), dtype=np.float64))
        eef_buf.append(np.asarray(action.get("abs_end_effector",
                                             obs["abs_end_effector"]), dtype=np.float64))
        dex_buf.append(np.asarray(action.get("dex_joint_position",
                                             obs["dex_joint_position"]), dtype=np.float64))
        obs_buf.append(obs)
        obs, _r, done, last_info = env.step(action)
        if done or last_info.get("timed_out"):
            break

    out: Dict[str, np.ndarray] = {
        "actions.arm_joint_position": np.stack(arm_buf),
        "actions.abs_end_effector": np.stack(eef_buf),
        "actions.dex_joint_position": np.stack(dex_buf),
        "success": bool(last_info.get("success", False)),
        "frames": len(arm_buf),
    }
    return out
