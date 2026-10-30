#!/usr/bin/env python3
"""Canonical BiDex-100 leaderboard evaluation driver.

Evaluates a trained checkpoint over the **10-combination × 3-mode** matrix:
  2 domains (In_Domain, Out-of-Domain) × 5 task categories
  (Memory, Long_Horizon, Generalization, Instruction_Following,
  Limited_Demonstrations) = 10 combos, each run in the
  trained control mode (joint / robot_ee / camera_ee) of the checkpoint.

Reproducibility / scoring contract (what the leaderboard displays):
  * 2 in-domain seeds and 2 out-of-domain seeds × 25 episodes/seed =
    100 rollouts per task; aggregated mean_rate +
    success_rate written to the leaderboard, with a per-seed breakdown kept
    for debugging.
  * Per-subtask latching scores weighted by ``task_annotations.json``
    (``bidex/eval/scoring.py``) — episode score = sum of score weights
    of subtasks that latched `done`.
  * Deterministic OOD: a fixed (seed, task, episode) deterministically
    reproduces wall/floor/table textures, shadow-casting key/fill lighting,
    camera extrinsics and object placement;
    the distractor set and structural OOD settings are fixed by (seed, task).
    Independent SHA-256-keyed RNG streams and a hash-validated texture manifest
    make the plan reproducible across computers, while all 50 canonical OOD
    combinations (seeds 2/3 x 25 episodes) receive distinct role textures.
  * Output is the JSON layout the leaderboard server
    (``bidex/scripts/baseline_leaderboard_server.py``) reads:
    ``results/<model_id>/<benchmark>_<robot>_<mode>_<domain>.json`` +
    ``mp4s/<model_id>/<setting>/<task_dir>/s{seed}_ep{ep:03d}.mp4``.

Control modes (env always runs in the env-native mode; ee/camera_ee are
converted to the env action by the ControlAdapter):
  * joint            -> env JOINT (passthrough)
  * robot_ee/camera_ee, astribot -> env JOINT + adapter IK (``make_full_fk_ik``)
  * robot_ee/camera_ee, franka   -> env EE    + adapter frame conversion
                                      (policy collection<->world via the
                                      deterministic franka_camera_frame M_*/T_CAM)

Policy backends implement the same observation/action contract so models can
be evaluated through one benchmark driver.

Usage (openpi, in the `openpi` conda env):
  python bidex/eval/run_leaderboard_eval.py \
    --model openpi --ckpt "$MODEL_CKPT" \
    --benchmark Generalization --robot astribot --control-mode camera_ee \
    --domain both --id-seeds 0,1 --ood-seeds 2,3 \
    --episodes-per-seed 25 --model-gpu 0 --render-gpus 0,1,2,3,4,5,6,7

Usage (gr00t, in the Isaac-GR00T .venv):
  python bidex/eval/run_leaderboard_eval.py \
    --model groot --ckpt "$MODEL_CKPT" \
    --benchmark Generalization --robot astribot --control-mode camera_ee \
    --domain both --id-seeds 0,1 --ood-seeds 2,3 \
    --episodes-per-seed 25 --model-gpu 0 --render-gpus 0,1,2,3,4,5,6,7
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, json, argparse, subprocess, time, shutil, zlib, threading, functools, hashlib, copy
from pathlib import Path
from typing import Optional

import numpy as np

# Make `import bidex` work whether or not it is pip-installed.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bidex import BiDexEnv, load_task
from bidex.tasks.benchmark import arm_for_task, BENCHMARKS
from bidex.eval.scoring import subtask_weights, aggregate
from bidex.eval import ood as oodmod
from scripts.evaluation.policy_protocol import packb, unpackb


def enumerate_task_ids(benchmark: str, robot: str, arm: str = "dual"):
    """Return current ``<category>/<task>`` IDs for one benchmark and robot.

    The YAML's ``robot_type`` is authoritative; task-name prefixes are not part
    of the public task identity.
    """
    cat = benchmark
    ids = []
    try:
        from bidex.tasks.registry import list_tasks as _lt
        for name in _lt(cat):
            if name.endswith("__ood"):
                continue
            task_spec = load_task(cat, name)
            robot_type = task_spec.robot_type or ""
            if robot_type.startswith("astribot_"):
                r = "astribot"
            elif robot_type.startswith("dual_fr3v2_"):
                r = "franka"
            else:
                raise ValueError(
                    f"Unsupported robot_type {robot_type!r} in {cat}/{name}"
                )
            if r != robot:
                continue
            if arm not in ("dual", "single_arm") and arm_for_task(r, cat, name) != arm:
                continue
            ids.append(f"{cat}/{name}")
    except FileNotFoundError:
        return []
    return sorted(ids)

# ---- consolidated eval-data h5 (sole data source) ----
# All eval data (init_states, demo_lengths, franka tcoll) is read from a single
# h5 file at bidex/eval/eval_dataset.h5 (resolved relative to this script).
# Schema: <Category>/<task_name>/init_states/success, demo_lengths, tcoll/{arm,ee}.
_EVAL_H5_PATH = Path(__file__).resolve().parent / "eval_dataset.h5"
_EVAL_H5 = None  # None until first use, then a validated read-only h5py.File


def _expected_eval_h5_tasks() -> dict[str, set[str]]:
    """Return the 106 base-task names defined by the benchmark YAML files."""
    suites_root = _REPO / "bidex" / "tasks_data" / "suites"
    return {
        suite: {
            path.stem
            for path in (suites_root / suite).glob("*.yaml")
            if not path.stem.endswith("__ood")
        }
        for suite in BENCHMARKS
    }


def _validate_eval_h5_schema(h5_file) -> None:
    """Fail before evaluation if H5 task names drift from the benchmark YAMLs."""
    expected = _expected_eval_h5_tasks()
    actual_suites = set(h5_file.keys())
    if actual_suites != set(expected):
        raise RuntimeError(
            "suite mismatch: "
            f"missing={sorted(set(expected) - actual_suites)}, "
            f"extra={sorted(actual_suites - set(expected))}"
        )

    errors = []
    for suite, task_names in sorted(expected.items()):
        actual = set(h5_file[suite].keys())
        missing = sorted(task_names - actual)
        extra = sorted(actual - task_names)
        if missing or extra:
            errors.append(f"{suite}: missing={missing}, extra={extra}")
            continue
        for task in sorted(task_names):
            group = h5_file[f"{suite}/{task}"]
            for dataset in ("init_states/success", "demo_lengths"):
                if dataset not in group or group[dataset].size == 0:
                    errors.append(f"{suite}/{task}: missing or empty {dataset}")
            init_states = group.get("init_states/success")
            if init_states is not None and (
                init_states.ndim != 2 or init_states.shape[1] != 58
            ):
                errors.append(
                    f"{suite}/{task}: init_states/success shape "
                    f"{init_states.shape}, expected (N, 58)"
                )
            task_spec = load_task(suite, task)
            if task_spec.name != task:
                errors.append(
                    f"{suite}/{task}: YAML name={task_spec.name!r} must match filename"
                )
            if (task_spec.robot_type or "").startswith("dual_fr3v2_"):
                arm = group.get("tcoll/arm")
                ee = group.get("tcoll/ee")
                if arm is None or ee is None or arm.size == 0 or ee.size == 0:
                    errors.append(f"{suite}/{task}: missing Franka tcoll data")
                elif arm.shape != ee.shape or arm.ndim != 2 or arm.shape[1] != 14:
                    errors.append(
                        f"{suite}/{task}: invalid tcoll shapes "
                        f"arm={arm.shape}, ee={ee.shape}"
                    )
    if errors:
        raise RuntimeError("eval_dataset.h5/YAML mismatch:\n  " + "\n  ".join(errors))


def _eval_h5():
    """Open and validate the consolidated evaluation dataset once per process."""
    global _EVAL_H5
    if _EVAL_H5 is None:
        if not _EVAL_H5_PATH.exists():
            raise FileNotFoundError(
                f"Eval dataset h5 not found: {_EVAL_H5_PATH}. "
                "This file is required for evaluation."
            )
        import h5py
        handle = h5py.File(str(_EVAL_H5_PATH), "r")
        try:
            _validate_eval_h5_schema(handle)
        except Exception:
            handle.close()
            raise
        _EVAL_H5 = handle
    return _EVAL_H5

CONTROL_FREQ_HZ = 25   # 25 Hz control (matches the rebuilt 25 fps bidex_qc3 dataset)
# An intentionally unreachable cap: clients clamp it to the actual action-chunk
# length returned by each model. This keeps the model-native maximum as the
# default without duplicating model-specific horizons in the transport layer.
FULL_CHUNK_REUSE = 1 << 20
LEADERBOARD_ROOT = Path(os.environ.get("BIDEX_LEADERBOARD_ROOT", "leaderboard"))
RESULTS_DIR = LEADERBOARD_ROOT / "results"
MP4_DIR = str(LEADERBOARD_ROOT / "mp4s")
MODEL_ID_MAP = {"openpi": "pi05", "groot": "gr00t_n1_7", "lingbot": "lingbot_v2", "galaxea": "galaxea_g05", "dm05": "dm05", "xr1": "xr1", "internvla": "internvla"}


def _configured_chunk_reuse():
    """Return an optional user cap; empty/zero means the full returned chunk."""
    raw = os.environ.get("BIDEX_VEC_CHUNK_REUSE", "").strip()
    if not raw:
        return FULL_CHUNK_REUSE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"BIDEX_VEC_CHUNK_REUSE must be an integer, got {raw!r}") from exc
    return FULL_CHUNK_REUSE if value <= 0 else value

# --------------------------------------------------------------------------- #
# streaming mp4 writer (per-episode head-cam rollout video, low memory)        #
# --------------------------------------------------------------------------- #


def _ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        f = shutil.which("ffmpeg")
        if f:
            return f
        raise RuntimeError("ffmpeg not found (install imageio-ffmpeg or put ffmpeg on PATH)")


class Mp4Writer:
    """Stream HxWx3 uint8 frames to an mp4 via an ffmpeg stdin pipe."""

    def __init__(self, path, fps=25):
        self.path = path
        self.fps = fps
        self.proc = None

    def write(self, frame):
        f = np.asarray(frame)
        f = np.ascontiguousarray(f.astype(np.uint8))
        if self.proc is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            h, w = f.shape[:2]
            cmd = [_ffmpeg_bin(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-an",
                   "-pix_fmt", "yuv420p",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                   "-tag:v", "avc1", "-movflags", "+faststart", self.path]
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.proc.stdin.write(f.tobytes())

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close(); self.proc.wait(timeout=30)
            except Exception:
                pass
            finally:
                if self.proc.poll() is None:
                    self.proc.kill()
        self.proc = None

# --------------------------------------------------------------------------- #
# collection init-state restore (proven, ported from the reference scripts)    #
# --------------------------------------------------------------------------- #


def load_collection_init_states(suite, task):
    """Load recorded ID robot poses from the matching benchmark task group."""
    _h = _eval_h5()
    try:
        arr = np.asarray(
            _h[f"{suite}/{task}/init_states/success"][()],
            dtype=np.float64,
        )
    except KeyError:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr if arr.size else np.zeros((0, 0), dtype=np.float64)


def _robot_jnt_ids(env):
    """All robot joint ids (qpos prefix) for init_state restore — arm+dex+torso+head.

    For franka: right/left_franka_jnt_ids + right/left_sharpa_jnt_ids = 58 (no
    torso/head — the MJCF starts with the arm joints). Matches the init_state row layout
    exactly.

    For astribot: the MJCF has [torso(4), head(2), left_arm(7), right_arm(7),
    left_dex(22), right_dex(22)] = 64 robot joints BEFORE any object joints. The
    franka/sharpa attrs cover only the 58 arm+dex joints — the 6 torso/head joints are
    at qposadr 0-5 (the row's first 6 entries). Without including them,``restore_state_from_collection``
    would write the torso values onto left-arm qpos slots → the arm starts at a WILDLY
    wrong pose → "initial-frame joints fly off". Fix: include all joints whose qposadr is below the
    max arm/dex qposadr — i.e., the robot is a contiguous qpos prefix in the MJCF."""
    ids = []
    for attr in ("right_franka_jnt_ids", "left_franka_jnt_ids",
                 "right_sharpa_jnt_ids", "left_sharpa_jnt_ids"):
        v = getattr(env, attr, None)
        if v:
            ids.extend(j for j in v if j >= 0)
    # astribot (+ any robot with torso/head joints BEFORE arm/dex in the MJCF):
    # include all joints at qposadr < min(arm/dex qposadr) that are NOT object joints.
    # Object joints (e.g. free-joint baskets) have range-state and come AFTER robot;
    # their names contain no "astribot_" / "franka_" / "link" / "sharpa" prefix.
    if ids:
        m = env.model
        min_arm_adr = min(m.jnt_qposadr[j] for j in ids)
        for j in range(m.njnt):
            if j in ids:
                continue
            adr = m.jnt_qposadr[j]
            if adr < min_arm_adr:
                name = m.jnt(j).name
                # robot prefix joints (torso, head, base) — NOT object free joints
                if name and not name.startswith("obj_") and "free" not in name.lower():
                    ids.append(j)
    return sorted(ids, key=lambda j: env.model.jnt_qposadr[j])


def restore_state_from_collection(env, state_row):
    """Start the episode from the recorded robot working pose (not T-pose).

    Restores ONLY the robot arm+dex joint POSITIONS from the saved init_state row
    — the first N entries (the robot is defined first in the MJCF, so its qpos is a
    stable contiguous prefix regardless of object edits). Object placement comes
    from the task YAML / env reset's region sampling; the policy input is only joint
    positions (state_arm + state_dex), so neither object state nor velocity is taken
    from the saved row. This makes the restore robust to ANY task-XML object
    add/remove (the robot prefix is stable) and it never raises on a stale
    init_states file — object info is never read from init_states.

    Trimmed h5 support (58D rows): when the row has fewer dims than the full robot
    joint count (64 for astribot), the missing leading joints are the fixed
    torso(4)+head(2) which are always ≈[0,0,0,0,0,1]. The function detects this
    case, sets the fixed joints to their known default, and restores only the
    arm+dex (58D) from the truncated row. Franka has no fixed joints so a 58D
    row matches n_robot=58 exactly and works unchanged.
    """
    row = np.asarray(state_row, dtype=np.float64)
    m = env.model
    robot_ids = _robot_jnt_ids(env)
    n_robot = len(robot_ids)
    if n_robot == 0:
        print(f"[restore] WARN no robot joint ids on env (robot_type="
              f"{getattr(env, 'robot_type', '?')}); skipping restore (env default init)", flush=True)
        return
    import mujoco
    adr = sorted(m.jnt_qposadr[j] for j in robot_ids)   # robot qposadr ascending = saved-row order
    n_fixed = n_robot - row.size  # leading fixed joints missing from the truncated row
    if n_fixed < 0:
        # Row has MORE dims than robot joints — only use the first n_robot.
        n_fixed = 0
    elif n_fixed > 0:
        # Trimmed 58D row on a 64-joint astribot: set the n_fixed leading joints (torso+head)
        # to their known default values [0,0,0,0,0,1], then restore arm+dex from the 58D row.
        # Verified: across all 1000 astribot init_states, torso ≈ [2e-4, 3e-4, 3e-4, -2e-6]
        # and head ≈ [0, 1.000011] — effectively fixed at [0,0,0,0, 0,1].
        _ASTIBOT_FIXED_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        for k in range(min(n_fixed, len(_ASTIBOT_FIXED_DEFAULTS))):
            env.data.qpos[adr[k]] = _ASTIBOT_FIXED_DEFAULTS[k]
        print(f"[restore] trimmed init_state {row.size}D on {n_robot}-joint robot: "
              f"set {n_fixed} fixed joints (torso+head) to default, restore arm+dex from row",
              flush=True)
    # Restore arm+dex from the row (offset by n_fixed in adr to skip the fixed joints).
    for k in range(row.size):
        target_k = n_fixed + k
        if target_k < len(adr):
            env.data.qpos[adr[target_k]] = row[k]
    mujoco.mj_forward(m, env.data)
    if row.size + n_fixed != m.nq + m.nv:
        print(f"[restore] init_state {row.size}D (+{n_fixed} fixed) vs env nq+nv {m.nq + m.nv}D: "
              f"restored robot pose only ({n_robot} joints); objects keep env reset placement",
              flush=True)


def _demo_episode_lengths(suite, task):
    """Read demonstration lengths from the matching benchmark task group."""
    _h = _eval_h5()
    try:
        return list(np.asarray(
            _h[f"{suite}/{task}/demo_lengths"][()]
        ).tolist())
    except KeyError:
        return []


@functools.lru_cache(maxsize=None)
def compute_max_steps_p90(suite, task, factor=1.2, fallback_seconds=30.0):
    """Max interaction horizon = 1.2 × P90(demo episode lengths) from the eval h5.
    Falls back to fallback_seconds × CONTROL_FREQ_HZ when no demo lengths exist."""
    lengths = _demo_episode_lengths(suite, task)
    if lengths:
        p90 = float(np.percentile(lengths, 90))
        return max(int(round(factor * p90)), 1)
    return max(int(round(fallback_seconds * CONTROL_FREQ_HZ)), 1)


def max_steps_for_task(suite, task):
    """Max interaction steps for a task. A task yaml with an explicit `max_steps:`
    field always wins; otherwise 1.2 × P90(demo_lengths) from the eval h5.
    Optional BIDEX_MAX_STEPS_CAP clamps the horizon."""
    try:
        yaml_steps = load_task(suite, task).max_steps
    except Exception:
        yaml_steps = None
    if yaml_steps:
        steps = int(yaml_steps)
    else:
        steps = compute_max_steps_p90(suite, task)
    cap = os.environ.get("BIDEX_MAX_STEPS_CAP", "").strip()
    if cap:
        steps = min(int(steps), int(cap))
    return steps


def stable_task_hash(suite: str, task: str) -> int:
    """Deterministic integer hash of a (suite, task) pair, stable across
    processes (unlike Python's randomized ``hash(str)``). Retained in result
    provenance for compatibility; random streams use SHA-256 component keys."""
    return zlib.crc32(f"{suite}/{task}".encode("utf-8")) % 1_000_000


def load_task_for_domain(suite: str, task: str, domain: str, eval_seed: int,
                         thash: int, rank_label: str = "",
                         episode: Optional[int] = None):
    """Load the task config for one eval domain.

    in_domain: the base yaml (fresh region sampling per episode; the caller may
    replay collection init states).

    out_of_domain: prefer the baked ``<task>__ood.yaml`` variant (region
    offsets / articulated scale / left-right mirroring are baked INTO that
    file by scripts/gen_ood_yamls.py) and apply its ``ood:`` block:
    ``scene_randomize`` (floor/wall/table textures + shadow-casting key/fill
    lighting, seeded deterministically per task+eval_seed+episode), per-episode
    camera extrinsics, and ``distractors`` (count
    range, fixed per task+eval_seed). When
    no variant file exists, fall back to the base yaml + legacy global OOD
    (scene randomize + post-reset object offset by the caller).

    Returns (loaded_task, distractor_names, apply_post_reset_offset).
    """
    loaded = load_task(suite, task)
    if domain != "out_of_domain":
        loaded._ood_metadata = {"schema": oodmod.RANDOMIZATION_SCHEMA,
                                "domain": domain, "eval_seed": int(eval_seed),
                                "episode": episode}
        return loaded, [], False
    has_variant = False
    try:
        loaded = load_task(suite, task + "__ood")
        loaded.name = task  # keep the base task identity (results/init keys)
        has_variant = True
    except FileNotFoundError:
        print(f"[{rank_label}][OOD] no __ood yaml for {suite}/{task}; "
              f"falling back to base config + legacy randomization", flush=True)
    ood_cfg = getattr(loaded, "ood", None) or {}
    distractor_names = []
    scene_metadata = {}
    scale_metadata = {}
    camera_plan = {}
    if episode is not None and ood_cfg.get("articraft_scale") is not None:
        # Only the Articraft object named by task.articulations is scaled.
        # Apply it before distractor packing so collision footprints use the
        # exact episode geometry. Ordinary objects retain their YAML scale.
        scale_metadata = oodmod.randomize_articraft_scale(
            loaded, oodmod.episode_rng(
                suite, task, int(eval_seed), int(episode), "articraft_scale"))
    dis = ood_cfg.get("distractors")
    if dis is None and not has_variant:
        # Legacy fallback (no __ood variant): Generalization got 3-6 distractors.
        dis = {"count_min": 3, "count_max": 6}
    if dis:
        n_min = int(dis.get("count", dis.get("count_min", 2)))
        n_max = int(dis.get("count", dis.get("count_max", 4)))
        distractor_names = oodmod.add_distractors(
            loaded, oodmod.randomization_rng(
                suite, task, int(eval_seed), "distractors"), n_min, n_max,
            eval_seed=int(eval_seed))
    if ood_cfg.get("scene_randomize", True):
        from bidex.scene_builder import resolve_scene_xml
        # Generate the managed XML only after all in-memory planning succeeds,
        # so a scale/packing error cannot leak a temporary ood_scene_*.xml.
        # Strictly fail on missing managed textures rather than silently
        # producing a non-comparable Libero-only scene.
        loaded._base_scene = loaded.scene
        loaded.scene, scene_metadata = oodmod.randomize_scene(
            resolve_scene_xml(loaded.scene), suite=suite, task=task,
            seed=int(eval_seed), episode=episode, return_metadata=True)
    if episode is not None:
        camera_plan = oodmod.camera_randomization_plan(
            loaded.robot_type, suite, task, int(eval_seed), int(episode)
        )
        # BiDexEnv consumes the plan after camera ids exist and before the first
        # render.  Keeping the exact committed deltas on the Task also makes the
        # Every server-client evaluation path shares one contract.
        loaded._ood_camera_plan = camera_plan
    # Baked variants already carry their own region offsets -> suppress the
    # legacy post-reset object offset (would double-shift + risk instability).
    apply_offset = not has_variant
    meta = {"schema": oodmod.RANDOMIZATION_SCHEMA,
            "domain": domain, "eval_seed": int(eval_seed),
            "episode": int(episode) if episode is not None else None,
            "task_key": f"{suite}/{task}", "scene": scene_metadata,
            "cameras": camera_plan,
            "distractors": list(distractor_names),
            "omitted_legacy_distractors": list(getattr(
                loaded, "_ood_omitted_legacy_distractors", []
            )),
            "object_scales": scale_metadata,
            "baked_layout": bool(ood_cfg.get("baked_layout", False)),
            "arm_mirror": bool(ood_cfg.get("arm_mirror", False)),
            "articraft_scale": ood_cfg.get("articraft_scale"),
            "apply_post_reset_offset": bool(apply_offset)}
    meta["plan_digest"] = hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    loaded._ood_metadata = meta
    return loaded, distractor_names, apply_offset


def _ood_result_metadata(domain: str, thash: int, ood_offset: float,
                         plans_by_episode: dict) -> dict:
    """Serializable provenance for reproducing every evaluated episode."""
    distractors = sorted({name for plan in plans_by_episode.values()
                          for name in plan.get("distractors", [])})
    return {
        "schema": oodmod.RANDOMIZATION_SCHEMA,
        "domain": domain,
        "plans_by_episode": plans_by_episode,
        # Kept as a union for consumers of the pre-v2 result schema.
        "distractors": distractors,
        "object_offset_m": ood_offset if domain == "out_of_domain" else 0.0,
        "rng": {
            "task_hash": thash,
            "appearance_key": "schema/suite/task/seed/component/episode",
            "distractor_key": "schema/suite/task/seed/distractors",
            "episode_key": "schema/suite/task/seed/component/episode",
        },
    }


def _close_eval_env(env, loaded_task) -> None:
    """Close an env and remove only its generated OOD scene XML."""
    try:
        env.close()
    finally:
        scene = Path(getattr(loaded_task, "scene", ""))
        if scene.name.startswith("ood_scene_") and scene.suffix == ".xml":
            try:
                scene.unlink(missing_ok=True)
            except TypeError:  # Python 3.7 compatibility for older workers.
                if scene.exists():
                    scene.unlink()


def env_native_mode(control_mode: str, robot: str) -> str:
    """The env-native control mode the env should be built in. ee/camera_ee on
    BOTH robots are IK'd by the ControlAdapter in the env's JOINT mode, mirroring
    the proven replay scripts (astribot: make_full_fk_ik; franka:
    FrankaEEPipeline per-arm Kabsch T_coll + standalone mink, env JOINT). The env
    EE mode is intentionally NOT used (warm-starts from sagged qpos → drift)."""
    return "joint" if control_mode == "joint" else "joint"

# --------------------------------------------------------------------------- #
# ControlAdapter: policy action (trained control mode) -> env action          #
# --------------------------------------------------------------------------- #


class ControlAdapter:
    """Converts a model's predicted action (in the TRAINED control mode) to the
    env's native action. Also converts the env's WORLD-frame ee obs back into
    the policy's training frame (for ee/camera_ee state inputs).

    joint            : passthrough (env JOINT).
    robot_ee/camera_ee, astribot: adapter IK via make_full_fk_ik -> env JOINT.
    robot_ee/camera_ee, franka  : adapter IK via FrankaEEPipeline (per-arm
                                  Kabsch T_coll fit from the task's H5 data +
                                  standalone mink, chained warm-start) -> env
                                  JOINT. Mirrors examples/franka_replay.py so
                                  inference matches the verified replay path
                                  (env EE mode is NOT used — it drifts). The
                                  B<->W frame reconciliation reads the LIVE
                                  env "base" body pose (as the deterministic FK
                                  rebuild does), NOT the gif-only
                                  FRANKA_BASE_WORLD constant.

    Articulated single-arm hold: when ``hold`` is set (per-episode), the static
    arm+hand is pinned to its dataset-default pose (joint action) each step."""

    def __init__(self, mode: str, robot: str, env):
        self.mode = mode; self.robot = robot; self.env = env
        self.hold = None  # {"active": "left"|"right", "arm14":[14], "dex44":[44]}
        self._astribot_ik = False; self._franka_ee = False
        if mode != "joint" and robot == "astribot":
            self._setup_astribot_ik(env); self._astribot_ik = True
        elif mode != "joint" and robot == "franka":
            self._franka_ee = True; self._setup_franka_ik(env)

    # --- astribot IK (env JOINT) ---
    def _setup_astribot_ik(self, env):
        from bidex.ik.retarget.astribot.full_fk_ik import make_full_fk_ik
        from bidex.ik.retarget.astribot import astribot_camera_frame as acf
        self._solve = make_full_fk_ik(env)
        self._acf = acf
        self._ee_camera_to_L = acf.ee_camera_to_L
        m = env.model; d = env.data; cb = m.body("chassis_base").id
        self._Rch = d.xmat[cb].reshape(3, 3).copy()
        self._tch = d.xpos[cb].copy()
        self._qch = d.xquat[cb].copy()
        lqa = [m.jnt_qposadr[j] for j in env.left_franka_jnt_ids if j >= 0]
        rqa = [m.jnt_qposadr[j] for j in env.right_franka_jnt_ids if j >= 0]
        self._q_prev = np.concatenate([[d.qpos[a] for a in lqa], [d.qpos[a] for a in rqa]])

    def reset_warmstart(self, env):
        m = env.model; d = env.data
        if self._astribot_ik:
            lqa = [m.jnt_qposadr[j] for j in env.left_franka_jnt_ids if j >= 0]
            rqa = [m.jnt_qposadr[j] for j in env.right_franka_jnt_ids if j >= 0]
            self._q_prev = np.concatenate([[d.qpos[a] for a in lqa], [d.qpos[a] for a in rqa]])
            cb = m.body("chassis_base").id
            self._Rch = d.xmat[cb].reshape(3, 3).copy()
            self._tch = d.xpos[cb].copy()
            self._qch = d.xquat[cb].copy()
        elif self._franka_ee:
            rqa = [m.jnt_qposadr[j] for j in env.right_franka_jnt_ids if j >= 0]
            lqa = [m.jnt_qposadr[j] for j in env.left_franka_jnt_ids if j >= 0]
            self._q_prev = {"right": np.array([d.qpos[a] for a in rqa]),
                            "left":  np.array([d.qpos[a] for a in lqa])}
            self._read_franka_base(env)

    # --- franka IK via FrankaEEPipeline (env JOINT), mirrors franka_replay.py ---
    def _setup_franka_ik(self, env):
        from bidex.ik.retarget.franka import franka_camera_frame as fcf
        from bidex.ik.retarget.franka.franka_ee_replay_lib import FrankaEEPipeline, ee_camera_to_B
        self._fcf = fcf
        self._ee_camera_to_B = ee_camera_to_B
        self._pipe = FrankaEEPipeline()
        self._arm_slices = {"right": (0, 7), "left": (7, 14)}
        self._Tcoll = None  # set per-task by set_franka_task_tcoll()
        m = env.model; d = env.data
        rqa = [m.jnt_qposadr[j] for j in env.right_franka_jnt_ids if j >= 0]
        lqa = [m.jnt_qposadr[j] for j in env.left_franka_jnt_ids if j >= 0]
        self._q_prev = {"right": np.array([d.qpos[a] for a in rqa]),
                        "left":  np.array([d.qpos[a] for a in lqa])}
        self._read_franka_base(env)

    def _read_franka_base(self, env):
        """Live pose of the dual-arm "base" body (the frame the deterministic FK
        rebuild expressed ee_B in). Read live so inference matches replay/rebuild
        regardless of the arena base placement (NOT the gif-only constant)."""
        m = env.model; d = env.data
        bbase = m.body("base").id
        self._Rb = d.xmat[bbase].reshape(3, 3).copy()
        self._tb = d.xpos[bbase].copy()
        self._qb = d.xquat[bbase].copy()  # wxyz

    def set_franka_task_tcoll(self, suite, task):
        """Fit per-arm T_coll from the task's recorded (state_arm, state_ee_B)
        trajectory in the consolidated eval h5, so the standalone mink IK
        reconciles its link0 frame with the parquet B frame. Rigid per arm, so
        fitting from any episode of the task is correct. The H5 schema is
        validated at startup, so every Franka task has an exact task-named
        trajectory and no legacy-name or cross-task fallback is allowed."""
        _h = _eval_h5()
        src = f"{suite}/{task}"
        Sarm = np.asarray(_h[f"{src}/tcoll/arm"][()], dtype=np.float64)
        See = np.asarray(_h[f"{src}/tcoll/ee"][()], dtype=np.float64)
        self._Tcoll = {
            side: self._pipe.fit(Sarm, See, self._arm_slices[side])
            for side in ("right", "left")
        }
        for side in ("right", "left"):
            print(
                f"  [franka IK] T_coll[{side}] resid "
                f"{self._Tcoll[side]['resid'].mean()*1e3:.2f} mm "
                f"[src: {src}]",
                flush=True,
            )

    def obs_ee_to_policy_frame(self, ee_W14):
        """Env WORLD-frame ee (14-D right-first [R xyz, R quat_wxyz, L xyz, L
        quat_wxyz]) -> the frame the policy was trained on.
        astribot camera_ee: W -> chassis-local L -> head-cam optical (xyzw).
        astribot robot_ee : W -> chassis-local L (wxyz).
        franka  robot_ee  : W -> B (live "base" body), wxyz.
        franka  camera_ee : W -> B -> head-cam optical (xyzw).
        joint / passthrough cases: n/a (state_arm path)."""
        ee = np.asarray(ee_W14, dtype=np.float64)
        if self.mode == "joint":
            return ee
        if self._astribot_ik:
            from bidex.ik.retarget.astribot.astribot_camera_frame import ee_L_to_camera, qmul, qinv, wxyz_to_xyzw
            qch_inv = qinv(self._qch)
            out = np.empty(14, dtype=np.float64)
            for s0 in (0, 7):
                pW = ee[s0:s0 + 3]; qW = ee[s0 + 3:s0 + 7]  # wxyz (env obs)
                pL = self._Rch.T @ (pW - self._tch)        # W -> L
                qL = qmul(qch_inv, qW)                     # wxyz
                if self.mode == "camera_ee":
                    out[s0:s0 + 7] = ee_L_to_camera(np.concatenate([pL, wxyz_to_xyzw(qL)]))
                else:  # robot_ee -> base (L) frame, wxyz
                    out[s0:s0 + 3] = pL; out[s0 + 3:s0 + 7] = qL
            return out
        # franka: W -> B (live "base" body), then B -> camera if camera_ee (matches
        # the deterministic FK rebuild: ee_B = Rb^T (xpos[link8]-xpos[base])).
        from bidex.ik.retarget.franka import franka_camera_frame as fcf
        qb_inv = fcf.qinv(self._qb)
        out = np.empty(14, dtype=np.float64)
        for s0 in (0, 7):
            pW = ee[s0:s0 + 3]; qW = ee[s0 + 3:s0 + 7]  # wxyz
            pB = self._Rb.T @ (pW - self._tb)                       # W -> B
            qB = fcf.qmul(qb_inv, qW)                               # wxyz
            if self.mode == "camera_ee":
                seg = fcf.ee_B_to_camera(np.concatenate([pB, fcf.wxyz_to_xyzw(qB)]))  # xyzw
                out[s0:s0 + 7] = seg
            else:  # robot_ee -> B frame; keep wxyz
                out[s0:s0 + 3] = pB; out[s0 + 3:s0 + 7] = qB
        return out

    def to_env_action(self, model_action, dex, env):
        """model_action: 14-D right-first in the TRAINED control mode's frame.
        Returns the env JOINT action dict {arm_joint_position, dex_joint_position}
        (ee/camera_ee are IK'd by the adapter, NOT env EE mode — matches replay)."""
        model_action = np.asarray(model_action, dtype=np.float64)
        dex = np.asarray(dex, dtype=np.float64)
        if self.mode == "joint":
            return self._apply_hold({"arm_joint_position": model_action,
                                     "dex_joint_position": dex})
        if self._astribot_ik:
            from bidex.ik.retarget.astribot.astribot_camera_frame import xyzw_to_wxyz, qmul
            eeW14 = np.empty(14, dtype=np.float64)
            for s0 in (0, 7):
                ee = model_action[s0:s0 + 7]
                if self.mode == "camera_ee":
                    ee = self._ee_camera_to_L(ee)  # -> chassis-local L (xyzw)
                eeW14[s0:s0 + 3] = self._Rch @ ee[:3] + self._tch          # L -> W
                eeW14[s0 + 3:s0 + 7] = qmul(self._qch, xyzw_to_wxyz(ee[3:7]))  # wxyz
            q14 = self._solve(eeW14, self._q_prev); self._q_prev = q14
            arm = np.concatenate([q14[7:14], q14[0:7]]).astype(np.float64)  # [R7,L7]
            return self._apply_hold({"arm_joint_position": arm, "dex_joint_position": dex})
        # franka ee / camera_ee: policy ee -> B (camera_ee via ee_camera_to_B) ->
        # FrankaEEPipeline.ee_to_joints (T_coll fit per task) -> q7 per arm.
        # Mirrors franka_replay.py: camera_ee rebases to B first, then same IK.
        if self._Tcoll is None:
            raise RuntimeError("franka T_coll not set for this task "
                               "(call set_franka_task_tcoll before stepping)")
        qR = np.empty(7); qL = np.empty(7)
        for s, (a0, a1) in self._arm_slices.items():
            s0 = 0 if s == "right" else 7
            seg = model_action[s0:s0 + 7]
            if self.mode == "camera_ee":
                eeB_xyzw = self._ee_camera_to_B(seg)        # camera optical -> B
            else:  # robot_ee: policy frame is already B (wxyz) -> xyzw for the pipe
                eeB_xyzw = np.concatenate([seg[:3], self._fcf.wxyz_to_xyzw(seg[3:7])])
            q7, ok = self._pipe.ee_to_joints(eeB_xyzw, self._Tcoll[s], self._q_prev[s])
            self._q_prev[s] = q7
            (qR if s == "right" else qL)[:] = q7
        arm = np.concatenate([qR, qL]).astype(np.float64)  # right-first [R7,L7]
        return self._apply_hold({"arm_joint_position": arm, "dex_joint_position": dex})

    def _apply_hold(self, action):
        """Pin the static arm+hand (joint action). Right-first packing:
        arm[0:7]/dex[0:22]=right, arm[7:14]/dex[22:44]=left."""
        h = self.hold
        if not h or h.get("active") in (None, "dual"):
            return action
        arm = np.array(action["arm_joint_position"], dtype=np.float64)
        dex = np.array(action["dex_joint_position"], dtype=np.float64)
        if h["active"] == "left":      # right arm+hand static
            arm[0:7] = h["arm14"][0:7]; dex[0:22] = h["dex44"][0:22]
        else:                          # left arm+hand static
            arm[7:14] = h["arm14"][7:14]; dex[22:44] = h["dex44"][22:44]
        action["arm_joint_position"] = arm; action["dex_joint_position"] = dex
        return action

# --------------------------------------------------------------------------- #
# policy backends — ported from the proven reference scripts                  #
# --------------------------------------------------------------------------- #


def _openpi_obs_to_input(o, mode, adapter, single=False, active=None):
    def img(cam):
        return np.asarray(o["images"][cam], dtype=np.uint8).transpose(2, 0, 1)  # HWC->CHW
    d = {
        "images": {
            "cam_high": img("head"),
            "cam_left_wrist": img("left_wrist"),
            "cam_right_wrist": img("right_wrist"),
        },
        "prompt": str(o.get("language", "")),
    }
    if mode == "joint":
        d["state_arm"] = np.asarray(o["arm_joint_position"], dtype=np.float32)
        d["state_dex"] = np.asarray(o["dex_joint_position"], dtype=np.float32)
    else:  # robot_ee / camera_ee: state_ee in the policy's TRAINED frame
        d["state_ee"] = adapter.obs_ee_to_policy_frame(o["abs_end_effector"]).astype(np.float32)
        d["state_dex"] = np.asarray(o["dex_joint_position"], dtype=np.float32)
    # Articulated single-arm (29D) policy: the Inputs transform slices state 14->7 / 44->22
    # by `arm` AND suffixes the prompt with "(<arm> arm)". Without d["arm"] it defaults to
    # "right" -> left-arm tasks get the wrong state slice + wrong prompt conditioning.
    if single and active in ("left", "right"):
        d["arm"] = active
    return d


def _openpi_action_keys(mode):
    return ("actions_ee", "actions_dex") if mode != "joint" else ("actions_arm", "actions_dex")


def _openpi_action_from_chunk(res, idx, mode, single=False, active=None, adapter=None):
    arm_key, dex_key = _openpi_action_keys(mode)
    arm = np.asarray(res[arm_key])[idx]    # [14] dual / [7] single-arm (active arm)
    dex = np.asarray(res[dex_key])[idx]    # [44] dual / [22] single-arm (active hand)
    if not (single and active in ("left", "right")):
        return arm, dex
    # single-arm: route the 7-D active-arm + 22-D active-hand output into 14/44 (right-first
    # [R7,L7]/[R22,L22]); the static slot is the held dataset-default so to_env_action's
    # _apply_hold keeps it pinned while the active slot carries the policy output.
    h = adapter.hold or {}
    arm14 = np.array(h.get("arm14", np.zeros(14)), dtype=np.float32).copy()
    dex44 = np.array(h.get("dex44", np.zeros(44)), dtype=np.float32).copy()
    if active == "right":
        arm14[0:7] = arm; dex44[0:22] = dex
    else:  # left
        arm14[7:14] = arm; dex44[22:44] = dex
    return arm14, dex44


def _openpi_chunk_length(chunk, mode):
    arm_key, _ = _openpi_action_keys(mode)
    return int(np.asarray(chunk[arm_key]).shape[0])


class _RobustOpenPIClient:
    """openpi policy client over raw websockets with ping_interval=None.

    The stock openpi_client.WebsocketClientPolicy uses the websockets default
    ping_interval=20s; under 7-worker load the server's asyncio loop is blocked
    (sync policy.infer) and misses pongs -> clients' connections time out and
    client.infer hangs forever on recv. ping_interval=None disables client pings
    so the connection survives server-side blocking bursts. Lazy-connect +
    reconnect-on-error self-heal if the socket does drop. Protocol matches the
    openpi WebsocketPolicyServer (metadata handshake, then msgpack obs->action)."""

    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._packer = None
        self._unpack = None

    def _connect(self):
        if self._ws is not None:
            return
        import websockets.sync.client as _wsc
        from openpi_client import msgpack_numpy as _mn
        self._packer = _mn.Packer()
        self._unpack = _mn.unpackb
        self._ws = _wsc.connect(self._uri, compression=None, max_size=None, ping_interval=None)
        try:
            self._unpack(self._ws.recv(timeout=None))  # server metadata handshake
        except Exception:
            pass

    def _close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def infer(self, obs):
        for _ in range(3):
            try:
                self._connect()
                self._ws.send(self._packer.pack(obs))
                resp = self._ws.recv(timeout=None)
                if isinstance(resp, str):
                    raise RuntimeError(f"server error: {resp[:200]}")
                return self._unpack(resp)
            except Exception as e:
                print(f"[openpi client] infer failed ({e}); reconnecting...", flush=True)
                self._close()
        raise RuntimeError("openpi server infer failed after 3 reconnects")

# --------------------------------------------------------------------------- #
# multi-GPU launcher + leaderboard JSON deposit                               #
# --------------------------------------------------------------------------- #


def _write_leaderboard_json(results_dir, model_id, benchmark, robot, mode, domain, payload):
    p = Path(results_dir) / model_id / f"{benchmark}_{robot}_{mode}_{domain}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(p, payload)
    return p


def _atomic_write_json(path, obj):
    """Write JSON atomically (temp + rename) so concurrent readers (the live
    leaderboard server, which re-reads results on every request/poll) never see a
    half-written file. This is what makes per-task scores + videos appear in real
    time as each task's shard is flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, default=str)
    # The progress thread and the main thread can write the same
    # path concurrently; if they share one .tmp, one os.replace can steal the
    # other's tmp -> FileNotFoundError, aborting the run and skipping the
    # whole domain (losing data). Retry: rebuild the tmp and replace, up to 5
    # times; on failure fall back to a direct write.
    for _ in range(5):
        try:
            tmp.write_text(data)
            os.replace(tmp, path)
            return
        except FileNotFoundError:
            continue
    try:
        path.write_text(data)
    except Exception:
        pass


def _merge_shards(out_dir, n_ranks):
    """Merge whatever shard_rank{r}.json files exist so far into one per-task dict.
    Ranks write their shard atomically after each completed task, so this returns
    the set of tasks finished so far (incremental). Unreadable/missing shards are
    skipped (a rank mid-write to .tmp is simply not visible until os.replace)."""
    out = Path(out_dir)
    merged = {}
    for rank in range(n_ranks):
        sp = out / f"shard_rank{rank}.json"
        if sp.exists():
            try:
                merged.update(json.load(open(sp)))
            except Exception:
                pass  # mid-write or corrupt -> skip this tick; next tick picks it up
    return merged


def _per_seed_summary(per_ep, seeds):
    """Per-seed {n, mean_rate, success_rate} computed from whatever episodes are
    in `per_ep` so far (used for live per-episode flushes, where a seed may be
    mid-way through its eps)."""
    out = {}
    for sd in seeds:
        es = [e for e in per_ep if int(e.get("seed", -1)) == int(sd)]
        if es:
            out[str(int(sd))] = {
                "n": len(es),
                "mean_rate": (sum(e["score"] for e in es) / len(es) / (es[0]["max_score"] or 1)),
                "success_rate": sum(1 for e in es if e["success"]) / len(es)}
    return out


def _build_overview(model, ckpt, benchmark, robot, control_mode, domain, seeds,
                    eps_per_seed, gpu_ids, arm, all_results):
    mean_rate = float(np.mean([v.get("mean_rate", 0) for v in all_results.values()])) if all_results else 0.0
    return {"benchmark": benchmark, "robot": robot, "control_mode": control_mode,
            "domain": domain, "seeds": list(map(int, seeds)), "model": model, "ckpt": ckpt,
            "episodes_per_seed": eps_per_seed, "n_gpus": len(gpu_ids), "arm": arm,
            "n_tasks": len(all_results), "mean_rate": mean_rate,
            "success_rate": float(np.mean([v.get("success_rate", 0) for v in all_results.values()])) if all_results else 0.0,
            "per_task": all_results}


def _summary_payload(overview):
    """Return the human-facing summary schema with normalized `score` fields.

    Evaluation result JSONs historically call the normalized score `mean_rate`.
    Keep those files backward-compatible, but do not copy that alias into
    summary.json. Episode-level `score` values are raw subtask points and remain
    untouched; only aggregate `mean_rate` fields are renamed.
    """
    summary = dict(overview)
    if "mean_rate" in summary:
        summary["score"] = summary.pop("mean_rate")
    per_task = {}
    for task_id, task_result in summary.get("per_task", {}).items():
        task_summary = dict(task_result)
        if "mean_rate" in task_summary:
            task_summary["score"] = task_summary.pop("mean_rate")
        if "per_seed" in task_summary:
            task_summary["per_seed"] = {
                seed: ({**seed_result, "score": seed_result.get("mean_rate", 0.0)}
                       if "mean_rate" in seed_result else dict(seed_result))
                for seed, seed_result in task_summary["per_seed"].items()
            }
            for seed_result in task_summary["per_seed"].values():
                seed_result.pop("mean_rate", None)
        per_task[task_id] = task_summary
    summary["per_task"] = per_task
    return summary


class MissingCheckpoint(Exception):
    """Raised when --ckpt does not resolve to any loadable checkpoint dir."""


def _ckpt_has_weights(p) -> bool:
    """Heuristic: does directory `p` look like a loadable model checkpoint?

    Covers Isaac-GR00T (model.safetensors / sharded model-* + config.json),
    openpi/jax (params/ dir) and Being-H (model.safetensors + config.json).
    """
    p = Path(p)
    if not p.is_dir():
        return False
    if (p / "model.safetensors").exists():
        return True
    if (p / "model.safetensors.index.json").exists():
        return True
    if (p / "params").is_dir():
        return True
    if (p / "config.json").exists() and any(p.glob("model-*.safetensors")):
        return True
    return False


def resolve_checkpoint(ckpt: str) -> str:
    """Resolve ``--ckpt`` to a concrete, loadable checkpoint directory.

    Accepts either a finished checkpoint dir (returned unchanged) or a
    training-output *base* dir; in the latter case the latest step subdir is
    selected, supporting both Isaac-GR00T HF naming (``checkpoint-<step>``) and
    OpenPI bare-number naming (``<step>``, e.g. ``9999`` / ``0010000``).
    Raises :class:`MissingCheckpoint` when nothing loadable is found, so the
    caller can skip this combo cleanly (mirrors the old per-yml MISSING guard).
    """
    if not ckpt:
        raise MissingCheckpoint("empty --ckpt")
    p = Path(ckpt)
    if not p.exists():
        raise MissingCheckpoint(f"--ckpt not found: {ckpt}")
    if not p.is_dir():
        return ckpt  # leave to the backend; it reports a clearer error
    # 1) latest checkpoint-<step> subdir (Isaac-GR00T / HF transformers).
    ck_sub = sorted(
        (d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
         and _ckpt_has_weights(d)),
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
    )
    if ck_sub:
        return str(ck_sub[-1])
    # 2) latest bare-<step> subdir (OpenPI, e.g. 9999 / 0010000).
    num_sub = sorted(
        (d for d in p.iterdir() if d.is_dir() and d.name.isdigit() and _ckpt_has_weights(d)),
        key=lambda d: int(d.name),
    )
    if num_sub:
        return str(num_sub[-1])
    # 2.5) lingbot DCP convention: weights are safetensors in a `hf_ckpt/` subdir of a
    #      `global_step_<N>` dir. Handles both --ckpt = <base>/ (contains global_step_N/)
    #      and --ckpt = <base>/global_step_N (weights in its hf_ckpt/).
    def _hf_ckpt(d):
        h = d / "hf_ckpt"
        return str(h) if h.is_dir() and _ckpt_has_weights(h) else None
    gs_sub = sorted(
        (d for d in p.iterdir() if d.is_dir() and d.name.startswith("global_step_")
         and d.name.split("_")[-1].isdigit() and _hf_ckpt(d)),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if gs_sub:
        return _hf_ckpt(gs_sub[-1])
    _h = _hf_ckpt(p)
    if _h:
        return _h
    # 2.6) xr1 / Xiaomi-Robotics-1 save a Lightning+DeepSpeed run dir: config.py +
    #      last.ckpt/checkpoint/mp_rank_00_model_states.pt (no safetensors at the top
    #      level). The _build_xr1_infer loader reads that layout directly.
    if (p / "config.py").exists() and (p / "last.ckpt").is_dir():
        return str(p)
    # 2.7) GalaxeaVLA G0.5: a hydra run dir — .hydra/config.yaml + last.pt (symlink to
    #      checkpoints/step_<N>.pt) + dataset_stats.json + checkpoints/step_*.pt (no
    #      safetensors, only .pt). _ckpt_has_weights only sees safetensors/params, so
    #      without this branch the run dir is wrongly rejected as MissingCheckpoint
    #      before _build_galaxea_infer can glob checkpoints/step_*.pt itself. Pass the
    #      run dir through unchanged (local_eval.sh + the cluster eval yml both hand
    #      GALAXEA_CKPT=<run_dir>, no single-.pt pinning). Mirrors the xr1 2.6 branch.
    if (p / ".hydra" / "config.yaml").exists() and (p / "last.pt").exists():
        return str(p)
    # 2.8) xr1 Lightning epoch snapshot: `epoch=N-step=M.ckpt/` is a DIRECTORY
    #      holding DeepSpeed `checkpoint/mp_rank_00_model_states.pt` (the run dir's
    #      last.ckpt/ is the same snapshot). Passed through unchanged — the xr1
    #      infer builders take either the run dir or the epoch dir (config.py is
    #      discovered in the parent, weights under <dir>/checkpoint/).
    if (p.name.startswith("epoch=") and p.name.endswith(".ckpt")
            and (p / "checkpoint" / "mp_rank_00_model_states.pt").exists()):
        return str(p)
    # 2.9) InternVLA-A1.5 saves `checkpoints/<step:06d>/pretrained_model/`
    #      (config.json + model.safetensors + stats.json). A step dir (or any dir
    #      in that layout) resolves to its pretrained_model/ — case 3 below then
    #      accepts it directly when the pretrained_model dir itself is passed.
    if (p / "pretrained_model" / "model.safetensors").exists():
        return str(p / "pretrained_model")
    # 3) the dir itself is the checkpoint.
    if _ckpt_has_weights(p):
        return str(p)
    raise MissingCheckpoint(f"no loadable checkpoint under --ckpt {ckpt}")

# --------------------------------------------------------------------------- #
# vectorized eval: one render-only worker per GPU and N envs per worker.       #
# Every model runs in a separate policy-server process.                       #
# --------------------------------------------------------------------------- #


class _ChunkReusedVecInfer:
    """Per-env action-chunk reuse wrapper for a policy-server client.

    model_forward(sub_obs, sub_ad) -> list of (arm_chunk(H',14), dex_chunk(H',44)) per env,
    the FULL predicted chunk (H' = the model's horizon). The wrapper caches the chunk per env,
    re-queries ONLY the envs whose chunk is exhausted (a subset forward -> amortization: one
    forward per H steps per env, not per step), and exposes reset_env (drop the chunk at episode
    start, so the env re-queries) + query_due (drives render-skip in VectorRolloutManager).

    `reuse` defaults to the full returned chunk. ``BIDEX_VEC_CHUNK_REUSE`` may
    cap it for an explicit ablation."""
    def __init__(self, model_forward, reuse, always_full=False, label="policy"):
        self._fwd = model_forward
        self.H = max(1, int(reuse))
        self._label = str(label)
        self._reported_horizon = False
        # always_full=True: ALWAYS query the FULL obs_list (constant batch B=N) and
        # store only envs whose chunk is due. For backends that CANNOT batch a
        # variable B (lingbot caches per-batch image split_sizes — any B != the
        # first forward's B crashes torch.split_with_sizes), this pins every server
        # query at constant B=N (= the first forward's B that set the cache) so it
        # is safe across aligned chunk re-queries AND ragged per-env resets.
        self._always_full = bool(always_full)
        self._arm = []   # per-env cached chunk (...,14)
        self._dex = []   # per-env cached chunk (...,44)
        self._ci = []    # per-env step cursor (ci==H -> exhausted -> due)

    def _grow(self, n):
        while len(self._arm) < n:
            self._arm.append(None); self._dex.append(None); self._ci.append(self.H)

    def __call__(self, obs_list, adapter_list):
        n = len(obs_list); self._grow(n)
        due = [i for i in range(n) if self._arm[i] is None or self._ci[i] >= self.H]
        if due:
            if self._always_full:
                # Constant-batch path: query the FULL obs_list (B=N, every env incl.
                # idle dummies) and store ONLY due envs' chunks; non-due envs' freshly
                # computed chunks are discarded. Keeps the server's per-batch split
                # cache (lingbot) valid — the B it sees every query is the SAME B it
                # cached on the first forward.
                res = self._fwd(obs_list, adapter_list)   # N chunks, indexed by env idx
                for i in due:
                    self._arm[i] = np.asarray(res[i][0], dtype=np.float32)
                    self._dex[i] = np.asarray(res[i][1], dtype=np.float32)
                    self._ci[i] = 0
                    _clen = int(self._arm[i].shape[0])
                    if _clen and _clen < self.H:
                        self.H = _clen
                    if not self._reported_horizon:
                        print(f"[chunk] {self._label}: native={_clen} effective={self.H}", flush=True)
                        self._reported_horizon = True
            else:
                sub_o = [obs_list[i] for i in due]; sub_a = [adapter_list[i] for i in due]
                res = self._fwd(sub_o, sub_a)            # list of (arm_chunk(H,14), dex_chunk(H,44))
                for k, i in enumerate(due):
                    self._arm[i] = np.asarray(res[k][0], dtype=np.float32)
                    self._dex[i] = np.asarray(res[k][1], dtype=np.float32)
                    self._ci[i] = 0
                    # Clamp the reuse horizon to the ACTUAL chunk length the backend
                    # returned. The galaxea server-client builder sets reuse=BIDEX_VEC
                    # _CHUNK_REUSE (default 50) WITHOUT capping at the model's action
                    # horizon (galaxea G0.5 action_size=32); the server caps its chunk
                    # at 32 and ships a (n,32,14) tensor, so with H=50 the per-env
                    # cursor would reach 32 and `self._arm[i][32]` below would IndexError
                    # at ~step 32 (t≈1.3s). All chunks share one length, so a single
                    # global clamp is exact because every response in the batch
                    # shares the same action horizon.
                    _clen = int(self._arm[i].shape[0])
                    if _clen and _clen < self.H:
                        self.H = _clen
                    if not self._reported_horizon:
                        print(f"[chunk] {self._label}: native={_clen} effective={self.H}", flush=True)
                        self._reported_horizon = True
        arms = np.zeros((n, 14), dtype=np.float32)
        dexs = np.zeros((n, 44), dtype=np.float32)
        for i in range(n):
            arms[i] = self._arm[i][self._ci[i], :14]
            dexs[i] = self._dex[i][self._ci[i], :44]
            self._ci[i] += 1
        return arms, dexs

    def query_due(self, i):
        if i >= len(self._arm):
            return True
        return self._arm[i] is None or self._ci[i] >= self.H

    def reset_env(self, i):
        self._grow(i + 1)
        self._arm[i] = None; self._dex[i] = None; self._ci[i] = self.H


def _repack_protocol_obs(sub_obs):
    """env obs -> the protocol obs dict the SimpleBatchedServer backends expect
    (same shape lingbot/galaxea clients send)."""
    out = []
    for o in sub_obs:
        img = o["images"]
        out.append({
            "images": {k: np.ascontiguousarray(np.asarray(v, dtype=np.uint8))
                       for k, v in img.items()},
            "arm_joint_position": np.asarray(o["arm_joint_position"], dtype=np.float32).reshape(-1),
            "dex_joint_position": np.asarray(o["dex_joint_position"], dtype=np.float32).reshape(-1),
            "language": str(o.get("language", "")),
        })
    return out


class _PolicyServerClient:
    """Reconnect-capable client for the shared batched policy protocol."""

    def __init__(self, name, host, port, robot=None):
        self.name = name
        self.robot = robot
        self.uri = f"ws://{host}:{port}"
        self._ws = None
        try:
            import websockets.sync.client as websocket_client
        except Exception as exc:
            raise RuntimeError(
                f"{name} client needs websockets + msgpack: {exc}"
            ) from exc
        self._websocket_client = websocket_client

    def _connect(self):
        if self._ws is not None:
            return
        for _ in range(120):
            try:
                self._ws = self._websocket_client.connect(
                    self.uri, compression=None, max_size=None, ping_interval=None
                )
                unpackb(self._ws.recv(timeout=None))
                return
            except Exception:
                time.sleep(1.0)
        raise RuntimeError(f"{self.name} server {self.uri} not reachable after 120s")

    def infer(self, obs_list):
        self._connect()
        request = {"obs": obs_list}
        if self.robot is not None:
            request["robot"] = self.robot
        for _ in range(3):
            try:
                self._ws.send(packb(request))
                response = self._ws.recv(timeout=None)
                if isinstance(response, str):
                    raise RuntimeError(
                        f"{self.name} server error: {response[:200]}"
                    )
                result = unpackb(response)
                arms = np.asarray(result["arm"], dtype=np.float32)
                dexs = np.asarray(result["dex"], dtype=np.float32)
                return [(arms[i], dexs[i]) for i in range(arms.shape[0])]
            except Exception as exc:
                print(
                    f"[{self.name}-client] infer failed ({exc}); reconnecting...",
                    flush=True,
                )
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
                self._connect()
        raise RuntimeError(f"{self.name} server infer failed after 3 reconnects")


def _policy_server_address(name, default_port):
    prefix = name.upper()
    host = os.environ.get(
        f"BIDEX_{prefix}_SERVER_HOST", "127.0.0.1"
    ).strip() or "127.0.0.1"
    port = int(
        os.environ.get(f"BIDEX_{prefix}_SERVER_PORT", str(default_port)).strip()
        or default_port
    )
    return host, port


def _build_protocol_client_batched_infer(
    name, default_port, *, robot=None, always_full=False
):
    host, port = _policy_server_address(name, default_port)
    client = _PolicyServerClient(name, host, port, robot=robot)

    def model_forward(sub_obs, _sub_adapters):
        return client.infer(_repack_protocol_obs(sub_obs))

    infer = _ChunkReusedVecInfer(
        model_forward,
        _configured_chunk_reuse(),
        always_full=always_full,
        label=name,
    )
    return infer, client


def _build_galaxea_client_batched_infer():
    return _build_protocol_client_batched_infer("galaxea", 8766)


def _build_lingbot_client_batched_infer():
    # LingBot caches its first batch's image split sizes and therefore requires
    # a constant batch size across requests.
    return _build_protocol_client_batched_infer(
        "lingbot", 8767, always_full=True
    )


def _build_gr00t_client_batched_infer(robot="franka"):
    return _build_protocol_client_batched_infer("gr00t", 8768, robot=robot)


def _build_xr1_client_batched_infer(robot="franka"):
    return _build_protocol_client_batched_infer("xr1", 8769, robot=robot)


def _build_internvla_client_batched_infer(robot="franka"):
    return _build_protocol_client_batched_infer("internvla", 8770, robot=robot)


def _build_dm05_client_batched_infer(robot="franka"):
    return _build_protocol_client_batched_infer("dm05", 8771, robot=robot)

# Multi-model-GPU server port helpers: run_vec_models_eval.sh starts ONE policy
# server per MODEL_GPUS entry (ports base, base+1, ...) and exports the list in
# BIDEX_<MODEL>_SERVER_PORTS / OPENPI_SERVER_PORTS. launch_vector spreads the
# render workers across those servers round-robin (worker rank r -> server
# r % n_servers) so the sim eval load is allocated by the model-GPU count.


def _model_server_port_envs(model):
    """(port_env_var, ports_env_var) for a server-client model; (None, None)
    for models without a policy-server protocol."""
    if model == "openpi":
        return "OPENPI_SERVER_PORT", "OPENPI_SERVER_PORTS"
    _prefix = {"galaxea": "GALAXEA", "lingbot": "LINGBOT", "groot": "GR00T",
               "xr1": "XR1", "internvla": "INTERNVLA", "dm05": "DM05"}
    if model in _prefix:
        return f"BIDEX_{_prefix[model]}_SERVER_PORT", f"BIDEX_{_prefix[model]}_SERVER_PORTS"
    return None, None


def _server_port_list(model):
    """Server ports for `model` from BIDEX_<PREFIX>_SERVER_PORTS (list) or the
    single *_SERVER_PORT env (backward compat). Empty when not in server mode."""
    port_var, ports_var = _model_server_port_envs(model)
    if not port_var:
        return []
    ports = [p.strip() for p in (os.environ.get(ports_var, "") or "").split(",") if p.strip()]
    if not ports:
        single = os.environ.get(port_var, "").strip()
        if single:
            ports = [single]
    return ports


class _OpenPIVecInfer:
    """Vectorized OpenPI inference through a shared policy server.

    The worker loads no model or JAX runtime; it renders and queries the server.

    Uses `_RobustOpenPIClient` (ping_interval=None, lazy-connect, self-heal) rather
    than the stock `WebsocketClientPolicy`: the openpi server runs `policy.infer`
    SYNCHRONOUSLY in its asyncio loop, so under N-worker load the loop is
    intermittently blocked and the stock client's default 20s pings go unanswered →
    connection closes → first/next infer hangs forever on recv (the documented
    vectorized first-infer-storm deadlock). ping_interval=None survives the
    blocking bursts while preserving reconnect behavior.

    Per-env action-chunk reuse (reuse=50 = pi05 action_horizon): each env re-queries
    the server only every `reuse` control steps, amortizing the server's serial
    infer. `reset_env(i)` is called by VectorRolloutManager at episode boundaries
    so a stale chunk is never carried into a new episode."""

    def __init__(self, control_mode, reuse=FULL_CHUNK_REUSE):
        self._host = os.environ.get("OPENPI_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self._port = int(os.environ.get("OPENPI_SERVER_PORT", "0").strip() or "0")
        assert self._port, "OPENPI_SERVER_PORT must be set for vectorized openpi server-client mode"
        # Lazy-connect on first infer (env build + black-frame recreate can take >40s;
        # connecting earlier then waiting risks a stale/dropped socket).
        self.client = _RobustOpenPIClient(self._host, self._port)
        self.mode = control_mode
        self.reuse = reuse
        self._chunks = []
        self._cis = []
        self._reported_horizon = False

    def _infer(self, obs_in):
        """client.infer with lazy-connect + self-heal (the robust client retries 3×
        with reconnect internally; raise so VectorRolloutManager broadcasts zeros)."""
        return self.client.infer(obs_in)

    def _ensure(self, n):
        while len(self._chunks) < n:
            self._chunks.append(None)
            self._cis.append(0)

    def reset_env(self, i):
        self._ensure(i + 1)
        self._chunks[i] = None
        self._cis[i] = 0

    def _horizon(self, i):
        if self._chunks[i] is None:
            return 0
        native = _openpi_chunk_length(self._chunks[i], self.mode)
        effective = min(self.reuse, native)
        if not self._reported_horizon:
            print(f"[chunk] openpi: native={native} effective={effective}", flush=True)
            self._reported_horizon = True
        return effective

    def __call__(self, obs_list, adapter_list):
        n = len(obs_list)
        self._ensure(n)
        arms = np.zeros((n, 14), dtype=np.float32)
        dexs = np.zeros((n, 44), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            ad = adapter_list[i]
            # Idle envs (dummy all-zero obs from VectorRolloutManager) never reach the
            # server: their action is discarded (is_real=False) so just return zeros and
            # leave the chunk untouched so a real episode triggers a fresh infer.
            head = obs.get("images", {}).get("head") if isinstance(obs, dict) else None
            if head is None or float(np.asarray(head).mean()) < 1.0:
                continue
            h = getattr(ad, "hold", None)
            active = h.get("active") if h else None
            single = active in ("left", "right")
            if self._chunks[i] is None or self._cis[i] >= self._horizon(i):
                self._chunks[i] = self._infer(
                    _openpi_obs_to_input(obs, self.mode, ad, single, active))
                self._cis[i] = 0
            arm, dex = _openpi_action_from_chunk(self._chunks[i], self._cis[i],
                                                 self.mode, single, active, ad)
            arms[i] = np.asarray(arm, dtype=np.float32).reshape(14)
            dexs[i] = np.asarray(dex, dtype=np.float32).reshape(44)
            self._cis[i] += 1
        return arms, dexs

    def query_due(self, i):
        """True iff the NEXT ``__call__`` will re-query env i (no cached chunk yet,
        or the chunk index reached reuse). Called by VectorRolloutManager AFTER
        ``__call__`` (so ``_chunks[i]/_cis[i]`` are post-call) to decide which steps
        to render (render-skip): render exactly the steps whose obs the next query
        will consume. Idle/unstarted envs (index past the filled prefix) -> due."""
        self._ensure(i + 1)
        return self._chunks[i] is None or self._cis[i] >= self._horizon(i)


def _build_openpi_client_batched_infer(control_mode):
    """Return an `_OpenPIVecInfer` callable for VectorRolloutManager. Model lives
    on a separately started policy server; this worker only renders and queries
    it. The model's returned chunk length is the default horizon."""
    reuse = _configured_chunk_reuse()
    infer = _OpenPIVecInfer(control_mode, reuse=reuse)
    return infer, infer.client


def _build_openpi_concurrent_infer(control_mode, n_workers=None):
    """Concurrent server-client infer: fire all DUE envs' infers CONCURRENTLY
    (each on its own websocket connection via a thread pool) so a dynamic-batching
    server (`scripts/evaluation/serve_policy_batched.py`) can accumulate them into one batched
    forward. Contrast with `_OpenPIVecInfer` which sends envs SERIALLY over one
    connection → server sees ~1 request at a time → no batching.

    Per-env chunk reuse (reuse=50) and idle/black-skip are identical to
    `_OpenPIVecInfer`; only the DUE envs' infers are parallelized. Results are
    collected in env-index order (1-to-1). Gated by `BIDEX_VEC_CONCURRENT=1` and
    `BIDEX_VEC_CONCURRENCY` (pool/threads, default 32)."""
    from concurrent.futures import ThreadPoolExecutor

    class _Conc:
        def __init__(self, mode, reuse, nw):
            self.mode = mode
            self.reuse = reuse
            host = os.environ.get("OPENPI_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
            port = int(os.environ.get("OPENPI_SERVER_PORT", "0").strip() or "0")
            assert port, "OPENPI_SERVER_PORT must be set"
            self._chunks = []
            self._cis = []
            self._nw = nw
            self._pool = [_RobustOpenPIClient(host, port) for _ in range(nw)]
            self._exec = ThreadPoolExecutor(max_workers=nw,
                                            thread_name_prefix="openpi-conc")

        def _ensure(self, n):
            while len(self._chunks) < n:
                self._chunks.append(None)
                self._cis.append(0)

        def reset_env(self, i):
            self._ensure(i + 1)
            self._chunks[i] = None
            self._cis[i] = 0

        def _horizon(self, i):
            if self._chunks[i] is None:
                return 0
            return min(self.reuse, _openpi_chunk_length(self._chunks[i], self.mode))

        def _one(self, j, obs, ad, single, active):
            return self._pool[j % self._nw].infer(
                _openpi_obs_to_input(obs, self.mode, ad, single, active))

        def __call__(self, obs_list, adapter_list):
            n = len(obs_list)
            self._ensure(n)
            arms = np.zeros((n, 14), dtype=np.float32)
            dexs = np.zeros((n, 44), dtype=np.float32)
            due = []
            for i, obs in enumerate(obs_list):
                head = obs.get("images", {}).get("head") if isinstance(obs, dict) else None
                if head is None or float(np.asarray(head).mean()) < 1.0:
                    continue
                if self._chunks[i] is None or self._cis[i] >= self._horizon(i):
                    due.append(i)
            if due:
                ad_meta = []
                for i in due:
                    ad = adapter_list[i]
                    h = getattr(ad, "hold", None)
                    active = h.get("active") if h else None
                    single = active in ("left", "right")
                    ad_meta.append((ad, single, active))
                futs = {i: self._exec.submit(self._one, j, obs_list[i], ad_meta[k][0],
                                             ad_meta[k][1], ad_meta[k][2])
                        for k, (j, i) in enumerate(enumerate(due))}
                for i in due:
                    self._chunks[i] = futs[i].result()
                    self._cis[i] = 0
            for i, obs in enumerate(obs_list):
                head = obs.get("images", {}).get("head") if isinstance(obs, dict) else None
                if head is None or float(np.asarray(head).mean()) < 1.0:
                    continue
                ad = adapter_list[i]
                h = getattr(ad, "hold", None)
                active = h.get("active") if h else None
                single = active in ("left", "right")
                arm, dex = _openpi_action_from_chunk(self._chunks[i], self._cis[i],
                                                     self.mode, single, active, ad)
                arms[i] = np.asarray(arm, dtype=np.float32).reshape(14)
                dexs[i] = np.asarray(dex, dtype=np.float32).reshape(44)
                self._cis[i] += 1
            return arms, dexs

        def query_due(self, i):
            self._ensure(i + 1)
            return self._chunks[i] is None or self._cis[i] >= self._horizon(i)

    reuse = _configured_chunk_reuse()
    nw = n_workers or int(os.environ.get("BIDEX_VEC_CONCURRENCY", "32"))
    infer = _Conc(control_mode, reuse, nw)
    return infer, infer._pool[0]


def _build_record_episodes(seeds, eps_per_seed, max_record_envs):
    """Which ``(seed, ep)`` episodes to record for one task, evenly across seeds.

    Replaces the old "record the first R *env indices*" scheme (which, because
    the episode queue is FIFO in seed-major order, biased almost every saved
    video onto seed0). The total is ~``max_record_envs`` episodes SPLIT EVENLY
    across the seeds this worker sees, taking the **first N episodes**
    (``ep 0..N-1``) of each seed — every seed is represented even when the budget
    is small (each seed gets >=1).

    ``max_record_envs < 0``  -> record EVERY episode (full coverage; the legacy
    "record all envs" meaning). ``max_record_envs == 0`` -> record nothing.
    """
    seeds = [int(s) for s in seeds]
    if max_record_envs < 0:
        return {(s, e) for s in seeds for e in range(eps_per_seed)}
    if max_record_envs == 0:
        return set()
    n = max(1, len(seeds))
    base, rem = divmod(max(1, max_record_envs), n)
    plan = set()
    for idx, s in enumerate(seeds):
        per = max(1, base + (1 if idx < rem else 0))   # >=1 per seed (split evenly)
        per = min(per, eps_per_seed)
        for e in range(per):
            plan.add((s, e))
    return plan


def eval_shard_vector(tasks, seeds, eps_per_seed, model, ckpt, robot, control_mode,
                      domain, out_file, rank, gpu, benchmark="Generalization",
                      record_video=True, model_id="", mp4_root=MP4_DIR,
                      ood_offset=0.05, arm="dual", envs_per_gpu=8,
                      max_record_envs=-1):
    """Run one render-only vector worker against a required policy server."""
    import queue as _queue
    from bidex.eval.vector_rollout import VectorRolloutManager

    assert model in ("lingbot", "openpi", "galaxea", "groot", "xr1", "internvla", "dm05"), \
        f"vectorized mode supports lingbot/openpi/galaxea/groot/xr1/internvla/dm05, got {model}"
    assert domain in ("in_domain", "out_of_domain"), f"vectorized domain must be concrete, got {domain}"
    native = env_native_mode(control_mode, robot)
    _clients = {
        "lingbot": lambda: _build_lingbot_client_batched_infer(),
        "galaxea": lambda: _build_galaxea_client_batched_infer(),
        "groot": lambda: _build_gr00t_client_batched_infer(robot),
        "xr1": lambda: _build_xr1_client_batched_infer(robot),
        "internvla": lambda: _build_internvla_client_batched_infer(robot),
        "dm05": lambda: _build_dm05_client_batched_infer(robot),
    }
    if model == "openpi":
        if int(os.environ.get("BIDEX_VEC_CONCURRENT", "0")) == 1:
            # concurrent client: fire due envs in parallel -> a dynamic-batching
            # server (scripts/evaluation/serve_policy_batched.py) can accumulate them into one
            # batched forward. Use with the batched server for throughput.
            infer_batch_fn, _server = _build_openpi_concurrent_infer(control_mode)
            model_loc = "concurrent server-client (batched policy server; worker renders only)"
        else:
            infer_batch_fn, _server = _build_openpi_client_batched_infer(control_mode)
            model_loc = "server-client (policy server; worker renders only)"
    else:
        ports = _server_port_list(model)
        if not ports:
            raise RuntimeError(f"{model} policy server is required for vectorized evaluation")
        infer_batch_fn, _server = _clients[model]()
        model_loc = f"server-client ({model} policy server port {ports[0]}; worker renders only)"
    print(f"[vec rank {rank} gpu{gpu}] {model_loc}; {len(tasks)} tasks, N_envs={envs_per_gpu}",
          flush=True)

    results = {}
    # Per-EPISODE recording plan (replaces the old "first R env indices" set that
    # biased every saved video onto seed0): ~max_record_envs episodes split evenly
    # across seeds, first N eps (ep 0..N-1) of each. <0 => record all episodes.
    record_episodes = _build_record_episodes(seeds, eps_per_seed, max_record_envs)
    setting = f"{benchmark}_{robot}_{control_mode}_{domain}"

    for tname in tasks:
        suite, task = tname.split("/")
        thash = stable_task_hash(suite, task)
        if domain == "in_domain":
            init_states = load_collection_init_states(suite, task)
            if not init_states.size:
                raise RuntimeError(
                    f"ID evaluation requires H5 robot init states: {suite}/{task}; "
                    "the H5 group must exactly match the benchmark YAML name"
                )
        else:
            # OOD always starts from the task's default robot pose.
            init_states = np.zeros((0, 0), dtype=np.float64)
        max_steps = max_steps_for_task(suite, task)
        t_arm = arm_for_task(robot, suite, task) if arm == "single_arm" else arm
        task_dir = f"{robot}_{suite}_{task}"
        mp4_path_fn = (lambda s, e, _td=task_dir:
                       os.path.join(mp4_root, model_id, setting, _td, f"s{s}_ep{e:03d}.mp4"))
        per_ep = []
        plans_by_episode = {}
        t0 = time.time()

        # ID envs share one compiled task and draw episodes from a common queue.
        # Every OOD episode has its own compiled appearance, so build a wave of
        # distinct envs and pin one episode to each env. The policy still sees a
        # true batch instead of the previous one-env-per-manager regression.
        work_units = []
        for seed in seeds:
            if domain == "out_of_domain":
                episode_ids = list(range(eps_per_seed))
                wave_size = max(1, int(envs_per_gpu))
                work_units.extend(
                    (int(seed), episode_ids[start:start + wave_size])
                    for start in range(0, len(episode_ids), wave_size)
                )
            else:
                work_units.append((int(seed), list(range(eps_per_seed))))

        for seed, unit_episodes in work_units:
            loaded_tasks = []
            if domain == "out_of_domain":
                for ep in unit_episodes:
                    loaded, _distractors, apply_offset = load_task_for_domain(
                        suite, task, domain, seed, thash, f"vec rank {rank}",
                        episode=int(ep))
                    plans_by_episode[f"{seed}/{int(ep)}"] = dict(loaded._ood_metadata)
                    loaded_tasks.append((loaded, bool(apply_offset)))
            else:
                loaded, _distractors, apply_offset = load_task_for_domain(
                    suite, task, domain, seed, thash, f"vec rank {rank}")
                plans_by_episode[f"{seed}/*"] = dict(loaded._ood_metadata)
                loaded_tasks.append((loaded, bool(apply_offset)))

            weights = subtask_weights(loaded_tasks[0][0], robot, suite, task)
            envs, adapters = [], []
            pool_size = (len(unit_episodes) if domain == "out_of_domain"
                         else min(envs_per_gpu, len(unit_episodes)))
            for env_idx in range(pool_size):
                construction_episode = (int(unit_episodes[env_idx])
                                        if domain == "out_of_domain" else 0)
                construction_seed = oodmod.episode_seed(
                    suite, task, seed, construction_episode, "environment")
                # Task carries mutable per-episode bookkeeping (button/order/
                # color-swap flags). Sharing one instance across vector envs
                # makes outcomes depend on thread scheduling, so every env gets
                # an independent copy. OOD also selects this env's appearance.
                source_task = (loaded_tasks[env_idx][0]
                               if domain == "out_of_domain" else loaded_tasks[0][0])
                env_task = copy.deepcopy(source_task)
                e = BiDexEnv(env_task, control_mode=native,
                             cameras=["head", "left_wrist", "right_wrist"],
                             seed=construction_seed)
                envs.append(e)
                ad = ControlAdapter(control_mode, robot, e)
                if control_mode != "joint" and robot == "franka":
                    ad.set_franka_task_tcoll(suite, task)
                adapters.append(ad)

            ep_queue = _queue.Queue()
            episode_assignments = None
            if domain == "out_of_domain":
                episode_assignments = [(seed, int(ep)) for ep in unit_episodes]
            else:
                for ep in unit_episodes:
                    ep_queue.put((seed, int(ep)))

            apply_ood_offset = any(flag for _loaded, flag in loaded_tasks)

            task_ctx = {
                "suite": suite, "task": task, "thash": thash,
                "init_states": init_states, "weights": weights,
                "max_steps": max_steps, "control_mode": control_mode,
                "robot": robot, "arm": arm, "t_arm": t_arm,
                "record_video": bool(record_video),
                "record_episodes": record_episodes,
                "mp4_path_fn": mp4_path_fn if record_video else None,
                "language": "", "rank": rank, "domain": domain,
                "episode_assignments": episode_assignments,
                "heterogeneous_models": domain == "out_of_domain",
                # Baked variants already carry their layout shift.
                "ood_offset": ood_offset if apply_ood_offset else 0.0,
                "settle_s": float(os.environ.get("BIDEX_SETTLE_S", "1.0")),
                "total_eps": len(unit_episodes),
            }
            appearance = (f" seed={seed} episodes={unit_episodes[0]}..{unit_episodes[-1]}"
                          if domain == "out_of_domain" else f" seed={seed}")
            print(f"[vec rank {rank}] {tname}{appearance}: {len(unit_episodes)} eps "
                  f"over {pool_size} envs (max_steps={max_steps})", flush=True)
            mgr = VectorRolloutManager(envs, adapters, infer_batch_fn, ep_queue, task_ctx)
            try:
                per_ep.extend(mgr.run())
            finally:
                for env_idx, e in enumerate(envs):
                    try:
                        loaded_for_env = (loaded_tasks[env_idx][0]
                                          if domain == "out_of_domain" else loaded_tasks[0][0])
                        _close_eval_env(e, loaded_for_env)
                    except Exception:
                        pass
        dt = time.time() - t0
        # aggregate (mirror eval_shard's result schema)
        per_seed_summary = {}
        for sd in seeds:
            es = [x for x in per_ep if int(x["seed"]) == int(sd)]
            per_seed_summary[str(int(sd))] = {
                "n": len(es),
                "mean_rate": (sum(x["score"] for x in es) / len(es) / (es[0]["max_score"] or 1)) if es else 0.0,
                "success_rate": (sum(1 for x in es if x["success"]) / len(es)) if es else 0.0,
            }
        agg = aggregate(per_ep) if per_ep else {"mean_rate": 0.0, "score": 0, "max_score": 1}
        agg["success_rate"] = (sum(1 for x in per_ep if x["success"]) / len(per_ep)) if per_ep else 0.0
        agg["max_steps"] = max_steps
        agg["control_mode"] = control_mode
        agg["seeds"] = list(map(int, seeds))
        agg["episodes_per_seed"] = eps_per_seed
        agg["per_seed"] = per_seed_summary
        agg["ood"] = _ood_result_metadata(domain, thash, ood_offset, plans_by_episode)
        results[f"{robot}/{suite}/{task}"] = agg
        try:
            (Path(out_file).parent / f"shard_rank{rank}.done").write_text(str(len(results)))
        except Exception:
            pass
        try:
            _atomic_write_json(Path(out_file), results)
        except Exception:
            pass
        print(f"[vec rank {rank}] DONE {tname}: mean_rate={agg['mean_rate']:.3f} "
              f"success_rate={agg['success_rate']:.3f} ({len(per_ep)} eps, {dt:.0f}s)", flush=True)
    _atomic_write_json(Path(out_file), results)
    print(f"[vec rank {rank}] wrote {out_file} ({len(results)} tasks)", flush=True)


def auto_envs_per_gpu(gid, per_env_mb=200, max_envs=None, min_envs=1):
    """Pick the concurrent-env count for one render GPU from its free VRAM.

    With the shared-Renderer architecture (one GL context per worker; see
    vector_rollout.py), each env no longer owns a ~300 MB EGL context — only the
    single shared renderer does (amortized), so the per-env envelope is roughly
    sim overhead (~150-200 MB). `nvidia-smi memory.free // per_env_mb`, clipped to
    [min_envs, max_envs] (max_envs=env BIDEX_MAX_ENVS_PER_GPU or 8).

    No CPU-core cap: the N-sweep (RENDERING_XID_MAX_ENVS.md §9) measured the
    throughput peak at N=32/card even though that exceeds ~18 true-parallel
    env-threads/worker (physics time-slices), because render-skip + server waits
    leave enough slack that oversubscription still wins on episode concurrency.
    So we honor BIDEX_MAX_ENVS_PER_GPU (the lever the user sets) instead of a
    (cpu_count-2)//n_workers cap that would have pinned N at 18 for 7 workers."""
    if max_envs is None:
        max_envs = int(os.environ.get("BIDEX_MAX_ENVS_PER_GPU", "8"))
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gid)], stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        free_mb = int(out.splitlines()[0])
    except Exception:
        return min(max_envs, max(min_envs, envs_per_gpu_default()))
    n = free_mb // per_env_mb
    return max(min_envs, min(max_envs, int(n)))


def envs_per_gpu_default():
    try:
        return int(os.environ.get("ENVS_PER_GPU", "6"))
    except Exception:
        return 6


def _wait_for_vector_workers(procs):
    """Wait for every worker and fail the setting if any worker failed."""
    failures = []
    for rank, proc in procs:
        returncode = proc.wait()
        if returncode:
            failures.append((rank, returncode))
    if failures:
        details = ", ".join(
            f"rank {rank} exited with code {returncode}"
            for rank, returncode in failures
        )
        raise RuntimeError(f"vector worker failure: {details}")


def launch_vector(model, ckpt, benchmark, control_mode, robot, domain, seeds,
                  eps_per_seed, render_gpus, out_dir, envs_per_gpu, max_record_envs,
                  record_video, model_id, mp4_root, results_dir, arm, model_gpu=0,
                  ood_offset=0.05, render_gl="egl", auto_envs=False,
                  single_card=False):
    """Spawn one render-only vector worker per render GPU.

    Every worker reaches its model through a required policy server. GPU overlap
    changes process placement only and never changes inference behavior.
    Each worker runs N=envs_per_gpu concurrent envs (batched forward). Tasks
    round-robin across workers; shards merged into one leaderboard JSON.

    Single-card (`single_card` or empty render_gpus): model and simulation share
    `model_gpu` while osmesa performs CPU rendering.
    `auto_envs`: per-render-GPU env count from free VRAM (auto_envs_per_gpu)."""
    render_gpu_ids = [int(g) for g in render_gpus.split(",") if g != ""]
    # Single-card placement: the server owns CUDA; the worker uses CPU rendering.
    if single_card or not render_gpu_ids:
        render_gpu_ids = [model_gpu]
        render_gl = "osmesa"
        if auto_envs or envs_per_gpu <= 0:
            os.environ.setdefault("BIDEX_MAX_ENVS_PER_GPU", "4")  # osmesa is slower
            envs_per_gpu = auto_envs_per_gpu(model_gpu)
        print(f"[eval-vec] SINGLE-CARD mode: server on gpu{model_gpu}, osmesa worker, "
              f"N_envs={envs_per_gpu}", flush=True)
    # Expose the worker count to children (auto_envs_per_gpu CPU-core bound).
    os.environ["BIDEX_VEC_WORKERS"] = str(len(render_gpu_ids))
    # in_domain and out_of_domain run sequentially via launch_vector in the SAME
    # caller-passed out_dir. To stop the OOD invocation from unlinking shard_rank*.json
    # that a still-finishing in_domain rank is writing (which dropped tasks whose
    # rank ran longer than the domain switch — e.g. Limited_Demonstrations rank5),
    # give each domain its own sub-dir. The final leaderboard JSON still lands at
    # results_dir (domain-independent), so this only isolates the transient shards.
    out = Path(out_dir) / ("in_domain" if domain == "in_domain" else "out_of_domain")
    out.mkdir(parents=True, exist_ok=True)
    for _r in range(len(render_gpu_ids)):
        for _ext in (".json", ".done", ".tmp"):
            try:
                (out / f"shard_rank{_r}{_ext}").unlink()
            except Exception:
                pass
    tasks = enumerate_task_ids(benchmark, robot, arm)
    _filter = os.environ.get("BIDEX_TASKS", "").strip()
    if _filter:
        _keep = {t.strip() for t in _filter.split(",") if t.strip()}
        tasks = [t for t in tasks if t in _keep]
    if not tasks:
        _write_leaderboard_json(results_dir, model_id, benchmark, robot, control_mode, domain,
            {"benchmark": benchmark, "robot": robot, "control_mode": control_mode, "domain": domain,
             "seeds": list(map(int, seeds)), "episodes_per_seed": eps_per_seed, "model": model,
             "ckpt": ckpt, "n_tasks": 0, "per_task": {}, "mean_rate": 0.0, "success_rate": 0.0,
             "note": "placeholder (no annotated tasks)"})
        print("[eval-vec] no tasks; wrote placeholder leaderboard.", flush=True)
        return
    if model not in ("lingbot", "openpi", "galaxea", "groot", "xr1", "internvla", "dm05"):
        raise ValueError(f"unsupported vectorized model: {model}")
    # Per-render-GPU env count (auto from free VRAM, else fixed envs_per_gpu).
    rank_envs = []
    for gid in render_gpu_ids:
        n = auto_envs_per_gpu(gid) if auto_envs else envs_per_gpu
        rank_envs.append(max(1, int(n)))
    max_record_envs_eff = max_record_envs if max_record_envs and max_record_envs > 0 else min(rank_envs) if rank_envs else 1
    _srv_msg_ports = _server_port_list(model)
    if not _srv_msg_ports:
        raise RuntimeError(f"{model} policy server is required; no server port was configured")
    print(f"[eval-vec] {len(tasks)} tasks; {len(render_gpu_ids)} render workers "
          f"({render_gpu_ids}); N_envs/worker={rank_envs}; model={model} "
          f"(server-client on ports {','.join(_srv_msg_ports)}); "
          f"render_gl={render_gl}", flush=True)
    shards = [[] for _ in render_gpu_ids]
    for i, t in enumerate(tasks):
        shards[i % len(render_gpu_ids)].append(t)
    procs = []
    for rank, gid in enumerate(render_gpu_ids):
        n_envs = rank_envs[rank]
        shard_path = out / f"shard_rank{rank}.json"
        sf = out / f"shard_rank{rank}.txt"
        sf.write_text("\n".join(shards[rank]))
        # Seeds are part of the leaderboard protocol and must not depend on the
        # number of render workers or which rank happens to own a task.
        seeds_arg = ",".join(str(int(s)) for s in seeds)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--vectorized-worker", "--model", model, "--ckpt", ckpt,
               "--benchmark", benchmark, "--control-mode", control_mode, "--robot", robot,
               "--domain", domain, "--seeds", seeds_arg, "--episodes-per-seed", str(eps_per_seed),
               "--out-file", str(shard_path), "--rank", str(rank), "--gpu", str(gid),
               "--task-file", str(sf), "--record-video", str(int(bool(record_video))),
               "--model-id", model_id, "--mp4-root", mp4_root,
               "--ood-object-offset", str(ood_offset), "--arm", arm,
               "--envs-per-gpu", str(n_envs), "--max-record-envs", str(max_record_envs_eff),
               ]
        # Workers never load a policy model, so CUDA stays hidden. EGL selects
        # the physical render card independently through MUJOCO_EGL_DEVICE_ID.
        _srv_ports = _server_port_list(model)
        if render_gl == "osmesa":
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MUJOCO_GL="osmesa")
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                       MUJOCO_EGL_DEVICE_ID=str(gid), BIDEX_EGL_DEVICE_ID=str(gid))
        if _srv_ports and render_gl != "osmesa":
            # Multi-model-GPU dynamic allocation: worker rank r talks to policy
            # server r % n_servers (run_vec_models_eval.sh started one per
            # MODEL_GPUS entry; ports exported as BIDEX_*_SERVER_PORTS /
            # OPENPI_SERVER_PORTS). The assigned port overrides the parent's
            # single-port env var inside THIS worker only — the per-model client
            # builders (_build_*_client_batched_infer, _openpi_server_client)
            # read *_SERVER_PORT from their own process env.
            _port_var, _ = _model_server_port_envs(model)
            env[_port_var] = str(_srv_ports[rank % len(_srv_ports)])
        log = open(out / f"vec_rank{rank}.log", "w")
        procs.append((rank, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        # Stagger worker spawns: openpi server-client infer is SERIAL (no dynamic
        # batching), so N workers × M envs all firing their first-infer at once
        # saturate the server → later workers' first recv hangs forever (the
        # documented first-infer-storm deadlock). Spawning one worker every few
        # seconds spreads the first-infer burst so each worker's M queries land
        # while the server is free.
        time.sleep(float(os.environ.get("BIDEX_VEC_SPAWN_DELAY", "3.0")))

    # Live per-setting progress: each worker writes shard_rank{r}.done
    # (= tasks completed by that rank)
    # after every task; sum them every 3s into results/<model_id>/<setting>
    # .progress.json so the live dashboard / leaderboard server shows a progress
    # bar before the setting's final .json lands. Deleted once the final JSON
    # is written (below).
    setting = f"{benchmark}_{robot}_{control_mode}_{domain}"
    prog_path = Path(results_dir) / model_id / f"{setting}.progress.json"
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(tasks)
    stop_ev = threading.Event()

    def _agg_progress_vec():
        while not stop_ev.is_set():
            done = 0
            for r in range(len(render_gpu_ids)):
                d = out / f"shard_rank{r}.done"
                if d.exists():
                    try:
                        done += int(d.read_text().strip() or 0)
                    except Exception:
                        pass
            try:
                # also re-flush the leaderboard JSON from whatever shards exist so
                # Each finished task's score/video shows up live rather than only
                # after the whole setting completes.
                merged = _merge_shards(out, len(render_gpu_ids))
                if merged:
                    overview = _build_overview(model, ckpt, benchmark, robot, control_mode,
                                               domain, seeds, eps_per_seed, render_gpu_ids, arm, merged)
                    _write_leaderboard_json(results_dir, model_id, benchmark, robot,
                                            control_mode, domain, overview)
                    done = len(merged)
                prog_path.write_text(json.dumps({"done": min(done, n_total), "total": n_total,
                                                 "setting": setting, "model_id": model_id}))
            except Exception:
                pass
            stop_ev.wait(3.0)
    prog_path.write_text(json.dumps({"done": 0, "total": n_total, "setting": setting, "model_id": model_id}))
    progress_thread = threading.Thread(target=_agg_progress_vec, daemon=True)
    progress_thread.start()

    try:
        _wait_for_vector_workers(procs)
    except Exception:
        for f in [prog_path] + [out / f"shard_rank{r}.done" for r in range(len(render_gpu_ids))]:
            try:
                f.unlink()
            except Exception:
                pass
        raise
    finally:
        stop_ev.set()
        progress_thread.join(timeout=5.0)
    all_results = _merge_shards(out, len(render_gpu_ids))
    if all_results:
        overall = _build_overview(model, ckpt, benchmark, robot, control_mode, domain,
                                  seeds, eps_per_seed, render_gpu_ids, arm, all_results)
        _atomic_write_json(out / "summary.json", _summary_payload(overall))
        lb_path = _write_leaderboard_json(results_dir, model_id, benchmark, robot, control_mode,
                                          domain, overall)
        print(f"[eval-vec] DONE {model} {benchmark}/{robot}/{control_mode}/{domain}: "
              f"{len(all_results)} tasks, score={overall['mean_rate'] * 100:.2f}")
        print(f"[eval-vec] leaderboard -> {lb_path}")
    else:
        print(f"[eval-vec] DONE {model} {benchmark}/{robot}/{control_mode}/{domain}: "
              f"0 tasks completed (killed early?); NOT overwriting leaderboard.")
    # final result written -> remove the in-progress marker + per-rank done counters
    for f in [prog_path] + [out / f"shard_rank{r}.done" for r in range(len(render_gpu_ids))]:
        try:
            f.unlink()
        except Exception:
            pass


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["openpi", "groot", "lingbot", "galaxea", "dm05", "xr1", "internvla"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--benchmark", default="Generalization")
    ap.add_argument("--control-mode", default="joint", choices=["joint", "robot_ee", "camera_ee"])
    ap.add_argument("--robot", default="astribot", choices=["astribot", "franka"])
    ap.add_argument("--domain", default="in_domain", choices=["in_domain", "out_of_domain", "both"])
    ap.add_argument(
        "--seeds",
        default=None,
        help=(
            "explicit seed override for a single-domain run "
            "(default: ID=0,1; OOD=2,3)"
        ),
    )
    ap.add_argument("--id-seeds", default="0,1",
                    help="in-domain seeds when --domain both (default: 0,1)")
    ap.add_argument("--ood-seeds", default="2,3",
                    help="out-of-domain seeds when --domain both (default: 2,3)")
    ap.add_argument("--episodes-per-seed", type=int, default=25,
                    help="rollouts per seed (default: 25)")
    ap.add_argument("--out-dir", default="leaderboard/runs/_tmp")
    ap.add_argument("--log-file", default="",
                    help="tee parent-process stdout/stderr to this file (dir auto-created); "
                         "replaces the shell `>> $OUT/runs/<name>.log` redirect so the yml "
                         "no longer needs to pre-create $OUT/runs/")
    ap.add_argument("--out-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="leaderboard results root (results/<model_id>/<bench>_<robot>_<mode>_<domain>.json)")
    ap.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--task-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--record-video", type=int, default=1, help="1 = write per-(seed,ep) head-cam mp4")
    ap.add_argument("--model-id", default="", help="leaderboard model_id (default openpi->pi05, groot->gr00t_n1_7)")
    ap.add_argument("--mp4-root", default=MP4_DIR)
    ap.add_argument("--ood-object-offset", type=float, default=0.05)
    ap.add_argument("--arm", default="dual", choices=["dual", "left", "right", "single_arm"],
                    help="dual=pure dual (mixed-all 58D ckpt, Articulated as dual); "
                         "single_arm=route per-task + pin static arm (per-arm Articulated); "
                         "left/right=legacy per-arm force")
    ap.add_argument("--vectorized-worker", action="store_true",
                    help=argparse.SUPPRESS)  # internal: set by launch_vector for its subprocesses
    ap.add_argument("--model-gpu", type=int, default=0,
                    help="physical GPU holding the (single) model copy (default 0)")
    ap.add_argument("--render-gpus", default="1",
                    help="comma list of physical render GPUs, one vectorized worker each (default '1'). "
                         "Multi-card e.g. '1,2,3,4,5,6,7'; model stays on --model-gpu.")
    ap.add_argument("--envs-per-gpu", type=int, default=8,
                    help="concurrent sim envs per render-GPU worker (batch size of one forward)")
    ap.add_argument("--max-record-envs", type=int, default=-1,
                    help="only the first k envs per worker record mp4 (ffmpeg CPU-contention relief); -1=all")
    ap.add_argument("--render-gl", default="egl", choices=["egl", "osmesa"],
                    help="MuJoCo render backend. egl=GPU offscreen (render on --render-gpus, default). "
                         "osmesa=CPU software render (no GPU render; use on shared boxes where multi-context "
                         "EGL Xid-31 faults, at some CPU cost).")
    ap.add_argument("--auto-envs", action="store_true",
                    help="vectorized: set per-render-GPU env count from free VRAM "
                         "(nvidia-smi memory.free // 400MB, capped by BIDEX_MAX_ENVS_PER_GPU). "
                         "Protects render smoothness (too many envs/card -> frame drops).")
    ap.add_argument("--single-card", action="store_true",
                    help="vectorized: policy server stays on --model-gpu and the render-only "
                         "worker uses osmesa CPU rendering")
    return ap


def main():
    args = build_arg_parser().parse_args()
    model_id = args.model_id or MODEL_ID_MAP.get(args.model, args.model)
    explicit_seeds = (
        [int(s) for s in args.seeds.split(",") if s != ""]
        if args.seeds is not None and args.seeds.strip()
        else None
    )
    # Per-domain seed sets for --domain both: ID tasks run --id-seeds, OOD tasks
    # run --ood-seeds (benchmark protocol: 2 seeds x 25 eps per domain = 100
    # rollouts per task). A concrete domain uses its protocol seed set unless
    # --seeds explicitly overrides it.
    id_seeds = ([int(s) for s in args.id_seeds.split(",") if s != ""]
                if args.id_seeds.strip()
                else (explicit_seeds if explicit_seeds is not None else [0, 1]))
    ood_seeds = ([int(s) for s in args.ood_seeds.split(",") if s != ""]
                 if args.ood_seeds.strip()
                 else (explicit_seeds if explicit_seeds is not None else [2, 3]))
    if args.domain == "in_domain":
        seeds = explicit_seeds if explicit_seeds is not None else id_seeds
    elif args.domain == "out_of_domain":
        seeds = explicit_seeds if explicit_seeds is not None else ood_seeds
    else:
        # Parent --domain both runs use _dom_seeds below. Keep a concrete list for
        # defensive compatibility with direct/internal worker invocations.
        seeds = explicit_seeds if explicit_seeds is not None else id_seeds

    def _dom_seeds(dom):
        if args.domain != "both":
            return seeds
        return id_seeds if dom == "in_domain" else ood_seeds

    # Worker shards already write to per-rank logs via Popen(stdout=...).
    if not args.vectorized_worker and args.log_file:
        try:
            lf = Path(args.log_file)
            lf.parent.mkdir(parents=True, exist_ok=True)
            _fd = os.open(lf, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.dup2(_fd, 1)
            os.dup2(_fd, 2)
            os.close(_fd)
            for _s in (sys.stdout, sys.stderr):
                try:
                    _s.reconfigure(line_buffering=True)
                except Exception:
                    pass
            print(f"[eval] logging parent process -> {lf}", flush=True)
        except Exception as e:
            sys.stderr.write(f"[eval] could not set up --log-file {args.log_file}: {e}\n")

    # Resolve --ckpt (base training dir -> latest checkpoint-<step>/<step>); if
    # nothing loadable is found, skip this combo cleanly and record it.
    try:
        args.ckpt = resolve_checkpoint(args.ckpt)
    except MissingCheckpoint:
        msg = f"MISSING {args.model} ckpt for {args.benchmark}/{args.robot} ({args.ckpt}); skip"
        print(msg, flush=True)
        try:
            with open(Path(LEADERBOARD_ROOT) / "eval_skipped.txt", "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass
        return
    print(f"[eval] {args.model} ckpt -> {args.ckpt}", flush=True)
    eval_h5 = _eval_h5()
    print(
        f"[eval] validated eval_dataset.h5 against "
        f"{sum(len(eval_h5[suite]) for suite in eval_h5)} benchmark tasks",
        flush=True,
    )

    if args.vectorized_worker:
        # Subprocess of launch_vector: CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID are
        # already set by the parent env dict -> do NOT override them here. The model loads
        # on cuda:0 (= physical model-gpu); this worker renders on MUJOCO_EGL_DEVICE_ID.
        tasks = [l.strip() for l in open(args.task_file)] if args.task_file else \
                enumerate_task_ids(args.benchmark, args.robot, args.arm)
        eval_shard_vector(tasks, seeds, args.episodes_per_seed, args.model, args.ckpt,
                          args.robot, args.control_mode, args.domain, args.out_file, args.rank,
                          args.gpu, args.benchmark, bool(args.record_video), model_id,
                          args.mp4_root, args.ood_object_offset, args.arm,
                          args.envs_per_gpu, args.max_record_envs)
        return

    domains = ["in_domain", "out_of_domain"] if args.domain == "both" else [args.domain]
    for dom in domains:
        launch_vector(args.model, args.ckpt, args.benchmark, args.control_mode,
                      args.robot, dom, _dom_seeds(dom), args.episodes_per_seed,
                      args.render_gpus, args.out_dir, args.envs_per_gpu,
                      args.max_record_envs, bool(args.record_video), model_id,
                      args.mp4_root, args.results_dir, args.arm, args.model_gpu,
                      args.ood_object_offset, args.render_gl, args.auto_envs,
                      args.single_card)

if __name__ == "__main__":
    main()
