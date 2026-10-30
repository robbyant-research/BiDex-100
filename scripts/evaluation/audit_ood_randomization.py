#!/usr/bin/env python3
"""Render and validate the deterministic OOD lighting/camera protocol.

The gallery contains one PNG per benchmark task. Rows are the three policy
cameras. Columns show nominal rendering, matched-texture dark and bright OOD
lighting extremes, and the actual scheduled lighting plus randomized camera
extrinsics. The dark/bright columns deliberately reuse the same texture,
object placement, light direction, and nominal camera so brightness changes
are directly visible rather than confounded by texture or viewpoint changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pyexpat  # noqa: F401 -- pin ABI-matched expat before MuJoCo loads EGL libs
import sys
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from PIL import Image, ImageDraw

# Make direct ``python scripts/evaluation/...`` execution work without an
# editable install, matching the canonical leaderboard driver's behaviour.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bidex import BiDexEnv
from bidex.eval import ood
from bidex.eval.run_leaderboard_eval import (
    load_task_for_domain,
    stable_task_hash,
)
from bidex.scene_builder import resolve_scene_xml
from bidex.tasks.registry import list_suites, list_tasks


CAMERAS = ("head", "left_wrist", "right_wrist")
VARIANTS = ("nominal", "lighting_dark", "lighting_bright", "lighting_camera")
VARIANT_LABELS = {
    "nominal": "nominal light + camera",
    "lighting_dark": "OOD dark + nominal camera",
    "lighting_bright": "OOD high-bright + nominal camera",
    "lighting_camera": "scheduled OOD light + safe random camera",
}
FORCED_LIGHTING_PROFILES = {
    "lighting_dark": "dark",
    "lighting_bright": "high_bright",
}


def benchmark_tasks() -> list[tuple[str, str]]:
    return [
        (suite, task)
        for suite in list_suites()
        for task in list_tasks(suite)
        if not task.endswith("__ood")
    ]


def _remove_generated_scene(task) -> None:
    scene = str(getattr(task, "scene", ""))
    if Path(scene).name.startswith("ood_scene_"):
        try:
            os.unlink(scene)
        except FileNotFoundError:
            pass


def load_variant(suite: str, task_name: str, seed: int, episode: int,
                 variant: str, forced_pose: dict | None = None):
    task, _, apply_offset = load_task_for_domain(
        suite, task_name, "out_of_domain", seed,
        stable_task_hash(suite, task_name), "ood-audit", episode,
    )
    if variant == "nominal":
        _remove_generated_scene(task)
        task.scene = task._base_scene
        task._ood_metadata["scene"] = {}
    elif variant in FORCED_LIGHTING_PROFILES:
        # Rebuild from the same base scene and with the same seed/episode. This
        # preserves texture choice, tint and light direction while changing
        # only the exposure profile, making the sweep visually comparable.
        _remove_generated_scene(task)
        task.scene, scene_meta = ood.randomize_scene(
            resolve_scene_xml(task._base_scene),
            suite=suite,
            task=task_name,
            seed=seed,
            episode=episode,
            lighting_profile=FORCED_LIGHTING_PROFILES[variant],
            return_metadata=True,
        )
        task._ood_metadata["scene"] = scene_meta
    if variant != "lighting_camera":
        task.__dict__.pop("_ood_camera_plan", None)
        task._ood_metadata["cameras"] = {}
    elif forced_pose is not None:
        task._ood_camera_plan = copy.deepcopy(task._ood_camera_plan)
        for camera in CAMERAS:
            task._ood_camera_plan["cameras"][camera] = ood.camera_pose_delta(
                ood.load_camera_pose_manifest(), task.robot_type, camera,
                forced_pose,
            )
        task._ood_metadata["cameras"] = copy.deepcopy(task._ood_camera_plan)
    return task, apply_offset


def _apply_episode_placement(env: BiDexEnv, task, suite: str, task_name: str,
                             seed: int, episode: int, apply_offset: bool) -> None:
    rng = ood.episode_rng(suite, task_name, seed, episode, "placement")
    ood.resample_distractors(env, task, rng)
    if apply_offset:
        ood.object_offset(
            env, task,
            ood.episode_rng(suite, task_name, seed, episode, "object_offset"),
            0.05,
        )
    mujoco.mj_forward(env.model, env.data)


def _table_points(task) -> np.ndarray:
    (xlo, xhi), (ylo, yhi), z = ood._table_world_for(task)
    # A conservative inset avoids declaring a safe camera invalid merely
    # because a decorative table edge is cropped by the nominal calibration.
    xs = np.linspace(xlo + 0.12 * (xhi - xlo), xhi - 0.12 * (xhi - xlo), 3)
    ys = np.linspace(ylo + 0.12 * (yhi - ylo), yhi - 0.12 * (yhi - ylo), 3)
    return np.asarray([(x, y, z + 0.015) for x in xs for y in ys])


def _project_points(env: BiDexEnv, camera: str, points: np.ndarray) -> np.ndarray:
    cid = env.camera_ids[camera]
    rotation = np.asarray(env.data.cam_xmat[cid]).reshape(3, 3)
    local = (points - env.data.cam_xpos[cid]) @ rotation
    depth = -local[:, 2]
    fy = math.tan(math.radians(float(env.model.cam_fovy[cid])) / 2.0)
    fx = fy * (env.img_width / env.img_height)
    return np.column_stack((local[:, 0] / (depth * fx),
                            local[:, 1] / (depth * fy), depth))


def _robot_body_ids(env: BiDexEnv) -> set[int]:
    model = env.model
    seeds = {int(model.jnt_bodyid[j]) for j in env._robot_jnt_ids}
    result = set(seeds)
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    # Include jointless mounting bodies above the first robot joints, without
    # swallowing world-attached task objects.
    for seed in tuple(seeds):
        body_id = seed
        while body_id > 0:
            name = model.body(body_id).name or ""
            if any(token in name.lower() for token in
                   ("fr3", "franka", "sharpa", "astribot", "torso", "head", "arm")):
                result.add(body_id)
            body_id = int(model.body_parentid[body_id])
    return result


def _robot_occlusion(env: BiDexEnv, camera: str) -> tuple[float, float]:
    renderer = env.renderer
    if renderer is None:
        return 1.0, 1.0
    renderer.enable_segmentation_rendering()
    try:
        # Segmentation ids are assigned while the scene is updated, so this
        # must happen after enabling segmentation mode.
        renderer.update_scene(env.data, camera=env.camera_ids[camera],
                              scene_option=env._scene_option)
        segmentation = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    geom_ids = segmentation[..., 0]
    geom_type = segmentation[..., 1]
    body_ids = _robot_body_ids(env)
    robot_geoms = {
        gid for gid in range(env.model.ngeom)
        if int(env.model.geom_bodyid[gid]) in body_ids
    }
    mask = (geom_type == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(geom_ids, list(robot_geoms))
    h, w = mask.shape
    central = mask[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(mask.mean()), float(central.mean())


def image_metrics(image: np.ndarray) -> dict:
    rgb = image.astype(np.float64)
    y = rgb @ np.asarray([0.2126, 0.7152, 0.0722])
    h, w = y.shape
    central = y[h // 8:7 * h // 8, w // 8:7 * w // 8]
    return {
        "luma_mean": float(central.mean()),
        "luma_p01": float(np.percentile(central, 1)),
        "luma_p99": float(np.percentile(central, 99)),
        "dark_fraction": float((central < 8).mean()),
        "bright_fraction": float((central > 247).mean()),
    }


def render_variant(suite: str, task_name: str, seed: int, episode: int,
                   variant: str, forced_pose: dict | None = None):
    task = None
    task, apply_offset = load_variant(
        suite, task_name, seed, episode, variant, forced_pose=forced_pose)
    env = None
    try:
        env_seed = ood.episode_seed(suite, task_name, seed, episode, "environment")
        env = BiDexEnv(task, control_mode="joint", cameras=list(CAMERAS),
                       render_depth=False, image_size=(240, 320), seed=env_seed)
        env.reset(seed=env_seed)
        _apply_episode_placement(
            env, task, suite, task_name, seed, episode, apply_offset)
        images = {}
        metrics = {}
        points = _table_points(task)
        for camera in CAMERAS:
            image = env.render(camera=camera, depth=False)
            images[camera] = image
            projected = _project_points(env, camera, points)
            visible = ((projected[:, 2] > 0.01)
                       & (np.abs(projected[:, 0]) <= 1.0)
                       & (np.abs(projected[:, 1]) <= 1.0))
            overall_occ, central_occ = _robot_occlusion(env, camera)
            metrics[camera] = {
                **image_metrics(image),
                "table_points_visible": int(visible.sum()),
                "table_center_visible": bool(visible[4]),
                "robot_fraction": overall_occ,
                "robot_central_fraction": central_occ,
            }
        return images, metrics, copy.deepcopy(env.camera_pose_metadata)
    finally:
        if env is not None:
            env.close()
        if task is not None:
            _remove_generated_scene(task)


def _tile(image: np.ndarray, title: str, width: int = 320, height: int = 270) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(Image.fromarray(image).resize((width, height - 30)), (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black")
    return canvas


def write_montage(path: Path, rendered: dict[str, dict[str, np.ndarray]]) -> None:
    montage = Image.new("RGB", (320 * len(VARIANTS), 270 * len(CAMERAS)), "white")
    for col, variant in enumerate(VARIANTS):
        for row, camera in enumerate(CAMERAS):
            title = f"{camera} | {VARIANT_LABELS[variant]}"
            montage.paste(_tile(rendered[variant][camera], title), (col * 320, row * 270))
    montage.save(path)


def camera_safe(camera: str, metrics: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not (18.0 <= metrics["luma_mean"] <= 238.0):
        reasons.append("mean_luminance")
    if metrics["dark_fraction"] > 0.60:
        reasons.append("too_dark")
    # A white tabletop can legitimately occupy close to half a wrist/head
    # frame while dark task objects remain fully resolved. Treat clipping as
    # invalid only when it dominates the observation; mean luminance and the
    # percentile span still guard globally washed-out frames.
    if metrics["bright_fraction"] > 0.60:
        reasons.append("too_bright")
    if metrics["luma_p99"] - metrics["luma_p01"] < 20.0:
        reasons.append("low_contrast")
    if camera == "head":
        if not metrics["table_center_visible"] or metrics["table_points_visible"] < 5:
            reasons.append("table_workarea_out_of_view")
    elif metrics["table_points_visible"] < 1:
        reasons.append("table_workarea_out_of_view")
    # Hands are intentionally present in wrist views. The central threshold
    # catches loss of the observation to the arm/hand rather than normal tool
    # visibility.
    occ_limit = 0.55 if camera == "head" else 0.72
    if metrics["robot_central_fraction"] > occ_limit:
        reasons.append("robot_occlusion")
    return not reasons, reasons


def iter_selected(tasks: list[tuple[str, str]], suite_filter: str,
                  task_filter: str, limit: int) -> Iterable[tuple[str, str]]:
    if suite_filter:
        tasks = [x for x in tasks if x[0] == suite_filter]
        if not tasks:
            raise SystemExit(f"suite not found: {suite_filter}")
    if task_filter:
        tasks = [x for x in tasks if f"{x[0]}/{x[1]}" == task_filter]
        if not tasks:
            raise SystemExit(f"task not found: {task_filter}")
    return tasks[:limit or None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task", default="")
    parser.add_argument("--suite", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pose-audit", action="store_true",
                        help="also test every committed pose on every selected task")
    parser.add_argument("--save-pose-images", action="store_true",
                        help="write a three-camera PNG for every forced pose")
    parser.add_argument("--pose", default="",
                        help="with --pose-audit, test only this pose id")
    args = parser.parse_args()

    output = Path(args.output)
    gallery = output / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    selected = list(iter_selected(
        benchmark_tasks(), args.suite, args.task, args.limit
    ))
    report = {
        "schema": "bidex-ood-visual-audit-v2",
        "seed": args.seed,
        "episode": args.episode,
        "task_count": len(selected),
        "camera_manifest": ood.load_camera_pose_manifest()["fingerprint"],
        "tasks": {},
        "pose_audit": {},
        "failures": [],
    }

    for index, (suite, task_name) in enumerate(selected, 1):
        key = f"{suite}/{task_name}"
        print(f"[gallery {index}/{len(selected)}] {key}", flush=True)
        rendered = {}
        task_report = {}
        for variant in VARIANTS:
            images, metrics, camera_meta = render_variant(
                suite, task_name, args.seed, args.episode, variant)
            rendered[variant] = images
            task_report[variant] = {"metrics": metrics, "camera_pose": camera_meta}
            for camera, values in metrics.items():
                safe, reasons = camera_safe(camera, values)
                if not safe:
                    report["failures"].append({
                        "task": key, "variant": variant,
                        "camera": camera, "reasons": reasons,
                    })
        filename = f"{index:03d}_{suite}_{task_name}.png"
        write_montage(gallery / filename, rendered)
        task_report["image"] = f"gallery/{filename}"
        report["tasks"][key] = task_report

    if args.pose_audit:
        poses = ood.load_camera_pose_manifest()["poses"]
        if args.pose:
            poses = [pose for pose in poses if pose["id"] == args.pose]
            if not poses:
                raise SystemExit(f"pose not found: {args.pose}")
        pose_gallery = output / "pose_gallery"
        if args.save_pose_images:
            pose_gallery.mkdir(parents=True, exist_ok=True)
        for task_index, (suite, task_name) in enumerate(selected, 1):
            key = f"{suite}/{task_name}"
            print(f"[poses {task_index}/{len(selected)}] {key}", flush=True)
            pose_rows = {}
            for pose_index, pose in enumerate(poses):
                pose_images, metrics, _ = render_variant(
                    suite, task_name, args.seed, pose_index,
                    "lighting_camera", forced_pose=pose)
                if args.save_pose_images:
                    canvas = Image.new("RGB", (320 * len(CAMERAS), 270), "white")
                    for camera_index, camera in enumerate(CAMERAS):
                        canvas.paste(_tile(
                            pose_images[camera], f"{pose['id']} | {camera}"
                        ), (camera_index * 320, 0))
                    canvas.save(pose_gallery / (
                        f"{task_index:03d}_{suite}_{task_name}_{pose['id']}.png"))
                pose_rows[pose["id"]] = metrics
                for camera, values in metrics.items():
                    safe, reasons = camera_safe(camera, values)
                    if not safe:
                        report["failures"].append({
                            "task": key, "variant": f"pose:{pose['id']}",
                            "camera": camera, "reasons": reasons,
                        })
            report["pose_audit"][key] = pose_rows

    report["passed"] = not report["failures"]
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "task_count": len(selected), "passed": report["passed"],
        "failure_count": len(report["failures"]), "output": str(output),
    }, sort_keys=True), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
