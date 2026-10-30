"""Domain randomization driver: ties Region samplers to per-task articulations.

This is the "scene assembly at reset()" piece of LIBERO. The flow is:

    1. RegionSampler.sample_for_task(task, model, data) -> {obj_name: (pos, quat)}
    2. For each free-jointed object, write the sampled (pos, quat) into qpos.
    3. For each `articulations:` entry, write joint qpos (open/closed, on/off).
    4. mj_forward to refresh kinematics so the publish snapshot is correct.

We do NOT regenerate the MJCF here — that's a one-time cost in SimEnv.__init__.
Reset is purely a state-reset; the structure of the world stays the same.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bidex.tasks.region_sampler import RegionSampler


def _set_freejoint_pose(model, data, joint_name: str, pos, quat):
    """Write a (pos, quat) onto a free joint's qpos slice."""
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return False
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr:qadr + 3] = pos
    if quat is None:
        # identity orientation
        data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    else:
        data.qpos[qadr + 3:qadr + 7] = quat
    return True


def _set_joint_qpos(model, data, joint_name: str, qpos: float):
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return False
    data.qpos[model.jnt_qposadr[jid]] = float(qpos)
    return True


def _set_static_body_pose(model, data, obj, pos, quat):
    """Write a (pos, quat) onto a WELDED (freejoint: false) object body.

    Welded objects have no free joint, so region sampling can't reposition them
    via the qpos path (the freejoint loop in ``apply_task_initial_state`` skips
    them), and without this they'd keep their build-time baked pose every
    episode -- no randomization. But ``model.body_pos`` / ``model.body_quat``
    (the static body pose baked into the MJCF) ARE settable at runtime, so we
    move the body directly. This delivers the semantics a `freejoint: false` +
    `region:` task asks for: a fresh random init pose each episode, but an
    immovable body during the episode (no joint to drift / be knocked). The
    body's subtree (articulated children included) moves rigidly with it, so
    drawer/hinge joints keep working relative to the new pose.
    """
    import mujoco
    prefix = f"obj{obj.instance_id}_"
    # Target the object's TOP-LEVEL body (the one whose parent is worldbody).
    # `model.body_pos` / `body_quat` are relative to the body's PARENT, so writing
    # a world-frame (pos, quat) onto a non-top body (e.g. obj1_object when it sits
    # under obj1_root) would compound with the parent's baked pose and launch the
    # object (0.85 + 0.795 = 1.645 m up). The top body is the freejoint body for
    # free objects; for welded objects it is the static root body that
    # _prepare_object_xml baked the (placeholder/declared) pose onto.
    bid = -1
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith(prefix) and model.body_parentid[b] == 0:  # 0 == worldbody
            bid = b
            break
    if bid < 0:
        # Fall back to the named candidates, then the first prefix body.
        for cand in (prefix + "root", prefix + "object", prefix + "main"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if bid >= 0:
                break
    if bid < 0:
        for b in range(model.nbody):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix):
                bid = b
                break
    if bid < 0:
        return False
    model.body_pos[bid] = pos
    if quat is None:
        model.body_quat[bid] = (1.0, 0.0, 0.0, 0.0)
    else:
        model.body_quat[bid] = quat
    return True


def repose_welded_objects(model, data, task, rng=None):
    """Re-pose WELDED (freejoint: false) objects that declare a `region`.

    This runs on the init-state RESTORE paths (``SimEnv._apply_initial_state``
    Path 1/2), which set qpos from a saved .npy and return early -- skipping
    ``apply_task_initial_state`` (Path 3). A welded object has no free joint, so
    its pose is NOT in qpos and the restore leaves it at its baked placeholder
    pose every episode. Without this re-pose, a `freejoint: false` + `region:`
    object ignores its region on restore (e.g. the welded cabinet stuck at
    (0,0,0.85) with no yaw, unreachable except its bottom drawer). Free-jointed
    objects are left alone -- their pose comes from the restored qpos.
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    sampled = RegionSampler(rng).sample_for_task(task, model=model, data=data)
    n = 0
    for obj in task.all_object_inits():
        if obj.freejoint:
            continue
        if obj.name not in sampled:
            continue
        pos, quat = sampled[obj.name]
        if _set_static_body_pose(model, data, obj, pos, quat):
            n += 1
        else:
            print(f"[domain] welded body for object '{obj.name}' not found; "
                  f"keeping baked pose")
    return n


def apply_task_initial_state(
    model,
    data,
    task,
    rng: Optional[np.random.Generator] = None,
):
    """Resample object poses (regions) and articulation qpos for this task.

    Mutates `data` in place; calls `mj_forward` at the end. Returns the
    `{obj_name: (pos, quat)}` dict that was applied so callers can log it.

    Objects without a region keep whatever pose was loaded by the MJCF (i.e.
    the static pos/quat from their YAML spec). Objects WITH a region are
    drawn from RegionSampler and written via their freejoint qpos.
    """
    import mujoco

    if rng is None:
        rng = np.random.default_rng()

    sampler = RegionSampler(rng)
    sampled = sampler.sample_for_task(task, model=model, data=data)

    for obj in task.all_object_inits():
        if not obj.freejoint:
            continue
        if obj.name not in sampled:
            continue
        pos, quat = sampled[obj.name]
        joint_name = f"obj{obj.instance_id}_freejoint"
        ok = _set_freejoint_pose(model, data, joint_name, pos, quat)
        if not ok:
            # MJCF builder uses freejoint name = "obj{i}_freejoint"; if the
            # object's xml renamed its free joint we silently skip rather
            # than crash. Loud-but-recoverable: print for visibility.
            print(f"[domain] free joint '{joint_name}' not found for object '{obj.name}'")

    # Free-body objects with an explicit declared pose but NO region: restore
    # their declared pos/quat directly. The region loop above only re-poses
    # objects RegionSampler emitted (region-based); explicit-pose freejoint
    # bodies -- e.g. the articulated bottle in the open-bottle tasks, or any
    # free object with a fixed declared pose and no region -- would otherwise
    # keep whatever pose they ended the previous episode in (knocked over,
    # dropped off the table, etc.) and never reset. The hammer/spray tool
    # families are re-posed again afterwards by their dedicated reset hooks
    # (_reset_hammer_nail_state / _reset_spray_state, idempotent / overridden
    # for sprays), so this generic restore is safe for them and is the ONLY
    # reset path for unprotected free bodies such as the bottle. Velocities
    # are already zeroed by the caller before this runs.
    for obj in task.all_object_inits():
        if not obj.freejoint:
            continue
        if obj.name in sampled:
            continue            # already re-posed by the region loop above
        if obj.pos is None:
            continue
        joint_name = f"obj{obj.instance_id}_freejoint"
        ok = _set_freejoint_pose(
            model, data, joint_name, tuple(obj.pos),
            tuple(obj.quat) if obj.quat is not None else None)
        if not ok:
            print(f"[domain] free joint '{joint_name}' not found for object '{obj.name}'")

    # Welded (freejoint: false) objects WITH a region: re-pose them per episode
    # via their static body pose (model.body_pos / body_quat), since they have
    # no free joint qpos to write. This is the `freejoint: false` + `region:`
    # contract: random init pose each episode, immovable body during the episode.
    # Objects without a region keep their baked (declared) pose.
    for obj in task.all_object_inits():
        if obj.freejoint:
            continue
        if obj.name not in sampled:
            continue
        pos, quat = sampled[obj.name]
        if not _set_static_body_pose(model, data, obj, pos, quat):
            print(f"[domain] welded body for object '{obj.name}' not found; "
                  f"keeping baked pose")

    # Articulation init (open/closed, on/off) -- delegated to the reusable
    # apply_articulations helper so BOTH the fresh-init path (here) and the
    # restore path (env._repose_welded_after_restore) reset drawer/door/cap
    # joints to their declared qpos. Without this, a Path-1 restored saved
    # state keeps the drawer open from the previous episode.
    apply_articulations(model, data, task)

    mujoco.mj_forward(model, data)
    return sampled


def apply_articulations(model, data, task):
    """Apply the task's `articulations:` init (open/closed/on/off) to joint qpos.

    Range-aware ``open``/``closed`` shorthand resolves to the range endpoint
    closest to (or farthest from) zero. Called by ``apply_task_initial_state``
    (Path 3 fresh init) AND by ``SimEnv._repose_welded_after_restore`` (Path 1
    saved-state restore) so a restored state still gets its drawer/door/cap
    reset to the declared init qpos -- otherwise a drawer left open in the
    saved state stays open after reset/refresh.
    """
    import mujoco
    for art in task.articulations:
        obj_name = art["object"]
        local_joint = art["joint"]
        qpos_value = art.get("qpos", 0.0)
        if isinstance(qpos_value, str):
            obj = task.find_object(obj_name)
            if obj is not None:
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"obj{obj.instance_id}_{local_joint}",
                )
                if jid >= 0 and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    if qpos_value == "open":
                        qpos_value = float(hi if abs(hi) >= abs(lo) else lo)
                    elif qpos_value == "closed":
                        qpos_value = float(lo if abs(lo) <= abs(hi) else hi)
                    else:
                        qpos_value = 0.0
        obj = task.find_object(obj_name)
        if obj is None:
            print(f"[domain] articulation refers to unknown object '{obj_name}'")
            continue
        jname = f"obj{obj.instance_id}_{local_joint}"
        if not _set_joint_qpos(model, data, jname, qpos_value):
            print(f"[domain] articulation joint '{jname}' not found")
