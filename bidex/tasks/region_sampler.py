"""Region samplers
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

import numpy as np

from bidex.tasks.task_spec import Region


def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions.

    Pure-python (no mujoco import) so the sampler stays light. R(q) = R(a)R(b),
    i.e. the product applies ``b`` first, then ``a`` -- so ``q_yaw ⊗ q_flip``
    applies the flip then the yaw. Used to compose a fixed per-object spawn
    orientation (e.g. an upside-down flip) onto the sampler's random yaw.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _get_visualize_scene():
    """Resolve the module providing resolve_object_xml + _prepare_object_xml
    (identical signatures). scripts/visualize_scene.py no longer exists — its
    functions moved into bidex.scene_builder; the old import path made
    _object_resting_offset silently return 0.0 (rest_on_surface was a no-op)."""
    from bidex import scene_builder
    return scene_builder


_RESTING_OFFSET_CACHE: dict = {}


def _object_resting_offset(obj_name: str, scale: float) -> float:
    """How high the object's body origin sits above the surface it NATURALLY
    rests on -- the z lift added to region.center.z so `rest_on_surface` spawns
    the object already at rest (no drop transient, no penetration).

    Measured by solo-dropping the object (at its task `scale`) onto a floor in
    a private sim and reading the settled freejoint origin z. This is robust to
    recessed bases / cradles / legs, where the resting plane (the rim) is ABOVE
    the lowest collision vertex (the recessed floor): the naive
    "lowest-collision-point" offset would spawn such objects floating by the
    recess depth and then drop them onto the table. Settling finds the TRUE
    resting plane.

    Geometry-static, so cached by (obj_name, scale) across resets. Returns 0.0
    on any failure (sampler then leaves origin at region.center.z -- the legacy
    behaviour -- rather than blocking task load)."""
    key = (obj_name, round(float(scale), 6))
    cached = _RESTING_OFFSET_CACHE.get(key)
    if cached is not None:
        return cached
    off = 0.0
    try:
        import os, tempfile, mujoco
        vs = _get_visualize_scene()
        xml_path, _ = vs.resolve_object_xml(obj_name)
        obj_dir = os.path.dirname(os.path.realpath(xml_path))
        fd, out = tempfile.mkstemp(prefix=".rest_", suffix=".xml", dir=obj_dir)
        os.close(fd)
        try:
            vs._prepare_object_xml(xml_path, 1, (0.0, 0.0, 2.0), None, True, out,
                                   scale=float(scale), mass_scale=1.0)
            with open(out) as f:
                xml = f.read()
            xml = xml.replace(
                "<worldbody>",
                '<worldbody><geom name="rest_floor" type="plane" size="5 5 0.1" '
                'contype="1" conaffinity="1" friction="0.8 0.01 0.001"/>', 1)
            with open(out, "w") as f:
                f.write(xml)
            m = mujoco.MjModel.from_xml_path(out)
            d = mujoco.MjData(m)
            fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj1_freejoint")
            qadr = int(m.jnt_qposadr[fj])
            mujoco.mj_forward(m, d)
            for _ in range(3000):
                mujoco.mj_step(m, d)
            off = float(d.qpos[qadr + 2])  # settled origin z above floor 0
        finally:
            try:
                os.remove(out)
            except OSError:
                pass
    except Exception:
        off = 0.0
    _RESTING_OFFSET_CACHE[key] = off
    return off


class RegionSampler:
    """Stateless sampler. Holds a numpy RNG so each reset is reproducible.

    Usage::

        rng = np.random.default_rng(seed)
        sampler = RegionSampler(rng)
        pos, quat = sampler.sample(region, model=model, data=data)
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(
        self,
        region: Region,
        model=None,
        data=None,
        layer_index: int = 0,
        base_pos: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Sample (pos, quat) inside the region, in world frame.

        If `region.target != "world"` the rectangle is interpreted as relative
        to that body's position. We do NOT compose orientations of the target
        body — yaw randomization always rotates around world-z, matching
        LIBERO's behavior for table-top tasks.

        The target body's position is resolved in this priority order:

          1. ``base_pos`` — the *fresh* xyz of the target object as just sampled
             in the same reset pass (passed in by ``sample_for_task``). This is
             the authoritative pose for a free-jointed target whose new (jittered)
             position has not yet been written to ``data`` (sampling runs before
             the apply loop in ``apply_task_initial_state``), so without it a
             body-relative spawn would read a stale ``data.xpos`` and land in the
             wrong place. Lets e.g. a fruit spawn centred over a bowl that
             jitters per episode — the fruit is guaranteed inside the bowl
             regardless of the bowl's random xy.
          2. ``data.xpos[target_body]`` — the live body position. Correct for a
             *static* target (fixture / welded body) whose pose never changes.

        `layer_index` (0-based) is the position of this object among the objects
        already sampled into the same region; when ``region.z_layer > 0`` the
        z is set to ``base + layer_index*z_layer`` so every object in the region
        sits at a distinct z layer (>= z_layer apart) -- a deterministic no-
        embed guarantee (two objects sharing xy stack instead of penetrating).
        """
        cx, cy, cz = region.center
        if region.target != "world":
            if base_pos is not None:
                bx, by, bz = base_pos
                cx, cy, cz = cx + bx, cy + by, cz + bz
            elif model is not None and data is not None:
                try:
                    import mujoco
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, region.target)
                    if bid >= 0:
                        bx, by, bz = data.xpos[bid]
                        cx, cy, cz = cx + bx, cy + by, cz + bz
                except Exception:
                    # Fall back to world-frame center; mismatch is loud at goal-check time.
                    pass

        dx, dy = region.xy_range
        x = cx + (self.rng.uniform(-dx, dx) if dx > 0 else 0.0)
        y = cy + (self.rng.uniform(-dy, dy) if dy > 0 else 0.0)
        z = cz + region.z_offset
        if region.z_layer > 0:
            # Deterministic distinct z per object -> no two share a z layer ->
            # no embedding; same-xy objects stack. Takes precedence over z_range.
            z += layer_index * region.z_layer
        elif region.z_range > 0:
            # Randomize spawn z so two objects sharing xy land at different z
            # (stack) instead of embedding at the same z.
            z += self.rng.uniform(-region.z_range, region.z_range)

        ymin, ymax = region.yaw_range
        if ymax > ymin:
            yaw = self.rng.uniform(ymin, ymax)
        else:
            # Equal bounds mean a pinned heading, including non-zero headings
            # such as [pi, pi]. The previous 0.0 fallback silently discarded it.
            yaw = float(ymin)

        # Yaw-only quaternion (w, x, y, z), rotation about world-z.
        quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        return (x, y, z), quat

    # ---- batch convenience ----

    def sample_for_task(self, task, model=None, data=None) -> dict:
        """Return `{object_name: (pos, quat)}` for every object that has a region.

        Objects without a region (or with explicit pos) are NOT returned — the
        caller falls back to their declared pose.

        If a region sets ``min_distance > 0``, objects sampled into it are
        rejection-sampled so no two land within that xy distance of each other
        (prevents cubes embedding / pre-stacking at spawn). For ordinary ID
        placement this remains the legacy best-effort constraint: after 256
        attempts the last deterministic sample is accepted. OOD geometry-safe
        placement is strict and raises if no valid placement can be found.
        Other regions keep the legacy independent-per-object sampling.
        """
        out = {}
        # region_name -> list of (x, y) already placed, for min_distance rejection.
        taken: dict = {}
        # OOD-only, geometry-static cross-region rejection. Region
        # min_distance only separates objects assigned to the same region;
        # widened OOD regions can otherwise overlap a neighbouring region.
        # Use parsed MJCF bounds rather than MuJoCo contacts so retry counts are
        # identical on every machine.
        ood_cfg = getattr(task, "ood", None) or {}
        geometry_safe = bool(ood_cfg.get("geometry_safe_placement", False))
        world_boxes = []

        def geometry_box(obj, pos, quat):
            if not geometry_safe:
                return None
            from bidex.scene_builder import object_footprint
            fp = object_footprint(obj.name, getattr(obj, "scale", 1.0))
            if fp is None:
                return None
            w, qx, qy, qz = map(float, quat)
            yaw = math.atan2(
                2.0 * (w * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            hx = c * float(fp[0]) + s * float(fp[1])
            hy = s * float(fp[0]) + c * float(fp[1])
            return (
                float(pos[0]), float(pos[1]),
                float(pos[2]) + float(fp[2]),
                float(pos[2]) + float(fp[3]),
                hx, hy,
            )

        def clears_world_boxes(box):
            if box is None:
                return True
            x, y, zlo, zhi, hx, hy = box
            for ox, oy, ozlo, ozhi, ohx, ohy in world_boxes:
                z_overlap = min(zhi, ozhi) - max(zlo, ozlo)
                if z_overlap <= 0.0:
                    continue
                # The current object may intentionally rest on an earlier
                # support object (plate/tray/shelf). Only its lower face near
                # the earlier object's upper face qualifies; shallow side
                # overlap between two table objects must still be rejected.
                if abs(zlo - ozhi) <= 0.03:
                    continue
                if (
                    abs(x - ox) < hx + ohx + 0.005
                    and abs(y - oy) < hy + ohy + 0.005
                ):
                    return False
            return True

        # Fixed-pose task objects are obstacles for subsequently sampled
        # world regions. Region-targeted objects are handled after sampling.
        if geometry_safe:
            for fixed in task.all_object_inits():
                if fixed.region is not None or getattr(fixed, "pos", None) is None:
                    continue
                quat = getattr(fixed, "quat", None) or (1.0, 0.0, 0.0, 0.0)
                box = geometry_box(fixed, fixed.pos, quat)
                if box is not None:
                    world_boxes.append(box)
        # instance_id -> (x, y, z) of objects already sampled this pass. A later
        # object whose region targets `obj{i}_<body>` is spawned relative to the
        # target's freshly-jittered pose (see `sample`'s `base_pos`). EVERY
        # region-sampled object is recorded -- free-jointed AND welded-with-
        # region -- because a welded fixture that carries a region is also
        # re-posed per episode (via model.body_pos in apply_task_initial_state),
        # so its live pose is equally not-yet-written to `data` at sampling
        # time. (Before, only free-jointed targets were recorded and welded
        # targets fell through to `data.xpos`, i.e. the STALE baked pose -- fine
        # for tray zones with ~2 cm jitter, but wrong for any larger fixture
        # jitter. init_states restores bypass this sampler, so collected data is
        # unaffected.) A fixture target with NO region is never sampled, keeps
        # its static baked pose, and correctly falls through to `data.xpos`.
        fresh_xyz: dict = {}
        # Opt-in region shuffle (Task.shuffle_regions): randomly PERMUTE the
        # region assignment among the named objects each reset, so the scene
        # re-shuffles which object sits in which zone (e.g. the "find the target
        # utensil in the tray" task -- the target no longer always spawns at the
        # same slot). The swapped regions must themselves be exchangeable
        # (same `target` / absolute-frame) and pinned/non-overlapping; the
        # sampler does NOT re-run min_distance across swapped regions. Off unless
        # the task declares a non-empty shuffle_regions, so tasks without the
        # field (and their collected init_states, which bypass this sampler) are
        # unaffected. `region_for[obj_name]` overrides that object's declared
        # region this pass.
        region_for: Dict[str, str] = {}
        shuffle_group = getattr(task, "shuffle_regions", None)
        if shuffle_group:
            _group = [o for o in task.all_object_inits() if o.name in set(shuffle_group)]
            if len(_group) > 1:
                _orig = [o.region for o in _group]
                if getattr(task, "shuffle_mode", None) == "cyclic":
                    # Cyclic shifts only: object i takes the region of object
                    # (i-k) mod N, preserving the group's declared cyclic order
                    # (e.g. bowls a->b->c stay the world-clockwise cycle around
                    # a triangle while the cycle starts at a random vertex).
                    _k = int(self.rng.integers(len(_orig)))
                    _perm = [((int(_i) + _k) % len(_orig)) for _i in range(len(_orig))]
                else:
                    _perm = self.rng.permutation(len(_orig))
                for _o, _j in zip(_group, _perm):
                    region_for[_o.name] = _orig[int(_j)]
        for obj in task.all_object_inits():
            if obj.region is None:
                continue
            rname = region_for.get(obj.name, obj.region)
            region = task.regions.get(rname)
            if region is None:
                raise KeyError(
                    f"Task '{task.name}': object '{obj.name}' refers to unknown "
                    f"region '{rname}'. Defined regions: {list(task.regions)}"
                )
            min_d = region.min_distance
            placed = taken.setdefault(rname, [])
            # Resolve a body-relative region against the target object's
            # just-sampled pose when the target is a free body declared earlier
            # in this pass (its body name is prefixed `obj{i}_` by the MJCF
            # builder, so we recover its instance id from the prefix). Falls
            # back to `data.xpos` when the target isn't a freshly-sampled free
            # body (e.g. a static fixture).
            base_pos = None
            if region.target != "world":
                m = re.match(r"obj(\d+)_", region.target)
                if m:
                    base_pos = fresh_xyz.get(int(m.group(1)))
            chosen = None
            attempts = 0
            check_world_geometry = (
                geometry_safe
                and region.target == "world"
                and not rname.startswith("_ood_distractor")
            )
            max_attempts = 4096 if check_world_geometry else 256
            while True:
                attempts += 1
                pos, quat = self.sample(region, model=model, data=data,
                                         layer_index=len(placed),
                                         base_pos=base_pos)
                xy = (pos[0], pos[1])
                distance_ok = (
                    min_d <= 0
                    or all(
                        ((xy[0] - p[0]) ** 2
                         + (xy[1] - p[1]) ** 2) ** 0.5 >= min_d
                        for p in placed
                    )
                )
                geometry_ok = (
                    not check_world_geometry
                    or clears_world_boxes(geometry_box(obj, pos, quat))
                )
                if distance_ok and geometry_ok:
                    chosen = (pos, quat)
                    break
                if attempts >= max_attempts:
                    if check_world_geometry:
                        raise RuntimeError(
                            f"Task '{task.name}': cannot place '{obj.name}' in "
                            f"region '{rname}' with deterministic geometry safety "
                            f"and min_distance={min_d} after "
                            f"{max_attempts} deterministic attempts"
                        )
                    # Preserve the original ID behavior. min_distance was a
                    # best-effort anti-overlap hint, not a reset-fatal
                    # constraint; existing task parameters rely on this.
                    chosen = (pos, quat)
                    break
            # Compose a fixed per-object orientation (e.g. an upside-down flip)
            # onto the sampled yaw so a region object can spawn flipped while
            # still jittering xy + heading and honoring `min_distance`. q_final
            # = q_yaw ⊗ spawn_quat (applies spawn_quat first, then the yaw) -- a
            # 180-deg-about-x flip stays "cap down" for any yaw, since rotating
            # about world-z preserves the inverted vertical axis.
            if obj.spawn_quat is not None:
                chosen = (chosen[0], _quat_mul(chosen[1], obj.spawn_quat))
            # rest_on_surface: lift the sampled origin by the object's
            # resting_offset (origin -> natural resting plane, found by a
            # solo settle) so the object spawns already at rest on
            # region.center.z -- not floating/dropping and not buried in it.
            # Prevents objects whose origin sits well above their base
            # (articraft hinged/pullout boxes) from penetrating the table.
            if region.rest_on_surface:
                off = _object_resting_offset(obj.name, obj.scale)
                if off != 0.0:
                    p = chosen[0]
                    chosen = ((p[0], p[1], p[2] + off), chosen[1])
            out[obj.name] = chosen
            placed.append((chosen[0][0], chosen[0][1]))
            if check_world_geometry:
                box = geometry_box(obj, chosen[0], chosen[1])
                if box is not None:
                    world_boxes.append(box)
            fresh_xyz[obj.instance_id] = tuple(chosen[0])
        return out
