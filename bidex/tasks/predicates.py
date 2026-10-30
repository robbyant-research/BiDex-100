"""Goal predicates.

Each predicate receives the live MjModel/MjData plus a *Task* and a
GoalPredicate spec, and returns a bool. Names mirror LIBERO's vocabulary
(`On`, `In`, `Open`, `TurnOn`, …) but the implementation is intentionally
shallow — we resolve symbolic names to MuJoCo body / joint ids by following
the obj{i}_ prefix convention from visualize_scene._prepare_object_xml.
"""

from __future__ import annotations

from typing import Callable, Dict


def _body_id_for_object(model, task, obj_name: str) -> int:
    """Map symbolic object name -> MjModel body id of its instance root."""
    import mujoco

    obj = task.find_object(obj_name)
    if obj is None:
        return -1
    # _prepare_object_xml prefixes every body name; the instance-root is the
    # first body in the source xml, which often is named "object" or "<obj>"
    # but may be unnamed. Try a small sequence of likely candidates.
    prefix = f"obj{obj.instance_id}_"
    for cand in (prefix + "object", prefix + obj_name, prefix + "root", prefix + "main"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
        if bid >= 0:
            return bid
    # Fallback: scan for any body whose name starts with the prefix.
    for bid in range(model.nbody):
        try:
            name = model.body(bid).name
        except Exception:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name.startswith(prefix):
            return bid
    return -1


def _joint_qpos_for_object(model, data, task, obj_name: str, local_joint: str) -> float:
    """Read qpos for a non-free joint declared inside an articulated object.

    BiDex-100 ``objects``/``fixtures`` (registered ObjectInits) name the joint
    ``obj{instance_id}_{local_joint}`` (the visualize_scene._prepare_object_xml
    prefix). RoboCasa *scene* fixtures -- e.g. the kitchen sink faucet, baked into
    the kitchen MJCF by build_robocasa_scene_xml and NOT registered as an
    ObjectInit -- get no instance id, so we fall back to the assembled fixture
    name: scan for a joint whose name starts with ``{obj_name}_`` and ends with
    ``_{local_joint}``. The upstream prefix nests a sub-group, e.g. layout 11's
    sink flow handle is ``sink_left_group_handle_joint``, so a literal
    ``f"{obj_name}_{local_joint}"`` would miss; the scan handles any prefix.

    Returns NaN when no matching joint exists (predicate then reads False), so a
    typo or an absent fixture degrades gracefully the same way an unknown
    ObjectInit does.
    """
    import mujoco

    obj = task.find_object(obj_name)
    if obj is not None:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{obj.instance_id}_{local_joint}"
        )
        if jid < 0:
            return float("nan")
        return float(data.qpos[model.jnt_qposadr[jid]])

    # Fixture fallback: obj_name is a RoboCasa scene fixture (no instance id).
    suffix = "_" + local_joint
    prefix = obj_name + "_"
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name.startswith(prefix) and name.endswith(suffix):
            return float(data.qpos[model.jnt_qposadr[jid]])
    return float("nan")


def _resolve_xyz(model, data, task, ref) -> tuple:
    """Resolve an `args` entry to a world-frame xyz.

    `ref` may be:
      - an object name (uses its body xpos),
      - a region name (uses the region center, optionally body-relative),
      - a 3-tuple/list of floats.
    """
    if isinstance(ref, (list, tuple)) and len(ref) == 3 and all(
        isinstance(v, (int, float)) for v in ref
    ):
        return tuple(float(v) for v in ref)

    if isinstance(ref, str) and ref in task.regions:
        r = task.regions[ref]
        cx, cy, cz = r.center
        if r.target != "world":
            import mujoco
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, r.target)
            if bid >= 0:
                bx, by, bz = data.xpos[bid]
                cx, cy, cz = cx + bx, cy + by, cz + bz
        return (cx, cy, cz + r.z_offset)

    bid = _body_id_for_object(model, task, ref) if isinstance(ref, str) else -1
    if bid >= 0:
        x, y, z = data.xpos[bid]
        return (float(x), float(y), float(z))
    return (float("nan"),) * 3


def _named_geom_id(model, task, object_name: str, geom_name: str) -> int:
    """Resolve a named geom inside a task object instance."""
    import mujoco

    obj = task.find_object(object_name)
    if obj is None:
        return -1
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"obj{obj.instance_id}_{geom_name}",
    )


def _named_site_id(model, task, object_name: str, site_name: str) -> int:
    """Resolve a named site inside a task object instance."""
    import mujoco

    obj = task.find_object(object_name)
    if obj is None:
        return -1
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        f"obj{obj.instance_id}_{site_name}",
    )


def _point_inside_geom(
    model, data, geom_id: int, point, clearance: float = 0.0
) -> bool:
    """Return whether a world point is inside a live box/cylinder/sphere geom.

    ``clearance`` shrinks the horizontal boundary. It is useful when the point
    is a spherical object's centre and its complete horizontal footprint must
    fit in the region. Vertical bounds remain the receiver's valid centre-height
    range at its floor and rim.
    """
    import mujoco
    import numpy as np

    if geom_id < 0:
        return False
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=np.float64) - center)
    size = model.geom_size[geom_id]
    margin = max(float(clearance), 0.0)
    # MuJoCo exposes this as numpy.int32.  Tuple membership against mjtGeom
    # IntEnums is asymmetric in some NumPy/MuJoCo combinations and can reject
    # a cylinder even though ``geom_type == mjGEOM_CYLINDER`` is true.
    geom_type = int(model.geom_type[geom_id])

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        bounds = np.asarray(size[:3], dtype=np.float64).copy()
        bounds[:2] -= margin
        return bool(np.all(bounds > 0.0) and np.all(np.abs(local) < bounds))
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = float(size[0]) - margin
        half_height = float(size[1])
        radial = float(np.hypot(local[0], local[1]))
        return (
            radius > 0.0
            and half_height > 0.0
            and radial < radius
            and abs(float(local[2])) < half_height
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        bounds = np.asarray(size[:3], dtype=np.float64).copy()
        bounds[:2] -= margin
        return bool(
            np.all(bounds > 0.0)
            and float(np.sum(np.square(local / bounds))) < 1.0
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = float(size[0]) - margin
        return radius > 0.0 and float(np.linalg.norm(local)) < radius
    return False


def _point_inside_site(model, data, site_id: int, point) -> bool:
    """Return whether a world point is inside a live box/cylinder/sphere site."""
    import mujoco
    import numpy as np

    if site_id < 0:
        return False
    center = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=np.float64) - center)
    size = model.site_size[site_id]
    site_type = model.site_type[site_id]

    if site_type == mujoco.mjtGeom.mjGEOM_BOX:
        return bool(np.all(np.abs(local) < np.asarray(size[:3], dtype=np.float64)))
    if site_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (
            float(np.hypot(local[0], local[1])) < float(size[0])
            and abs(float(local[2])) < float(size[1])
        )
    if site_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(np.linalg.norm(local)) < float(size[0])
    return False


def _body_subtree_ids(model, root_body_id: int) -> set[int]:
    """Collect a body and all descendants."""
    children = {}
    for body_id in range(model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id >= 0:
            children.setdefault(parent_id, []).append(body_id)
    subtree = set()
    stack = [root_body_id]
    while stack:
        body_id = stack.pop()
        if body_id < 0 or body_id in subtree:
            continue
        subtree.add(body_id)
        stack.extend(children.get(body_id, []))
    return subtree


def _has_upward_support_contact(
    model, data, object_body_id: int, support_body_id: int
) -> bool:
    """Return whether the support subtree is pushing the object upward."""
    normal_z_min = 0.5
    support_bodies = _body_subtree_ids(model, support_body_id)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        if body_1 == object_body_id and body_2 in support_bodies:
            support_to_object_nz = -float(contact.frame[2])
        elif body_2 == object_body_id and body_1 in support_bodies:
            support_to_object_nz = float(contact.frame[2])
        else:
            continue
        if support_to_object_nz > normal_z_min:
            return True
    return False


def _pred_sunk(model, data, task, args, threshold) -> bool:
    """sunk(nail, [depth]): a mocap nail has been driven downward by >= `depth`.

    Mirrors dexjoco's hammer-nail success condition ``nail_depth >= 0.04``.
    The nail's initial z (``z0``) is read from its :class:`ObjectInit` ``pos``
    (set in the task YAML); the current z is read from the live body world
    position. Because the nail is a mocap body, ``mj_step``/``mj_forward``
    propagate the env-written ``mocap_pos`` into ``data.xpos[bid]``, so the
    sunk depth is simply ``z0 - z_now``. ``threshold`` overrides the default
    0.04 m success depth.
    """
    if not args:
        return False
    obj = task.find_object(args[0])
    if obj is None or obj.pos is None:
        return False
    z0 = float(obj.pos[2])
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if obj_xyz[2] != obj_xyz[2]:  # NaN check
        return False
    depth = float(threshold) if threshold is not None else 0.04
    return (z0 - obj_xyz[2]) >= depth


def _pred_not_sunk(model, data, task, args, threshold) -> bool:
    """not_sunk(nail, [depth]): nail has NOT been driven down by `depth` -- negation of :func:`_pred_sunk`.

    Same depth threshold as `sunk` (default 0.04m). Use with ``must_hold:
    true`` to forbid hammering a nail: without it the subtask latches true on
    frame 0 (the nail starts undriven) and stays "done" even if the robot
    later strikes it. Because the env's hammer-nail script only drives mocap
    nails downward (monotonic), a nail driven at any frame stays down to the
    end, so the final-frame `must_hold` check catches any-frame driving --
    no "driven then undone" escape as with not_lifted.
    """
    return not _pred_sunk(model, data, task, args, threshold)


def _pred_watered(model, data, task, args, threshold) -> bool:
    """watered(plant, [radius]): some spray's trigger is pressed AND its nozzle
    is currently within `radius` (xy) of the plant and within 0.20 m in z.

    Mirrors dexjoco's water-plant success (trigger pulled + ``ref_point`` near
    the plant), but stateless: it reports live truth, not history. Give each
    plant its OWN subtask so :class:`SubtaskTracker` latches it the first frame
    a pressed spray nozzle reaches it -- this lets the robot water plants one
    at a time and still satisfy all `watered` goals, exactly as `sunk` goals
    latch per nail.

    The trigger state is read from the spray's ``obj{i}_joint_0`` hinge qpos
    (declared in dexjoco_spray.xml), pull threshold 0.34 (dexjoco
    _TRIGGER_PULL_THRESHOLD). The nozzle position is read from the spray
    body's ``obj{i}_ref_point`` site (fallback: body origin). `threshold`
    overrides the default 0.12 m xy radius.
    """
    import mujoco
    if not args:
        return False
    plant_bid = _body_id_for_object(model, task, args[0])
    if plant_bid < 0:
        return False
    px, py, pz = (float(v) for v in data.xpos[plant_bid][:3])
    radius = float(threshold) if threshold is not None else 0.12
    z_half = 0.20
    pull_thresh = 0.05   # light press counts (matches env _SPRAY_TRIGGER_PULL_THRESHOLD)
    for obj in task.all_object_inits():
        if not obj.is_spray:
            continue
        prefix = f"obj{obj.instance_id}_"
        # Trigger must be pressed (tq > pull_thresh) — same threshold as the env's
        # _SPRAY_TRIGGER_PULL_THRESHOLD, so watering fires iff the blue cone is on.
        tjid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "joint_0")
        if tjid < 0:
            continue
        tq = float(data.qpos[model.jnt_qposadr[tjid]])
        if tq < pull_thresh:
            continue
        # Nozzle position: prefer the declared site, fall back to body origin.
        sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, prefix + "ref_point")
        if sid >= 0:
            nx, ny, nz = (float(v) for v in data.site_xpos[sid][:3])
        else:
            sbid = -1
            for cand in (prefix + "spray_body", prefix + "root", prefix + obj.name):
                sbid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, cand)
                if sbid >= 0:
                    break
            if sbid < 0:
                continue
            nx, ny, nz = (float(v) for v in data.xpos[sbid][:3])
        dx, dy = nx - px, ny - py
        if (dx * dx + dy * dy) <= radius * radius and abs(nz - pz) < z_half:
            return True
    return False


def _pred_not_watered(model, data, task, args, threshold) -> bool:
    """not_watered(plant): plant is NOT being watered this frame -- negation of :func:`_pred_watered`.

    Watering is *transient*: `watered` is true only while a trigger is pressed
    AND the nozzle is near the plant. By the final frame the robot has moved
    on, so a plain ``must_hold`` end-check would read `not_watered` = True
    again and miss a plant that was watered mid-episode. Pair this with
    ``latch_violation: true`` instead -- :class:`SubtaskTracker` latches a
    permanent failure the first frame this predicate is False (i.e. the
    forbidden plant got watered), so any momentary watering fails the task.
    """
    return not _pred_watered(model, data, task, args, threshold)


def _pred_pressed(model, data, task, args, threshold) -> bool:
    """pressed(button, [count]): a spring push-button has been pressed at least
    `count` times this episode.

    The button is a welded-base, spring-return cap on a slide joint (see the
    dexjoco_button_* object family). A *press* is a transient event: the cap
    deflects down under a finger, then springs back, so a stateless per-frame
    joint-angle check (like `open`) cannot count repeated presses. Instead the
    env's per-step button script (:func:`SimEnv._apply_button_interaction`)
    does hysteresis edge-detection on the cap's slide qpos and increments a
    runtime counter stored on the object (``ObjectInit.press_count``), which we
    read here. Give each button its OWN subtask so :class:`SubtaskTracker`
    latches it the first frame the count reaches the target -- exactly as
    `sunk` latches per nail. `count` defaults to 1; `threshold` is unused (the
    press depth threshold lives in the env script so it is shared with the
    edge detector).
    """
    if not args:
        return False
    obj = task.find_object(args[0])
    if obj is None:
        return False
    try:
        target = int(args[1]) if len(args) > 1 and args[1] is not None else 1
    except (TypeError, ValueError):
        target = 1
    return int(getattr(obj, "press_count", 0)) >= max(target, 1)


def _pred_pressed_exact(model, data, task, args, threshold) -> bool:
    """pressed_exact(button, count): a spring push-button has been pressed
    EXACTLY ``count`` times this episode -- not fewer AND not more.

    Strict variant of :func:`_pred_pressed` for tasks whose spec demands a
    precise press tally (e.g. "press the red button exactly ten times": 9 or
    11 both fail). ``press_count`` is monotonic, so the predicate is True only
    in the window where the count equals the target. Pair with ``must_hold:
    true``: :class:`SubtaskTracker` latches ``done`` the frame the count first
    reaches the target, and the final-frame ``holds_now`` check then requires
    the count to STILL equal the target. That makes an N+1th press (count
    overshoots) fail -- ``done`` stayed latched but ``holds_now`` is False --
    while stopping short at N-1 never latches ``done`` at all. ``count``
    defaults to 1; ``threshold`` is unused (press depth lives in the env script
    so it is shared with the edge detector).
    """
    if not args:
        return False
    obj = task.find_object(args[0])
    if obj is None:
        return False
    try:
        target = int(args[1]) if len(args) > 1 and args[1] is not None else 1
    except (TypeError, ValueError):
        target = 1
    return int(getattr(obj, "press_count", 0)) == max(target, 1)


def _pred_pressed_after(model, data, task, args, threshold) -> bool:
    """pressed_after(button, count, prereq_button, prereq_count): ``button`` has
    been pressed EXACTLY ``count`` times AND every one of those presses occurred
    AFTER ``prereq_button`` had already reached ``prereq_count`` presses (no
    press of ``button`` happened while the prerequisite was still short).

    Models the "do A exactly N times, THEN do B exactly once" contract: B must
    not be touched before A is complete. Proof of "after" comes from the
    env-populated ``press_log`` on ``button`` -- one snapshot per recorded
    press, each mapping every OTHER button's name -> its press_count at that
    instant (monotonic, so a snapshot showing ``prereq >= prereq_count`` means
    the prerequisite was already met when this press fired). If any snapshot
    shows the prereq still short, this predicate is permanently False and the
    subtask never latches -> premature press fails the task.

    Pair with ``must_hold: true`` so the exact count must STILL hold at the
    final frame: a second press of ``button`` makes ``press_count != count``
    so ``holds_now`` goes False and the task fails, while zero presses never
    latch. ``threshold`` is unused.
    """
    if len(args) < 4:
        return False
    obj = task.find_object(args[0])
    if obj is None:
        return False
    try:
        target = int(args[1])
        prereq_name = str(args[2])
        prereq_count = int(args[3])
    except (TypeError, ValueError):
        return False
    if int(getattr(obj, "press_count", 0)) != max(target, 1):
        return False
    # press_count and press_log are written together on the same rising edge,
    # so a correct run has one snapshot per press. A missing/short log means
    # we cannot prove every press was post-prerequisite -> fail safe.
    log = list(getattr(obj, "press_log", []))
    if len(log) != int(getattr(obj, "press_count", 0)):
        return False
    for snap in log:
        if int(snap.get(prereq_name, -1)) < prereq_count:
            return False  # a press fired before the prerequisite was met
    return True


def _count_orderable(task) -> int:
    """Number of `orderable` objects declared on the task (the items placed into
    the basket). Used by `placed_n` / `order_match` so the expected count is not
    hard-coded to a fixed 3."""
    return sum(1 for o in task.all_object_inits() if getattr(o, "orderable", False))


def _pred_placed_n(model, data, task, args, threshold) -> bool:
    """placed_n(basket, n): at least `n` objects have been placed into the
    basket so far this episode.

    "Placed" = recorded by the env's per-step order-tracking script as a first
    outside->inside transition. The count is read from the basket
    :class:`ObjectInit`'s ``place_seq`` list (env-populated, same pattern as
    `pressed` reading ``press_count``), so the predicate stays stateless. Give
    it its own subtask so :class:`SubtaskTracker` latches the placement
    milestone the frame the n-th object drops in; pair with `order_match`
    (gated by ``ordered_subtasks``) so the removal check only arms after the
    placement phase. `n` defaults to the number of `orderable` objects on the
    task.
    """
    if not args:
        return False
    basket = task.find_object(args[0])
    if basket is None:
        return False
    try:
        n = int(args[1]) if len(args) > 1 and args[1] is not None else _count_orderable(task)
    except (TypeError, ValueError):
        n = _count_orderable(task)
    return len(getattr(basket, "place_seq", [])) >= max(n, 1)


def _pred_order_match(model, data, task, args, threshold) -> bool:
    """order_match(basket): every orderable object has been placed into the
    basket AND removed again, AND the removal order equals the placement order
    (first-in-first-out -- "take them out in the order they were put in").

    Reads the basket :class:`ObjectInit`'s env-populated ``place_seq`` /
    ``remove_seq`` catalog-name lists (stateless). A mismatched (or incomplete)
    sequence is never True, so the subtask never latches and the task fails --
    the desired "wrong removal order = error" outcome. Pair with
    ``must_hold: true``: the sequences are fixed once the last object is
    removed, so a correct order latches and keeps holding at the final frame.
    """
    if not args:
        return False
    basket = task.find_object(args[0])
    if basket is None:
        return False
    place_seq = list(getattr(basket, "place_seq", []))
    remove_seq = list(getattr(basket, "remove_seq", []))
    expected = _count_orderable(task)
    if expected == 0:
        return False
    return len(place_seq) == expected and place_seq == remove_seq


def _pred_drawn_n(model, data, task, args, threshold) -> bool:
    """drawn_n(container, n): at least `n` objects have been removed from the
    container so far this episode.

    "Removed" = recorded by the env's basket order-tracker as a first
    inside->outside transition (after being placed). This reads the container
    :class:`ObjectInit`'s ``remove_seq`` length, so the predicate stays stateless
    (same pattern as `order_match`). In the "draw paper, press the matching
    number" task the container is the paper holder and ``remove_seq`` length is
    exactly the number of sheets drawn. `n` defaults to 1; give it its own
    subtask so :class:`SubtaskTracker` latches the "left arm is drawing" milestone
    the moment the first sheet leaves the holder.
    """
    if not args:
        return False
    container = task.find_object(args[0])
    if container is None:
        return False
    try:
        n = int(args[1]) if len(args) > 1 and args[1] is not None else 1
    except (TypeError, ValueError):
        n = 1
    return len(getattr(container, "remove_seq", [])) >= max(n, 1)


def _pred_press_matches_drawn(model, data, task, args, threshold) -> bool:
    """press_matches_drawn(container, btn1, btn2, ...): the press-button whose
    ``button_value`` equals the number of objects removed from `container` (the
    draw count) has been pressed, AND no other listed button has been pressed.

    Models the "draw N sheets, press the number N" task: the left arm draws
    ``N = len(container.remove_seq)`` sheets; the right arm must press the button
    whose declared ``value`` == N. Strict -- a wrong-button press (or pressing
    when nothing was drawn) fails:

      * ``1 <= N <= <number of listed buttons>`` (a button must exist for N);
      * the button with ``button_value == N`` has ``press_count >= 1``;
      * every OTHER listed button has ``press_count == 0``.

    ``press_count`` is monotonic (the env button script only increments it), so
    once a wrong button is pressed this predicate is permanently False and the
    subtask never latches -> task fails. Pair with ``must_hold: true`` so the
    match must still hold at the final frame. All reads are off the
    env-populated ``remove_seq`` / ``press_count`` (stateless predicate).
    """
    if len(args) < 2:
        return False
    container = task.find_object(args[0])
    if container is None:
        return False
    buttons = [task.find_object(a) for a in args[1:]]
    if any(b is None for b in buttons):
        return False
    n_buttons = len(buttons)
    draw_count = len(getattr(container, "remove_seq", []))
    if not (1 <= draw_count <= n_buttons):
        return False
    # Exactly one button may match the draw count (its value), and only it may be pressed.
    pressed = [int(getattr(b, "press_count", 0)) for b in buttons]
    values = [getattr(b, "button_value", None) for b in buttons]
    matching = [i for i, v in enumerate(values) if v == draw_count]
    if len(matching) != 1:
        return False  # no unique button for this count (shouldn't happen with values 1..N)
    for i, p in enumerate(pressed):
        if i == matching[0]:
            if p < 1:
                return False          # matching number not pressed
        else:
            if p >= 1:
                return False          # a wrong number was pressed (strict)
    return True


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _pred_flyover_n(model, data, task, args, threshold) -> bool:
    """flyover_n(obj, n): obj has entered the task's flyover zone at least n
    times this episode.

    Reads the env-populated ``flyover_count`` on the object's ObjectInit
    (incremented by the per-tick flyover tracker on each outside->inside
    transition into the zone's live xy radius + z window), so the predicate
    stays stateless -- same pattern as `pressed` reading ``press_count``. `n`
    defaults to 1. Monotonic count, so once reached it stays reached;
    ``must_hold`` is unnecessary. Pair with ``ordered_subtasks`` gating (e.g.
    grasp first, then count the passes) so early passes do not pre-latch the
    milestone.
    """
    if not args:
        return False
    obj = task.find_object(args[0])
    if obj is None:
        return False
    try:
        n = int(args[1]) if len(args) > 1 and args[1] is not None else 1
    except (TypeError, ValueError):
        n = 1
    return int(getattr(obj, "flyover_count", 0)) >= max(n, 1)


def _pred_swaps_n(model, data, task, args, threshold) -> bool:
    """swaps_n(obj, n): the cup colors have been swapped at least n times.

    Reads the env-populated ``color_swap_count`` on the color-cube ObjectInit
    (incremented by the per-tick color-swap hook each time the cube enters the
    currently-red cup's zone, capped at 3), so the predicate stays stateless
    -- same pattern as `flyover_n`. Monotonic; `must_hold` unnecessary.
    """
    if not args:
        return False
    obj = task.find_object(args[0])
    if obj is None:
        return False
    try:
        n = int(args[1]) if len(args) > 1 and args[1] is not None else 1
    except (TypeError, ValueError):
        n = 1
    return int(getattr(obj, "color_swap_count", 0)) >= max(n, 1)


def _pred_color_match_place(model, data, task, args, threshold) -> bool:
    """color_match_place(obj): the color-cube was placed into a cup whose
    CURRENT color equals the cube's (remembered) initial color.

    Reads the env-populated ``color_restored`` / ``color_matched`` flags: the
    color-swap hook restores the cube's initial rgba the first frame it rests
    inside a cup (tight placement zone, below the rim) and records whether
    that cup's current color matches. A placement into the wrong-colored cup
    sets restored=True but matched=False, so this predicate never turns true
    -- the desired permanent failure. Pair with ``must_hold: true``.
    """
    if not args:
        return False
    obj = task.find_object(args[0])
    if obj is None:
        return False
    return bool(getattr(obj, "color_restored", False)
                and getattr(obj, "color_matched", False))


def _pred_on(model, data, task, args, threshold) -> bool:
    """on(obj, target): obj is within an xy-tolerance of target's xy and slightly above its z.

    `target` may be another object, a region, or a literal xyz. `threshold`
    overrides the default xy radius (default 0.06m).
    """
    if len(args) < 2:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    tgt_xyz = _resolve_xyz(model, data, task, args[1])
    if any(v != v for v in obj_xyz + tgt_xyz):  # NaN check
        return False
    radius = float(threshold) if threshold is not None else 0.06
    dxy = ((obj_xyz[0] - tgt_xyz[0]) ** 2 + (obj_xyz[1] - tgt_xyz[1]) ** 2) ** 0.5
    above = obj_xyz[2] > tgt_xyz[2] - 0.02
    return dxy < radius and above


def _pred_stacked_on(model, data, task, args, threshold) -> bool:
    """stacked_on(obj, target, [z_min, z_max]): obj is stacked directly ON TOP of target.

    Stricter than :func:`_pred_on`: obj must be one layer above the target within
    an xy radius AND within a z window -- a real stack, not two objects side by
    side (rejected by the z window's lower bound) or two layers up (rejected by
    the upper bound).

    ``threshold`` is the xy radius (default 0.05 m). The z window defaults to
    ``(0.02, 0.06)`` -- one sugar-cube layer (~3 cm tall). For taller stackable
    bodies pass an explicit window as ``args[2]``/``args[3]``: e.g. full-size
    beverage cans (~11 cm tall) use ``(0.08, 0.13)`` so a neat can-on-can layer
    (centres ~0.10-0.11 m apart) reads true while a two-layer gap (~0.22 m) and
    a same-height neighbour (Δz ~0) do not. A single ``args[2]`` is treated as
    the expected layer delta with a ±0.03 window. Pair with ``must_hold: true``
    so a stack that slips at episode end (obj no longer on top) does not count.
    """
    if len(args) < 2:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    tgt_xyz = _resolve_xyz(model, data, task, args[1])
    if any(v != v for v in obj_xyz + tgt_xyz):  # NaN check
        return False
    radius = float(threshold) if threshold is not None else 0.05
    if len(args) >= 4:
        z_lo = float(args[2])
        z_hi = float(args[3])
    elif len(args) == 3:
        zc = float(args[2])
        z_lo = zc - 0.03
        z_hi = zc + 0.03
    else:
        z_lo, z_hi = 0.02, 0.06
    dxy = ((obj_xyz[0] - tgt_xyz[0]) ** 2 + (obj_xyz[1] - tgt_xyz[1]) ** 2) ** 0.5
    one_layer_above = (obj_xyz[2] > tgt_xyz[2] + z_lo) and (obj_xyz[2] < tgt_xyz[2] + z_hi)
    return (dxy < radius) and one_layer_above


def _pred_on_table(model, data, task, args, threshold) -> bool:
    """on_table(obj): the scene table directly supports the object from below.

    Direct support is used instead of an xy drop zone so every point on the
    physical tabletop is valid, while an object held at tabletop height or
    resting inside another object is not accepted.
    """
    if not args:
        return False
    import mujoco

    object_body_id = _body_id_for_object(model, task, args[0])
    table_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "table"
    )
    if object_body_id < 0 or table_body_id < 0:
        return False
    return _has_upward_support_contact(
        model, data, object_body_id, table_body_id
    )


def _pred_inside(model, data, task, args, threshold) -> bool:
    """inside(obj, container): obj xy within container xy radius (default 0.1m), and z within +/-0.25m.

    The z window is 0.25 m (not 0.15): tall containers whose body origin sits at
    the base (e.g. short_cabinet at scale 1.7, whose drawer interior is ~0.17 m
    ABOVE the body origin) need the extra headroom or an item correctly placed
    in a raised drawer/shelf reads dz > 0.15 and never counts as inside.
    """
    if len(args) < 2:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    tgt_xyz = _resolve_xyz(model, data, task, args[1])
    radius = float(threshold) if threshold is not None else 0.10
    dxy = ((obj_xyz[0] - tgt_xyz[0]) ** 2 + (obj_xyz[1] - tgt_xyz[1]) ** 2) ** 0.5
    dz = abs(obj_xyz[2] - tgt_xyz[2])
    return dxy < radius and dz < 0.25


def _pred_inside_box(model, data, task, args, threshold) -> bool:
    """inside_box(obj, container, [margin]): obj is inside the container's XY BOX
    footprint (separate x/y tolerances) and within a z window.

    Unlike :func:`_pred_inside` (a circular ``dxy < radius``), this uses the
    container's own (box-collision) half-extents on each axis, so a long, narrow
    container -- e.g. a 3-compartment desk caddy (~0.42 m long, ~0.13 m wide)
    -- accepts an item in ANY compartment (along the long axis) while rejecting
    an item resting BESIDE it (off the short/narrow axis). The circular
    ``inside`` cannot do this: a radius big enough to reach the end compartments
    (dxy ~0.21) also covers 'beside the middle' (dxy ~0.1), so it lights green
    before the item is actually in.

    ``threshold`` (default 0.04 m) is a margin added to each half-extent. The z
    window is the same fixed 0.15 m as `inside`. The box is taken in WORLD axes
    against the container body's xpos (accurate while the container's long axis
    stays near a world axis, e.g. the caddy's yaw ~pi keeps its length along
    world y); for a heavily rotated container use `inside` instead.
    """
    if len(args) < 2:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    tgt_xyz = _resolve_xyz(model, data, task, args[1])
    if any(v != v for v in obj_xyz + tgt_xyz):  # NaN check
        return False
    margin = float(threshold) if threshold is not None else 0.04

    import mujoco
    cbid = _body_id_for_object(model, task, args[1])
    if cbid < 0:
        # No resolvable container body -> fall back to the circular test.
        dxy = ((obj_xyz[0] - tgt_xyz[0]) ** 2 + (obj_xyz[1] - tgt_xyz[1]) ** 2) ** 0.5
        return dxy < margin and abs(obj_xyz[2] - tgt_xyz[2]) < 0.25

    # Container subtree (body + descendants) -- the caddy may put walls on child
    # bodies. Collect box-collision geom world extents relative to the body xpos.
    children = {}
    for b in range(model.nbody):
        p = model.body_parentid[b]
        if p >= 0:
            children.setdefault(p, []).append(b)
    subtree = set()
    stack = [cbid]
    while stack:
        b = stack.pop()
        if b < 0 or b in subtree:
            continue
        subtree.add(b)
        stack.extend(children.get(b, []))
    hx = hy = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in subtree:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        gx, gy, _ = data.geom_xpos[g]
        sx, sy = float(model.geom_size[g][0]), float(model.geom_size[g][1])
        hx = max(hx, abs(gx - tgt_xyz[0]) + sx)
        hy = max(hy, abs(gy - tgt_xyz[1]) + sy)
    if hx <= 0 and hy <= 0:
        # No box geoms -> fall back to circular.
        dxy = ((obj_xyz[0] - tgt_xyz[0]) ** 2 + (obj_xyz[1] - tgt_xyz[1]) ** 2) ** 0.5
        return dxy < margin and abs(obj_xyz[2] - tgt_xyz[2]) < 0.25

    dx = abs(obj_xyz[0] - tgt_xyz[0])
    dy = abs(obj_xyz[1] - tgt_xyz[1])
    dz = abs(obj_xyz[2] - tgt_xyz[2])
    return dx < hx + margin and dy < hy + margin and dz < 0.25


def _pred_inside_site(model, data, task, args, threshold) -> bool:
    """inside_site(obj, container, site, [margin, z_lo, z_hi]): obj is inside the
    container's named SITE box -- xy tested in the site's LIVE local frame, z
    tested as a WORLD-frame height band around the site's world z.

    Needed for pull-out drawers. The drawer slides ~0.16 m past the cabinet
    body, so ``inside`` (circular, vs body origin) can't reach the open drawer
    without also swallowing the spawn zones; ``inside_box`` (body-xy box + wide
    z window) accepts an item sitting on the TABLE directly beneath the hovering
    open drawer. The drawer's interior ``<site>`` (e.g. ``middle_region``) moves
    WITH the drawer, so its xy footprint always matches the tray; and the world
    height band separates 'resting in the tray' (obj z ~= site z, within a few
    cm) from 'on the table under/behind it' (obj z ~7-10 cm BELOW site z).

    Note: the site frame is rotated, so dl.z is NOT the vertical -- the z band
    is applied to the raw world-z difference, not to the site-local dl.z.
    ``threshold`` is the xy margin (default 0.02 m); ``z_lo`` / ``z_hi`` (extra
    args, defaults -0.05 / +0.10) bound ``obj_z - site_z``. At spawn (drawer
    closed) the site box sits inside the cabinet body, so spawn-zone items are
    outside in xy -- no false latch either way.

    Additionally the obj must be IN CONTACT with the site-owner body (the
    drawer). Volume tests alone can't separate 'resting in the tray' from
    'carried over the tray at the same height' (a tall bottle's centre passes
    through the z band while still descending / in transit) -- contact can: a
    resting item touches the drawer floor/walls, a hovering one doesn't.
    """
    if len(args) < 3:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if any(v != v for v in obj_xyz):  # NaN check
        return False
    import mujoco
    cont = task.find_object(args[1])
    if cont is None:
        return False
    sid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"obj{cont.instance_id}_{args[2]}")
    if sid < 0:
        return False
    margin = float(threshold) if threshold is not None else 0.02
    z_lo = float(args[3]) if len(args) > 3 else -0.05
    z_hi = float(args[4]) if len(args) > 4 else 0.10
    import numpy as np
    sx, sy, _sz = (float(v) for v in model.site_size[sid][:3])
    spos = np.asarray(data.site_xpos[sid], dtype=np.float64)
    smat = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    dl = smat.T @ (np.asarray(obj_xyz, dtype=np.float64) - spos)
    dz_world = float(obj_xyz[2]) - float(spos[2])
    if not (abs(float(dl[0])) < sx + margin
            and abs(float(dl[1])) < sy + margin
            and z_lo <= dz_world <= z_hi):
        return False
    # Contact gate: obj must touch the site-owner body (the drawer), so a bottle
    # merely CARRIED over the open tray (in the box, in the z band, but not yet
    # released) does not latch. A resting item always touches the drawer floor.
    obj_bid = _body_id_for_object(model, task, args[0])
    drawer_bid = int(model.site_bodyid[sid])
    if obj_bid < 0:
        return False
    for c in range(data.ncon):
        b1 = model.geom_bodyid[data.contact[c].geom1]
        b2 = model.geom_bodyid[data.contact[c].geom2]
        if (b1 == obj_bid and b2 == drawer_bid) or (b2 == obj_bid and b1 == drawer_bid):
            return True
    return False


def _pred_inside_site_loose(model, data, task, args, threshold) -> bool:
    """inside_site_loose(obj, container, site, [z_lo, z_hi]): like
    :func:`_pred_inside_site` (obj inside the container's named SITE box -- xy in
    the site's LIVE local frame, z a world band around the site's world z) but
    WITHOUT the contact gate.

    For a mesh receiver whose support contact flickers at rest -- e.g. tomatoes in
    a wire strainer: a tomato rests IN the bowl (positionally inside the bowl
    <site> box) but its mesh contact is momentarily lost at the final frame, so the
    contact-gated inside_site reads false even though the tomato is genuinely in
    the bowl. The site box already localises the object to the receiver, so the
    contact gate is dropped here. Pair with must_hold so the object must still be
    in the site box at the final frame (a carried/hovering object inside the box
    also reads true, so this suits released/resting objects at the end-check).
    ``threshold`` is the xy margin (default 0.02); ``z_lo`` / ``z_hi`` (extra args,
    defaults -0.05 / +0.10) bound ``obj_z - site_z`` -- widen z_lo for a deep bowl.
    """
    if len(args) < 3:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if any(v != v for v in obj_xyz):  # NaN check
        return False
    import mujoco
    import numpy as np
    cont = task.find_object(args[1])
    if cont is None:
        return False
    sid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"obj{cont.instance_id}_{args[2]}")
    if sid < 0:
        return False
    margin = float(threshold) if threshold is not None else 0.02
    z_lo = float(args[3]) if len(args) > 3 else -0.05
    z_hi = float(args[4]) if len(args) > 4 else 0.10
    sx, sy, _sz = (float(v) for v in model.site_size[sid][:3])
    spos = np.asarray(data.site_xpos[sid], dtype=np.float64)
    smat = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
    dl = smat.T @ (np.asarray(obj_xyz, dtype=np.float64) - spos)
    dz_world = float(obj_xyz[2]) - float(spos[2])
    return (abs(float(dl[0])) < sx + margin
            and abs(float(dl[1])) < sy + margin
            and z_lo <= dz_world <= z_hi)


def _pred_inside_site_any(model, data, task, args, threshold) -> bool:
    """inside_site_any(obj, container, site1, site2, ...): obj is inside ANY of
    the listed drawer sites. OR over :func:`_pred_inside_site` -- the contact
    gate naturally picks the drawer the item actually rests in, so exactly one
    alternative succeeds. Use to accept "put the item in either the top OR the
    middle drawer" without expressing a disjunction in the goal DSL.
    """
    if len(args) < 3:
        return False
    obj_name, cont_name = args[0], args[1]
    for site_name in args[2:]:
        if _pred_inside_site(
            model, data, task, [obj_name, cont_name, site_name], threshold
        ):
            return True
    return False


def _pred_inside_site_any_loose(model, data, task, args, threshold) -> bool:
    """inside_site_any_loose(obj, container, site..., [z_lo, z_hi]).

    Accept an object whose centre is in any listed live site without requiring
    direct floor contact.  The final two arguments may be numeric world-z
    bounds relative to each site; otherwise ``inside_site_loose`` defaults are
    used.  This is appropriate for true container interiors where an object
    can be wedged against a wall or supported by another contained object.
    Pair it with ``must_hold`` for final-state evaluation.
    """
    if len(args) < 3:
        return False
    obj_name, cont_name = args[0], args[1]
    site_names = list(args[2:])
    z_bounds = []
    if len(site_names) >= 3:
        try:
            z_bounds = [float(site_names[-2]), float(site_names[-1])]
        except (TypeError, ValueError):
            z_bounds = []
        else:
            site_names = site_names[:-2]
    for site_name in site_names:
        call_args = [obj_name, cont_name, site_name, *z_bounds]
        if _pred_inside_site_loose(model, data, task, call_args, threshold):
            return True
    return False


def _pred_inside_region(model, data, task, args, threshold) -> bool:
    """inside_region(obj, container, geom): obj centre is inside a named geom.

    The geom is evaluated in its live frame, so the predicate follows a moving
    or tilted container. ``threshold`` is an inward clearance from the geom
    horizontal boundary. For a spherical object, setting it to the sphere
    radius requires the complete horizontal footprint to be inside.
    """
    if len(args) < 3:
        return False
    object_xyz = _resolve_xyz(model, data, task, args[0])
    if any(value != value for value in object_xyz):
        return False
    geom_id = _named_geom_id(model, task, args[1], args[2])
    clearance = float(threshold) if threshold is not None else 0.0
    return _point_inside_geom(model, data, geom_id, object_xyz, clearance)


def _pred_inside_site_volume_any(model, data, task, args, threshold) -> bool:
    """inside_site_volume_any(obj, container, site1, ...): live 3-D interiors.

    Unlike the drawer-specific world-height band, all three local coordinates
    are bounded. Rotated sites must not leave a horizontal direction unbounded.
    No direct floor contact is required for a tool wedged inside a compartment.
    """
    if len(args) < 3:
        return False
    point = _resolve_xyz(model, data, task, args[0])
    return any(
        _point_inside_site(model, data, _named_site_id(model, task, args[1], site), point)
        for site in args[2:]
    )


def _pred_not_inside_region(model, data, task, args, threshold) -> bool:
    """Negation of :func:`_pred_inside_region`."""
    return not _pred_inside_region(model, data, task, args, threshold)


def _pred_keypoint_in_site(model, data, task, args, threshold) -> bool:
    """keypoint_in_site(obj, container, site, source_geom, axis): test one end.

    The keypoint starts at the source geom centre and moves along one local axis
    by ``threshold`` times that geom's half-size. This supports elongated tools:
    their functional end may be inside a receiver while the handle remains out.
    Axis values are ``x+``, ``x-``, ``y+``, ``y-``, ``z+``, and ``z-``.
    """
    if len(args) < 5:
        return False
    import numpy as np

    source_geom_id = _named_geom_id(model, task, args[0], args[3])
    target_site_id = _named_site_id(model, task, args[1], args[2])
    if source_geom_id < 0 or target_site_id < 0:
        return False
    axes = {
        "x+": (0, 1.0),
        "x-": (0, -1.0),
        "y+": (1, 1.0),
        "y-": (1, -1.0),
        "z+": (2, 1.0),
        "z-": (2, -1.0),
    }
    axis = axes.get(str(args[4]).lower())
    if axis is None:
        return False
    axis_index, direction = axis
    fraction = float(threshold) if threshold is not None else 0.75
    local_offset = np.zeros(3, dtype=np.float64)
    local_offset[axis_index] = (
        direction * fraction * float(model.geom_size[source_geom_id][axis_index])
    )
    rotation = np.asarray(
        data.geom_xmat[source_geom_id], dtype=np.float64
    ).reshape(3, 3)
    keypoint = (
        np.asarray(data.geom_xpos[source_geom_id], dtype=np.float64)
        + rotation @ local_offset
    )
    return _point_inside_site(model, data, target_site_id, keypoint)


def _pred_supported_in_region(model, data, task, args, threshold) -> bool:
    """supported_in_region(obj, container, region...): contained and supported.

    The object centre must be inside at least one named live region geom and the
    container must simultaneously provide an upward support contact. This
    rejects side bumps and objects resting outside or on top of a shelf.
    ``threshold`` optionally shrinks the region boundary.
    """
    if len(args) < 3:
        return False
    object_xyz = _resolve_xyz(model, data, task, args[0])
    if any(value != value for value in object_xyz):
        return False
    clearance = float(threshold) if threshold is not None else 0.0
    inside = any(
        _point_inside_geom(
            model,
            data,
            _named_geom_id(model, task, args[1], region_name),
            object_xyz,
            clearance,
        )
        for region_name in args[2:]
    )
    if not inside:
        return False
    object_body_id = _body_id_for_object(model, task, args[0])
    container_body_id = _body_id_for_object(model, task, args[1])
    if object_body_id < 0 or container_body_id < 0:
        return False
    return _has_upward_support_contact(
        model, data, object_body_id, container_body_id
    )


def _point_inside_named_region_band(
    model,
    data,
    task,
    container_name: str,
    region_name: str,
    point,
    z_lo: float,
    z_hi: float,
    clearance: float,
) -> bool:
    """Test an arbitrary world point against a named horizontal region."""
    import mujoco
    import numpy as np

    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return False
    geom_id = _named_geom_id(model, task, container_name, region_name)
    if geom_id < 0:
        return False
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (point - center)
    size = model.geom_size[geom_id]
    # MuJoCo exposes this as numpy.int32. Tuple membership against mjtGeom
    # IntEnums is asymmetric in some NumPy/MuJoCo combinations.
    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        half_x = float(size[0]) - clearance
        half_y = float(size[1]) - clearance
        horizontal_inside = (
            half_x > 0.0
            and half_y > 0.0
            and abs(float(local[0])) < half_x
            and abs(float(local[1])) < half_y
        )
    elif geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0]) - clearance
        horizontal_inside = (
            radius > 0.0
            and float(np.hypot(local[0], local[1])) < radius
        )
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius_x = float(size[0]) - clearance
        radius_y = float(size[1]) - clearance
        horizontal_inside = (
            radius_x > 0.0
            and radius_y > 0.0
            and (float(local[0]) / radius_x) ** 2
            + (float(local[1]) / radius_y) ** 2
            < 1.0
        )
    else:
        return False
    return bool(
        horizontal_inside
        and float(z_lo) <= float(local[2]) <= float(z_hi)
    )


def _inside_named_region_band(model, data, task, args, clearance: float) -> bool:
    """Shared object-origin implementation for named horizontal regions."""
    if len(args) < 5:
        return False
    object_xyz = _resolve_xyz(model, data, task, args[0])
    return _point_inside_named_region_band(
        model,
        data,
        task,
        args[1],
        args[2],
        object_xyz,
        float(args[3]),
        float(args[4]),
        clearance,
    )


def _pred_inside_region_band(model, data, task, args, threshold) -> bool:
    """inside_region_band(obj, container, geom, z_lo, z_hi).

    Require the object's centre inside a named region's horizontal footprint
    and an explicit local-height band, without requiring contact with the
    container floor.  This handles planar ``reg_int`` opening geoms and true
    container interiors where an object may be wedged against a wall or rest
    on another contained object.  ``threshold`` shrinks the horizontal region.
    """
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    return _inside_named_region_band(model, data, task, args, clearance)


def _pred_axis_in_region_band(model, data, task, args, threshold) -> bool:
    """axis_in_region_band(obj, container, region, source_geom, z_lo, z_hi).

    Accept an elongated object when any sampled point on its full long axis is
    inside the receiver region and height band.  This models an inserted spoon
    or fork whose lower end is in a cup while its centre and handle legitimately
    remain outside.  Orientation remains a separate predicate so a horizontal
    utensil laid across the rim cannot pass.
    """
    if len(args) < 6:
        return False
    import numpy as np

    source_geom_id = _named_geom_id(model, task, args[0], args[3])
    if source_geom_id < 0:
        return False
    sizes = np.asarray(model.geom_size[source_geom_id], dtype=np.float64)
    if sizes.size < 3 or not np.all(np.isfinite(sizes[:3])):
        return False
    long_axis = int(np.argmax(sizes[:3]))
    half_length = float(sizes[long_axis])
    if half_length <= 0.0:
        return False
    center = np.asarray(data.geom_xpos[source_geom_id], dtype=np.float64)
    rotation = np.asarray(
        data.geom_xmat[source_geom_id], dtype=np.float64
    ).reshape(3, 3)
    axis = rotation[:, long_axis]
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    for fraction in np.linspace(-1.0, 1.0, 21):
        point = center + axis * (fraction * half_length)
        if _point_inside_named_region_band(
            model,
            data,
            task,
            args[1],
            args[2],
            point,
            float(args[4]),
            float(args[5]),
            clearance,
        ):
            return True
    return False


def _pred_axis_in_region(model, data, task, args, threshold) -> bool:
    """axis_in_region(obj, container, region, source_geom).

    Accept an elongated object when any sampled point on its full long axis is
    strictly inside the receiver's live 3-D region geom.  Unlike the banded
    variant, the receiver geom supplies the floor and rim bounds, so a shallow
    insertion counts while a utensil crossing above or beside the opening does
    not.  No orientation constraint is implied.
    """
    if len(args) < 4:
        return False
    import numpy as np

    source_geom_id = _named_geom_id(model, task, args[0], args[3])
    region_geom_id = _named_geom_id(model, task, args[1], args[2])
    if source_geom_id < 0 or region_geom_id < 0:
        return False
    sizes = np.asarray(model.geom_size[source_geom_id], dtype=np.float64)
    if sizes.size < 3 or not np.all(np.isfinite(sizes[:3])):
        return False
    long_axis = int(np.argmax(sizes[:3]))
    half_length = float(sizes[long_axis])
    if half_length <= 0.0:
        return False
    center = np.asarray(data.geom_xpos[source_geom_id], dtype=np.float64)
    rotation = np.asarray(
        data.geom_xmat[source_geom_id], dtype=np.float64
    ).reshape(3, 3)
    axis = rotation[:, long_axis]
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    return any(
        _point_inside_geom(
            model,
            data,
            region_geom_id,
            center + axis * (fraction * half_length),
            clearance,
        )
        for fraction in np.linspace(-1.0, 1.0, 21)
    )


def _pred_supported_in_region_band(model, data, task, args, threshold) -> bool:
    """supported_in_region_band(obj, container, geom, z_lo, z_hi).

    Require :func:`_pred_inside_region_band` plus physical upward support from
    the container.  Use this for exposed surfaces such as boards and trays;
    use ``inside_region_band`` for a genuine container interior.
    """
    if not _pred_inside_region_band(model, data, task, args, threshold):
        return False
    object_body_id = _body_id_for_object(model, task, args[0])
    container_body_id = _body_id_for_object(model, task, args[1])
    if object_body_id < 0 or container_body_id < 0:
        return False
    return _has_upward_support_contact(
        model, data, object_body_id, container_body_id
    )


def _pred_supported_in_region_band_stack(model, data, task, args, threshold) -> bool:
    """A contained item supported by the receiver, directly or through contents.

    The object and every intermediate task object must be inside the same live
    horizontal region and height band. Directed upward support contacts must
    form a path to the receiver. A held/hovering item, a nearby table, and a
    stack outside the bowl cannot supply that path.
    """
    if not _pred_inside_region_band(model, data, task, args, threshold):
        return False
    object_body = _body_id_for_object(model, task, args[0])
    receiver_body = _body_id_for_object(model, task, args[1])
    if object_body < 0 or receiver_body < 0 or object_body == receiver_body:
        return False
    candidates = {receiver_body}
    for obj in task.all_object_inits():
        if obj.name in args[:2]:
            continue
        if _pred_inside_region_band(
            model, data, task, [obj.name, *args[1:]], threshold
        ):
            body = _body_id_for_object(model, task, obj.name)
            if body >= 0:
                candidates.add(body)
    pending, visited = [object_body], {object_body}
    while pending:
        current = pending.pop()
        for support in candidates - visited:
            if not _has_upward_support_contact(model, data, current, support):
                continue
            if support == receiver_body:
                return True
            visited.add(support)
            pending.append(support)
    return False


def _pred_axis_in_local_frustum(model, data, task, args, threshold) -> bool:
    """axis_in_local_frustum(tool, bowl, bbox, floor_z, rim_z, floor_r, rim_r).

    Sample the tool's long axis as for ``axis_in_region``. The receiver-local
    tapered interior supplies floor/rim bounds for bowls without a named
    interior geom. A handle may remain outside, but some actual axis point
    must be below the rim and inside the bowl, rather than above or beside it.
    """
    if len(args) != 7:
        return False
    import numpy as np

    source = _named_geom_id(model, task, args[0], args[2])
    receiver = _body_id_for_object(model, task, args[1])
    if source < 0 or receiver < 0:
        return False
    z0, z1, r0, r1 = map(float, args[3:])
    if not all(np.isfinite([z0, z1, r0, r1])) or not (z0 < z1 and min(r0, r1) > 0):
        return False
    sizes = np.asarray(model.geom_size[source], dtype=np.float64)
    if not np.all(np.isfinite(sizes)) or np.max(sizes) <= 0:
        return False
    axis_index = int(np.argmax(sizes))
    source_rotation = np.asarray(data.geom_xmat[source]).reshape(3, 3)
    receiver_rotation = np.asarray(data.xmat[receiver]).reshape(3, 3)
    center = np.asarray(data.geom_xpos[source])
    origin = np.asarray(data.xpos[receiver])
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    for fraction in np.linspace(-1.0, 1.0, 21):
        point = center + source_rotation[:, axis_index] * (fraction * sizes[axis_index])
        local = receiver_rotation.T @ (point - origin)
        if z0 < local[2] < z1:
            radius = r0 + (r1 - r0) * (local[2] - z0) / (z1 - z0) - clearance
            if float(np.hypot(local[0], local[1])) < radius:
                return True
    return False


def _pred_supported_geom_in_region_band(model, data, task, args, threshold) -> bool:
    """supported_geom_in_region_band(obj, container, obj_geom, region, z_lo, z_hi).

    Like ``supported_in_region_band``, but evaluates the visible object's named
    geometry centre instead of its sometimes-offset root body origin.  This is
    useful for asymmetric assets such as cheese wedges while retaining the
    physical upward-support requirement for an exposed board.
    """
    if len(args) < 6:
        return False
    object_geom_id = _named_geom_id(model, task, args[0], args[2])
    if object_geom_id < 0:
        return False
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    if not _point_inside_named_region_band(
        model,
        data,
        task,
        args[1],
        args[3],
        data.geom_xpos[object_geom_id],
        float(args[4]),
        float(args[5]),
        clearance,
    ):
        return False
    object_body_id = _body_id_for_object(model, task, args[0])
    container_body_id = _body_id_for_object(model, task, args[1])
    if object_body_id < 0 or container_body_id < 0:
        return False
    return _has_upward_support_contact(
        model, data, object_body_id, container_body_id
    )


def _pred_supported_in_region_band_near(model, data, task, args, threshold) -> bool:
    """supported_in_region_band_near(obj, container, geom, z_lo, z_hi, [slop]).

    Require live named-region containment, upward support, and a coarse
    horizontal distance from the container root. ``threshold`` is that
    distance (default 0.2 m). ``slop`` (default 0.01 m) expands both the named
    horizontal footprint and coarse radius for a released object that settles
    a few millimetres over the nominal edge. The support and height gates stay
    strict, so a cup on the nearby table still fails.
    """
    if len(args) < 5:
        return False
    slop = max(float(args[5]) if len(args) > 5 else 0.01, 0.0)
    if not _inside_named_region_band(model, data, task, args[:5], -slop):
        return False
    object_body_id = _body_id_for_object(model, task, args[0])
    container_body_id = _body_id_for_object(model, task, args[1])
    if object_body_id < 0 or container_body_id < 0:
        return False
    if not _has_upward_support_contact(
        model, data, object_body_id, container_body_id
    ):
        return False
    object_xyz = _resolve_xyz(model, data, task, args[0])
    container_xyz = _resolve_xyz(model, data, task, args[1])
    if any(value != value for value in object_xyz + container_xyz):
        return False
    radius = float(threshold) if threshold is not None else 0.2
    dxy = (
        (object_xyz[0] - container_xyz[0]) ** 2
        + (object_xyz[1] - container_xyz[1]) ** 2
    ) ** 0.5
    return dxy <= radius + slop


def _pred_supported_in_local_box(model, data, task, args, threshold) -> bool:
    """supported_in_local_box(obj, container, x0, x1, y0, y1, z0, z1).

    Test an explicit box in the container root body's live local frame and
    require upward support contact from that container's subtree. This is for
    articulated fixtures whose assets expose no named interior region. The
    explicit bounds exclude their exterior top, front face, and nearby table.

    ``threshold`` optionally shrinks every face inward by that clearance.
    """
    if len(args) < 8:
        return False
    import numpy as np

    object_xyz = _resolve_xyz(model, data, task, args[0])
    object_body_id = _body_id_for_object(model, task, args[0])
    container_body_id = _body_id_for_object(model, task, args[1])
    if (
        object_body_id < 0
        or container_body_id < 0
        or any(value != value for value in object_xyz)
    ):
        return False
    center = np.asarray(data.xpos[container_body_id], dtype=np.float64)
    rotation = np.asarray(data.xmat[container_body_id], dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (np.asarray(object_xyz, dtype=np.float64) - center)
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    lows = np.asarray([float(args[2]), float(args[4]), float(args[6])]) + clearance
    highs = np.asarray([float(args[3]), float(args[5]), float(args[7])]) - clearance
    if not bool(np.all(lows < highs) and np.all(local >= lows) and np.all(local <= highs)):
        return False
    return _has_upward_support_contact(
        model, data, object_body_id, container_body_id
    )


def _pred_inside_local_box(model, data, task, args, threshold) -> bool:
    """inside_local_box(obj, container, x0, x1, y0, y1, z0, z1).

    Test an explicit box in the container root body's live local frame without
    requiring an upward contact.  This is intended for a narrow holder where a
    long object can remain physically wedged against the side walls rather than
    resting on a horizontal floor.  Pair it with ``must_hold`` and an
    orientation predicate so a transient pass-through or a horizontal object
    cannot count as the final state.

    ``threshold`` optionally shrinks every face inward by that clearance.
    """
    if len(args) < 8:
        return False
    import numpy as np

    object_xyz = _resolve_xyz(model, data, task, args[0])
    container_body_id = _body_id_for_object(model, task, args[1])
    if container_body_id < 0 or any(value != value for value in object_xyz):
        return False
    center = np.asarray(data.xpos[container_body_id], dtype=np.float64)
    rotation = np.asarray(data.xmat[container_body_id], dtype=np.float64).reshape(3, 3)
    local = rotation.T @ (np.asarray(object_xyz, dtype=np.float64) - center)
    clearance = max(float(threshold) if threshold is not None else 0.0, 0.0)
    lows = np.asarray([float(args[2]), float(args[4]), float(args[6])]) + clearance
    highs = np.asarray([float(args[3]), float(args[5]), float(args[7])]) - clearance
    return bool(
        np.all(lows < highs)
        and np.all(local >= lows)
        and np.all(local <= highs)
    )


def _pred_inside_any(model, data, task, args, threshold) -> bool:
    """inside_any(obj, c1, c2, ...): obj is inside ANY of the listed containers.

    OR over :func:`_pred_inside` -- needed when several identical containers
    (e.g. three matching bowls) cover objects and the goal must express
    "this cube is under SOME bowl" without naming which instance: the robot
    pairs bowls to cubes by position, so the covering bowl is whichever one
    sits at the cube's slot this episode. Combine with ``not_inside`` over the
    full bowl list to express "covered" / "uncovered" for choreography gating
    (e.g. the cover-and-reveal memory tasks).
    """
    if len(args) < 2:
        return False
    for cont_name in args[1:]:
        if _pred_inside(model, data, task, [args[0], cont_name], threshold):
            return True
    return False


def _pred_inside_contact(model, data, task, args, threshold) -> bool:
    """inside_contact(obj, container, [z_min]): obj is SUPPORTED by the container
    -- there is a contact between the obj and the container's body subtree whose
    NORMAL is vertical (the container pushes the obj up from underneath, as a
    slab does a resting item), and the obj's centre z > z_min (``threshold``,
    default 0).

    For multi-level shelves / cabinets where the geometric ``inside`` /
    ``inside_box`` tests are unreliable -- a large rotated shelf whose body
    origin sits off-centre, or whose slabs span far beyond a small xy radius --
    contact is the robust discriminator. But bare contact alone over-triggers:
    an item that merely BUMPS the shelf's EXTERIOR (a side panel, a wall) while
    being carried past it also touches the shelf, and so does an item that
    landed on a slab and then ROLLED OFF onto the table but is still grazing the
    shelf base. So this predicate requires a SUPPORT contact -- one whose normal
    is vertical (``|normal_z| > NORMAL_Z_MIN``), i.e. the shelf is beneath the
    obj holding it up, not beside it. The normal (not the contact-point height)
    is used so the test is independent of the obj's size: a short can resting on
    a slab (contact barely below its centre) still reads vertical-normal, while
    a side bump reads horizontal-normal regardless of where along the side it
    touches. The roll-off case is handled by the caller with ``must_hold: true``
    (the support contact must still exist at episode end). Rotation- and
    offset-invariant, unlike the xy-radius tests.
    """
    if len(args) < 2:
        return False
    obj_bid = _body_id_for_object(model, task, args[0])
    cbid = _body_id_for_object(model, task, args[1])
    if obj_bid < 0 or cbid < 0:
        return False
    z_min = float(threshold) if threshold is not None else 0.0
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if obj_xyz[2] != obj_xyz[2] or obj_xyz[2] <= z_min:  # NaN or below z_min
        return False
    return _has_upward_support_contact(model, data, obj_bid, cbid)


def _pred_not_inside(model, data, task, args, threshold) -> bool:
    """not_inside(obj, container): obj is NOT inside the container.

    Exact negation of :func:`_pred_inside` -- same default xy radius (0.10m)
    and z window (0.15m), so the inside/outside boundary matches and an obj
    sitting on the rim is consistently classified by both. Use this with
    ``must_hold: true`` to express an exclusion constraint ("this mango must
    not end up in a basket"): without ``must_hold`` a ``not_inside`` subtask
    would latch true on frame 0 (the obj starts on the table, outside) and
    stay "done" even if the robot later drops it in, defeating the point.
    """
    return not _pred_inside(model, data, task, args, threshold)


def _pred_hung_on(model, data, task, args, threshold) -> bool:
    """hung_on(obj, holder, [contact_dz]): obj is HANGING on the holder (mug
    tree / hook rack) -- not merely touching it.

    A plain ``on``/proximity test fires the moment a mug bumps the tree, even
    though it then topples back onto the table. Hanging means the holder
    CARRIES the obj's weight at a raised point:

      1. SUPPORT CONTACT: some contact between the obj and the holder's body
         subtree has a vertical normal (|normal_z| > ``normal_z_min``, default
         0.4 -- tunable via an optional 3rd arg, ``args[2]``) pointing holder->obj
         -- the peg pushes the handle up from underneath. A mug leaning on the
         trunk or grazing a peg from the side reads a horizontal normal and is
         rejected. The default was lowered from 0.5 to 0.4 so a genuinely-hanging
         mug that tilts/swings a little still latches (it is support-contacted at
         peg height); pass a smaller ``args[2]`` to accept more tilt.
      2. RAISED CONTACT: the contact point sits at least ``contact_dz``
         (``threshold``, default 0.08 m) ABOVE the holder's origin. This
         rejects an obj resting on the holder's low base/feet (a flat
         base-plate support contact is vertical too, but sits at base height).
         On MugTree001 @ scale 1.1 the base feet top out at ~+0.03 while the
         lowest hanging peg is at ~+0.14, so 0.08 separates them.
      3. The obj's own centre must be above holder_origin_z + 0.03 (a mug on
         the table beside the tree fails even if it somehow touches a peg).

    Pair with ``must_hold: true`` so the mug must still be hanging at episode
    end (a touch-and-fall never counts).
    """
    NORMAL_Z_MIN_DEFAULT = 0.4
    if len(args) < 2:
        return False
    obj_bid = _body_id_for_object(model, task, args[0])
    hold_bid = _body_id_for_object(model, task, args[1])
    if obj_bid < 0 or hold_bid < 0:
        return False
    contact_dz = float(threshold) if threshold is not None else 0.08
    # |normal_z| above this => vertical (support) contact. Tunable via args[2]:
    # lower accepts a more tilted/swung mug (reduces false negatives for a
    # genuinely-hanging mug); default 0.4 (~66 deg from vertical).
    normal_z_min = float(args[2]) if len(args) > 2 else NORMAL_Z_MIN_DEFAULT
    holder_z = float(data.xpos[hold_bid][2])
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if obj_xyz[2] != obj_xyz[2] or obj_xyz[2] <= holder_z + 0.03:  # NaN / too low
        return False
    # Holder subtree (body + descendants).
    children = {}
    for b in range(model.nbody):
        p = model.body_parentid[b]
        if p >= 0:
            children.setdefault(p, []).append(b)
    subtree = set()
    stack = [hold_bid]
    while stack:
        b = stack.pop()
        if b < 0 or b in subtree:
            continue
        subtree.add(b)
        stack.extend(children.get(b, []))
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        # con.frame[0:3] is the contact normal (world) pointing geom1 -> geom2.
        if b1 == obj_bid and b2 in subtree:
            holder_to_obj_nz = -float(con.frame[2])
        elif b2 == obj_bid and b1 in subtree:
            holder_to_obj_nz = float(con.frame[2])
        else:
            continue
        if holder_to_obj_nz > normal_z_min and float(con.pos[2]) >= holder_z + contact_dz:
            return True
    return False


def _pred_near(model, data, task, args, threshold) -> bool:
    """near(a, b, threshold): 3D euclidean distance < threshold (default 0.05m)."""
    if len(args) < 2:
        return False
    a = _resolve_xyz(model, data, task, args[0])
    b = _resolve_xyz(model, data, task, args[1])
    radius = float(threshold) if threshold is not None else 0.05
    d = sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
    return d < radius


def _pred_open(model, data, task, args, threshold) -> bool:
    """open(obj, joint, [min_qpos]): articulated joint past `min_qpos` (default mid of its range)."""
    if len(args) < 2:
        return False
    obj_name, joint_name = args[0], args[1]
    qpos = _joint_qpos_for_object(model, data, task, obj_name, joint_name)
    if qpos != qpos:  # NaN
        return False
    if threshold is not None:
        return abs(qpos) > float(threshold)
    # Fallback: use joint range midpoint if available.
    import mujoco
    obj = task.find_object(obj_name)
    if obj is None:
        return False
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{obj.instance_id}_{joint_name}"
    )
    if jid < 0:
        return False
    if model.jnt_limited[jid]:
        lo, hi = model.jnt_range[jid]
        mid = 0.5 * (lo + hi)
        # "closed" is the range endpoint closest to zero -- the SAME convention
        # domain.py uses to initialise `qpos: closed` (range-aware shorthand:
        # closed = the bound with smaller |value|). This matters for joints
        # whose range runs [-X, 0], e.g. the microwave microjoint [-2.094, 0]
        # where closed = hi = 0 and open = lo = -2.094: the old
        # `abs(qpos - lo) > half_range` hard-coded closed = lo and read such a
        # joint as "open" while shut (qpos 0 -> |0-(-2.094)| > 1.047 -> True)
        # and "closed" while open -- the "Open the microwave door" green light
        # lit at frame 0 and "Close ..." lit once the door was opened. "open"
        # = qpos has crossed the midpoint toward the FAR end (away from the
        # closed endpoint). For the common [0, X] range (closed = lo = 0) this
        # reduces to `qpos > mid`, identical to the old behaviour, so existing
        # doors/lids/drawers are unchanged; symmetric [-X, X] is also unchanged.
        closed_is_lo = abs(lo) <= abs(hi)
        return (qpos > mid) if closed_is_lo else (qpos < mid)
    return abs(qpos) > 0.3


def _pred_closed(model, data, task, args, threshold) -> bool:
    return not _pred_open(model, data, task, args, threshold)


def _pred_open_any(model, data, task, args, threshold) -> bool:
    """open_any(obj, joint1, joint2, ...): ANY of the listed articulated joints
    is open. OR over :func:`_pred_open` (each joint tested with the same
    threshold). Use to accept "open either the top OR the middle drawer".
    """
    if len(args) < 2:
        return False
    obj_name = args[0]
    for joint_name in args[1:]:
        if _pred_open(model, data, task, [obj_name, joint_name], threshold):
            return True
    return False


def _pred_lifted(model, data, task, args, threshold) -> bool:
    """lifted(obj, [min_z]): object body z above min_z (default obj.pos.z + 0.1)."""
    if not args:
        return False
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if obj_xyz[2] != obj_xyz[2]:
        return False
    obj = task.find_object(args[0])
    if threshold is not None:
        z_thresh = float(threshold)
    elif obj and obj.pos is not None:
        z_thresh = float(obj.pos[2]) + 0.1
    else:
        z_thresh = 0.95
    return obj_xyz[2] > z_thresh


def _pred_not_lifted(model, data, task, args, threshold) -> bool:
    """not_lifted(obj, [min_z]): obj is NOT lifted -- negation of :func:`_pred_lifted`.

    Same z threshold as `lifted` (default obj.pos.z + 0.1), so the boundary
    matches. Use with ``must_hold: true`` for a final-frame "not held at end"
    check, or ``latch_violation: true`` for an any-frame "never picked up"
    check (the latter latches a permanent failure the first frame the obj is
    lifted, so a grab-then-release cannot recover -- matching the not_watered
    pattern for transient violations).
    """
    return not _pred_lifted(model, data, task, args, threshold)


def _pred_in_bottle(model, data, task, args, threshold) -> bool:
    """in_bottle(cube, bottle, [radius]): cube has fallen INSIDE the bottle body.

    Resolves the bottle body via :func:`_body_id_for_object` (task instance-id
    prefix aware), so the bottle's *live, settled* xpos is used -- no
    region/``target`` is needed (the region ``target`` lookup uses the raw
    object name and misses the prefixed body, so a target-relative zone would
    silently collapse to world origin). Requires the cube to be:

      (a) within ``radius`` (default 0.13m) of the bottle in xy, AND
      (b) in the body height: ``origin_z - 0.05 < cube_z < origin_z + 0.20``
          (base through neck). The bottle origin sits near the base/table, so
          a cube resting inside is at roughly ``origin_z`` (NOT below it); the
          window excludes a cube perched on the mouth rim (above origin+0.20).
          A cube beside the bottle on the table is excluded by the xy radius
          (it sits outside the footprint).

    All checks are relative to the live bottle origin, so it is scale-tolerant
    and tracks the freejoint bottle wherever it settles.
    """
    if len(args) < 2:
        return False
    cube_xyz = _resolve_xyz(model, data, task, args[0])
    if any(v != v for v in cube_xyz):  # NaN
        return False
    bbid = _body_id_for_object(model, task, args[1])
    if bbid < 0:
        print(f"[in_bottle] DEBUG: bottle body not found for {args[1]!r}")
        return False
    bx, by, bz = (float(v) for v in data.xpos[bbid][:3])
    # Default radius 0.13 covers the bottle's max xy half-extent (~0.126 at
    # scale 0.65, the elliptical y-axis) so a cube resting against the inner
    # wall (centre dxy up to ~0.111) still counts as inside; a cube BESIDE the
    # bottle touching the outer wall (dxy >= ~0.141) stays outside.
    radius = float(threshold) if threshold is not None else 0.13
    dxy = ((cube_xyz[0] - bx) ** 2 + (cube_xyz[1] - by) ** 2) ** 0.5
    # The bottle body origin is above the base (the model origin sits ~0.14 m
    # above the resting base), so a cube at the bottle-base interior is well
    # BELOW the origin. A window of origin-0.05 would miss it (and would flicker
    # when the bottle is freejoint and settles with the origin near the base).
    # Use origin-0.25 .. origin+0.20: reaches the base regardless of whether the
    # bottle is freejoint:settled (origin near base) or welded (origin high),
    # and still excludes a cube perched on the mouth rim (above origin+0.20) and
    # a cube beside the bottle on the table (excluded by the xy radius).
    in_body_z = (bz - 0.25) < cube_xyz[2] < (bz + 0.20)
    result = (dxy < radius) and in_body_z
    return result


def _pred_not_in_bottle(model, data, task, args, threshold) -> bool:
    """not_in_bottle(cube, bottle): cube is NOT inside the bottle body.

    Negation of :func:`_pred_in_bottle`. Use with ``latch_violation: true``: a
    forbidden cube entering the bottle is a transient event (the cube can be
    pulled back out), so a plain ``must_hold`` end-check would let the
    violation recover -- ``latch_violation`` latches a permanent failure the
    first frame the cube is inside, so once a forbidden cube goes in the
    subtask stays failed even if it is later removed (no green-light relight).
    """
    return not _pred_in_bottle(model, data, task, args, threshold)


def _pred_turned_on(model, data, task, args, threshold) -> bool:
    """turned_on(obj, joint, [min_qpos]): a knob/switch joint is in the "on" position.

    Semantically identical to ``open`` but used for knobs, dials, and switches
    (e.g. stove knob, faucet handle) to distinguish "turning on" from "opening
    a door".  The implementation reuses the same joint-angle check as ``open``.
    """
    return _pred_open(model, data, task, args, threshold)


def _pred_upright(model, data, task, args, threshold) -> bool:
    """upright(obj): object's z-axis is approximately vertical (within ~30 deg).

    Uses the body xmat to check whether the local z-axis points upward.
    If the body cannot be found, falls back to checking that the object's
    world z-position is above a reasonable threshold.
    """
    if not args:
        return False
    bid = _body_id_for_object(model, task, args[0])
    if bid >= 0:
        # xmat is 3x3 rotation matrix stored as 9-element row-major array
        xmat = data.xmat[bid].reshape(3, 3)
        # local z-axis in world frame
        local_z_world = xmat[:, 2]
        # upright means local z is close to world z = (0, 0, 1)
        dot = local_z_world[2]  # dot product with (0,0,1)
        angle_threshold = 0.5  # cos(60deg) ≈ 0.5, allow ~60 deg tilt
        if threshold is not None:
            angle_threshold = float(threshold)
        return dot > angle_threshold
    # Fallback: check z position
    obj_xyz = _resolve_xyz(model, data, task, args[0])
    if obj_xyz[2] != obj_xyz[2]:
        return False
    obj = task.find_object(args[0])
    if obj and obj.pos is not None:
        return obj_xyz[2] > float(obj.pos[2]) + 0.05
    return obj_xyz[2] > 0.9


def _pred_long_axis_vertical(model, data, task, args, threshold) -> bool:
    """long_axis_vertical(obj, [geom]): an elongated object's long axis is raised.

    The longest half-extent of the named geom (``reg_bbox`` by default) selects
    the object's intrinsic length axis.  The absolute dot product between that
    live axis and world up must exceed ``threshold`` (default 0.35).  Absolute
    value intentionally accepts either end pointing upward.  A threshold of
    0.35 allows roughly 70 degrees of tilt away from vertical while rejecting
    a utensil that is horizontal or merely laid across a receiver's rim.
    """
    if not args:
        return False
    import numpy as np

    geom_name = str(args[1]) if len(args) > 1 else "reg_bbox"
    geom_id = _named_geom_id(model, task, args[0], geom_name)
    if geom_id < 0:
        return False
    sizes = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if sizes.size < 3 or not np.all(np.isfinite(sizes[:3])):
        return False
    long_axis = int(np.argmax(sizes[:3]))
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    vertical_component = abs(float(rotation[2, long_axis]))
    minimum = float(threshold) if threshold is not None else 0.35
    return vertical_component > minimum


def _pred_on_edge(model, data, task, args, threshold) -> bool:
    """on_edge(obj, [threshold]): object stands on its SIDE -- its local z-axis
    (the nominal up-axis, e.g. a plate's disk-normal) is approximately HORIZONTAL.

    The complement of :func:`_pred_upright`: `upright` is true when local-z points
    up (object lying flat / in its normal orientation); `on_edge` is true when
    local-z points sideways (object tipped ~90 deg, standing on its rim/edge).
    Use for a plate INSERTED VERTICALLY into a dish-rack slot: a correctly
    inserted plate stands on its rim so its disk-normal is horizontal
    (``on_edge`` true), while a plate merely lying flat on the rack tray reads
    ``on_edge`` false. Pair with ``must_hold: true`` so the plate must still be
    on-edge at episode end.

    ``threshold`` is ``|local_z . world_z|`` -- the cosine of the angle between
    the object's up-axis and the world up. ``on_edge`` is true when this is BELOW
    ``threshold`` (the axis is closer to horizontal than to vertical). Default
    0.5 = within ~30 deg of perfectly on-edge (i.e. tilted more than 60 deg from
    flat); a smaller threshold is stricter (closer to a perfect 90-deg stand).
    """
    if not args:
        return False
    bid = _body_id_for_object(model, task, args[0])
    if bid >= 0:
        xmat = data.xmat[bid].reshape(3, 3)
        dot = float(xmat[2, 2])  # local-z axis . world-z
        thr = float(threshold) if threshold is not None else 0.5
        return abs(dot) < thr
    # No resolvable body -> cannot assess orientation; fail safe (False).
    return False


PREDICATE_REGISTRY: Dict[str, Callable] = {
    "on": _pred_on,
    "stacked_on": _pred_stacked_on,  # cube directly on top of another (strict)
    "on_table": _pred_on_table,  # direct support anywhere on the scene table
    "inside": _pred_inside,
    "in": _pred_inside,         # alias
    "inside_contact": _pred_inside_contact,  # contact + z_min (multi-level shelves)
    "inside_box": _pred_inside_box,
    "in_box": _pred_inside_box,  # alias
    "inside_site": _pred_inside_site,  # container's named <site> box (drawers)
    "inside_site_loose": _pred_inside_site_loose,  # same but no contact gate (mesh receivers, e.g. strainer bowl)
    "inside_site_any": _pred_inside_site_any,  # obj inside ANY of several drawer sites
    "inside_site_any_loose": _pred_inside_site_any_loose,  # any named site, no floor-contact requirement
    "inside_site_volume_any": _pred_inside_site_volume_any,  # full local 3-D site volumes
    "inside_region": _pred_inside_region,  # inside a named live geom
    "not_inside_region": _pred_not_inside_region,  # negation of inside_region
    "keypoint_in_site": _pred_keypoint_in_site,  # one end of an elongated object inside a site
    "supported_in_region": _pred_supported_in_region,  # named region plus upward support
    "inside_region_band": _pred_inside_region_band,  # planar/3-D horizontal region + height band, no contact
    "axis_in_region_band": _pred_axis_in_region_band,  # any point on an elongated object's axis inside region band
    "axis_in_region": _pred_axis_in_region,  # any point on an elongated object's axis inside a live 3-D region
    "supported_in_region_band": _pred_supported_in_region_band,  # planar/3-D region + height band + support
    "supported_in_region_band_stack": _pred_supported_in_region_band_stack,
    "axis_in_local_frustum": _pred_axis_in_local_frustum,
    "supported_geom_in_region_band": _pred_supported_geom_in_region_band,  # visible geom centre in region + support
    "supported_in_region_band_near": _pred_supported_in_region_band_near,  # region/support gate plus coarse radius with settling slop
    "supported_in_local_box": _pred_supported_in_local_box,  # explicit container-local interior + support
    "inside_local_box": _pred_inside_local_box,  # explicit container-local interior; side-wall wedging allowed
    "inside_any": _pred_inside_any,  # obj inside ANY of several containers (OR)
    "not_inside": _pred_not_inside,
    "outside": _pred_not_inside,  # alias
    "hung_on": _pred_hung_on,     # obj hanging on a holder (peg support contact, raised)
    "near": _pred_near,
    "open": _pred_open,
    "open_any": _pred_open_any,  # ANY of several articulated joints is open
    "closed": _pred_closed,
    "close": _pred_closed,      # alias
    "lifted": _pred_lifted,
    "not_lifted": _pred_not_lifted,
    "in_bottle": _pred_in_bottle,         # cube fallen inside bottle body
    "not_in_bottle": _pred_not_in_bottle,  # negation; use with latch_violation
    "turned_on": _pred_turned_on,
    "upright": _pred_upright,
    "long_axis_vertical": _pred_long_axis_vertical,
    "on_edge": _pred_on_edge,        # object on its side (local-z horizontal): plate vertical in a dish rack
    "vertical": _pred_on_edge,       # alias
    "sunk": _pred_sunk,           # dexjoco hammer-nail: mocap nail driven down
    "not_sunk": _pred_not_sunk,   # dexjoco hammer-nail: nail must NOT be driven
    "watered": _pred_watered,      # dexjoco water-plant: spray nozzle near plant
    "not_watered": _pred_not_watered,  # negation; use with latch_violation
    "pressed": _pred_pressed,       # dexjoco touch: spring button pressed N times
    "touch": _pred_pressed,         # alias
    "touched": _pred_pressed,       # alias
    "pressed_exact": _pred_pressed_exact,   # spring button pressed EXACTLY N times
    "pressed_after": _pred_pressed_after,   # button pressed exactly N times, all after a prereq
    "flyover_n": _pred_flyover_n,      # obj entered the flyover zone >= N times
    "swaps_n": _pred_swaps_n,          # cup colors swapped >= N times (color-memory)
    "color_match_place": _pred_color_match_place,  # cube placed into the matching-color cup
    "placed_n": _pred_placed_n,     # >= N objects recorded as placed into a basket
    "order_match": _pred_order_match,  # basket removal order == placement order
    "drawn_n": _pred_drawn_n,           # >= N objects removed from a container (draw count)
    "press_matches_drawn": _pred_press_matches_drawn,  # press button whose value == draw count (strict)
}


def evaluate_goal(model, data, task) -> dict:
    """Evaluate every predicate against the current frame.

    Returns ``{"success": bool, "predicates": [...], "subtasks": [...]}``.

    `success` requires every subtask's predicates to all be true *now* —
    this is the strict, non-latching view used by tooling that just wants
    a single boolean. For latching semantics (subtask once-true stays
    done) use :class:`SubtaskTracker` instead, which keeps history across
    frames.
    """
    pred_results = []
    for g in task.goal:
        fn = PREDICATE_REGISTRY.get(g.type.lower())
        if fn is None:
            ok = False
        else:
            try:
                ok = bool(fn(model, data, task, g.args, g.threshold))
            except Exception:
                ok = False
        pred_results.append({
            "type": g.type, "args": g.args, "ok": ok,
            "subtask": g.subtask, "must_hold": g.must_hold,
            "latch_violation": g.latch_violation,
        })

    # Subtask roll-up: predicates with the same .subtask label are AND-ed.
    subtask_results = []
    all_subtask_ok = True
    for sname in task.subtask_names():
        members = [p for p in pred_results if p["subtask"] == sname]
        s_ok = bool(members) and all(p["ok"] for p in members)
        subtask_results.append({"name": sname, "ok": s_ok,
                                "predicates": members})
        all_subtask_ok = all_subtask_ok and s_ok

    return {
        "success": bool(all_subtask_ok and subtask_results),
        "predicates": pred_results,
        "subtasks": subtask_results,
    }


class SubtaskTracker:
    """Latching subtask completion tracker.

    Use one tracker per episode. Call :meth:`update` every frame (or every
    K frames — sub-Hz polling is fine for slow predicates). Each subtask
    transitions from `pending` → `done` the first frame all of its
    predicates are true. Subtasks declared with `must_hold: true` on any
    of their predicates also require *current-frame* truthiness at the
    final ``status()`` call to count as success.

    Designed so collection (`env._drive_task_cycle`) and inference
    (`scripts/eval_success.py`) can share one schema:

    >>> tracker = SubtaskTracker(task)
    >>> while running:
    ...     tracker.update(model, data)
    >>> status = tracker.status()
    >>> status["success"]                  # whole task
    >>> status["subtasks"][0]["latched"]   # individual subtask history
    """

    def __init__(self, task):
        self.task = task
        self._names = task.subtask_names()
        self._latched: dict = {n: False for n in self._names}
        self._latched_at: dict = {n: None for n in self._names}
        self._must_hold: dict = {
            n: any(p.must_hold for p in task.predicates_by_subtask().get(n, []))
            for n in self._names
        }
        # Subtasks that own a `latch_violation` predicate: the first frame any
        # such predicate is False permanently fails the subtask. Used for
        # negation guards over transient events (e.g. not_watered), where a
        # final-frame must_hold check would miss a mid-episode violation.
        self._latch_violation: dict = {
            n: any(p.latch_violation for p in task.predicates_by_subtask().get(n, []))
            for n in self._names
        }
        self._violated: dict = {n: False for n in self._names}
        self._violated_at: dict = {n: None for n in self._names}
        self._last_eval: Optional[dict] = None
        self._frame: int = 0
        # Ordered-subtask mode (opt-in via Task.ordered_subtasks): a subtask may
        # only latch (or latch a violation) once every subtask earlier in
        # declaration order has latched. Default False -> independent latching,
        # preserving the behaviour every existing task relies on.
        self._ordered = bool(getattr(task, "ordered_subtasks", False))
        self._order_idx = {n: i for i, n in enumerate(self._names)}
        # Order-group resolution: subtasks sharing a non-None order_group may
        # latch in any order among themselves (they do NOT gate one another).
        # A subtask becomes eligible once every EARLIER subtask that is NOT in
        # its own group has latched. Subtasks whose group is None each get a
        # unique singleton group, which makes `_eligible` reduce to the original
        # strict "all earlier latched" rule -- existing tasks are unchanged.
        raw_groups = task.subtask_order_groups() if hasattr(task, "subtask_order_groups") else {}
        self._order_group = {n: raw_groups.get(n) for n in self._names}
        # Assign a unique sentinel to each None-grouped subtask so the
        # group-comparison below treats them as mutually distinct (ordered).
        _sentinel = 0
        for n in self._names:
            if self._order_group[n] is None:
                self._order_group[n] = ("__unique__", _sentinel)
                _sentinel += 1

    def _eligible(self, subtask_name: str) -> bool:
        """Ordered gate: is `subtask_name` allowed to latch this frame?

        In unordered mode always True. In ordered mode a subtask is eligible
        once every EARLIER subtask that is NOT in its own order_group has
        latched -- subtasks sharing an order_group may complete in any order
        (they don't gate each other), but the group as a whole must finish
        before any later subtask in a different group becomes eligible. A later
        subtask whose predicate is incidentally true at frame 0 (e.g. the
        "restored" step of a swap-then-restore dance) must wait for its
        predecessors. Subtasks without an explicit order_group each form their
        own singleton group, so the default is the original strict total order.
        """
        if not self._ordered:
            return True
        i = self._order_idx[subtask_name]
        my_grp = self._order_group[subtask_name]
        for j in range(i):
            if self._order_group[self._names[j]] == my_grp:
                continue
            latched_at = self._latched_at[self._names[j]]
            # A predecessor that first becomes true in this same update must
            # not unlock the next sequential milestone immediately. Without
            # this frame boundary an ordered chain can cascade through several
            # already-true states in one frame.
            if latched_at is None or latched_at >= self._frame:
                return False
        return True

    def update(self, model, data) -> dict:
        self._frame += 1
        ev = evaluate_goal(model, data, self.task)
        for s in ev["subtasks"]:
            if s["ok"] and not self._latched[s["name"]] and self._eligible(s["name"]):
                self._latched[s["name"]] = True
                self._latched_at[s["name"]] = self._frame
        # Latch failures: a latch_violation predicate going False (even once)
        # permanently marks its subtask violated. `not_watered` is False only
        # in the frame(s) the forbidden plant is actually being watered, so we
        # must catch it here, not at the final status() call. Gated by the same
        # ordering rule: a later subtask's violation guard shouldn't fire before
        # the subtask is even active.
        for p in ev["predicates"]:
            if p.get("latch_violation") and not p["ok"] and self._eligible(p["subtask"]):
                sn = p["subtask"]
                if not self._violated[sn]:
                    self._violated[sn] = True
                    self._violated_at[sn] = self._frame
        self._last_eval = ev
        return ev

    def status(self) -> dict:
        """Final status: latched + (optional) must_hold sticky check.

        - `subtasks[i].done`      — latched at any point
        - `subtasks[i].holds_now` — last-eval predicate truth (live)
        - `subtasks[i].violated`  — a latch_violation predicate went False at
          any frame (permanent failure)
        - `success`               — every subtask `done`, every must_hold
          subtask `holds_now`, and no subtask `violated`
        """
        live = self._last_eval["subtasks"] if self._last_eval else []
        live_by_name = {s["name"]: s for s in live}
        per_subtask = []
        all_done = True
        for n in self._names:
            holds = bool(live_by_name.get(n, {}).get("ok", False))
            done = self._latched[n]
            need_hold = self._must_hold[n]
            violated = self._violated[n]
            need_no_violate = self._latch_violation[n]
            ok = done and (holds if need_hold else True) \
                and (not violated if need_no_violate else True)
            per_subtask.append({
                "name": n,
                "done": done,
                "holds_now": holds,
                "must_hold": need_hold,
                "violated": violated,
                "latch_violation": need_no_violate,
                "latched_at_frame": self._latched_at[n],
                "violated_at_frame": self._violated_at[n],
                "ok": ok,
            })
            all_done = all_done and ok
        return {
            "success": bool(all_done and per_subtask),
            "subtasks": per_subtask,
            "frames_observed": self._frame,
        }


# Imported lazily to avoid a top-level Optional import in older versions.
from typing import Optional  # noqa: E402
