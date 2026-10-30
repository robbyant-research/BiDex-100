"""Task spec dataclasses + YAML loader.

A task YAML follows this shape (see tasks/suites/*/*.yaml for examples)::

    name: pick_bowl_to_plate
    language: "Pick up the akita black bowl and place it on the plate."
    scene: libero_kitchen_tabletop_base_style
    robot_pose:           # optional — falls back to arena defaults
      pos: [0.0, -0.4, 0.8]
      quat: [0.382683, 0, 0, 0.923880]
    arm_init_qpos:        # optional — falls back to SimEnv defaults
      left:  [0.7854, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.7854]
      right: [-0.7854, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.7854]
    regions:
      bowl_zone:   { center: [-0.15, 0.0, 0.85], xy_range: [0.05, 0.05], yaw_range: [-0.3, 0.3] }
      plate_zone:  { center: [0.10,  0.0, 0.85], xy_range: [0.02, 0.02] }
    objects:
      - { name: akita_black_bowl, region: bowl_zone }
      - { name: plate,            region: plate_zone, freejoint: false }
    fixtures: []          # static (non-freejoint) objects, e.g. cabinets
    articulations: []     # joint init for articulated objects: [{object: microwave, joint: microjoint, qpos: 0.0}]
    goal:
      - { type: on, args: [akita_black_bowl, plate_zone] }
    obj_of_interest: [akita_black_bowl, plate]
    init_states_file: pick_bowl_to_plate.npy   # optional — relative to tasks/init_states/<suite>/

The dataclasses below are intentionally thin wrappers around dict so that
adding a new predicate or a new region attribute is one line in the YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import yaml


@dataclass
class Region:
    name: str
    target: str = "world"   # "world" or a fixture/object name (xy resolved relative to its body)
    center: Tuple[float, float, float] = (0.0, 0.0, 0.85)
    xy_range: Tuple[float, float] = (0.0, 0.0)            # half-extents on x, y
    yaw_range: Tuple[float, float] = (0.0, 0.0)           # min, max yaw rotation (rad)
    z_offset: float = 0.0                                  # extra z lift over center
    # Half-extent of uniform z randomization applied on top of (center_z +
    # z_offset). 0 = all objects in this region spawn at the same z (legacy).
    # >0 lets two objects that share xy end up at different z (stack) instead of
    # embedding into each other at the same z.
    z_range: float = 0.0
    # Per-object z layer step (z_layer metres between consecutive objects sampled
    # in this region). The Nth object gets z = base + (N-1)*z_layer, so any two
    # objects differ in z by >= z_layer. Set z_layer >= the object's height and
    # two objects can NEVER overlap in z -> no embedding, even if they share xy
    # (they stack instead). Unlike z_range (random, not guaranteed) this is a
    # deterministic no-embed guarantee. 0 = off (use z_offset / z_range).
    z_layer: float = 0.0
    # Min centre-to-centre xy separation between objects sampled in this region
    # (rejection sampling). 0 = no rejection (objects may overlap in xy). >0
    # forces distinct xy positions so cubes neither embed nor pre-stack.
    min_distance: float = 0.0
    # When True, `center.z` is interpreted as the height the object's BASE
    # (lowest collision point) should rest at, not its origin z. The sampler
    # lifts the sampled origin by the object's `resting_offset` (origin -> lowest
    # collision geom, measured at the compiled qpos0 pose) so the base lands on
    # `center.z` instead of penetrating it. Use for objects whose body origin
    # sits well above their base (e.g. articraft hinged/pullout boxes whose
    # origin is 5-13 cm above the floor) -- without this they spawn buried in
    # the table. Default False preserves the legacy origin-z semantics that
    # object-relative regions (fruit-in-bowl, ice-in-cup) hand-tune against.
    rest_on_surface: bool = False


@dataclass
class ObjectInit:
    name: str                                              # catalog name (e.g. akita_black_bowl)
    region: Optional[str] = None                           # region name -> sample on reset
    pos: Optional[Tuple[float, float, float]] = None       # explicit pose override
    quat: Optional[Tuple[float, float, float, float]] = None
    # Fixed orientation composed onto the region sampler's random yaw at spawn
    # (q_final = q_yaw ⊗ spawn_quat -- see RegionSampler.sample_for_task).
    # Lets a REGION-placed object spawn with a fixed flip/pitch -- e.g. a
    # bottle upside-down (quat [0,1,0,0] = 180 deg about world-x) -- while still
    # jittering xy + heading per episode and honoring the region's
    # `min_distance` rejection. Ignored for objects without a region (use
    # ``quat`` above for a one-shot fixed orientation). (w, x, y, z).
    spawn_quat: Optional[Tuple[float, float, float, float]] = None
    freejoint: bool = True                                 # set false for fixtures
    joint_init: dict = field(default_factory=dict)         # {joint_local_name: qpos}
    instance_id: int = 0                                   # filled by Task.__post_init__
    # Per-instance MJCF rewrites applied by scripts/visualize_scene._prepare_object_xml.
    # scale=1.0 means no-op; mass=None means leave inertials alone. Both go through
    # scripts/scale_object_xml.scale_tree so size/mass stay physically consistent.
    scale: float = 1.0
    mass: Optional[float] = None
    # Hammer/nail markers for the dexjoco hammer-nail task family. `is_nail`
    # marks a mocap-driven nail body whose depth the env's contact script
    # advances on hammer strikes; `is_hammer` marks a free-body hammer the
    # script treats as the striking tool. Both default False so every other
    # task is unaffected.
    is_nail: bool = False
    is_hammer: bool = False
    # dexjoco water-plant markers: `is_spray` marks a free-body watering can the
    # robot grasps; the `watered` predicate reads its obj{i}_ref_point site to
    # check proximity to a plant. Plants need no flag -- they are passed as the
    # `watered(plant)` goal arg. Defaults False so every other task is unaffected.
    is_spray: bool = False
    # dexjoco touch/press button marker: `is_button` marks a welded-base,
    # spring-return push button (bell). The env's per-step button script counts
    # distinct presses (cap slide joint descending past the press threshold then
    # releasing) by incrementing `press_count`; the `pressed` predicate reads it.
    # Defaults False so every other task is unaffected.
    is_button: bool = False
    # Runtime press counter (NOT loaded from YAML). Incremented by
    # env._apply_button_interaction on each rising edge of the button cap; zeroed
    # by env._reset_button_state at the start of every episode. Lives on the
    # ObjectInit so the stateless `pressed` predicate can read it via
    # task.find_object(name).press_count without a handle to the env.
    press_count: int = 0
    # Runtime per-press snapshot log (NOT loaded from YAML). On every rising
    # edge of this button's cap, env._apply_button_interaction appends a dict
    # mapping every OTHER button's name -> its press_count at that instant. This
    # lets the stateless `pressed_after` predicate prove that every press of
    # this button happened AFTER a prerequisite button reached its own target
    # (no premature press), which a bare monotonic count cannot express. Cleared
    # by env._reset_button_state at the start of every episode.
    press_log: list = field(default_factory=list)
    # dexjoco-style "place into a container in order, then remove in the same
    # order" task family. `is_basket` marks the open container whose interior
    # the env's per-step order script monitors; `orderable` marks each small
    # object that gets placed into / removed from it. Both default False so every
    # other task is unaffected. The script records, on the basket ObjectInit,
    # the ordered catalog-name lists `place_seq` (objects in the order they first
    # entered) and `remove_seq` (order they first left after entering); the
    # stateless `placed_n` / `order_match` predicates read these via
    # task.find_object(name).place_seq / .remove_seq without an env handle
    # (same pattern as `pressed` reading press_count).
    is_basket: bool = False
    orderable: bool = False
    place_seq: list = field(default_factory=list)   # NOT loaded from YAML; env-populated
    remove_seq: list = field(default_factory=list)  # NOT loaded from YAML; env-populated
    # "Appear on button press" marker for the cover-and-reveal memory tasks:
    # a `reveal` object (e.g. a cap that must hide a cube) is NOT spawned at
    # its region at reset -- the env parks it at its explicit `pos`/`quat`
    # (e.g. under the table, invisible) and stashes the region-sampled pose;
    # the first time any button's press_count crosses 1, the env teleports it
    # to that stashed pose (the "cups appear" moment), after which it is a
    # normal free body the robot can lift. Default False so every other task
    # is unaffected. NOTE: pose teleporting only runs on the fresh (region-
    # sampled) reset path; init_states restores keep whatever the saved state
    # holds and skip the reveal.
    reveal: bool = False
    # Flyover counter ("cycle over the other cup N times"): `flyover_zone`
    # marks the watched container (e.g. the cup the held cube must pass over),
    # `flyover` marks the moving object (the held cube). The env's per-tick
    # flyover tracker edge-detects the object entering the zone's live xy
    # radius (0.06 m) + in_bottle-style z window and increments
    # `flyover_count` on each outside->inside transition; the stateless
    # `flyover_n` predicate reads it (same pattern as `pressed` reading
    # press_count). Both default False; `flyover_count` is runtime-only
    # (zeroed at every reset, NOT loaded from YAML).
    flyover_zone: bool = False
    flyover: bool = False
    flyover_count: int = 0
    # Color-memory task markers ("cube loses its color after N presses, cups
    # swap colors as the cube passes over them, place into the matching cup").
    # `color_cube` marks the cube whose visual rgba the env randomizes (red or
    # blue) at reset, greys out after the press milestone, and restores on
    # placement; `color_cup` marks each of the two cups whose colors swap.
    # All runtime fields below are env-populated, NOT loaded from YAML:
    #   cube_initial_color  -- the randomized spawn color ("red"/"blue")
    #   cup_color           -- the cup's CURRENT color (swapped by the env)
    #   color_swap_count    -- how many cup-color swaps have fired (cap 3)
    #   color_greyed        -- cube rgba already greyed out post-press
    #   color_restored      -- cube rgba already restored on a placement
    #   color_matched       -- the placement cup's color == cube_initial_color
    color_cube: bool = False
    color_cup: bool = False
    cube_initial_color: str = ""
    cup_color: str = ""
    color_swap_count: int = 0
    color_greyed: bool = False
    color_restored: bool = False
    color_matched: bool = False
    # Integer label for a press-button in the "draw paper, press the matching
    # number" family: the button's numeric value (1..N). The `press_matches_drawn`
    # predicate reads the env-populated `remove_seq` length (sheets drawn) and
    # checks that the button whose `button_value` equals that count was pressed.
    # None for every non-numbered object. Loaded from the YAML `value:` key.
    button_value: Optional[int] = None


@dataclass
class FixtureInit(ObjectInit):
    """Convenience alias: an object with freejoint=False."""

    def __post_init__(self):
        self.freejoint = False


@dataclass
class InitPredicate:
    """Reserved for future use (forced placement constraints that don't fit Region).

    LIBERO has e.g. `(:init (Open microwave))` — we instead express that via
    `articulations:` in the YAML. This dataclass is kept so future BDDL-style
    predicates can be added without a schema break.
    """
    type: str
    args: List[Any] = field(default_factory=list)


@dataclass
class GoalPredicate:
    type: str                                              # "on", "inside", "open", "closed", "near"
    args: List[Any] = field(default_factory=list)
    threshold: Optional[float] = None                      # tolerance for "near"
    # Subtask label: predicates with the same label are AND-grouped into a
    # single subtask. Defaults to "default" so legacy single-goal YAMLs still
    # produce one valid subtask. Subtask success uses *latching* semantics
    # (once true at any frame, stays true) unless `must_hold` is set, in
    # which case the predicate must also be true at episode end.
    subtask: str = "default"
    # Optional order-group tag (only consulted when Task.ordered_subtasks is
    # True). Subtasks sharing an order_group may latch in ANY order among
    # themselves; a subtask becomes eligible once every EARLIER subtask that is
    # NOT in its own group has latched. Subtasks without a tag each form their
    # own singleton group, so the default behaviour stays a strict total order
    # over declaration order -- existing tasks are unchanged. Use this to mark a
    # run of interchangeable subtasks (e.g. "pick & place snack A" vs "snack B")
    # that sit between ordered milestones (open the cabinet first, close it last).
    order_group: Optional[str] = None
    must_hold: bool = False
    # Latch a PERMANENT failure the first frame this predicate is False. For
    # negation predicates whose violation is transient (e.g. `not_watered`:
    # watering is a momentary trigger-press, so by the final frame the nozzle
    # has moved and a plain `must_hold` end-check reads True again, missing
    # the mid-episode watering). `must_hold` is a final-frame stickiness check;
    # `latch_violation` is an any-frame failure latch. They compose: a subtask
    # succeeds only if latched-true, (optionally) holds at end, and never
    # violated.
    latch_violation: bool = False


@dataclass
class Task:
    name: str
    suite: str
    language: str
    scene: str
    objects: List[ObjectInit]
    fixtures: List[FixtureInit] = field(default_factory=list)
    regions: dict = field(default_factory=dict)            # name -> Region
    articulations: List[dict] = field(default_factory=list)
    goal: List[GoalPredicate] = field(default_factory=list)
    obj_of_interest: List[str] = field(default_factory=list)
    robot_pose: Optional[dict] = None                      # {pos, quat} override
    arm_init_qpos: Optional[dict] = None                   # {left, right}
    init_states_file: Optional[str] = None
    yaml_path: Optional[str] = None
    # RoboCasa procedural kitchens: when `scene == "robocasa_kitchen"` the
    # scene XML is built on the fly from a (layout, style) pair via
    # bidex.robocasa.build_robocasa_scene_xml.
    layout_id: Optional[int] = None
    style_id: Optional[int] = None
    enable_fixtures: List[str] = field(default_factory=list)
    clutter_mode: int = 0
    scene_seed: int = 0
    # Wall-clock budget for one episode (seconds). When --collect-cycle is on
    # and this is exceeded, env auto-saves the current episode as a failure.
    # Inference scripts (`scripts/eval_success.py`) use the same field to
    # decide when to stop a rollout. `max_steps` overrides if set.
    max_seconds: Optional[float] = None
    max_steps: Optional[int] = None
    # Robot assembly identity for this task. Resolved against
    # bidex.robots.registry.ROBOT_REGISTRY. None means "fall back to
    # config.robot_config.robot_type"; env / replay honour the priority
    # --robot CLI > Task.robot_type > config default.
    robot_type: Optional[str] = None
    # When true, subtasks must complete in declaration order: a subtask only
    # latches once every earlier subtask (first-appearance order in `goal`) has
    # latched. Used by sequential tasks whose milestones are individually
    # reachable from the start state -- e.g. a swap-then-restore dance where the
    # "restored" subtask's predicate is already true at frame 0 and must NOT
    # latch until its predecessors are done. Default False preserves the
    # independent (parallel) latching every existing task relies on.
    ordered_subtasks: bool = False
    # When True (default), env._reset_button_state jitters each button's mocap
    # base around its declared pos (rejection-sampled so no two overlap) for
    # per-episode placement variety. Set False to keep buttons FIXED at their
    # declared positions -- needed when a task places many buttons close together
    # in a known row (e.g. the "numbers 1-5" row, where value is encoded by
    # position and the button min-separation rejection would scatter them).
    randomize_buttons: bool = True
    # Opt-in region shuffle: a list of object names whose declared `region`s are
    # randomly PERMUTED among themselves each reset, so the scene re-shuffles
    # which object sits in which zone while every zone stays a distinct,
    # non-overlapping spawn box (e.g. a "find the target utensil in the tray"
    # task where the target no longer always spawns at the same slot). The
    # permuted regions must themselves be exchangeable (same `target` /
    # absolute-frame semantics) and pinned/non-overlapping -- the sampler does
    # not re-run min_distance rejection ACROSS swapped regions. Default [] (off)
    # preserves every existing task's behaviour; collected init_states restore
    # bypass RegionSampler entirely, so previously-collected episodes are
    # unaffected even for a task that later opts in.
    shuffle_regions: List[str] = field(default_factory=list)
    # Shuffle mode for `shuffle_regions`: default None / "full" draws any
    # uniform permutation; "cyclic" restricts the permutation to the N cyclic
    # shifts (object i takes the region of object (i-k) mod N), which PRESERVES
    # the group's declared cyclic order. Needed when the choreography is defined
    # relative to the objects' order around a layout (e.g. the triangle
    # rotation task: a->b->c must stay the world-clockwise cycle for the fixed
    # instance-level goal sequence to always be the clockwise one), while the
    # whole cycle still starts at a random position each episode. Off by
    # default; only read when shuffle_regions is non-empty.
    shuffle_mode: Optional[str] = None
    # Scored subtask declarations synced from task_annotations.json
    # (scripts/sync_annotations_to_yamls.py): [{name, score}]. First-appearance
    # order matches the goal[*].subtask group order == SubtaskTracker index
    # order, so eval scoring weights come straight from this yaml (the yaml is
    # the single source of truth; task_annotations.json is only the fallback).
    subtasks: List[dict] = field(default_factory=list)
    # OOD (out-of-domain) directives. Present ONLY in `<task>__ood.yaml`
    # variant configs; consumed by the eval runner when domain=out_of_domain
    # (see bidex/eval/ood.py). Keys:
    #   scene_randomize: bool   -- swap floor/wall/table textures + jitter lights
    #   distractors: {count_min, count_max} -- inject N objects from sibling tasks
    # (Region offsets / object scale / left-right mirroring are NOT runtime
    # directives -- they are baked directly into the OOD yaml's own
    # regions/objects by scripts/gen_ood_yamls.py.)
    ood: Optional[dict] = None
    # Optional mid-episode instruction schedule for Instruction_Following
    # "switch" tasks: [{until_step: int, text: str}, ..., {text: str}]. When
    # non-empty, env.get_obs() serves the text of the FIRST segment whose
    # until_step exceeds the current control step (last entry = final
    # instruction). Empty -> obs["language"] = task.language every step.
    language_schedule: List[dict] = field(default_factory=list)

    def __post_init__(self):
        # Stable instance ids in declaration order. The MJCF builder uses these
        # as the obj{i}_ prefix, so downstream code can predict joint / body
        # names without re-reading the assembled XML.
        for i, obj in enumerate(self.objects + self.fixtures, start=1):
            obj.instance_id = i

    # ---- access helpers ----

    def all_object_inits(self) -> List[ObjectInit]:
        """objects + fixtures in instance-id order."""
        return list(self.objects) + list(self.fixtures)

    def subtask_names(self) -> List[str]:
        """Stable, declaration-order list of subtask labels in `goal`.

        Predicates without an explicit `subtask:` tag fall under "default".
        Returned order is the order in which each subtask label first
        appears in the YAML — keeping it stable lets the inference
        success-rate report and the env progress publication speak the
        same indices.
        """
        seen: List[str] = []
        for g in self.goal:
            if g.subtask not in seen:
                seen.append(g.subtask)
        return seen if seen else ["default"]

    def predicates_by_subtask(self) -> dict:
        out: dict = {}
        for g in self.goal:
            out.setdefault(g.subtask, []).append(g)
        return out

    def subtask_order_groups(self) -> dict:
        """Map each subtask label to its order_group.

        A subtask's group is the first non-None ``order_group`` among its
        predicates (a subtask's predicates are expected to share one group; if
        they disagree, the first wins). Subtasks with no group map to None, which
        :class:`SubtaskTracker` treats as a unique singleton group so they stay
        fully ordered relative to their neighbours.
        """
        groups: dict = {}
        for sname in self.subtask_names():
            grp = next(
                (g.order_group for g in self.goal
                 if g.subtask == sname and g.order_group is not None),
                None,
            )
            groups[sname] = grp
        return groups

    def find_object(self, name: str) -> Optional[ObjectInit]:
        for o in self.all_object_inits():
            if o.name == name:
                return o
        return None

    def to_object_specs(
        self,
        sampled_poses: Optional[dict] = None,
    ) -> List[Tuple[str, Tuple[float, float, float], Optional[Tuple[float, float, float, float]], float, Optional[float]]]:
        """Convert to the (name, pos, quat, scale, mass) tuples that visualize_scene expects.

        `sampled_poses` is an optional `{object_name: (pos, quat)}` mapping
        produced by RegionSampler at reset time. If provided, sampled poses
        win over the spec's static pos/quat for objects that have a region.
        `scale` and `mass` are per-instance MJCF rewrites (see ObjectInit).
        """
        out = []
        for o in self.all_object_inits():
            if sampled_poses and o.name in sampled_poses:
                pos, quat = sampled_poses[o.name]
            elif o.pos is not None:
                pos = tuple(o.pos)
                quat = tuple(o.quat) if o.quat is not None else None
            else:
                # No region, no explicit pose -> drop on origin of region center if any
                if o.region:
                    pos = (0.0, 0.0, 0.85)  # placeholder; will be re-sampled at reset
                else:
                    pos = (0.0, 0.0, 0.85)
                quat = None
            out.append((o.name, pos, quat, float(o.scale), o.mass))
        return out

    # ---- YAML loader ----

    @classmethod
    def from_yaml(cls, path: str, suite: Optional[str] = None) -> "Task":
        with open(path, "r") as f:
            d = yaml.safe_load(f)

        # Suite: explicit > parent dir name > "default"
        if suite is None:
            suite = os.path.basename(os.path.dirname(os.path.abspath(path))) or "default"

        regions_raw = d.get("regions", {}) or {}
        regions = {
            rname: Region(
                name=rname,
                target=rspec.get("target", "world"),
                center=tuple(rspec.get("center", (0.0, 0.0, 0.85))),
                xy_range=tuple(rspec.get("xy_range", (0.0, 0.0))),
                yaw_range=tuple(rspec.get("yaw_range", (0.0, 0.0))),
                z_offset=float(rspec.get("z_offset", 0.0)),
                z_range=float(rspec.get("z_range", 0.0)),
                z_layer=float(rspec.get("z_layer", 0.0)),
                min_distance=float(rspec.get("min_distance", 0.0)),
                rest_on_surface=bool(rspec.get("rest_on_surface", False)),
            )
            for rname, rspec in regions_raw.items()
        }

        def _parse_objs(raw, fixture: bool) -> list:
            out = []
            for spec in raw or []:
                init_cls = FixtureInit if fixture else ObjectInit
                out.append(
                    init_cls(
                        name=spec["name"],
                        region=spec.get("region"),
                        pos=tuple(spec["pos"]) if spec.get("pos") is not None else None,
                        quat=tuple(spec["quat"]) if spec.get("quat") is not None else None,
                        spawn_quat=(tuple(spec["spawn_quat"]) if spec.get("spawn_quat") is not None else None),
                        freejoint=bool(spec.get("freejoint", not fixture)),
                        joint_init=dict(spec.get("joint_init", {})),
                        scale=float(spec.get("scale", 1.0)),
                        mass=(float(spec["mass"]) if spec.get("mass") is not None else None),
                        is_nail=bool(spec.get("nail", False)),
                        is_hammer=bool(spec.get("hammer", False)),
                        is_spray=bool(spec.get("spray", False)),
                        is_button=bool(spec.get("button", False)),
                        is_basket=bool(spec.get("basket", False)),
                        orderable=bool(spec.get("orderable", False)),
                        reveal=bool(spec.get("reveal", False)),
                        flyover_zone=bool(spec.get("flyover_zone", False)),
                        flyover=bool(spec.get("flyover", False)),
                        color_cube=bool(spec.get("color_cube", False)),
                        color_cup=bool(spec.get("color_cup", False)),
                        button_value=(int(spec["value"]) if spec.get("value") is not None else None),
                    )
                )
            return out

        objects = _parse_objs(d.get("objects", []), fixture=False)
        fixtures = _parse_objs(d.get("fixtures", []), fixture=True)

        goal = [
            GoalPredicate(
                type=str(g["type"]).lower(),
                args=list(g.get("args", [])),
                threshold=g.get("threshold"),
                subtask=str(g.get("subtask", "default")),
                must_hold=bool(g.get("must_hold", False)),
                latch_violation=bool(g.get("latch_violation", False)),
                order_group=(str(g["order_group"]) if g.get("order_group") is not None else None),
            )
            for g in (d.get("goal") or [])
        ]

        return cls(
            name=d.get("name") or os.path.splitext(os.path.basename(path))[0],
            suite=suite,
            language=d.get("language", ""),
            scene=d["scene"],
            objects=objects,
            fixtures=fixtures,
            regions=regions,
            articulations=list(d.get("articulations", []) or []),
            goal=goal,
            obj_of_interest=list(d.get("obj_of_interest", []) or []),
            robot_pose=d.get("robot_pose"),
            arm_init_qpos=d.get("arm_init_qpos"),
            init_states_file=d.get("init_states_file"),
            yaml_path=os.path.abspath(path),
            layout_id=d.get("layout_id"),
            style_id=d.get("style_id"),
            enable_fixtures=list(d.get("enable_fixtures", []) or []),
            clutter_mode=int(d.get("clutter_mode", 0)),
            scene_seed=int(d.get("scene_seed", 0)),
            max_seconds=(float(d["max_seconds"]) if d.get("max_seconds") is not None else None),
            max_steps=(int(d["max_steps"]) if d.get("max_steps") is not None else None),
            robot_type=(str(d["robot_type"]) if d.get("robot_type") is not None else None),
            ordered_subtasks=bool(d.get("ordered_subtasks", False)),
            randomize_buttons=bool(d.get("randomize_buttons", True)),
            shuffle_regions=list(d.get("shuffle_regions", []) or []),
            shuffle_mode=(str(d["shuffle_mode"]) if d.get("shuffle_mode") is not None else None),
            subtasks=[dict(s) for s in (d.get("subtasks") or [])],
            ood=(dict(d["ood"]) if d.get("ood") is not None else None),
            language_schedule=[dict(s) for s in (d.get("language_schedule") or [])],
        )
