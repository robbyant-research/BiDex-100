from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import mujoco
import yaml

from bidex.tasks import load_task, predicates


SUITES = Path(__file__).resolve().parents[1] / "bidex" / "tasks_data" / "suites"


def _load(suite: str, task_name: str):
    with (SUITES / suite / f"{task_name}.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class _Task:
    def __init__(self, objects):
        self._objects = {obj.name: obj for obj in objects}
        self.regions = {}

    def find_object(self, name):
        return self._objects.get(name)


def _compile(body_xml: str):
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{body_xml}</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _task(*names):
    return _Task(
        SimpleNamespace(name=name, instance_id=index)
        for index, name in enumerate(names, start=1)
    )


def test_inside_region_requires_the_whole_ball_inside_cylinder():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.03 0 0">
          <geom name="obj1_ice_sphere" type="sphere" size="0.025"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="cylinder" size="0.04 0.05"/>
        </body>
        """
    )

    assert not predicates._pred_inside_region(
        model, data, _task("ice", "cup"), ["ice", "cup", "reg_int"], 0.025
    )

    model.body("obj1_object").pos[0] = 0.01
    mujoco.mj_forward(model, data)
    assert predicates._pred_inside_region(
        model, data, _task("ice", "cup"), ["ice", "cup", "reg_int"], 0.025
    )


def test_inside_region_without_clearance_accepts_settled_ice_near_cup_wall():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.023 0 0">
          <geom name="obj1_ice_sphere" type="sphere" size="0.025"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="cylinder" size="0.042 0.06"/>
        </body>
        """
    )
    task = _task("ice", "cup")

    assert predicates._pred_inside_region(
        model, data, task, ["ice", "cup", "reg_int"], None
    )
    assert not predicates._pred_inside_region(
        model, data, task, ["ice", "cup", "reg_int"], 0.025
    )

    model.body("obj1_object").pos[0] = 0.068
    mujoco.mj_forward(model, data)
    assert not predicates._pred_inside_region(
        model, data, task, ["ice", "cup", "reg_int"], None
    )


def test_inside_region_supports_ellipsoid_bowl_regions():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.02 0 0">
          <geom name="obj1_item" type="sphere" size="0.01"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="ellipsoid" size="0.06 0.05 0.03"/>
        </body>
        """
    )
    task = _task("item", "bowl")

    assert predicates._pred_inside_region(
        model, data, task, ["item", "bowl", "reg_int"], None
    )

    model.body("obj1_object").pos[0] = 0.07
    mujoco.mj_forward(model, data)
    assert not predicates._pred_inside_region(
        model, data, task, ["item", "bowl", "reg_int"], None
    )


def test_on_table_requires_direct_upward_table_support():
    model, data = _compile(
        """
        <body name="table" pos="0 0 0.4">
          <geom name="table_collision" type="box" size="0.4 0.4 0.4"/>
        </body>
        <body name="obj1_object" pos="0 0 0.824">
          <freejoint/>
          <geom type="sphere" size="0.025"/>
        </body>
        """
    )
    task = _task("milk")

    assert predicates._pred_on_table(model, data, task, ["milk"], None)

    qpos_address = model.jnt_qposadr[0]
    data.qpos[qpos_address : qpos_address + 3] = [0.6, 0.0, 0.824]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_on_table(model, data, task, ["milk"], None)


def test_lifted_default_is_relative_to_spawn_height():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0 0 0.85">
          <geom type="sphere" size="0.01"/>
        </body>
        """
    )
    task = _task("scissors")
    task.find_object("scissors").pos = [0.0, 0.0, 0.85]

    assert not predicates._pred_lifted(model, data, task, ["scissors"], None)
    assert predicates._pred_lifted(model, data, task, ["scissors"], 0.82)

    model.body("obj1_object").pos[2] = 0.96
    mujoco.mj_forward(model, data)
    assert predicates._pred_lifted(model, data, task, ["scissors"], None)


def test_keypoint_in_site_accepts_spoon_head_with_handle_outside():
    model, data = _compile(
        """
        <body name="obj1_object">
          <geom name="obj1_reg_bbox" type="box" size="0.02 0.16 0.01"/>
        </body>
        <body name="obj2_object">
          <site name="obj2_contain_region" type="box"
                pos="0 0.12 0" size="0.05 0.05 0.05"/>
        </body>
        """
    )

    assert predicates._pred_keypoint_in_site(
        model,
        data,
        _task("spoon", "tray"),
        ["spoon", "tray", "contain_region", "reg_bbox", "y+"],
        0.75,
    )

    model.body("obj1_object").pos[2] = 0.2
    mujoco.mj_forward(model, data)
    assert not predicates._pred_keypoint_in_site(
        model,
        data,
        _task("spoon", "tray"),
        ["spoon", "tray", "contain_region", "reg_bbox", "y+"],
        0.75,
    )


def test_supported_in_region_rejects_support_outside_region(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.14 0 0">
          <geom name="obj1_item" type="sphere" size="0.02"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_spawn_1" type="cylinder" size="0.12 0.1"/>
        </body>
        """
    )
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: True)

    assert not predicates._pred_supported_in_region(
        model, data, _task("item", "shelf"), ["item", "shelf", "spawn_1"], None
    )

    model.body("obj1_object").pos[0] = 0.08
    mujoco.mj_forward(model, data)
    assert predicates._pred_supported_in_region(
        model, data, _task("item", "shelf"), ["item", "shelf", "spawn_1"], None
    )


def test_supported_in_region_band_accepts_planar_interior_and_rejects_rim(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="0 0 0.04">
          <geom name="obj1_item" type="sphere" size="0.01"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="box" size="0.05 0.04 0.00001"/>
        </body>
        """
    )
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: True)
    task = _task("item", "cup")
    args = ["item", "cup", "reg_int", -0.02, 0.08]

    assert predicates._pred_supported_in_region_band(model, data, task, args, 0.005)

    model.body("obj1_object").pos[:] = [0.048, 0.0, 0.10]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_supported_in_region_band(model, data, task, args, 0.005)


def test_inside_region_band_accepts_contained_object_without_bottom_contact(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.01 0 0.06">
          <geom name="obj1_item" type="sphere" size="0.01"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="box" size="0.05 0.04 0.00001"/>
        </body>
        """
    )
    task = _task("spoon", "cup")
    args = ["spoon", "cup", "reg_int", -0.08, 0.18]
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: False)

    assert predicates._pred_inside_region_band(model, data, task, args, 0.0)

    model.body("obj1_object").pos[:] = [0.07, 0.0, 0.06]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_inside_region_band(model, data, task, args, 0.0)


def test_axis_in_region_band_accepts_tilted_utensil_with_tip_inside():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0 0.06 0.04" euler="60 0 0">
          <geom name="obj1_reg_bbox" type="box" size="0.01 0.10 0.01"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="cylinder" size="0.04 0.00001"/>
        </body>
        """
    )
    task = _task("spoon", "cup")
    args = ["spoon", "cup", "reg_int", "reg_bbox", -0.08, -0.01]

    # The utensil centre is outside the opening, but its lower end is inside.
    assert not predicates._pred_inside_region_band(
        model, data, task, ["spoon", "cup", "reg_int", -0.08, -0.01], 0.0
    )
    assert predicates._pred_long_axis_vertical(
        model, data, task, ["spoon", "reg_bbox"], 0.35
    )
    assert predicates._pred_axis_in_region_band(model, data, task, args, 0.0)

    # A similarly tilted utensil wholly beside the cup must still fail.
    model.body("obj1_object").pos[1] = 0.12
    mujoco.mj_forward(model, data)
    assert not predicates._pred_axis_in_region_band(model, data, task, args, 0.0)

    # Crossing the opening above the rim is not an insertion.  The previous
    # task configuration used z_hi=+0.18 and incorrectly accepted this state.
    model.body("obj1_object").pos[:] = [0.0, 0.06, 0.12]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_axis_in_region_band(model, data, task, args, 0.0)


def test_axis_in_region_accepts_any_shallow_inside_part_but_rejects_outside():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0 0.06 0.049">
          <geom name="obj1_reg_bbox" type="box" size="0.01 0.10 0.01"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="cylinder" size="0.04 0.05"/>
        </body>
        """
    )
    task = _task("spoon", "cup")
    args = ["spoon", "cup", "reg_int", "reg_bbox"]

    # Part of the horizontal spoon is 1 mm below the cup's true interior top.
    # The old fixed z_hi=0.03 band rejects it even though it is inside.
    assert not predicates._pred_axis_in_region_band(
        model, data, task, args + [-0.08, 0.03], 0.0
    )
    assert predicates._pred_axis_in_region(model, data, task, args, 0.0)

    # Crossing the opening above the rim is outside, not an insertion.
    model.body("obj1_object").pos[2] = 0.051
    mujoco.mj_forward(model, data)
    assert not predicates._pred_axis_in_region(model, data, task, args, 0.0)

    # The same shallow pose wholly beside the cup is also outside.
    model.body("obj1_object").pos[:] = [0.041, 0.06, 0.049]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_axis_in_region(model, data, task, args, 0.0)


def test_spoon_task_accepts_normal_two_centimeter_insertion():
    task_config = _load("Generalization", "pick_up_spoon_from_tray_into_cup")
    placement = next(
        goal for goal in task_config["goal"]
        if goal["type"] == "axis_in_region"
    )
    spoon_name, cup_name = placement["args"][:2]
    model, data = _compile(
        """
        <body name="obj1_object" pos="0 0 0.129" euler="90 0 0">
          <geom name="obj1_reg_bbox" type="box" size="0.01 0.10 0.01"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_int" type="cylinder" size="0.04 0.05"/>
        </body>
        """
    )
    task = _task(spoon_name, cup_name)

    # The cup rim is z=0.05 and the spoon's lower end is z=0.029: a normal
    # 2.1 cm insertion that should not require reaching 6 cm below the rim.
    assert predicates._pred_axis_in_region(
        model, data, task, placement["args"], placement.get("threshold", 0.0)
    )


def test_supported_geom_in_region_band_uses_visible_object_center(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.09 0 0.04">
          <geom name="obj1_reg_bbox" pos="-0.02 0 0"
                type="box" size="0.03 0.04 0.02"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_reg_bbox" type="box" size="0.08 0.12 0.005"/>
        </body>
        """
    )
    task = _task("cheese", "board")
    args = ["cheese", "board", "reg_bbox", "reg_bbox", -0.03, 0.12]
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: True)

    # The body origin is outside x=0.08, while the visible cheese bbox centre
    # is at x=0.07 and physically supported by the board.
    assert not predicates._pred_supported_in_region_band(
        model,
        data,
        task,
        ["cheese", "board", "reg_bbox", -0.03, 0.12],
        0.0,
    )
    assert predicates._pred_supported_geom_in_region_band(
        model, data, task, args, 0.0
    )

    model.body("obj1_object").pos[0] = 0.13
    mujoco.mj_forward(model, data)
    assert not predicates._pred_supported_geom_in_region_band(
        model, data, task, args, 0.0
    )


def test_inside_site_any_loose_accepts_either_slot_without_contact(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.03 0 0.01">
          <geom type="box" size="0.01 0.01 0.01"/>
        </body>
        <body name="obj2_object">
          <site name="obj2_slot_a" pos="-0.03 0 0" size="0.02 0.02 0.02"/>
          <site name="obj2_slot_b" pos="0.03 0 0" size="0.02 0.02 0.02"/>
        </body>
        """
    )
    task = _task("cube", "holder")
    args = ["cube", "holder", "slot_a", "slot_b", -0.02, 0.015]
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: False)

    assert predicates._pred_inside_site_any_loose(model, data, task, args, 0.0)

    model.body("obj1_object").pos[:] = [0.0, 0.0, 0.01]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_inside_site_any_loose(model, data, task, args, 0.0)


def test_supported_in_region_band_near_allows_settling_slop_but_rejects_outside_region(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.206 0 0.17">
          <geom name="obj1_item" type="cylinder" size="0.045 0.055"/>
        </body>
        <body name="obj2_object">
          <geom name="obj2_tray_shell" type="box"
                pos="0.15 0 0.03" size="0.13 0.10 0.01"/>
        </body>
        """
    )
    task = _task("cup", "espresso_machine")
    args = ["cup", "espresso_machine", "tray_shell", -0.02, 0.23, 0.01]
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: True)

    # A cup can settle 6 mm beyond the nominal 0.2 m radial boundary while
    # remaining physically supported in the front tray.
    assert predicates._pred_supported_in_region_band_near(
        model, data, task, args, 0.2
    )

    # The same broad radial distance must not accept a cup beside the machine.
    model.body("obj1_object").pos[:] = [0.0, 0.19, 0.08]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_supported_in_region_band_near(
        model, data, task, args, 0.2
    )


def test_supported_in_local_box_rejects_exterior_top(monkeypatch):
    model, data = _compile(
        """
        <body name="obj1_object" pos="-0.1 0 0.04">
          <geom name="obj1_item" type="sphere" size="0.01"/>
        </body>
        <body name="obj2_object" pos="0 0 0"/>
        """
    )
    monkeypatch.setattr(predicates, "_has_upward_support_contact", lambda *args: True)
    task = _task("item", "cabinet")
    args = ["item", "cabinet", -0.2, 0.0, -0.1, 0.1, -0.1, 0.1]

    assert predicates._pred_supported_in_local_box(model, data, task, args, None)

    model.body("obj1_object").pos[2] = 0.15
    mujoco.mj_forward(model, data)
    assert not predicates._pred_supported_in_local_box(model, data, task, args, None)


def test_inside_local_box_uses_live_container_frame_without_requiring_floor_contact():
    model, data = _compile(
        """
        <body name="obj1_object" pos="0.1 0.12 0.04">
          <geom name="obj1_item" type="sphere" size="0.01"/>
        </body>
        <body name="obj2_object" euler="0 0 90"/>
        """
    )
    task = _task("spoon", "rack")
    args = ["spoon", "rack", 0.03, 0.19, -0.17, -0.09, -0.05, 0.18]

    assert predicates._pred_inside_local_box(model, data, task, args, 0.005)

    model.body("obj1_object").pos[:] = [0.0, 0.0, 0.04]
    mujoco.mj_forward(model, data)
    assert not predicates._pred_inside_local_box(model, data, task, args, 0.005)


def test_grill_task_accepts_stacked_food_below_closed_lid_crown():
    task_config = _load("Long_Horizon", "grill_drumsticks_fish_and_squash")
    placement = next(
        goal for goal in task_config["goal"]
        if goal["type"] == "inside_local_box"
    )
    food_name, grill_name = placement["args"][:2]
    model, data = _compile(
        """
        <body name="obj1_object" pos="0 0 0.42">
          <geom name="obj1_reg_bbox" type="box" size="0.025 0.04 0.02"/>
        </body>
        <body name="obj2_object"/>
        """
    )
    task = _task(food_name, grill_name)

    # A food root at z=0.42 represents an upper item in a short stack. It is
    # above the cookbox wall (0.39) but below the closed lid crown (~0.446).
    assert predicates._pred_inside_local_box(
        model, data, task, placement["args"], placement.get("threshold", 0.0)
    )

    # Keep rejecting a root above the physical lid envelope.
    model.body("obj1_object").pos[2] = 0.451
    mujoco.mj_forward(model, data)
    assert not predicates._pred_inside_local_box(
        model, data, task, placement["args"], placement.get("threshold", 0.0)
    )


def test_closed_container_tasks_accept_interior_without_support_in_id_and_ood():
    tasks = {
        "Generalization": ("store_snacks_in_slide_cabinet", 2),
        "Long_Horizon": ("restock_fridge_with_condiments_and_bottles", 4),
    }
    for suite, (name, expected_count) in tasks.items():
        for suffix in ("", "__ood"):
            task = _load(suite, name + suffix)
            placements = [
                goal for goal in task["goal"]
                if goal["type"] == "inside_local_box"
            ]
            assert len(placements) == expected_count
            assert all(goal.get("must_hold") is True for goal in placements)
            assert all(
                goal["type"] != "supported_in_local_box"
                for goal in task["goal"]
            )


def test_long_axis_vertical_rejects_horizontal_and_allows_large_tilt():
    model, data = _compile(
        """
        <body name="obj1_object">
          <geom name="obj1_reg_bbox" type="box" size="0.01 0.10 0.01"/>
        </body>
        """
    )
    task = _task("spoon")
    args = ["spoon", "reg_bbox"]

    # The longest local axis is y. At the dataset spawn pose it is horizontal.
    assert not predicates._pred_long_axis_vertical(model, data, task, args, 0.35)

    # A 30 degree tilt from vertical (60 degrees above horizontal) is valid.
    model.body("obj1_object").quat[:] = [0.8660254, 0.5, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    assert predicates._pred_long_axis_vertical(model, data, task, args, 0.35)

    # Direction is intentionally ignored: handle-up and handle-down are both
    # vertical placements, so the absolute long-axis/world-z dot must pass.
    model.body("obj1_object").quat[:] = [0.7071068, -0.7071068, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    assert predicates._pred_long_axis_vertical(model, data, task, args, 0.35)


def test_ordered_tracker_does_not_cascade_multiple_steps_in_one_frame(monkeypatch):
    goal_a = SimpleNamespace(
        subtask="step a", must_hold=False, latch_violation=False, order_group=None
    )
    goal_b = SimpleNamespace(
        subtask="step b", must_hold=False, latch_violation=False, order_group=None
    )
    task = SimpleNamespace(
        ordered_subtasks=True,
        goal=[goal_a, goal_b],
        subtask_names=lambda: ["step a", "step b"],
        predicates_by_subtask=lambda: {"step a": [goal_a], "step b": [goal_b]},
        subtask_order_groups=lambda: {"step a": None, "step b": None},
    )
    both_true = {
        "success": True,
        "predicates": [],
        "subtasks": [
            {"name": "step a", "ok": True, "predicates": []},
            {"name": "step b", "ok": True, "predicates": []},
        ],
    }
    monkeypatch.setattr(predicates, "evaluate_goal", lambda *_: both_true)
    tracker = predicates.SubtaskTracker(task)

    tracker.update(None, None)
    first = tracker.status()["subtasks"]
    assert first[0]["done"]
    assert not first[1]["done"]

    tracker.update(None, None)
    second = tracker.status()["subtasks"]
    assert second[1]["done"]


def test_ordered_tracker_allows_parallel_group_but_not_next_group_same_frame(monkeypatch):
    goals = [
        SimpleNamespace(subtask="left", must_hold=False, latch_violation=False, order_group="phase_1"),
        SimpleNamespace(subtask="right", must_hold=False, latch_violation=False, order_group="phase_1"),
        SimpleNamespace(subtask="next", must_hold=False, latch_violation=False, order_group="phase_2"),
    ]
    names = [goal.subtask for goal in goals]
    task = SimpleNamespace(
        ordered_subtasks=True,
        goal=goals,
        subtask_names=lambda: names,
        predicates_by_subtask=lambda: {goal.subtask: [goal] for goal in goals},
        subtask_order_groups=lambda: {goal.subtask: goal.order_group for goal in goals},
    )
    all_true = {
        "success": True,
        "predicates": [],
        "subtasks": [
            {"name": name, "ok": True, "predicates": []} for name in names
        ],
    }
    monkeypatch.setattr(predicates, "evaluate_goal", lambda *_: all_true)
    tracker = predicates.SubtaskTracker(task)

    tracker.update(None, None)
    first = {row["name"]: row["done"] for row in tracker.status()["subtasks"]}
    assert first == {"left": True, "right": True, "next": False}

    tracker.update(None, None)
    second = {row["name"]: row["done"] for row in tracker.status()["subtasks"]}
    assert second == {"left": True, "right": True, "next": True}


def test_bimanual_and_utensil_fix_semantics_are_declared_in_id_and_ood():
    utensil_tasks = {
        "Generalization": (
            "drop_two_spoons_from_tray_into_two_cups",
            "pick_up_spoon_from_tray_into_cup",
        ),
        "Limited_Demonstrations": ("pick_up_fork_from_tray_into_cup",),
    }
    for suite, names in utensil_tasks.items():
        for name in names:
            for suffix in ("", "__ood"):
                task = _load(suite, name + suffix)
                placements = [
                    goal for goal in task["goal"]
                    if goal["type"] == "axis_in_region"
                ]
                assert placements
                for placement in placements:
                    assert placement["args"][2:] == ["reg_int", "reg_bbox"]
                    assert placement.get("must_hold") is True
                    orientations = [
                        goal for goal in task["goal"]
                        if goal["subtask"] == placement["subtask"]
                        and goal["type"] == "long_axis_vertical"
                    ]
                    assert orientations == []
                if name == "drop_two_spoons_from_tray_into_two_cups":
                    assert not task.get("ordered_subtasks", False)

    for suffix in ("", "__ood"):
        rack = _load(
            "Long_Horizon",
            "place_strainer_with_tomatoes_and_spoon_on_dish_rack" + suffix,
        )
        spoon_goals = [
            goal for goal in rack["goal"]
            if goal["subtask"] == "Place the spoon onto the dish rack"
        ]
        placement = next(goal for goal in spoon_goals if goal["type"] == "inside_local_box")
        orientation = next(goal for goal in spoon_goals if goal["type"] == "long_axis_vertical")
        assert placement["args"][2:] == [0.03, 0.19, 0.095, 0.17, -0.05, 0.18]
        assert placement["threshold"] == 0.005
        assert orientation["threshold"] == 0.35
        assert all(goal.get("must_hold") is True for goal in spoon_goals)

    for suffix in ("", "__ood"):
        scale = _load(
            "Generalization",
            "measure_two_paprika_shakers_on_digital_scale" + suffix,
        )
        assert [goal.get("order_group") for goal in scale["goal"]] == [
            "on_scale", "on_scale", "on_scale", "on_scale",
            "off_scale", "off_scale",
        ]

        for name in ("swap_fruits_between_bowls", "swap_candles_between_bowls"):
            swap = _load("Memory", name + suffix)
            assert [goal.get("order_group") for goal in swap["goal"]] == [
                "swap_out", "swap_out", "swap_back", "swap_back", "swap_back",
            ]
        candles = _load("Memory", "swap_candles_between_bowls" + suffix)
        assert all(goal["type"] == "supported_in_region_band" for goal in candles["goal"])
        assert all(goal["args"][2] == "reg_bbox" for goal in candles["goal"])


def test_drain_strainer_and_tomato_lift_thresholds_match_reference_task():
    reference = _load(
        "Long_Horizon",
        "place_strainer_with_tomatoes_and_spoon_on_dish_rack",
    )
    reference_lifts = {
        goal["args"][0].rsplit("/", 1)[-1]: goal["threshold"]
        for goal in reference["goal"]
        if goal["type"] == "lifted"
    }
    assert reference_lifts["Strainer002"] == 0.92
    assert reference_lifts["tomato_0"] == 0.91

    for suffix in ("", "__ood"):
        task = _load(
            "Limited_Demonstrations",
            "drain_three_tomatoes_in_strainer" + suffix,
        )
        lifts = {
            goal["args"][0].rsplit("/", 1)[-1]: goal["threshold"]
            for goal in task["goal"]
            if goal["type"] == "lifted"
        }
        assert lifts == {
            "Strainer001": 0.92,
            "tomato_0": 0.91,
            "tomato_1": 0.91,
            "tomato_2": 0.91,
        }


def test_make_espresso_uses_supported_upper_recess_in_id_and_ood():
    for suffix in ("", "__ood"):
        task = _load("Generalization", "make_espresso" + suffix)
        placement = task["goal"][-1]
        assert placement["type"] == "supported_in_local_box"
        assert placement["args"] == [
            "rc/lightwheel/glass_cup/GlassCup003",
            "espresso_machine_with_articulated_components",
            0.08,
            0.31,
            -0.06,
            0.06,
            0.18,
            0.29,
        ]
        assert placement.get("must_hold") is True


def test_reported_false_positive_and_false_negative_fixes_are_declared_in_id_and_ood():
    for suffix in ("", "__ood"):
        grill = _load("Long_Horizon", "grill_drumsticks_fish_and_squash" + suffix)
        assert all(
            goal["type"] == "inside_local_box"
            for goal in grill["goal"][1:6]
        )
        assert all(goal["args"][7] == 0.45 for goal in grill["goal"][1:6])

        recycle = _load("Long_Horizon", "recycle_five_cans_into_bin" + suffix)
        assert all(goal["type"] == "inside_local_box" for goal in recycle["goal"][1:6])

        fill = _load("Long_Horizon", "fill_pot_and_bottle_with_sugar_cubes_and_veg" + suffix)
        pot_lift = next(goal for goal in fill["goal"] if goal["subtask"] == "Left arm lifts the pot")
        assert pot_lift["threshold"] == 0.90

        drainer = _load("Generalization", "put_peeler_and_pizza_cutter_in_drainer" + suffix)
        assert all(
            goal["type"] == "inside_site_volume_any"
            for goal in (drainer["goal"][1], drainer["goal"][3])
        )

        tools = _load("Generalization", "insert_spatula_and_rolling_pin_into_glass_cup" + suffix)
        assert tools["goal"][1]["type"] == "supported_in_region_band"
        assert tools["goal"][1]["args"][2] == "reg_bbox"

        cubes = _load("Generalization", "insert_two_cubes_into_slot" + suffix)
        assert all(goal["type"] == "inside_site_any_loose" for goal in cubes["goal"])
        assert all(goal["args"][2:4] == ["slot_a", "slot_b"] for goal in cubes["goal"])

        for suite, name in (
            ("Generalization", "drop_two_spoons_from_tray_into_two_cups"),
            ("Generalization", "pick_up_spoon_from_tray_into_cup"),
            ("Limited_Demonstrations", "pick_up_fork_from_tray_into_cup"),
        ):
            task = _load(suite, name + suffix)
            assert all(
                goal["type"] != "supported_in_region_band"
                for goal in task["goal"]
            )
            assert any(goal["type"] == "axis_in_region" for goal in task["goal"])

        cheese = _load("Limited_Demonstrations", "serve_cheese_on_two_cutting_boards" + suffix)
        assert cheese["goal"][1]["type"] == "supported_geom_in_region_band"
        assert cheese["goal"][3]["type"] == "supported_geom_in_region_band"


def test_bottle_tasks_only_shrink_the_requested_last_sugar_cube():
    tasks = (
        ("Long_Horizon", "fill_pot_and_bottle_with_sugar_cubes_and_veg"),
        ("Instruction_Following", "open_bottle_and_drop_two_sugar_cubes_in_bottle"),
    )
    expected_scales = {
        "SugarCube001": 3,
        "SugarCube002": 3,
        "SugarCube003": 3,
        "SugarCube004": 2.8,
    }
    for suite, name in tasks:
        for suffix in ("", "__ood"):
            task = _load(suite, name + suffix)
            sugar_cube_scales = {
                obj["name"].rsplit("/", 1)[-1]: obj["scale"]
                for obj in task["objects"]
                if "/sugar_cube/" in obj["name"]
            }
            assert sugar_cube_scales == expected_scales


def test_bottle_drop_goals_must_remain_inside_in_id_and_ood():
    tasks = (
        ("open_bottle_and_drop_one_sugar_cube_in_bottle", 1),
        ("open_bottle_and_drop_two_sugar_cubes_in_bottle", 2),
    )
    for name, expected_count in tasks:
        for suffix in ("", "__ood"):
            task = _load("Instruction_Following", name + suffix)
            placements = [
                goal for goal in task["goal"]
                if goal["type"] == "in_bottle"
            ]
            assert len(placements) == expected_count
            assert all(goal.get("must_hold") is True for goal in placements)


def test_ood_parallel_task_semantics_match_id():
    milk_id = _load("Generalization", "take_milk_from_microwave")
    milk_ood = _load("Generalization", "take_milk_from_microwave__ood")
    assert milk_ood["language"] == milk_id["language"]
    assert not milk_id.get("ordered_subtasks", False)
    assert not milk_ood.get("ordered_subtasks", False)
    assert milk_id["goal"][2]["type"] == "on_table"
    assert milk_ood["goal"][2]["type"] == "on_table"
    assert milk_id["goal"][2].get("must_hold") is True
    assert milk_ood["goal"][2].get("must_hold") is True

    for task_name in (
        "pour_ice_between_two_cups",
        "pour_ice_between_two_cups__ood",
    ):
        ice = _load("Memory", task_name)
        region_goals = ice["goal"][:4]
        assert all(
            goal["type"] in {"inside_region", "not_inside_region"}
            for goal in region_goals
        )
        assert all("threshold" not in goal for goal in region_goals)

    for task_name in (
        "organize_items_in_desk_caddy",
        "organize_items_in_desk_caddy__ood",
    ):
        caddy = _load("Generalization", task_name)
        assert caddy["goal"][0]["type"] == "lifted"
        assert caddy["goal"][0]["threshold"] == 0.9

    grill_id = _load("Generalization", "grill_steak_and_corn")
    grill_ood = _load("Generalization", "grill_steak_and_corn__ood")
    assert grill_ood["language"] == grill_id["language"]
    assert grill_ood["goal"][0]["order_group"] == grill_id["goal"][0]["order_group"]

    freezer_id = _load("Limited_Demonstrations", "store_frozen_food_in_freezer")
    freezer_ood = _load("Limited_Demonstrations", "store_frozen_food_in_freezer__ood")
    assert freezer_ood["goal"][0]["order_group"] == freezer_id["goal"][0]["order_group"]


def test_confirmed_fix_id_and_ood_predicate_semantics_match():
    tasks = {
        "Generalization": (
            "drop_two_spoons_from_tray_into_two_cups",
            "heat_sandwich_bread_in_microwave",
            "insert_two_cubes_into_slot",
            "measure_two_paprika_shakers_on_digital_scale",
            "pick_up_spoon_from_tray_into_cup",
            "put_marshmallow_and_cookie_dough_ball_in_bowl",
            "put_sweet_potato_and_onion_in_bowl",
            "store_snacks_in_slide_cabinet",
        ),
        "Instruction_Following": (
            "pack_four_mangoes_into_baskets",
            "pack_two_mangoes_into_two_baskets",
        ),
        "Limited_Demonstrations": (
            "pick_up_fork_from_tray_into_cup",
            "rotate_two_cubes_around_plates",
        ),
        "Long_Horizon": (
            "place_strainer_with_tomatoes_and_spoon_on_dish_rack",
            "restock_fridge_with_condiments_and_bottles",
            "sort_mixed_food_into_boxes",
        ),
        "Memory": ("swap_fruits_between_bowls",),
    }

    def semantics(task):
        return [
            (
                goal["type"],
                goal["args"],
                goal.get("threshold"),
                goal.get("must_hold", False),
                goal.get("order_group"),
            )
            for goal in task["goal"]
        ]

    for suite, names in tasks.items():
        for name in names:
            assert semantics(_load(suite, name)) == semantics(_load(suite, name + "__ood"))


def test_instruction_following_goal_labels_are_declared():
    for path in (SUITES / "Instruction_Following").glob("*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = {subtask["name"] for subtask in task["subtasks"]}
        referenced = {goal.get("subtask", "default") for goal in task["goal"]}
        assert referenced <= declared, path.name


def test_instruction_following_shelf_goals_require_the_canonical_tier():
    expected_tiers = {
        "put_jam_and_lemon_from_tray_onto_shelf": {
            "rc/objaverse/jam/jam_0": "spawn_2",
            "rc/objaverse/lemon/lemon_2": "spawn_2",
        },
        "put_lemon_on_shelf_then_take_jam_into_tray": {
            "rc/objaverse/lemon/lemon_2": "spawn_2",
        },
        "put_orange_on_shelf_then_take_back_into_tray": {
            "rc/objaverse/orange/orange_3": "spawn_2",
        },
        "take_jam_into_tray_and_put_milk_on_shelf": {
            "rc/objaverse/milk/milk_3": "spawn_1",
        },
        "take_jam_into_tray_then_put_back_on_shelf": {
            "rc/objaverse/jam/jam_0": "spawn_2",
        },
        "take_teapot_into_tray_and_put_orange_on_shelf": {
            "rc/objaverse/orange/orange_3": "spawn_2",
        },
        "take_teapot_into_tray_then_put_back_on_shelf": {
            "rc/objaverse/teapot/teapot_0": "spawn_1",
        },
    }

    for task_name, expected in expected_tiers.items():
        for suffix in ("", "__ood"):
            task = _load("Instruction_Following", task_name + suffix)
            shelf_goals = {
                goal["args"][0]: goal
                for goal in task["goal"]
                if goal["type"] == "supported_in_region"
            }
            assert shelf_goals.keys() == expected.keys(), task_name + suffix
            for item, tier in expected.items():
                assert shelf_goals[item]["args"][2:] == [tier], (
                    task_name + suffix,
                    item,
                )


def _placement_goal(task, model, data, item):
    goal = next(g for g in task.goal if g.args[0] == item and g.must_hold)
    return predicates.PREDICATE_REGISTRY[goal.type](
        model, data, task, goal.args, goal.threshold
    )


def test_cereal_spoon_accepts_rim_insertion_but_rejects_outside():
    for suffix in ("", "__ood"):
        task = load_task("Generalization", "make_cereal_in_bowl" + suffix)
        model, data = _compile('''
            <body name="obj1_object"><geom type="sphere" size="0.001"/></body>
            <body name="obj4_object" pos="0 0.105 0.09125"
                  quat="0 0 -0.382683 0.923880">
              <geom name="obj4_reg_bbox" type="box" size="0.0191 0.09535 0.0157"/>
            </body>
        ''')
        assert _placement_goal(task, model, data, "spoon")
        for pos in ([0, 0.105, 0.25], [0.25, 0.105, 0.09125], [0, 0.105, -0.1]):
            model.body("obj4_object").pos[:] = pos
            mujoco.mj_forward(model, data)
            assert not _placement_goal(task, model, data, "spoon")


def test_coffee_mug_accepts_tray_end_support_but_rejects_hovering():
    for suffix in ("", "__ood"):
        task = load_task("Generalization", "place_coffee_mug_and_spoon_on_tray" + suffix)
        model, data = _compile('''
            <body name="obj2_object"><geom type="box" size="0.145 0.08 0.005"/></body>
            <body name="obj1_object" pos="0.115 0 0.029">
              <freejoint/><geom type="box" size="0.02 0.02 0.025"/>
            </body>
        ''')
        assert _placement_goal(task, model, data, "red_coffee_mug")
        for pos in ([0.115, 0, 0.15], [0.22, 0, 0.029]):
            data.qpos[:3] = pos
            mujoco.mj_forward(model, data)
            assert not _placement_goal(task, model, data, "red_coffee_mug")


def test_drainer_bounds_all_three_rotated_site_dimensions():
    for suffix in ("", "__ood"):
        task = load_task("Generalization", "put_peeler_and_pizza_cutter_in_drainer" + suffix)
        model, data = _compile('''
            <body name="obj3_object">
              <site name="obj3_left_region" type="box" pos="0 0.086954 0.094598"
                    quat="0.5 0.5 -0.5 -0.5" size="0.07231 0.081312 0.131922"/>
              <site name="obj3_right_region" type="box" pos="0 -0.072142 0.094598"
                    quat="0.5 0.5 -0.5 -0.5" size="0.07231 0.081312 0.131922"/>
            </body>
            <body name="obj1_object"><geom type="sphere" size="0.005"/></body>
        ''')
        cases = [([0.3, 0.08, 0.05], False), ([0, 0.08, 0.06], True),
                 ([0, -0.07, 0.06], True), ([0, 0.08, 0.2], False),
                 ([0, 0.3, 0.06], False)]
        for pos, expected in cases:
            model.body("obj1_object").pos[:] = pos
            mujoco.mj_forward(model, data)
            assert _placement_goal(task, model, data, "pizza_cutter") == expected


def test_mustard_accepts_supported_overhang_but_rejects_outside():
    for suffix in ("", "__ood"):
        task = load_task("Generalization", "put_mustard_and_honey_bottle_on_tray" + suffix)
        model, data = _compile('''
            <body name="obj3_object"><geom type="box" size="0.0667 0.1106 0.018"/></body>
            <body name="obj1_object" pos="0 0.109 0.076">
              <freejoint/><geom type="box" size="0.029 0.021 0.059"/>
            </body>
        ''')
        assert _placement_goal(task, model, data, "mustard")
        for pos in ([0, 0.109, 0.18], [0.16, 0, 0.076]):
            data.qpos[:3] = pos
            mujoco.mj_forward(model, data)
            assert not _placement_goal(task, model, data, "mustard")


def test_bowl_stack_accepts_either_order_and_revokes_removed_support():
    for suffix in ("", "__ood"):
        task = load_task("Generalization", "put_marshmallow_and_cookie_dough_ball_in_bowl" + suffix)
        for bottom, top in (("marshmallow", "cookie_dough_ball"),
                            ("cookie_dough_ball", "marshmallow")):
            bottom_id = task.find_object(bottom).instance_id
            top_id = task.find_object(top).instance_id
            model, data = _compile(f'''
                <body name="obj3_object">
                  <geom name="obj3_reg_int" type="ellipsoid" pos="0 0 0.018"
                        size="0.058 0.058 0.025" contype="0" conaffinity="0"/>
                  <geom type="box" pos="0 0 -0.008" size="0.065 0.065 0.008"/>
                </body>
                <body name="obj{bottom_id}_object" pos="0 0 0.024">
                  <freejoint/><geom type="sphere" size="0.025"/>
                </body>
                <body name="obj{top_id}_object" pos="0 0 0.073">
                  <freejoint/><geom type="sphere" size="0.025"/>
                </body>
            ''')
            assert _placement_goal(task, model, data, bottom)
            assert _placement_goal(task, model, data, top)
            tracker = predicates.SubtaskTracker(task)
            tracker.update(model, data)
            # The upper item is still inside the bowl footprint but loses its
            # support path when the lower item is removed.
            data.qpos[0] = 0.2
            mujoco.mj_forward(model, data)
            assert not _placement_goal(task, model, data, top)
            tracker.update(model, data)
            label = next(g.subtask for g in task.goal if g.args[0] == top and g.must_hold)
            status = next(s for s in tracker.status()["subtasks"] if s["name"] == label)
            assert status["done"] and not status["holds_now"] and not status["ok"]
