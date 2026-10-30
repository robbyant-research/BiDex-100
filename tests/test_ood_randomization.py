import os
import pyexpat
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from bidex.eval import ood
from bidex.eval import run_leaderboard_eval as evaluation
from bidex.env import BiDexEnv


def test_camera_manifest_fingerprint_and_limits():
    # MuJoCo/EGL must not replace the Python runtime's expat ABI. This parser
    # construction mirrors the OOD scene rewrite used during evaluation.
    assert pyexpat.ParserCreate() is not None
    manifest = ood.load_camera_pose_manifest()
    assert manifest["name"] == ood.CAMERA_RANDOMIZATION_SCHEMA
    assert len(manifest["poses"]) >= 8
    pos_limit = np.asarray(manifest["translation_limit_m"])
    rot_limit = np.asarray(manifest["rotation_limit_deg"])
    for pose in manifest["poses"]:
        assert np.all(np.abs(pose["delta_pos"]) <= pos_limit)
        assert np.all(np.abs(pose["delta_euler_deg"]) <= rot_limit)


def test_camera_plan_is_stable_and_has_exact_occurrence_counts():
    args = ("dual_fr3v2_sharpa", "Memory", "pour_ice_between_two_cups")
    first = ood.camera_randomization_plan(*args, 2, 0)
    assert first == ood.camera_randomization_plan(*args, 2, 0)
    manifest = ood.load_camera_pose_manifest()
    expected = {p["id"] for p in manifest["poses"]}
    plans = [
        ood.camera_randomization_plan(*args, seed, episode)
        for seed in (2, 3)
        for episode in range(25)
    ]
    assert Counter(plan["occurrence_mode"] for plan in plans) == {
        "nominal": 20,
        "head_only": 10,
        "wrists_only": 10,
        "head_and_wrists": 10,
    }
    for seed in (2, 3):
        assert Counter(
            ood.camera_randomization_plan(*args, seed, episode)["occurrence_mode"]
            for episode in range(25)
        ) == {
            "nominal": 10,
            "head_only": 5,
            "wrists_only": 5,
            "head_and_wrists": 5,
        }
    for plan in plans:
        mode = plan["occurrence_mode"]
        head = plan["cameras"]["head"]
        wrists = [plan["cameras"][name]
                  for name in ("left_wrist", "right_wrist")]
        assert (head["pose_id"] != "nominal") == (
            mode in {"head_only", "head_and_wrists"}
        )
        assert all((delta["pose_id"] != "nominal") == (
            mode in {"wrists_only", "head_and_wrists"}
        ) for delta in wrists)
        for delta in (head, *wrists):
            if delta["pose_id"] == "nominal":
                assert delta["delta_pos"] == [0.0, 0.0, 0.0]
                assert delta["delta_euler_deg"] == [0.0, 0.0, 0.0]
            else:
                assert delta["pose_id"] in expected
    astribot = ood.camera_randomization_plan(
        "astribot_s1_fixed_sharpa", "Generalization", "task", 2, 0
    )
    assert astribot["cameras"]["head"]["delta_pos"][0] >= 0.0
    assert astribot["cameras"]["left_wrist"]["delta_pos"][0] <= 0.0
    assert astribot["cameras"]["right_wrist"]["delta_pos"][0] >= 0.0
    franka = ood.camera_randomization_plan(
        "dual_fr3v2_sharpa", "Generalization", "task", 2, 0
    )
    for camera in ("left_wrist", "right_wrist"):
        delta = franka["cameras"][camera]
        assert delta["delta_pos"][2] >= 0.0
        assert max(abs(v) for v in delta["delta_pos"]) <= 0.0135
        assert max(abs(v) for v in delta["delta_euler_deg"]) <= 6.0


def test_id_domain_keeps_nominal_camera_extrinsics(monkeypatch):
    task = SimpleNamespace()
    monkeypatch.setattr(evaluation, "load_task", lambda *_args: task)

    loaded, distractors, apply_offset = evaluation.load_task_for_domain(
        "Memory", "task", "in_domain", 0, 0, episode=0
    )

    assert loaded is task
    assert not hasattr(loaded, "_ood_camera_plan")
    assert distractors == []
    assert apply_offset is False


def _minimal_scene(path: str):
    Path(path).write_text(
        """<mujoco><asset>
        <texture name="texplane" type="2d" file="old_floor.png"/>
        <texture name="tex-wall" type="2d" file="old_wall.png"/>
        <texture name="tex-table" type="cube" file="old_table.png"/>
        </asset><worldbody>
        <light name="light1" pos="1 1 4" diffuse=".8 .8 .8" castshadow="false"/>
        <light name="light2" pos="-3 -3 4" diffuse=".8 .8 .8" castshadow="false"/>
        </worldbody></mujoco>""",
        encoding="utf-8",
    )


def test_scene_randomization_is_reproducible_and_casts_one_shadow(tmp_path):
    scene = tmp_path / "libero_tabletop_base_style.xml"
    _minimal_scene(str(scene))
    pools = {
        "floor": [("floor", str(tmp_path / "floor.png"))],
        "wall": [("wall", str(tmp_path / "wall.png"))],
        "table": [("table", str(tmp_path / "table.png"))],
        "fingerprint": "test-textures",
    }
    with mock.patch.object(ood, "validate_texture_manifest", return_value=pools):
        a_path, a_meta = ood.randomize_scene(
            str(scene), suite="Memory", task="task", seed=2, episode=7,
            return_metadata=True,
        )
        b_path, b_meta = ood.randomize_scene(
            str(scene), suite="Memory", task="task", seed=2, episode=7,
            return_metadata=True,
        )
    try:
        assert Path(a_path).read_bytes() == Path(b_path).read_bytes()
        assert a_meta == b_meta
        lights = list(ET.parse(a_path).getroot().iter("light"))
        assert lights[0].get("castshadow") == "true"
        assert lights[1].get("castshadow") == "false"
        assert a_meta["lighting_schema"] == ood.LIGHTING_SCHEMA
        total = sum(np.mean([float(v) for v in light.get("diffuse").split()])
                    for light in lights)
        assert 0.1 <= total <= 0.8
        assert a_meta["lights"][0]["profile"] == "high_bright"
    finally:
        os.unlink(a_path)
        os.unlink(b_path)


def test_lighting_profiles_cover_visible_dark_to_bright_span(tmp_path):
    scene = tmp_path / "libero_tabletop_base_style.xml"
    _minimal_scene(str(scene))
    pools = {
        "floor": [("floor", str(tmp_path / "floor.png"))],
        "wall": [("wall", str(tmp_path / "wall.png"))],
        "table": [("table", str(tmp_path / "table.png"))],
        "fingerprint": "test-textures",
    }
    paths = []
    profiles = []
    levels = []
    try:
        with mock.patch.object(ood, "validate_texture_manifest", return_value=pools):
            for episode in range(5):
                path, meta = ood.randomize_scene(
                    str(scene), suite="Memory", task="task", seed=0,
                    episode=episode, return_metadata=True,
                )
                paths.append(path)
                profiles.append(meta["lights"][0]["profile"])
                levels.append(sum(np.mean(light["diffuse"])
                                  for light in meta["lights"]))
        assert profiles == ["dark", "dim", "balanced", "bright", "high_bright"]
        assert levels == sorted(levels)
        assert levels[-1] / levels[0] >= 4.0
    finally:
        for path in paths:
            os.unlink(path)


def test_apply_camera_plan_changes_only_extrinsics():
    xml = """
    <mujoco><worldbody><body name="base">
      <camera name="head" pos="0 0 1" quat="1 0 0 0" fovy="80"/>
      <camera name="left_wrist" pos="0 0 .1" quat="1 0 0 0" fovy="80"/>
      <camera name="right_wrist" pos="0 0 .1" quat="1 0 0 0" fovy="80"/>
    </body></worldbody></mujoco>
    """
    env = BiDexEnv.__new__(BiDexEnv)
    env.model = mujoco.MjModel.from_xml_string(xml)
    env.data = mujoco.MjData(env.model)
    env.robot_type = "dual_fr3v2_sharpa"
    env.camera_ids = {
        name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in ("head", "left_wrist", "right_wrist")
    }
    fovy = env.model.cam_fovy.copy()
    plan = ood.camera_randomization_plan(
        env.robot_type, "Memory", "task", 2, 0
    )
    applied = env._apply_camera_pose_plan(plan)
    assert set(applied["cameras"]) == set(env.camera_ids)
    assert np.array_equal(env.model.cam_fovy, fovy)
    for name, cid in env.camera_ids.items():
        delta = np.asarray(plan["cameras"][name]["delta_pos"])
        nominal = np.asarray(applied["cameras"][name]["nominal_pos"])
        assert np.allclose(env.model.cam_pos[cid], nominal + delta)
        assert np.isclose(np.linalg.norm(env.model.cam_quat[cid]), 1.0)
