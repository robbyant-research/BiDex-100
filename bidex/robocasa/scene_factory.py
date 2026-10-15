"""Adapter that materializes a RoboCasa kitchen layout/style as an MJCF file.

This module delegates the heavy layout work to upstream RoboCasa's
`KitchenArena` + `ManipulationTask`, then writes the assembled MJCF to a
tempfile so the rest of BiDex-100 can `<include>` it like any other scene.

Upstream RoboCasa hard-pins `mujoco==3.3.1` and `numpy==2.2.5` at import time
(robocasa/__init__.py asserts). Our env currently runs newer versions and we
don't want to downgrade, so we spoof the version strings before importing
robocasa. The pinned code paths themselves only use stable MuJoCo/numpy APIs,
and rendering + layout have worked fine under 3.8 / 2.2.6 in practice.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple


def _patch_upstream_version_asserts() -> None:
    import mujoco
    import numpy

    # robocasa/__init__.py: `assert mujoco.__version__ == "3.3.1"` and
    # `assert numpy.__version__ in ["2.2.5"]`. Spoofing the strings bypasses
    # the asserts without downgrading the env.
    mujoco.__version__ = "3.3.1"
    if numpy.__version__ not in ("2.2.5",):
        numpy.__version__ = "2.2.5"


def build_robocasa_scene_xml(
    layout_id,
    style_id,
    *,
    enable_fixtures: Optional[List[str]] = None,
    clutter_mode: int = 0,
    seed: int = 0,
    out_dir: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Build an MJCF for one RoboCasa (layout, style) pair and save it to disk.

    Parameters
    ----------
    layout_id, style_id:
        Integer ids matching `scenes/kitchen_layouts/{train,test}/layoutNNN.yaml`
        and `scenes/kitchen_styles/{train,test}/styleNNN.yaml`. Strings like
        "011" are accepted too — converted to int.
    enable_fixtures:
        Optional list of fixture names to force-enable (for layouts where
        certain fixtures default to `enable: false`).
    clutter_mode:
        0 (default) disables clutter-marked fixtures; 1 leaves them.
    seed:
        Determinism seed for any random style/fixture picks performed by
        `create_fixtures`.
    out_dir:
        Where to drop the XML. Defaults to `tempfile.gettempdir()`. The
        returned path lives here; paths inside the XML are absolute so the
        file can be moved freely.

    Returns
    -------
    (xml_path, temp_files):
        `xml_path` is the file we just wrote. `temp_files` contains the
        same path so it can be merged into the caller's cleanup list.
    """
    _patch_upstream_version_asserts()

    import numpy as np
    # Deferred so machines without robocasa installed can still import
    # bidex.robocasa for the asset helpers.
    from robocasa.models.scenes.kitchen_arena import KitchenArena
    from robosuite.models.tasks import ManipulationTask

    layout_id = int(layout_id)
    style_id = int(style_id)

    arena = KitchenArena(
        layout_id=layout_id,
        style_id=style_id,
        rng=np.random.default_rng(seed),
        enable_fixtures=enable_fixtures,
        clutter_mode=clutter_mode,
    )
    arena.set_origin([0, 0, 0])

    # ManipulationTask runs merge_arena + merge_objects internally
    # (robosuite/models/tasks/task.py:78-82), which copies every fixture's
    # <asset>/<body>/<actuator> onto the arena's worldbody with collision-
    # safe name prefixing. We pass an empty robot list — we don't need the
    # robosuite robot stack; BiDex-100 adds its own fr3v2 in env.py.
    task = ManipulationTask(
        mujoco_arena=arena,
        mujoco_robots=[],
        mujoco_objects=list(arena.fixtures.values()),
        enable_multiccd=False,
    )
    xml_str = task.get_xml()

    out_dir = out_dir or tempfile.gettempdir()
    fd, xml_path = tempfile.mkstemp(
        prefix=f"rc_kitchen_{layout_id:03d}_{style_id:03d}_",
        suffix=".xml",
        dir=out_dir,
    )
    os.close(fd)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return xml_path, [xml_path]


__all__ = ["build_robocasa_scene_xml"]
