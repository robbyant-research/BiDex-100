"""RoboCasa assets, mirrored into BiDex-100.

Layout (under ``robocasa/assets/``):

- ``arenas/``                   empty arena XML
- ``fixtures/``
    - ``cabinets/``             cabinet hinge / housing / panel / drawer XMLs
    - ``handles/``              bar / knob / boxed handle XMLs
    - ``counters/``             counter XML stubs (mesh files come from the
                                downloadable asset packs, see README.md)
    - ``accessories/``          light-switch / outlet stubs
    - ``fixture_registry/``     YAML registries that map a fixture *category*
                                (microwave, sink, stove, …) to one or more
                                downloadable XML packages plus default sizing
- ``scenes/``
    - ``kitchen_layouts/{train,test}/layoutNNN.yaml``  walls + fixture slots
    - ``kitchen_styles/{train,test}/styleNNN.yaml``    style → fixture/texture
                                                       picks consumed at scene
                                                       build time
- ``box_links/``                URLs and dataset descriptors used by RoboCasa's
                                ``download_kitchen_assets.py``
- ``novel_instructions/``       task instruction CSVs
- ``groot_dataset_assets/``     embodiment / modality JSON for the GROOT eval
                                pipeline

Most large mesh and texture files are NOT shipped here — they are several GB
and live in RoboCasa's downloadable Box packages. Use
``robocasa.scripts.download_kitchen_assets`` (in the upstream repo) to populate
``assets/{textures,fixtures,objects}/...`` if you need full-fidelity scenes.

Helpers in this module:

- ``ASSETS_ROOT`` — absolute path to ``robocasa/assets/``.
- ``layout_path(name, split="train")`` / ``style_path(...)`` — resolve a layout
  or style YAML by name (``"layout011"``) or stem (``"011"``).
- ``load_fixture_registry(category)`` — read e.g. ``microwave`` / ``cabinet``
  YAML registry as a dict.
- ``list_layouts(split="train")`` / ``list_styles(...)`` — discovery.
"""

from __future__ import annotations

import os
from typing import List

import yaml

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_ROOT = os.path.join(PKG_ROOT, "assets")

ARENAS_DIR = os.path.join(ASSETS_ROOT, "arenas")
FIXTURES_DIR = os.path.join(ASSETS_ROOT, "fixtures")
FIXTURE_REGISTRY_DIR = os.path.join(FIXTURES_DIR, "fixture_registry")
SCENES_DIR = os.path.join(ASSETS_ROOT, "scenes")
LAYOUTS_DIR = os.path.join(SCENES_DIR, "kitchen_layouts")
STYLES_DIR = os.path.join(SCENES_DIR, "kitchen_styles")


def _resolve_split_file(base: str, name: str, split: str, prefix: str) -> str:
    if not name.endswith((".yaml", ".yml")):
        if not name.startswith(prefix):
            name = prefix + name
        name += ".yaml"
    cand = os.path.join(base, split, name)
    if os.path.isfile(cand):
        return cand
    # Fall back: search the other split too — useful for ad-hoc lookups.
    for s in ("train", "test"):
        cand = os.path.join(base, s, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"{prefix} '{name}' not found under {base}/{{train,test}}"
    )


def layout_path(name: str, split: str = "train") -> str:
    """Resolve a layout YAML, e.g. ``layout_path("011")`` -> ``.../layout011.yaml``."""
    return _resolve_split_file(LAYOUTS_DIR, name, split, "layout")


def style_path(name: str, split: str = "train") -> str:
    """Resolve a style YAML, e.g. ``style_path("style018")``."""
    return _resolve_split_file(STYLES_DIR, name, split, "style")


def list_layouts(split: str = "train") -> List[str]:
    d = os.path.join(LAYOUTS_DIR, split)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".yaml"))


def list_styles(split: str = "train") -> List[str]:
    d = os.path.join(STYLES_DIR, split)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".yaml"))


def list_fixture_categories() -> List[str]:
    if not os.path.isdir(FIXTURE_REGISTRY_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(FIXTURE_REGISTRY_DIR)
        if f.endswith(".yaml")
    )


def load_fixture_registry(category: str) -> dict:
    """Load ``fixture_registry/<category>.yaml`` as a dict.

    Raises FileNotFoundError if the category does not exist; available names
    can be listed via :func:`list_fixture_categories`.
    """
    path = os.path.join(FIXTURE_REGISTRY_DIR, f"{category}.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"unknown fixture category '{category}'. Available: "
            f"{list_fixture_categories()}"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_layout(name: str, split: str = "train") -> dict:
    with open(layout_path(name, split=split), "r") as f:
        return yaml.safe_load(f) or {}


def load_style(name: str, split: str = "train") -> dict:
    with open(style_path(name, split=split), "r") as f:
        return yaml.safe_load(f) or {}


__all__ = [
    "ASSETS_ROOT",
    "ARENAS_DIR",
    "FIXTURES_DIR",
    "FIXTURE_REGISTRY_DIR",
    "SCENES_DIR",
    "LAYOUTS_DIR",
    "STYLES_DIR",
    "layout_path",
    "style_path",
    "list_layouts",
    "list_styles",
    "list_fixture_categories",
    "load_fixture_registry",
    "load_layout",
    "load_style",
    "build_robocasa_scene_xml",
]


# Lazy re-export so the bare-asset helpers above stay importable on machines
# that don't have upstream robocasa/robosuite installed.
def __getattr__(name):
    if name == "build_robocasa_scene_xml":
        from .scene_factory import build_robocasa_scene_xml
        return build_robocasa_scene_xml
    raise AttributeError(name)
