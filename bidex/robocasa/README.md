# RoboCasa Assets (mirrored)

This directory contains a verbatim copy of RoboCasa's `models/assets/` tree —
the XML / YAML *index* of arenas, fixtures, scene layouts, and scene styles
used by the upstream RoboCasa kitchen environment. Source:
<https://github.com/robocasa/robocasa>, MIT License (see `LICENSE`).

## What is here

- `assets/arenas/` — empty kitchen arena MJCF.
- `assets/fixtures/` — cabinet / handle / counter / accessory XML stubs and
  the per-category `fixture_registry/*.yaml` that maps logical fixtures
  (microwave, sink, stove, …) to specific downloadable XML packs plus default
  sizes.
- `assets/scenes/kitchen_layouts/{train,test}/layoutNNN.yaml` — wall geometry
  + fixture placement slots for ~80 kitchen layouts.
- `assets/scenes/kitchen_styles/{train,test}/styleNNN.yaml` — per-style
  selection of which fixture variants and textures to use.
- `assets/box_links/` — URLs that RoboCasa's downloader uses for the heavy
  mesh / texture / object packs.
- `assets/novel_instructions/`, `assets/groot_dataset_assets/` — auxiliary
  CSV / JSON metadata.

## What is NOT here

The actual mesh files and textures referenced by these XML stubs (microwaves,
fridges, ovens, fixture meshes, environment textures, Objaverse, and
AI-generated objects) are **not** committed to the repository. They are
distributed in the `assets/robocasa_assets/robocasa_assets.tar.gz` component
of the BiDex-100 Hugging Face dataset repository.

Download and install the asset release from the repository root:

```bash
bash scripts/download_release.sh assets downloads/BiDex-100
bash scripts/setup_assets_links.sh downloads/BiDex-100
```

The setup script verifies and extracts the archive, then links
`downloads/BiDex-100/assets/robocasa_assets` to `bidex/robocasa/assets`.

## Rendering RoboCasa fixtures in BiDex-100 tasks

A minimal RoboCasa-style scene is provided at `scenes_simple/robocasa_simple_kitchen.xml`
(wood floor + warm-white walls + wood countertop, all using textures from
`assets/textures/`). Task YAMLs reference it by scene name:

```yaml
scene: robocasa_simple_kitchen
```

RoboCasa fixtures and objects are addressable as task objects with an
`rc/` prefix. Resolution rules in `scripts/visualize_scene.resolve_object_xml`:

- `rc/Microwave011`              walks fixture categories (microwaves/, stoves/,
                                 cabinets/, …) to find
                                 `assets/fixtures/<cat>/Microwave011/model.xml`.
- `rc/microwaves/Microwave011`   explicit category prefix.
- `rc/objaverse/orange/orange_4` explicit object path.
- `rc/lightwheel/<...>`          same, for the lightwheel object set.

Three example tasks under `tasks_data/suites/robocasa/` demonstrate the wiring
(see `pick_bowl_in_robocasa_kitchen`, `open_robocasa_microwave`,
`stove_and_kettle_on_counter`). They can be rendered via:

```bash
python scripts/render_task_images.py --suite robocasa
```

Snapshots land in `bidex/images/robocasa/*.png`.

## Python helpers

`bidex.robocasa` exposes a tiny convenience layer over this tree (no
robosuite dependency):

```python
from bidex import robocasa as rc

rc.list_layouts("train")                    # -> ["layout011", ...]
rc.list_styles("train")                     # -> ["style011", ...]
rc.list_fixture_categories()                # -> ["microwave", "cabinet", ...]
rc.load_layout("011")                       # parsed YAML dict
rc.load_style("011")
rc.load_fixture_registry("microwave")
```

`rc.ASSETS_ROOT`, `rc.LAYOUTS_DIR`, `rc.STYLES_DIR`, `rc.FIXTURES_DIR`, and
`rc.FIXTURE_REGISTRY_DIR` are absolute paths suitable for joining with the
relative `xml:` / `texture:` entries inside the YAML files.

For task YAMLs that want to reference one of these scenes, set `scene:` to
either `robocasa_simple_kitchen` (the minimal scene shipped here) or to any
LIBERO scene name. To use a RoboCasa fixture or object as a task object,
prefix its name with `rc/` in the `objects:` list. See the example tasks
under `bidex/tasks_data/suites/robocasa/` for the full pattern.
