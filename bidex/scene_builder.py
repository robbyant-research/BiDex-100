"""MJCF scene-building library for BiDex-100.

Slimmed down from the original ``scripts/visualize_scene.py``: only the
library functions that ``bidex.env`` needs to resolve a scene XML,
resolve + per-object-rewrite object XMLs, and write a robot assembly at a
given base pose are kept. The CLI / ``visualize()`` / ``make_combined_xml``
top-level assembler live in the env now (``build_combined_xml``), which
also injects cameras and inlines ``<include>`` subtrees.

Path layout (this repo):
    PKG_ROOT = bidex/            (parent of this file)
    PKG_ROOT/libero/{scenes,objects,textures}
    PKG_ROOT/robocasa/{scenes_simple,assets}

``libero/`` and ``robocasa/assets`` are symlinks to the large asset packs
(see README) so they resolve transparently here.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# PKG_ROOT = parent of this file's directory = the bidex package root.
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
LIBERO_ROOT = os.path.join(PKG_ROOT, "libero")
SCENES_DIR = os.path.join(LIBERO_ROOT, "scenes")
OBJECTS_ROOT = os.path.join(LIBERO_ROOT, "objects")

ROBOCASA_ROOT = os.path.join(PKG_ROOT, "robocasa")
ROBOCASA_SCENES_DIR = os.path.join(ROBOCASA_ROOT, "scenes_simple")
ROBOCASA_ASSETS_ROOT = os.path.join(ROBOCASA_ROOT, "assets")
ROBOCASA_FIXTURES_ROOT = os.path.join(ROBOCASA_ASSETS_ROOT, "fixtures")
ROBOCASA_OBJECTS_ROOT = os.path.join(ROBOCASA_ASSETS_ROOT, "objects")

OBJECT_CATEGORIES = (
    "stable_hope_objects",
    "stable_scanned_objects",
    "turbosquid_objects",
    "articulated_objects",
    "articraft_objects",  # Articraft-10K MJCF pack (linked at libero/objects/articraft_objects)
    "dexjoco_objects",    # dexjoco hammer-nail / spray-plant / button / ice-ball assets
                           # (libero/objects/dexjoco_objects); referenced by tabletop dexjoco tasks
    "assembly_objects",   # hand-built MuJoCo primitive parts/fixtures for assemble_* tasks
                           # (libero/objects/assembly_objects -> $ASSETS/assembly_objects).
                           # aliases (cube_a..d -> assembly_cube, ball_a..c -> assembly_ball,
                           # tower_post_a/b & pier_a/b & stool_leg_a/b -> assembly_block_short,
                           # threading_rod_left/mid/right -> threading_rod, ...) are RELATIVE
                           # symlinks inside the pack: one canonical XML shared by many names.
)


def list_scenes() -> List[str]:
    out: List[str] = []
    if os.path.isdir(SCENES_DIR):
        for f in sorted(os.listdir(SCENES_DIR)):
            if f.startswith("libero_") and f.endswith(".xml"):
                out.append(os.path.splitext(f)[0])
    if os.path.isdir(ROBOCASA_SCENES_DIR):
        for f in sorted(os.listdir(ROBOCASA_SCENES_DIR)):
            if f.endswith(".xml"):
                out.append(os.path.splitext(f)[0])
    return out


def list_objects() -> Dict[str, List[str]]:
    """Return ``{category: [obj_name, ...]}`` for LIBERO catalogs and
    RoboCasa objaverse/lightwheel buckets. The bare names match what
    ``resolve_object_xml`` accepts and what task YAMLs reference."""
    out: Dict[str, List[str]] = {}
    for cat in OBJECT_CATEGORIES:
        cat_dir = os.path.join(OBJECTS_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        names: List[str] = []
        for entry in sorted(os.listdir(cat_dir)):
            full = os.path.join(cat_dir, entry)
            if os.path.isdir(full):
                cand = os.path.join(full, entry + ".xml")
                if os.path.isfile(cand):
                    names.append(entry)
                else:
                    xmls = [f for f in os.listdir(full) if f.endswith(".xml")]
                    if xmls:
                        names.append(entry)
            elif entry.endswith(".xml"):
                names.append(os.path.splitext(entry)[0])
        out[cat] = names
    for bucket in ("objaverse", "lightwheel"):
        bucket_dir = os.path.join(ROBOCASA_OBJECTS_ROOT, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        names = []
        for bare in sorted(os.listdir(bucket_dir)):
            bare_dir = os.path.join(bucket_dir, bare)
            if not os.path.isdir(bare_dir):
                continue
            for inst in os.listdir(bare_dir):
                inst_dir = os.path.join(bare_dir, inst)
                if os.path.isdir(inst_dir) and os.path.isfile(
                    os.path.join(inst_dir, "model.xml")
                ):
                    names.append(bare)
                    break
        if names:
            out[f"robocasa/{bucket}"] = names
    return out


def resolve_scene_xml(scene_name: str) -> str:
    if scene_name.endswith(".xml") and os.path.isfile(scene_name):
        return os.path.abspath(scene_name)
    candidate = os.path.join(SCENES_DIR, scene_name + ".xml")
    if os.path.isfile(candidate):
        return candidate
    rc_candidate = os.path.join(ROBOCASA_SCENES_DIR, scene_name + ".xml")
    if os.path.isfile(rc_candidate):
        return rc_candidate
    raise FileNotFoundError(
        f"Scene '{scene_name}' not found. Available:\n  " + "\n  ".join(list_scenes())
    )


def _is_env_prepared_temp(filename: str) -> bool:
    """True if ``filename`` is a leftover prepared-object temp written by the env.

    ``env._build_combined_xml_with_cameras`` writes each prepared object as
    ``tempfile.mkstemp(prefix="obj{i}_{name}_", suffix=".xml", dir=obj_dir)``
    INSIDE the object's own source directory, so on a killed/crashed run these
    ``obj<digit>_<name>_<hash>.xml`` files accumulate next to the canonical xml.
    An unsorted ``os.listdir[0]`` fallback that latches onto one of these -- they
    can even be 0 bytes -- makes the env ``ET.parse`` a leftover and either
    ParseError out or silently pose a stale body, dropping the object from the
    scene (e.g. threading_rod disappearing). Skip them during resolution.
    """
    head, sep, _rest = filename.partition("_")
    return sep == "_" and head.startswith("obj") and len(head) > 3 and head[3:].isdigit()


def resolve_object_xml(obj_name: str) -> Tuple[str, str]:
    """Resolve an object name to ``(xml_path, category)``.

    Naming conventions:
      - ``akita_black_bowl`` / ``microwave``          LIBERO catalogs
      - ``rc/<FixtureName>``                          RoboCasa fixture
      - ``rc/objaverse/<cat>/<name>`` / ``rc/lightwheel/<...>``  RoboCasa object

    The returned path is realpath'd so downstream temp-file copies land in
    the actual asset directory (required for symlinked catalogs like
    ``articraft_objects``).
    """
    if obj_name.endswith(".xml") and os.path.isfile(obj_name):
        return os.path.realpath(os.path.abspath(obj_name)), ""

    if obj_name.startswith("rc/"):
        path, cat = _resolve_robocasa_xml(obj_name[3:])
        return os.path.realpath(path), cat

    for cat in OBJECT_CATEGORIES:
        cat_dir = os.path.join(OBJECTS_ROOT, cat)
        sub = os.path.join(cat_dir, obj_name)
        if os.path.isdir(sub):
            inner = os.path.join(sub, obj_name + ".xml")
            if os.path.isfile(inner):
                return os.path.realpath(inner), cat
            # No <dirname>.xml canonical: fall back to another real XML in the
            # dir, deterministically and safely. Skip the env's own prepared-object
            # leftovers (obj<digit>_*_*.xml, incl. 0-byte) and prefer a
            # RoboCasa-style model.xml; otherwise the alphabetically-first survivor.
            # The old unsorted os.listdir[0] could grab a 0-byte leftover
            # (ParseError -> object vanishes from the scene).
            xmls = sorted(
                f for f in os.listdir(sub)
                if f.endswith(".xml")
                and not _is_env_prepared_temp(f)
                and os.path.getsize(os.path.join(sub, f)) > 0
            )
            if xmls:
                chosen = "model.xml" if "model.xml" in xmls else xmls[0]
                return os.path.realpath(os.path.join(sub, chosen)), cat
        flat = os.path.join(cat_dir, obj_name + ".xml")
        if os.path.isfile(flat):
            return os.path.realpath(flat), cat

    # Fallback: bare name might be a RoboCasa objaverse/lightwheel category.
    for sub in ("objaverse", "lightwheel"):
        cat_dir = os.path.join(ROBOCASA_OBJECTS_ROOT, sub, obj_name)
        if not os.path.isdir(cat_dir):
            continue
        instances = sorted(
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d))
            and os.path.isfile(os.path.join(cat_dir, d, "model.xml"))
        )
        if instances:
            return (
                os.path.realpath(os.path.join(cat_dir, instances[0], "model.xml")),
                f"robocasa/{sub}",
            )

    raise FileNotFoundError(
        f"Object '{obj_name}' not found. Call list_objects() for the catalog."
    )


# RoboCasa fixture categories that group instance dirs containing model.xml.
ROBOCASA_FIXTURE_CATS = (
    "microwaves", "stoves", "stovetops", "sinks", "ovens", "fridges",
    "dishwashers", "cabinets", "coffee_machines", "electric_kettles",
    "toaster_ovens", "toasters", "stand_mixers", "blenders", "hoods", "windows",
    "handles", "counters",
)
ROBOCASA_ACCESSORY_ROOTS = (
    os.path.join(ROBOCASA_FIXTURES_ROOT, "accessories"),
    os.path.join(ROBOCASA_OBJECTS_ROOT, "objaverse"),
    os.path.join(ROBOCASA_OBJECTS_ROOT, "lightwheel"),
)


# ---------------------------------------------------------------------------
# Static object footprint (collision-geometry AABB), no mujoco needed
# ---------------------------------------------------------------------------

_FOOTPRINT_CACHE: Dict[Tuple[str, float], Optional[Tuple[float, float, float, float]]] = {}


def _fp_quat_mat(w: float, x: float, y: float, z: float):
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _fp_floats(s: Optional[str], default):
    if s is None:
        return default
    return [float(v) for v in s.replace(",", " ").split()]


def object_footprint(obj_name: str, scale: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    """Collision-geometry extents ``(hx, hy, z_lo, z_hi)`` (metres) of the
    object relative to its root-body origin, multiplied by ``scale`` --
    xy AABB half-extents plus the min/max collision z.

    Parsed statically from the object's canonical MJCF (primitive geoms plus
    the ``reg_bbox`` marker box that RoboCasa-style packs ship; mesh collision
    geoms carry no explicit size and are skipped -- for those packs reg_bbox
    covers the whole body anyway). Nested bodies (lids/drawers at their
    declared, i.e. closed, pose) are walked with static transforms.

    Used by the OOD distractor placer: a big freejoint container (chest
    freezer, cutting board) whose placement REGION is only a few cm wide would
    otherwise get a distractor slot that overlaps its real body. Returns None
    when nothing parseable is found (caller falls back to region-box-only
    behaviour). Geometry-static, so cached per (name, scale).
    """
    key = (obj_name, round(float(scale), 6))
    if key in _FOOTPRINT_CACHE:
        return _FOOTPRINT_CACHE[key]
    out: Optional[Tuple[float, float, float, float]] = None
    try:
        xml_path, _cat = resolve_object_xml(obj_name)
        root = ET.parse(xml_path).getroot()
        wb = root.find("worldbody")
        if wb is None:
            raise ValueError("no worldbody")
        hx = hy = 0.0
        z_lo, z_hi = 1e9, -1e9
        n = 0

        def add_geom(g, bpos, bmat):
            nonlocal hx, hy, z_lo, z_hi, n
            name = g.get("name", "")
            is_bbox = name == "reg_bbox" and g.get("type") == "box"
            if not is_bbox:
                # skip pure-visual geoms (never collide)
                if int(g.get("contype", "1")) == 0 and int(g.get("conaffinity", "1")) == 0:
                    return
            gtype = g.get("type", "sphere")
            size = _fp_floats(g.get("size"), None)
            if gtype == "box":
                if not size or len(size) < 3:
                    return
                half = (size[0], size[1], size[2])
            elif gtype == "sphere":
                half = (size[0], size[0], size[0]) if size else (0.01, 0.01, 0.01)
            elif gtype in ("cylinder", "capsule"):
                if not size or len(size) < 2:
                    return
                half = (size[0], size[0], size[1])
            elif gtype == "ellipsoid":
                if not size or len(size) < 3:
                    return
                half = (size[0], size[1], size[2])
            else:
                return  # mesh etc.: no explicit size; reg_bbox covers these packs
            gpos = _fp_floats(g.get("pos"), [0.0, 0.0, 0.0])[:3]
            gquat = _fp_floats(g.get("quat"), [1.0, 0.0, 0.0, 0.0])[:4]
            gm = _fp_quat_mat(*gquat)
            # world centre = bpos + bmat @ gpos ; world half-extents = |bmat @ gm| @ half
            wc = [bpos[i] + sum(bmat[i][j] * gpos[j] for j in range(3)) for i in range(3)]
            am = [[sum(bmat[i][k] * gm[k][j] for k in range(3)) for j in range(3)]
                  for i in range(3)]
            ex = sum(abs(am[0][j]) * half[j] for j in range(3))
            ey = sum(abs(am[1][j]) * half[j] for j in range(3))
            ez = sum(abs(am[2][j]) * half[j] for j in range(3))
            hx = max(hx, abs(wc[0]) + ex)
            hy = max(hy, abs(wc[1]) + ey)
            z_lo = min(z_lo, wc[2] - ez)
            z_hi = max(z_hi, wc[2] + ez)
            n += 1

        def walk(body, pos, mat):
            bpos = _fp_floats(body.get("pos"), [0.0, 0.0, 0.0])[:3]
            bquat = _fp_floats(body.get("quat"), [1.0, 0.0, 0.0, 0.0])[:4]
            bm = _fp_quat_mat(*bquat)
            wpos = [pos[i] + sum(mat[i][j] * bpos[j] for j in range(3)) for i in range(3)]
            wmat = [[sum(mat[i][k] * bm[k][j] for k in range(3)) for j in range(3)]
                    for i in range(3)]
            for g in body.findall("geom"):
                add_geom(g, wpos, wmat)
            for child in body.findall("body"):
                walk(child, wpos, wmat)

        ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for body in wb.findall("body"):
            walk(body, (0.0, 0.0, 0.0), ident)
        if n > 0:
            s = float(scale)
            out = (hx * s, hy * s, z_lo * s, z_hi * s)
    except Exception:
        out = None
    _FOOTPRINT_CACHE[key] = out
    return out


def _resolve_robocasa_xml(rel_name: str) -> Tuple[str, str]:
    """Resolve ``rc/<rel_name>`` to a RoboCasa fixture or object XML."""
    if "/" in rel_name:
        for base in (ROBOCASA_FIXTURES_ROOT, ROBOCASA_OBJECTS_ROOT):
            cand = os.path.join(base, rel_name, "model.xml")
            if os.path.isfile(cand):
                return cand, f"robocasa/{rel_name.split('/')[0]}"
    for cat in ROBOCASA_FIXTURE_CATS:
        cand = os.path.join(ROBOCASA_FIXTURES_ROOT, cat, rel_name, "model.xml")
        if os.path.isfile(cand):
            return cand, f"robocasa/{cat}"
    for root in ROBOCASA_ACCESSORY_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "model.xml" in files and os.path.basename(dirpath) == rel_name:
                return os.path.join(dirpath, "model.xml"), "robocasa/objects"
    raise FileNotFoundError(
        f"RoboCasa resource 'rc/{rel_name}' not found. Asset root: {ROBOCASA_ASSETS_ROOT}"
    )


# ---------------------------------------------------------------------------
# Robot base-pose rewriting
# ---------------------------------------------------------------------------

def _write_robot_with_pose(
    robot_xml_path: str,
    pos: Tuple[float, float, float],
    quat: Tuple[float, float, float, float],
    out_path: str,
) -> None:
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError(f"{robot_xml_path} has no <worldbody>")
    base = None
    for b in wb.findall("body"):
        if b.get("name") == "base":
            base = b
            break
    if base is None:
        base = wb.find("body")
    if base is None:
        raise RuntimeError(f"{robot_xml_path} <worldbody> has no body")
    base.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    # Combine the body's built-in orientation with the task-specified quat
    # (task quat applied on top, e.g. astribot chassis_base euler="0 0 -1.57").
    import numpy as np
    orig_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
    if "quat" in base.attrib:
        orig_quat = np.array([float(x) for x in base.get("quat").split()])
    elif "euler" in base.attrib:
        from scipy.spatial.transform import Rotation
        e = [float(x) for x in base.get("euler").split()]
        r = Rotation.from_euler('XYZ', e)
        xyzw = r.as_quat()
        orig_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    tq = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)
    combined_quat = np.zeros(4)
    import mujoco as _mj
    _mj.mju_mulQuat(combined_quat, tq, orig_quat)
    for orient_attr in ("euler", "axisangle", "xyaxes", "zaxis"):
        if orient_attr in base.attrib:
            del base.attrib[orient_attr]
    base.set("quat", f"{combined_quat[0]} {combined_quat[1]} {combined_quat[2]} {combined_quat[3]}")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Object rewriting: prefix assets to avoid name clashes + set pose + freejoint
# ---------------------------------------------------------------------------

_ASSET_TAGS_WITH_NAME = ("texture", "material", "mesh", "hfield", "skin")
_REFERENCE_ATTRS = {
    "mesh": "mesh",
    "material": "material",
    "texture": "texture",
    "hfield": "hfield",
}
_NON_MJCF_TAGS = {"region"}
_STRIP_TOP_LEVEL_TAGS = ("option", "size", "statistic", "visual")


def _strip_top_level(root: ET.Element) -> None:
    for tag in _STRIP_TOP_LEVEL_TAGS:
        for el in list(root.findall(tag)):
            root.remove(el)


def _strip_non_mjcf(root: ET.Element) -> None:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for elem in list(root.iter()):
        if elem.tag in _NON_MJCF_TAGS:
            parent = parent_map.get(elem)
            if parent is not None:
                parent.remove(elem)


def _prepare_object_xml(
    obj_xml_path: str,
    instance_id: int,
    pos: Tuple[float, float, float],
    quat: Optional[Tuple[float, float, float, float]],
    add_freejoint: bool,
    out_path: str,
    scale: float = 1.0,
    mass_scale: float = 1.0,
) -> None:
    """Read an object XML, prefix all asset names with ``obj{i}_`` to avoid
    clashes, set the outer body's pos/quat, and optionally add a freejoint.
    The output file must sit next to the source XML to keep relative mesh
    refs valid.

    ``scale`` / ``mass_scale`` are relative multipliers (1.0 = no-op); when
    either != 1.0, ``scale_object_xml.scale_tree`` resizes mesh/geom size,
    mass, inertia, and slide-joint range together (physics-consistent)."""
    prefix = f"obj{instance_id}_"
    tree = ET.parse(obj_xml_path)
    root = tree.getroot()

    if scale != 1.0 or mass_scale != 1.0:
        from bidex.scale_object_xml import scale_tree
        scale_tree(root, size_scale=float(scale), mass_scale=float(mass_scale))

    comp = root.find("compiler")
    if comp is not None and "inertiagrouprange" in comp.attrib:
        del comp.attrib["inertiagrouprange"]

    _strip_non_mjcf(root)
    _strip_top_level(root)

    # Rename <default class="..."> to avoid clashes with robot/other objects.
    default_root = root.find("default")
    default_rename: Dict[str, str] = {}
    if default_root is not None:
        for d in default_root.iter("default"):
            cls = d.get("class")
            if cls is not None:
                new_cls = prefix + cls
                default_rename[cls] = new_cls
                d.set("class", new_cls)
    if default_rename:
        for elem in root.iter():
            cls = elem.get("class")
            if cls is not None and cls in default_rename:
                elem.set("class", default_rename[cls])
            childclass = elem.get("childclass")
            if childclass is not None and childclass in default_rename:
                elem.set("childclass", default_rename[childclass])

    rename_map_by_tag: Dict[str, Dict[str, str]] = {t: {} for t in _ASSET_TAGS_WITH_NAME}
    asset_root = root.find("asset")
    if asset_root is not None:
        for tag in _ASSET_TAGS_WITH_NAME:
            seen_count: Dict[str, int] = {}
            for elem in asset_root.findall(tag):
                old = elem.get("name")
                if not old:
                    continue
                base_new = prefix + old
                count = seen_count.get(base_new, 0)
                new = base_new if count == 0 else f"{base_new}_dup{count}"
                seen_count[base_new] = count + 1
                elem.set("name", new)
                rename_map_by_tag[tag].setdefault(old, new)

    for elem in root.iter():
        for attr, tag in _REFERENCE_ATTRS.items():
            v = elem.get(attr)
            if v is not None and v in rename_map_by_tag.get(tag, {}):
                elem.set(attr, rename_map_by_tag[tag][v])

    joint_rename: Dict[str, str] = {}
    for elem in root.iter():
        if elem.tag in ("body", "site", "joint", "geom", "camera", "light"):
            n = elem.get("name")
            if n is not None:
                new_n = prefix + n
                if elem.tag == "joint":
                    joint_rename[n] = new_n
                elem.set("name", new_n)

    # Defensive: joints without an explicit range get limited="false" so the
    # robot's default-class limited="true" doesn't clamp articraft hinges to (0,0).
    for j in root.iter("joint"):
        if "range" not in j.attrib and j.get("type") != "free":
            j.set("limited", "false")

    _JOINT_REF_ATTRS = ("joint", "joint1", "joint2")
    for elem in root.iter():
        for attr in _JOINT_REF_ATTRS:
            v = elem.get(attr)
            if v is not None and v in joint_rename:
                elem.set(attr, joint_rename[v])

    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError(f"{obj_xml_path} has no <worldbody>")
    outer_body = wb.find("body")
    if outer_body is None:
        raise RuntimeError(f"{obj_xml_path} <worldbody> has no body")
    if outer_body.get("name") is None:
        outer_body.set("name", prefix + "root")
    outer_body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    if quat is not None:
        outer_body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
    if add_freejoint:
        fj = ET.Element("freejoint", {"name": prefix + "freejoint"})
        outer_body.insert(0, fj)

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Object spec parsing
# ---------------------------------------------------------------------------

ObjectSpec = Tuple[str, Tuple[float, float, float], Optional[Tuple[float, float, float, float]]]


def _parse_object_spec(spec: str, default_z: float = 0.85) -> ObjectSpec:
    """Parse a ``NAME[@x,y,z[,qw,qx,qy,qz]]`` string."""
    if "@" not in spec:
        return spec, (0.0, 0.0, default_z), None
    name, pose = spec.split("@", 1)
    parts = [p.strip() for p in pose.split(",") if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"Bad object pose '{spec}'; expected NAME@x,y,z or NAME@x,y,z,qw,qx,qy,qz"
        )
    if len(nums) == 3:
        return name, (nums[0], nums[1], nums[2]), None
    if len(nums) == 7:
        return name, (nums[0], nums[1], nums[2]), (nums[3], nums[4], nums[5], nums[6])
    raise ValueError(
        f"Object pose needs 3 or 7 numbers, got {len(nums)}: '{spec}'"
    )
