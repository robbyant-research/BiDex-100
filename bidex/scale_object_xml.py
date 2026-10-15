#!/usr/bin/env python3
"""Scale an MJCF object's geometry and (optionally) extra-multiply its mass.

Built for the LIBERO articulated objects under
``bidex/bidex/libero/objects/articulated_objects/`` — the
default sizes are tuned for a tabletop scene and sometimes too small to
interact with comfortably from a teleoperated dexterous hand, or too
light so contacts knock them around. This script edits an existing MJCF
in place (or to a new file) and:

  * scales every ``pos`` / ``size`` / ``fromto`` triplet and every mesh
    ``scale`` attribute by a single linear factor ``s`` (=``--size``);
  * scales every explicit ``mass="..."`` by ``s**3 * --mass``
    (volumetric scaling first, then an extra dimensionless multiplier
    for "make it heavier without making it bigger" use-cases);
  * scales every ``diaginertia`` by ``s**5 * --mass`` (mass · length²,
    so it follows mass × size² growth);
  * scales prismatic ``<joint range="...">`` (the joint's travel
    distance grows with the object) but leaves hinge joint angle ranges
    alone (rotation limits aren't length-dependent);
  * leaves topology, joint axes, materials, friction, solref / solimp,
    contype / conaffinity, names, classes untouched.

Examples
--------

Scale a single articulated object 1.5× larger and 2× heavier, output to
``microwave_big.xml`` next to the input::

    python3 scripts/scale_object_xml.py \\
        bidex/libero/objects/articulated_objects/microwave.xml \\
        --size 1.5 --mass 2.0

In-place (overwrite the source), with size scale only::

    python3 scripts/scale_object_xml.py \\
        bidex/libero/objects/articulated_objects/wooden_cabinet.xml \\
        --size 1.3 --inplace

Batch a whole directory recursively, leaving the originals untouched::

    python3 scripts/scale_object_xml.py \\
        bidex/libero/objects/articulated_objects/ \\
        --size 1.4 --mass 1.5 --suffix _scaled

Notes
-----

* Geoms that derive their shape from a ``<mesh>`` (``geom type="mesh"``)
  get their visible size from the mesh's ``scale`` attribute — so
  scaling the mesh scale is the right knob, not the geom's ``size``
  (which mesh geoms typically don't have). Box / sphere / capsule /
  cylinder geoms have explicit ``size``, which we do scale.

* If a geom has no explicit ``mass`` attribute, MuJoCo computes mass
  from its volume × density (default density 1000 kg/m³). Volume grows
  with ``s**3`` automatically when we scale ``size`` (or mesh scale),
  so no extra work is needed for the volumetric component. The
  ``--mass`` factor in that case is applied via the ``density`` knob
  if a default density is present, otherwise we just emit a warning so
  the user knows the multiplier wasn't applicable.

* Output is pretty-printed with 2-space indent so diffs against the
  source stay readable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import xml.etree.ElementTree as ET


# Attributes that contain triplets of position/length values and need
# uniform linear scaling.
_LENGTH_VEC_ATTRS = ("pos", "fromto")
# Attributes that contain a vector of sizes (1-3 floats); all scaled uniformly.
_SIZE_VEC_ATTRS = ("size",)
# Attributes that contain mesh scale (3 floats) — multiplied component-wise.
_SCALE_VEC_ATTRS = ("scale",)


def _split_floats(s: str) -> List[float]:
    return [float(x) for x in s.replace(",", " ").split() if x]


def _fmt_floats(vs: Sequence[float]) -> str:
    # Print 6 sig figs; collapse trailing zeros so the diff stays small.
    return " ".join(f"{v:.6g}" for v in vs)


def _scale_attr(elem: ET.Element, attr: str, factor: float) -> bool:
    if attr not in elem.attrib:
        return False
    vs = _split_floats(elem.attrib[attr])
    elem.attrib[attr] = _fmt_floats([v * factor for v in vs])
    return True


def _scale_mass(elem: ET.Element, mass_mult: float) -> bool:
    if "mass" not in elem.attrib:
        return False
    try:
        m = float(elem.attrib["mass"])
    except ValueError:
        return False
    elem.attrib["mass"] = f"{m * mass_mult:.6g}"
    return True


def _scale_diaginertia(elem: ET.Element, factor: float) -> bool:
    if "diaginertia" not in elem.attrib:
        return False
    vs = _split_floats(elem.attrib["diaginertia"])
    elem.attrib["diaginertia"] = _fmt_floats([v * factor for v in vs])
    return True


def _scale_fullinertia(elem: ET.Element, factor: float) -> bool:
    # fullinertia is 6 entries (3 diagonal + 3 off-diagonal); all carry
    # length² × mass, so they scale identically with `factor`.
    if "fullinertia" not in elem.attrib:
        return False
    vs = _split_floats(elem.attrib["fullinertia"])
    elem.attrib["fullinertia"] = _fmt_floats([v * factor for v in vs])
    return True


def _scale_joint_range(elem: ET.Element, length_factor: float) -> bool:
    """Slide joints get their range scaled; hinge joints don't."""
    jtype = (elem.attrib.get("type") or "hinge").lower()
    if jtype != "slide":
        return False
    if "range" not in elem.attrib:
        return False
    vs = _split_floats(elem.attrib["range"])
    elem.attrib["range"] = _fmt_floats([v * length_factor for v in vs])
    return True


def _scale_default_density(elem: ET.Element, mass_extra: float) -> bool:
    """If a ``geom`` default carries an explicit ``density=...``, bump it
    by the extra mass multiplier so geoms without their own ``mass``
    attribute inherit a heavier-per-volume baseline."""
    if elem.tag != "geom":
        return False
    if "density" not in elem.attrib:
        return False
    try:
        d = float(elem.attrib["density"])
    except ValueError:
        return False
    elem.attrib["density"] = f"{d * mass_extra:.6g}"
    return True


def scale_tree(root: ET.Element, size_scale: float, mass_scale: float) -> dict:
    """Walk the MJCF tree and apply scaling. Returns a stats dict."""
    s = size_scale
    s3 = s ** 3
    s5 = s ** 5
    mass_factor = s3 * mass_scale          # for explicit `mass=` attrs
    inertia_factor = s5 * mass_scale       # for `diaginertia` / `fullinertia`

    stats = {
        "elements_visited": 0,
        "length_scaled": 0,
        "size_scaled": 0,
        "mesh_scaled": 0,
        "mass_scaled": 0,
        "inertia_scaled": 0,
        "joint_range_scaled": 0,
        "density_scaled": 0,
        "warn_no_mass_no_density": 0,
    }
    no_mass_warned_in: set = set()

    for elem in root.iter():
        stats["elements_visited"] += 1
        tag = elem.tag
        # Geometry-bearing attributes
        for a in _LENGTH_VEC_ATTRS:
            if _scale_attr(elem, a, s):
                stats["length_scaled"] += 1
        for a in _SIZE_VEC_ATTRS:
            if _scale_attr(elem, a, s):
                stats["size_scaled"] += 1
        for a in _SCALE_VEC_ATTRS:
            # `scale=` lives on <mesh> elements. RoboCasa lightwheel/objaverse
            # meshes often omit it (implicit MJCF default of "1 1 1"); without
            # synthesising one here, `<mesh file="visual/Pot052.obj"/>` would
            # silently keep its native size even when the task YAML asks for
            # scale != 1.0 — that's the bug behind "pot ignores scale".
            if tag == "mesh" and s != 1.0:
                if a not in elem.attrib:
                    elem.attrib[a] = _fmt_floats([s, s, s])
                    stats["mesh_scaled"] += 1
                elif _scale_attr(elem, a, s):
                    stats["mesh_scaled"] += 1
        # Mass-bearing attributes
        if _scale_mass(elem, mass_factor):
            stats["mass_scaled"] += 1
        if _scale_diaginertia(elem, inertia_factor):
            stats["inertia_scaled"] += 1
        if _scale_fullinertia(elem, inertia_factor):
            stats["inertia_scaled"] += 1
        # Joint range (length-dependent for slide joints only)
        if tag == "joint" and _scale_joint_range(elem, s):
            stats["joint_range_scaled"] += 1
        # Default density bump (so volumetric mass picks up --mass too)
        if mass_scale != 1.0 and _scale_default_density(elem, mass_scale):
            stats["density_scaled"] += 1

    # Walk again for warnings about geoms that have neither `mass` nor
    # a parent density (so the --mass multiplier didn't reach them).
    if mass_scale != 1.0:
        for elem in root.iter("geom"):
            if "mass" in elem.attrib or "density" in elem.attrib:
                continue
            # Look up the class default chain for density.
            cls = elem.attrib.get("class")
            if cls and _class_has_density(root, cls):
                continue
            stats["warn_no_mass_no_density"] += 1
            no_mass_warned_in.add(elem.attrib.get("name", "<unnamed>"))
    if no_mass_warned_in:
        sample = ", ".join(sorted(no_mass_warned_in)[:5])
        more = (f" (+{len(no_mass_warned_in) - 5} more)"
                if len(no_mass_warned_in) > 5 else "")
        print(f"  [warn] {len(no_mass_warned_in)} geom(s) have no explicit "
              f"mass/density and no class with density; --mass multiplier "
              f"won't affect their inertia: {sample}{more}",
              file=sys.stderr)
    return stats


def _class_has_density(root: ET.Element, class_name: str) -> bool:
    """Best-effort search: does any <default class="X"> ancestor have a
    <geom density="..."/> child?"""
    for d in root.iter("default"):
        if d.attrib.get("class") != class_name:
            continue
        for child in d:
            if child.tag == "geom" and "density" in child.attrib:
                return True
    return False


# ---------------------------------------------------------------------------
# Output formatting — keep diffs tidy
# ---------------------------------------------------------------------------

def _indent(elem: ET.Element, level: int = 0, step: str = "  ") -> None:
    """In-place ``ElementTree`` pretty-printer (mimics minidom but
    without losing attribute ordering)."""
    i = "\n" + level * step
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = i + step
        if not (elem.tail or "").strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1, step)
        if not (elem[-1].tail or "").strip():
            elem[-1].tail = i
    elif level and not (elem.tail or "").strip():
        elem.tail = i


# ---------------------------------------------------------------------------
# File / batch driver
# ---------------------------------------------------------------------------

def _resolve_output(src: Path, args) -> Path:
    if args.inplace:
        return src
    if args.out:
        return Path(args.out)
    suffix = args.suffix
    return src.with_name(src.stem + suffix + src.suffix)


def _process_one(src: Path, args) -> bool:
    try:
        tree = ET.parse(src)
    except ET.ParseError as e:
        print(f"[skip] {src}: parse failed: {e}", file=sys.stderr)
        return False
    root = tree.getroot()
    stats = scale_tree(root, args.size, args.mass)
    dst = _resolve_output(src, args)
    _indent(root)
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    print(f"[done] {src.name}  ->  {dst}")
    print(f"        size={args.size}  mass={args.mass}  "
          f"({stats['length_scaled']} length attrs, "
          f"{stats['size_scaled']} size attrs, "
          f"{stats['mesh_scaled']} mesh scales, "
          f"{stats['mass_scaled']} mass attrs, "
          f"{stats['inertia_scaled']} inertia attrs, "
          f"{stats['joint_range_scaled']} slide joint ranges, "
          f"{stats['density_scaled']} densities)")
    return True


def _iter_xmls(paths: Sequence[str]):
    for p in paths:
        path = Path(p)
        if path.is_dir():
            yield from sorted(path.rglob("*.xml"))
        elif path.suffix.lower() == ".xml" and path.is_file():
            yield path
        else:
            print(f"[skip] {path}: not an XML file or directory",
                  file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full notes.")
    parser.add_argument("paths", nargs="+",
                        help="One or more XML files OR directories to walk "
                             "(recursive *.xml). Each file gets scaled and "
                             "written out independently.")
    parser.add_argument("-s", "--size", type=float, default=1.0,
                        help="Linear scale factor for positions, sizes, "
                             "mesh scales, prismatic joint ranges. Mass "
                             "and inertia auto-track this (mass *= s^3, "
                             "diaginertia *= s^5). Default 1.0 (no scale).")
    parser.add_argument("-m", "--mass", type=float, default=1.0,
                        help="Extra dimensionless mass multiplier applied "
                             "on top of the volumetric s^3 scaling. Use "
                             "to make objects heavier WITHOUT enlarging "
                             "them (set --size 1.0 --mass 2.0). Inertia "
                             "also picks this up. Default 1.0.")
    parser.add_argument("-o", "--out", default=None,
                        help="Explicit output path. Only valid when exactly "
                             "one input file is given. Mutually exclusive "
                             "with --inplace and --suffix.")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the input file(s) in place. "
                             "Mutually exclusive with --out and --suffix.")
    parser.add_argument("--suffix", default="_scaled",
                        help="Suffix inserted before .xml for non-inplace "
                             "non-explicit-out runs. Default '_scaled'.")
    args = parser.parse_args()

    n_explicit = sum([bool(args.inplace), bool(args.out)])
    if n_explicit > 1:
        parser.error("--inplace and --out are mutually exclusive")
    if args.out and len(args.paths) > 1:
        parser.error("--out requires exactly one input path")
    if args.size <= 0 or args.mass <= 0:
        parser.error("--size and --mass must be > 0")

    files = list(_iter_xmls(args.paths))
    if not files:
        parser.error("no XML files found in inputs")

    ok = 0
    for f in files:
        if _process_one(f, args):
            ok += 1
    print(f"\nProcessed {ok}/{len(files)} file(s).")


if __name__ == "__main__":
    main()
