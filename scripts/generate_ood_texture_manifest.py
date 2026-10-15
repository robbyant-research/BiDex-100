#!/usr/bin/env python3
"""Generate the canonical, role-aware OOD texture manifest.

The manifest stores logical paths and content hashes, never machine-specific
absolute paths. Exact duplicate images are collapsed while their role sets are
merged. Run this deliberately whenever the benchmark texture inventory changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def roles_for(logical: str) -> set[str]:
    p = logical.lower()
    if "/prev/" in p or "/window_bkgs/" in p:
        return set()
    if "robocasa_assets/generative_textures/" in p:
        role_dir = p.split("robocasa_assets/generative_textures/", 1)[1].split("/", 1)[0]
        return {"wall"} if role_dir == "wall" else ({"floor"} if role_dir == "floor" else ({"table"} if role_dir in {"counter", "cabinet"} else set()))

    # Normalize the RoboCasa rotated variants back to their material family.
    stem = Path(p).stem.replace("_rot_rot", "").replace("_rot", "")
    text = f"{p} {stem}"
    # An explicit surface name wins over generic material words. For example,
    # ``wood_planks_wall.png`` is wood, but is not a sensible table texture.
    if any(k in stem for k in ("wall", "plaster", "stucco")):
        return {"wall"}
    if any(k in stem for k in ("floor", "ground")):
        return {"floor"}
    if any(k in stem for k in ("table", "counter", "cabinet")):
        return {"table"}
    roles = set()
    if any(k in text for k in ("wall", "plaster", "brick", "stucco", "flat/", "stone_wall")):
        roles.add("wall")
    if any(k in text for k in ("floor", "tile", "ground", "plank", "parquet", "marble", "concrete", "brick")):
        roles.add("floor")
    if any(k in text for k in ("wood", "tile", "ceramic", "metal", "fabric", "marble", "laminate", "flat/", "steel", "granite")):
        roles.add("table")
    return roles


def build_manifest(asset_root: Path) -> dict:
    roots = {"libero": asset_root / "libero",
             "robocasa_assets": asset_root / "robocasa_assets"}
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"missing asset root {name}: {root}")

    candidates = []
    for root_name, root in roots.items():
        search_roots = ([root / "textures"] if root_name == "libero" else
                        [root / "textures", root / "generative_textures"])
        for search_root in search_roots:
            for path in sorted(search_root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                logical = f"{root_name}/{path.relative_to(root).as_posix()}"
                roles = roles_for(logical)
                if roles:
                    data = path.read_bytes()
                    candidates.append({"path": logical, "size": len(data),
                                       "sha256": hashlib.sha256(data).hexdigest(),
                                       "roles": roles})

    # One sampling entry per unique visual asset. Prefer the lexicographically
    # first logical path and merge roles from any byte-identical aliases.
    by_hash = {}
    for item in sorted(candidates, key=lambda x: x["path"]):
        current = by_hash.get(item["sha256"])
        if current is None:
            by_hash[item["sha256"]] = item
        else:
            current["roles"].update(item["roles"])
    files = []
    for item in sorted(by_hash.values(), key=lambda x: x["path"]):
        files.append({"path": item["path"], "size": item["size"],
                      "sha256": item["sha256"], "roles": sorted(item["roles"])})
    rows = [(f["path"], f["size"], f["sha256"], tuple(f["roles"])) for f in files]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return {"schema": 1, "fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.asset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = {role: sum(role in f["roles"] for f in manifest["files"])
              for role in ("wall", "floor", "table")}
    print(f"wrote {args.output}: {len(manifest['files'])} unique textures, "
          f"roles={counts}, fingerprint={manifest['fingerprint']}")


if __name__ == "__main__":
    main()
