"""OOD (Out-Of-Domain) perturbations for the unified eval harness.

Four env-build-time OOD pieces + runtime hooks:
  1. Wall/floor/table texture + key/fill lighting randomization (scene XML ET rewrite).
  2. Distractor object injection (Generalization / Limited_Demonstrations):
     N objects from similar tasks, no-overlap sampling, grab-wrong-object
     early termination.
  3. Per-episode articulated-object scale variation.
  4. Per-episode camera-extrinsic selection from a validated finite pose bank.

Plus the post-reset object-position offset (legacy fallback for OOD runs whose
task has no baked `<task>__ood.yaml` variant; the generated OOD variants bake
their own region offsets into the yaml, see scripts/gen_ood_yamls.py).

Which perturbations apply is driven by the task yaml's top-level `ood:` block
(only present in `__ood` variant configs): {scene_randomize, distractors,
baked_layout}.
"""
from __future__ import annotations
import hashlib
import os, tempfile, json
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import numpy as np

ASSETS = Path(os.environ.get("BIDEX_ASSETS_ROOT") or str(Path(__file__).resolve().parent.parent))
ARTICRAFT_DIR = ASSETS / "libero" / "objects" / "articraft_objects"
_PKG_ROOT = Path(__file__).resolve().parents[1]
_TEXTURE_MANIFEST = Path(__file__).with_name("ood_texture_manifest.json")
_LEGACY_DISTRACTOR_MANIFEST = Path(__file__).with_name(
    "ood_legacy_distractors.json"
)
_CAMERA_POSE_MANIFEST = Path(__file__).with_name("ood_camera_pose_manifest.json")
RANDOMIZATION_SCHEMA = "bidex-ood-v8"
LIGHTING_SCHEMA = "bidex-lighting-v4"
CAMERA_RANDOMIZATION_SCHEMA = "bidex-camera-v4"
_MANIFEST_CACHE = None
_LEGACY_DISTRACTOR_CACHE = None
_VALIDATED_TEXTURES = None
_CAMERA_POSE_CACHE = None

# Texture names used by the two benchmark scenes:
#   dexjoco_table_arena.xml       : texplane / tex-light-gray-plaster / tex-ceramic
#   libero_tabletop_base_style.xml: texplane / tex-wall / tex-table (table top is a
#                                   CUBE texture -> converted to 2d on swap)
_FLOOR_TEX_NAMES = ["texplane"]                              # floor material floorplane
_WALL_TEX_NAMES = ["tex-light-gray-plaster", "tex-wall"]     # walls_mat (both scenes)
_TABLE_TEX_NAMES = ["tex-ceramic", "tex-table"]              # table-top (both scenes; legs untouched)
_LIGHT_TAGS = ["light"]
# table_collision world x,y bounds + tabletop z per scene:
#   dexjoco_table_arena       : body (0.3,0) half (0.4,0.8) top z=0.85
#   libero_tabletop_base_style: body (0,0)  half (0.4,0.4) top z=0.80
_TABLE_WORLD = ((-0.08, 0.66), (-0.74, 0.74), 0.85)
_TABLE_WORLD_LIBERO = ((-0.35, 0.35), (-0.35, 0.35), 0.80)

# The lighting schedule deliberately spans low/high elevations and the full
# azimuth circle.  One shadow-casting key light produces meaningful shadow
# direction changes; a broad, non-shadowing fill light enforces a conservative
# illumination floor so an OOD episode never becomes an unusable near-black
# observation.  RGB tints are luminance-normalised below before use.
_LIGHT_ELEVATION_DEG = (25.0, 38.0, 52.0, 66.0, 76.0)
_LIGHT_DISTANCE_M = (3.0, 3.5, 4.0, 4.4)
# Five explicit exposure profiles make the episode-to-episode brightness
# change visible while retaining a non-zero fill/ambient floor.  The brightest
# profile stays close to the previously audited maximum total illumination;
# most of the added span comes from making the dark profiles meaningfully
# darker, rather than pushing highlights into clipping.
_LIGHT_PROFILES = (
    {"id": "dark",       "key": 0.11, "fill": 0.035, "ambient": 0.014},
    {"id": "dim",        "key": 0.18, "fill": 0.060, "ambient": 0.022},
    {"id": "balanced",   "key": 0.29, "fill": 0.100, "ambient": 0.040},
    {"id": "bright",     "key": 0.40, "fill": 0.145, "ambient": 0.058},
    {"id": "high_bright","key": 0.48, "fill": 0.180, "ambient": 0.072},
)
_LIGHT_TINTS = (
    (1.0, 1.0, 1.0),        # neutral
    (1.0, 0.90, 0.78),      # warm
    (0.78, 0.90, 1.0),      # cool
)
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

# Each consecutive five-episode block contains this exact multiset in a
# deterministic task/seed-specific order.  Consequently the benchmark's two
# OOD seeds (2 and 3), with 25 episodes each, contain exactly 20 nominal,
# 10 head-only, 10 wrists-only and 10 head+wrists camera occurrences.
_CAMERA_OCCURRENCE_BLOCK = (
    "nominal",
    "nominal",
    "head_only",
    "wrists_only",
    "head_and_wrists",
)


def _table_world_for(task):
    # randomize_scene replaces task.scene with a temp xml path — the original
    # scene name is stashed on _base_scene by the eval loader beforehand.
    scene = getattr(task, "_base_scene", None) or getattr(task, "scene", "") or ""
    if "dexjoco" in scene:
        return _TABLE_WORLD
    return _TABLE_WORLD_LIBERO


def _distractor_center_bounds(task):
    """Reachable centre bounds for injected clutter (inside table bounds)."""
    (xlo, xhi), (ylo, yhi), _ = _table_world_for(task)
    robot = str(getattr(task, "robot_type", "") or "").lower()
    if "astribot" in robot:
        # FK audit: Astribot tabletop reach starts near x=0.20 m. Keep clutter
        # away from the rear/side table edges so it remains a meaningful,
        # reachable distractor instead of visual background.
        return ((max(xlo, 0.18), min(xhi, 0.63)),
                (max(ylo, -0.65), min(yhi, 0.65)))
    return (xlo, xhi), (ylo, yhi)


def _distractor_protection_margin(task, obj) -> float:
    """Extra keep-out around assembly surfaces that must stay unobstructed."""
    distractor_cfg = (
        (getattr(task, "ood", None) or {}).get("distractors") or {}
    )
    configured = float(distractor_cfg.get("object_clearance", 0.0))
    if configured > 0.0:
        protected = set(getattr(task, "obj_of_interest", []) or [])
        protected.update(
            a.get("object") if isinstance(a, dict) else getattr(a, "object", None)
            for a in getattr(task, "articulations", [])
        )
        protected.discard(None)
        if obj.name in protected:
            return configured
    if (getattr(task, "name", "") == "assemble_arch_bridge"
            and getattr(obj, "name", "") in {"assembly_base", "assembly_bridge_deck"}):
        return 0.05
    return 0.0


def randomization_seed(suite: str, task: str, seed: int, component: str,
                       episode: Optional[int] = None) -> int:
    """Stable, component-scoped seed independent of process and machine paths."""
    parts = [RANDOMIZATION_SCHEMA, suite, task, str(int(seed)), component]
    if episode is not None:
        parts.append(str(int(episode)))
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def randomization_rng(suite: str, task: str, seed: int, component: str,
                      episode: Optional[int] = None) -> np.random.Generator:
    """Return an independent RNG stream for one named randomization component."""
    return np.random.Generator(np.random.PCG64(
        randomization_seed(suite, task, seed, component, episode)))


def episode_rng(suite: str, task: str, seed: int, episode: int,
                component: str = "placement") -> np.random.Generator:
    return randomization_rng(suite, task, seed, component, episode)


def episode_seed(suite: str, task: str, seed: int, episode: int,
                 component: str = "environment") -> int:
    """Stable non-negative 63-bit seed for APIs that accept an integer seed."""
    return randomization_seed(suite, task, seed, component, episode) & ((1 << 63) - 1)


def _camera_manifest_payload(manifest: dict) -> bytes:
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def load_camera_pose_manifest() -> dict:
    """Load and strictly validate the finite, cross-machine camera pose bank."""
    global _CAMERA_POSE_CACHE
    if _CAMERA_POSE_CACHE is not None:
        return _CAMERA_POSE_CACHE
    manifest = json.loads(_CAMERA_POSE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("name") != CAMERA_RANDOMIZATION_SCHEMA:
        raise RuntimeError(f"unsupported OOD camera manifest: {_CAMERA_POSE_MANIFEST}")
    actual = hashlib.sha256(_camera_manifest_payload(manifest)).hexdigest()
    if actual != manifest.get("fingerprint"):
        raise RuntimeError(
            f"OOD camera manifest fingerprint mismatch: expected "
            f"{manifest.get('fingerprint')}, got {actual}"
        )
    poses = manifest.get("poses") or []
    if not poses:
        raise RuntimeError("OOD camera pose manifest contains no poses")
    pos_limit = np.asarray(manifest["translation_limit_m"], dtype=np.float64)
    rot_limit = np.asarray(manifest["rotation_limit_deg"], dtype=np.float64)
    ids = set()
    for pose in poses:
        pid = str(pose.get("id", ""))
        if not pid or pid in ids:
            raise RuntimeError(f"invalid/duplicate OOD camera pose id: {pid!r}")
        ids.add(pid)
        pos = np.asarray(pose.get("delta_pos"), dtype=np.float64)
        rot = np.asarray(pose.get("delta_euler_deg"), dtype=np.float64)
        if pos.shape != (3,) or rot.shape != (3,):
            raise RuntimeError(f"OOD camera pose {pid} must contain two 3-vectors")
        if np.any(np.abs(pos) > pos_limit + 1e-12):
            raise RuntimeError(f"OOD camera pose {pid} exceeds translation limits")
        if np.any(np.abs(rot) > rot_limit + 1e-12):
            raise RuntimeError(f"OOD camera pose {pid} exceeds rotation limits")
    for camera, scales in (manifest.get("camera_scales") or {}).items():
        for key in ("translation", "rotation"):
            value = float(scales.get(key, 1.0))
            if not (0.0 < value <= 1.0):
                raise RuntimeError(
                    f"invalid OOD camera {key} scale for {camera}: {value}"
                )
    for robot, camera_rules in (manifest.get("robot_camera_rules") or {}).items():
        if not isinstance(camera_rules, dict):
            raise RuntimeError(f"invalid OOD camera rules for {robot}")
        for camera, rule in camera_rules.items():
            positive = rule.get("positive_translation_axes", [])
            negative = rule.get("negative_translation_axes", [])
            for sign, axes in (("positive", positive), ("negative", negative)):
                if any(not isinstance(axis, int) or axis < 0 or axis > 2
                       for axis in axes):
                    raise RuntimeError(
                        f"invalid {sign} translation axes for {robot}/{camera}"
                    )
            if set(positive) & set(negative):
                raise RuntimeError(
                    f"conflicting translation axes for {robot}/{camera}"
                )
    _CAMERA_POSE_CACHE = manifest
    return manifest


def camera_pose_delta(manifest: dict, robot_type: str, camera_name: str,
                      pose: dict) -> dict:
    """Adapt a bank pose to mount-specific collision-safe directions."""
    scales = (manifest.get("camera_scales") or {}).get(str(camera_name), {})
    pos = (np.asarray(pose["delta_pos"], dtype=np.float64)
           * float(scales.get("translation", 1.0)))
    rot = (np.asarray(pose["delta_euler_deg"], dtype=np.float64)
           * float(scales.get("rotation", 1.0)))
    rule = ((manifest.get("robot_camera_rules") or {}).get(str(robot_type), {})
            .get(str(camera_name), {}))
    for axis in rule.get("positive_translation_axes", []):
        pos[int(axis)] = abs(pos[int(axis)])
    for axis in rule.get("negative_translation_axes", []):
        pos[int(axis)] = -abs(pos[int(axis)])
    return {
        "pose_id": pose["id"],
        "delta_pos": [float(v) for v in pos],
        "delta_euler_deg": [float(v) for v in rot],
    }


def camera_occurrence_mode(suite: str, task: str, seed: int,
                           episode: int) -> str:
    """Choose an exact-ratio camera occurrence mode for one OOD episode.

    A fresh deterministic permutation is used for every five-episode block.
    This preserves the requested 2:1:1:1 ratio in every complete block while
    avoiding a globally synchronized mode order across tasks and seeds.
    """
    episode = int(episode)
    if episode < 0:
        raise ValueError("benchmark episode must be non-negative")
    block, slot = divmod(episode, len(_CAMERA_OCCURRENCE_BLOCK))
    order = randomization_rng(
        suite, task, int(seed),
        f"{CAMERA_RANDOMIZATION_SCHEMA}.occurrence",
        block,
    ).permutation(len(_CAMERA_OCCURRENCE_BLOCK))
    return _CAMERA_OCCURRENCE_BLOCK[int(order[slot])]


def _nominal_camera_delta() -> dict:
    return {
        "pose_id": "nominal",
        "delta_pos": [0.0, 0.0, 0.0],
        "delta_euler_deg": [0.0, 0.0, 0.0],
    }


def camera_randomization_plan(robot_type: str, suite: str, task: str,
                              seed: int, episode: int,
                              camera_names=("head", "left_wrist", "right_wrist")) -> dict:
    """Select nominal or pre-audited camera extrinsics for an OOD episode.

    Selection uses only stable SHA-256/PCG64 streams and the committed manifest;
    it never depends on renderer output, GPU type or model behaviour.
    """
    manifest = load_camera_pose_manifest()
    poses = manifest["poses"]
    ordinal = _texture_episode_ordinal(int(seed), int(episode))
    occurrence_mode = camera_occurrence_mode(suite, task, seed, episode)
    change_head = occurrence_mode in {"head_only", "head_and_wrists"}
    change_wrists = occurrence_mode in {"wrists_only", "head_and_wrists"}
    selected = {}
    for camera_name in camera_names:
        should_change = change_head if camera_name == "head" else change_wrists
        if not should_change:
            selected[camera_name] = _nominal_camera_delta()
            continue
        order_rng = randomization_rng(
            suite, task, 0,
            f"{CAMERA_RANDOMIZATION_SCHEMA}.schedule.{robot_type}.{camera_name}",
        )
        order = order_rng.permutation(len(poses))
        pose = poses[int(order[ordinal % len(poses)])]
        selected[camera_name] = camera_pose_delta(
            manifest, robot_type, camera_name, pose
        )
    return {
        "schema": CAMERA_RANDOMIZATION_SCHEMA,
        "manifest_fingerprint": manifest["fingerprint"],
        "robot_type": str(robot_type),
        "suite": str(suite),
        "task": str(task),
        "seed": int(seed),
        "episode": int(episode),
        "occurrence_mode": occurrence_mode,
        "cameras": selected,
    }


def _texture_episode_ordinal(seed: int, episode: int) -> int:
    """Injectively map non-negative (seed, episode) pairs to an integer."""
    seed = int(seed)
    episode = int(episode)
    if seed < 0 or episode < 0:
        raise ValueError("benchmark seed and episode must be non-negative")
    total = seed + episode
    return total * (total + 1) // 2 + episode


def _texture_schedule_slot(seed: int, episode: int, pool_size: int) -> int:
    """Return a no-repeat slot for the canonical seeds 0..3.

    Each seed owns a disjoint quarter of the role pool. The real manifest's
    smallest post-filter pool has 160 entries, so every seed gets at least 40
    unique episode textures. Other seeds/longer runs use the stable paired
    fallback and remain deterministic, though finite pools may eventually wrap.
    """
    if pool_size <= 0:
        raise ValueError("texture pool must be non-empty")
    seed = int(seed)
    episode = int(episode)
    per_seed = pool_size // 4
    if 0 <= seed < 4 and 0 <= episode < per_seed:
        return seed * per_seed + episode
    return _texture_episode_ordinal(seed, episode) % pool_size


def _load_texture_manifest() -> dict:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        if not _TEXTURE_MANIFEST.is_file():
            raise FileNotFoundError(
                f"OOD texture manifest missing: {_TEXTURE_MANIFEST}; run "
                "scripts/generate_ood_texture_manifest.py")
        _MANIFEST_CACHE = json.loads(_TEXTURE_MANIFEST.read_text(encoding="utf-8"))
        if _MANIFEST_CACHE.get("schema") != 1:
            raise RuntimeError(f"unsupported OOD texture manifest schema: "
                               f"{_MANIFEST_CACHE.get('schema')!r}")
    return _MANIFEST_CACHE


def _texture_roots() -> dict[str, Path]:
    """Resolve logical asset roots without making absolute paths part of RNG."""
    explicit = os.environ.get("BIDEX_ASSETS_ROOT", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(ASSETS)
    for root in candidates:
        libero = root / "libero"
        robocasa = root / "robocasa_assets"
        if libero.is_dir() and robocasa.is_dir():
            return {"libero": libero, "robocasa_assets": robocasa}
    # Development checkout: both asset packs are normally symlinked into bidex/.
    libero = _PKG_ROOT / "libero"
    robocasa = _PKG_ROOT / "robocasa" / "assets"
    if libero.is_dir() and robocasa.is_dir():
        return {"libero": libero.resolve(), "robocasa_assets": robocasa.resolve()}
    raise FileNotFoundError(
        "OOD requires both libero/ and robocasa_assets/. Set BIDEX_ASSETS_ROOT "
        "to their common parent or run scripts/setup_assets_links.sh")


def validate_texture_manifest(force: bool = False) -> dict:
    """Strictly validate every managed texture and return role -> resolved files."""
    global _VALIDATED_TEXTURES
    if _VALIDATED_TEXTURES is not None and not force:
        return _VALIDATED_TEXTURES
    manifest = _load_texture_manifest()
    roots = _texture_roots()
    resolved = {"floor": [], "wall": [], "table": []}
    fingerprint_rows = []
    for entry in manifest["files"]:
        logical = entry["path"]
        root_name, rel = logical.split("/", 1)
        path = roots[root_name] / rel
        if not path.is_file():
            raise FileNotFoundError(f"OOD texture missing: {logical} ({path})")
        size = path.stat().st_size
        if size != int(entry["size"]):
            raise RuntimeError(f"OOD texture size mismatch: {logical}: "
                               f"expected {entry['size']}, got {size}")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise RuntimeError(f"OOD texture hash mismatch: {logical}")
        fingerprint_rows.append((logical, size, sha, tuple(entry["roles"])))
        for role in entry["roles"]:
            resolved[role].append((logical, str(path.resolve())))
    payload = json.dumps(fingerprint_rows, separators=(",", ":"), ensure_ascii=True)
    actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if actual != manifest["fingerprint"]:
        raise RuntimeError("OOD texture manifest fingerprint mismatch")
    for role, files in resolved.items():
        files.sort(key=lambda item: item[0])
        if not files:
            raise RuntimeError(f"OOD texture role has no files: {role}")
    resolved["fingerprint"] = manifest["fingerprint"]
    _VALIDATED_TEXTURES = resolved
    return resolved


def texture_manifest_fingerprint() -> str:
    return str(validate_texture_manifest()["fingerprint"])


def _scene_table_target(scene_xml: str) -> np.ndarray:
    """Return the centre of the useful tabletop workspace in world metres."""
    if "dexjoco" in Path(scene_xml).name:
        (xlo, xhi), (ylo, yhi), z = _TABLE_WORLD
    else:
        (xlo, xhi), (ylo, yhi), z = _TABLE_WORLD_LIBERO
    return np.asarray([(xlo + xhi) * 0.5, (ylo + yhi) * 0.5, z + 0.12],
                      dtype=np.float64)


def _normalised_tint(rgb) -> np.ndarray:
    """Keep warm/cool variants at equal Rec.709 luminance."""
    arr = np.asarray(rgb, dtype=np.float64)
    lum = float(np.dot(arr, np.asarray([0.2126, 0.7152, 0.0722])))
    return arr / max(lum, 1e-9)


def _fmt3(values) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _scheduled_lighting(scene_xml: str, suite: str, task: str,
                        seed: int, episode: int,
                        profile_id: Optional[str] = None) -> list[dict]:
    """Build a reproducible key/fill plan with a hard illumination floor."""
    ordinal = _texture_episode_ordinal(int(seed), int(episode))
    phase_rng = randomization_rng(
        suite, task, 0, f"{LIGHTING_SCHEMA}.azimuth_phase"
    )
    phase = float(phase_rng.uniform(-np.pi, np.pi))
    azimuth = phase + ordinal * _GOLDEN_ANGLE
    elevation = np.deg2rad(
        _LIGHT_ELEVATION_DEG[ordinal % len(_LIGHT_ELEVATION_DEG)]
    )
    distance = _LIGHT_DISTANCE_M[(ordinal * 3 + int(seed)) % len(_LIGHT_DISTANCE_M)]
    target = _scene_table_target(scene_xml)

    horizontal = distance * np.cos(elevation)
    key_pos = target + np.asarray([
        horizontal * np.cos(azimuth),
        horizontal * np.sin(azimuth),
        distance * np.sin(elevation),
    ])
    # Put the fill high and approximately opposite the key.  Its non-zero
    # diffuse + ambient contribution prevents black observations even for a
    # low grazing key-light angle.
    fill_azimuth = azimuth + np.pi
    fill_distance = 4.2
    fill_elevation = np.deg2rad(68.0)
    fill_horizontal = fill_distance * np.cos(fill_elevation)
    fill_pos = target + np.asarray([
        fill_horizontal * np.cos(fill_azimuth),
        fill_horizontal * np.sin(fill_azimuth),
        fill_distance * np.sin(fill_elevation),
    ])

    tint_idx = (ordinal + int(seed)) % len(_LIGHT_TINTS)
    key_tint = _normalised_tint(_LIGHT_TINTS[tint_idx])
    # Use the complementary temperature for the fill, but at much lower
    # saturation; this maximises appearance variation without destroying
    # colour identity of task objects.
    fill_tint = 0.75 * np.ones(3) + 0.25 * _normalised_tint(
        _LIGHT_TINTS[(tint_idx + 1) % len(_LIGHT_TINTS)]
    )
    if profile_id is None:
        # Every canonical 25-episode seed run visits all five exposure levels
        # exactly five times. Keep this independent from the Cantor ordinal
        # used by textures/angles so no profile can be accidentally skipped.
        profile = _LIGHT_PROFILES[
            (int(seed) + int(episode)) % len(_LIGHT_PROFILES)
        ]
    else:
        matches = [p for p in _LIGHT_PROFILES if p["id"] == profile_id]
        if not matches:
            raise ValueError(f"unknown OOD lighting profile: {profile_id!r}")
        profile = matches[0]
    key_level = float(profile["key"])
    fill_level = float(profile["fill"])
    ambient_level = float(profile["ambient"])

    def make(kind, pos, diffuse, castshadow, ambient, specular):
        direction = target - pos
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        return {
            "kind": kind,
            "profile": profile["id"],
            "pos": [float(v) for v in pos],
            "dir": [float(v) for v in direction],
            "diffuse": [float(v) for v in diffuse],
            "ambient": [float(v) for v in ambient],
            "specular": [float(v) for v in specular],
            "castshadow": bool(castshadow),
            "directional": False,
            "cutoff": 80.0,
            "exponent": 1.0,
        }

    return [
        make("key", key_pos, key_tint * key_level, True,
             np.zeros(3), key_tint * min(0.18, key_level * 0.20)),
        make("fill", fill_pos, fill_tint * fill_level, False,
             fill_tint * ambient_level,
             fill_tint * min(0.040, fill_level * 0.25)),
    ]


def randomize_scene(scene_xml: str, rng: Optional[np.random.Generator] = None, *,
                    suite: str = "", task: str = "", seed: Optional[int] = None,
                    episode: Optional[int] = None,
                    lighting_profile: Optional[str] = None,
                    return_metadata: bool = False):
    """Rewrite scene XML with managed textures and safe key/fill lighting.

    This randomizes actual floor/wall/table-top texture files and uses a
    shadow-casting key light plus an illumination-floor fill light. Writes a
    temp .xml (absolute pool
    paths so the swap resolves regardless of the temp file's dir)."""
    tree = ET.parse(scene_xml); root = tree.getroot()
    pools = validate_texture_manifest()
    metadata = {"schema": RANDOMIZATION_SCHEMA,
                "seed": int(seed) if seed is not None else None,
                "episode": int(episode) if episode is not None else None,
                "texture_manifest": pools["fingerprint"], "textures": {}, "lights": []}

    def component_rng(component):
        if seed is not None and suite and task:
            return randomization_rng(suite, task, seed, component, episode)
        if rng is None:
            raise ValueError("randomize_scene needs either rng or suite/task/seed")
        return rng

    def swap_textures(tex_names, role, allow_cube_to_2d=False):
        sel = pools[role]
        for tex_idx, asset in enumerate(root.iter("texture")):
            name = asset.get("name", "")
            if name not in tex_names:
                continue
            if not asset.get("file"):
                continue  # only file-based textures (skip skybox/builtin)
            if asset.get("type", "2d") != "2d":
                # libero_tabletop_base_style's table-top `tex-table` is a file-based
                # CUBE texture; a 2d png cannot drop in as a cube map, so convert
                # the element to a plain 2d texture (the table-top box maps it per
                # face; sides may stretch — acceptable for OOD variety).
                if not allow_cube_to_2d:
                    continue
                asset.set("type", "2d")
            cur = Path(asset.get("file")).name
            alts = [p for p in sel if Path(p[1]).name != cur] or sel
            if seed is not None and episode is not None and suite and task:
                # One stable permutation per task/role. Index it with the
                # (seed, episode) slot so textures never repeat in any
                # canonical 25-episode run for seeds 0..3.
                schedule_rng = randomization_rng(
                    suite, task, 0, f"texture_schedule.{role}.{tex_idx}")
                order = schedule_rng.permutation(len(alts))
                slot = _texture_schedule_slot(seed, episode, len(alts))
                selected = int(order[slot])
            else:
                role_rng = component_rng(f"texture.{role}.{tex_idx}")
                selected = int(role_rng.integers(0, len(alts)))
            logical, chosen = alts[selected]
            asset.set("file", chosen)
            metadata["textures"][role] = logical

    swap_textures(_FLOOR_TEX_NAMES, "floor")
    swap_textures(_WALL_TEX_NAMES, "wall")
    swap_textures(_TABLE_TEX_NAMES, "table", allow_cube_to_2d=True)

    lights = list(root.iter("light"))
    if seed is not None and episode is not None and suite and task:
        plans = _scheduled_lighting(
            scene_xml, suite, task, int(seed), int(episode), lighting_profile
        )
    else:
        # Compatibility for direct callers that supply only an RNG.  Convert
        # the caller's stream to stable integer inputs, then use the same safe
        # key/fill implementation as leaderboard evaluation.
        compat_rng = component_rng(f"{LIGHTING_SCHEMA}.compat")
        compat_seed = int(compat_rng.integers(0, 1 << 31))
        compat_episode = int(compat_rng.integers(0, 1 << 20))
        plans = _scheduled_lighting(
            scene_xml, suite or "direct", task or Path(scene_xml).stem,
            compat_seed, compat_episode, lighting_profile,
        )
    for light_idx, light in enumerate(lights):
        plan = plans[min(light_idx, len(plans) - 1)]
        light.set("pos", _fmt3(plan["pos"]))
        light.set("dir", _fmt3(plan["dir"]))
        light.set("diffuse", _fmt3(plan["diffuse"]))
        light.set("ambient", _fmt3(plan["ambient"]))
        light.set("specular", _fmt3(plan["specular"]))
        light.set("castshadow", "true" if plan["castshadow"] else "false")
        light.set("directional", "true" if plan["directional"] else "false")
        light.set("cutoff", f"{plan['cutoff']:.3f}")
        light.set("exponent", f"{plan['exponent']:.3f}")
        metadata["lights"].append({"index": light_idx, **plan})
    metadata["lighting_schema"] = LIGHTING_SCHEMA

    # write to temp (next to the original so ../textures/ relative paths resolve)
    orig_dir = Path(scene_xml).parent
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=str(orig_dir), prefix="ood_scene_")
    os.close(fd)
    tree.write(tmp)
    return (tmp, metadata) if return_metadata else tmp


def _collect_similar_objects(suite: str, current_task: str, exclude: set[str], rng, n: int):
    """Pick `n` object names from OTHER tasks in the same category dir as
    `current_task` (similar tasks). After the category reorganization the
    historical `suite` argument (tabletop/kitchen/...) is no longer a
    directory, so the pool dir is resolved from the task's own yaml path.
    `__ood` variant yamls are excluded from the pool.

    Pool is restricted to small desktop objects: rc/* catalog or dexjoco_*
    primitives, excluding big fixture-like pieces (baskets/cabinets/racks/...)
    and any bare articraft-appliance names (no rc/ prefix)."""
    import re as _re
    from bidex.tasks.registry import resolve_task_path
    from bidex import load_task
    _BIG = _re.compile(
        r"(tiered_basket|cabinet|shelf|rack|machine|drawer|fridge|microwave"
        r"|oven(?!_tray)|_stand|knife_block|mug_tree)", _re.IGNORECASE)

    def _ok(name):
        if not (name.startswith("rc/") or name.startswith("dexjoco_")):
            return False
        return not _BIG.search(name)

    try:
        pool_dir = Path(resolve_task_path(suite, current_task)).parent
    except Exception:
        pool_dir = None
    task_names = []
    if pool_dir is not None:
        task_names = sorted(p.stem for p in pool_dir.glob("*.yaml")
                            if not p.stem.endswith("__ood"))
    candidates = []
    for t in task_names:
        if t == current_task:
            continue
        try:
            task = load_task(suite, t)
        except Exception:
            continue
        for obj in task.objects:
            if obj.name not in exclude and _ok(obj.name):
                candidates.append(obj.name)
    candidates = sorted(set(candidates))
    if not candidates:
        return []
    n = min(n, len(candidates))
    order = rng.permutation(len(candidates))
    return [candidates[int(i)] for i in order[:n]]


def _legacy_distractor_entry(task, eval_seed: Optional[int]) -> dict:
    """Return the frozen seed-2/3 compatibility set for one task.

    The old sampler depended on historical task inventory and task-specific
    count branches.  Freezing its already-audited outputs in a manifest keeps
    cross-machine replay stable without mixing migration policy into packing.
    """
    global _LEGACY_DISTRACTOR_CACHE
    if eval_seed is None:
        return {"preserve": [], "omitted": []}
    if _LEGACY_DISTRACTOR_CACHE is None:
        payload = json.loads(_LEGACY_DISTRACTOR_MANIFEST.read_text())
        if payload.get("schema") != "bidex-legacy-distractors-v1":
            raise RuntimeError(
                f"unsupported legacy distractor manifest: "
                f"{payload.get('schema')!r}"
            )
        _LEGACY_DISTRACTOR_CACHE = payload.get("seeds", {})
    return dict(
        _LEGACY_DISTRACTOR_CACHE
        .get(f"{task.suite}/{task.name}", {})
        .get(str(int(eval_seed)), {"preserve": [], "omitted": []})
    )


_ARTICRAFT_EPISODE_SPAN = 0.18



def randomize_articraft_scale(task, rng: np.random.Generator) -> dict:
    """Vary only the object controlled by ``ood.articraft_scale``.

    Generated OOD YAMLs already bake ``ID scale * articraft_scale`` into the
    object. Recover the ID scale, then choose a visibly different per-episode
    OOD factor while preserving the configured direction (smaller or larger
    than ID). Ordinary task objects and distractors retain their YAML scales.
    """
    cfg = (getattr(task, "ood", None) or {}).get("articraft_scale")
    if cfg is None:
        return {}
    cfg = float(cfg)
    if cfg < 1.0:
        lo = max(0.65, cfg - _ARTICRAFT_EPISODE_SPAN)
        hi = min(0.96, cfg + _ARTICRAFT_EPISODE_SPAN)
    elif cfg > 1.0:
        lo = max(1.04, cfg - _ARTICRAFT_EPISODE_SPAN)
        hi = min(1.35, cfg + _ARTICRAFT_EPISODE_SPAN)
    else:
        lo, hi = 0.90, 1.10
    factor = float(rng.uniform(lo, hi))
    articulated = {
        (a.get("object") if isinstance(a, dict) else getattr(a, "object", None))
        for a in getattr(task, "articulations", [])
    }
    articulated.discard(None)
    result = {}
    for obj in task.objects:
        if obj.name not in articulated:
            continue
        baked_scale = float(getattr(obj, "scale", 1.0))
        id_scale = baked_scale / cfg
        obj.scale = round(id_scale * factor, 8)
        result[f"obj{obj.instance_id}:{obj.name}"] = {
            "id_scale": round(id_scale, 8),
            "configured_factor": cfg,
            "episode_factor": round(factor, 8),
            "factor_range": [round(lo, 8), round(hi, 8)],
            "scale": obj.scale,
        }
    if not result:
        raise RuntimeError(
            f"{task.suite}/{task.name}: articraft_scale is set but no articulated object was found"
        )
    return result


def distractor_slots(task) -> list:
    """Table positions where a distractor could go for this task (cleared of
    the original objects' placement boxes, mutually separated). Used by
    add_distractors at eval time AND by scripts/gen_ood_yamls.py to decide
    whether a task has room for distractors at all."""
    return _distractor_slots_impl(task)


def _distractor_slots_impl(task, restrict_reach: bool = True):
    from bidex.scene_builder import object_footprint
    (xlo, xhi), (ylo, yhi), table_z = _table_world_for(task)
    if restrict_reach:
        (cxlo, cxhi), (cylo, cyhi) = _distractor_center_bounds(task)
    else:
        (cxlo, cxhi), (cylo, cyhi) = (xlo, xhi), (ylo, yhi)
    big = (yhi - ylo) > 1.0
    DXY = 0.03
    CLEARANCE = 0.30 if big else 0.12
    forbidden = []

    def static_center(o, r):
        """Best-effort world xy of the object's placement at build time."""
        if r is None:
            p = getattr(o, "pos", None)
            return (float(p[0]), float(p[1])) if p is not None else None
        cx, cy, _cz = r.center
        if r.target == "world":
            return (cx, cy)
        # Body-relative region (e.g. target: obj1_root): resolve the target's
        # own static centre and add the offset. Unresolvable -> skip (rare; the
        # distractor-free colour-swap task is the main user of such regions).
        import re as _re
        m = _re.match(r"obj(\d+)_", r.target)
        if m:
            tid = int(m.group(1))
            for t in list(task.objects) + list(getattr(task, "fixtures", [])):
                if t.instance_id == tid:
                    tr = task.regions.get(getattr(t, "region", None))
                    base = static_center(t, tr)
                    if base is not None:
                        return (cx + base[0], cy + base[1])
        return None

    def forbidden_box(o, r):
        """(x0, x1, y0, y1) the distractor CENTRE must stay out of, or None."""
        c = static_center(o, r)
        if c is None:
            return None
        cx, cy = c
        dx = dy = 0.0
        if r is not None:
            dx, dy = r.xy_range
        fp = object_footprint(o.name, getattr(o, "scale", 1.0))
        if fp is not None:
            # Exact physical extent of the object's collision body (measured
            # from its MJCF), on top of the placement-jitter box. A big
            # freejoint container (chest freezer, cutting board) whose region
            # is only a few cm wide would otherwise get a distractor slot that
            # overlaps its real body (the box-overlaps-new-object bug).
            fx, fy = fp[0], fp[1]
            if r is not None and (r.yaw_range[1] - r.yaw_range[0]) > 1.6:
                fx = fy = (fx * fx + fy * fy) ** 0.5  # wide yaw: worst-case rotation
            # The forbidden box forbids the distractor CENTRE. A distractor's
            # own body extends ~0.03-0.05 m past its centre, so shrink each
            # object's physical extent by that allowance (clamped >= 0) and add
            # a modest safety gap. Exact footprints replace the legacy blanket
            # clearance, which either over- or under-protected.
            fx = max(0.0, fx - 0.03)
            fy = max(0.0, fy - 0.03)
            clear = 0.05
            return (cx - dx - fx - clear, cx + dx + fx + clear,
                    cy - dy - fy - clear, cy + dy + fy + clear)
        # No parseable geometry -> legacy region-box + heuristic clearance.
        # Many large welded fixtures (slide_cabinet / short_cabinet / microwave
        # / tiered_basket...) declare a tiny placement region (xy_range ~0.02)
        # while their physical collision body spans 0.3-0.5 m; for those clamp
        # the half-extent up and bump the clearance.
        is_big_fixture = (not getattr(o, "freejoint", True)) or getattr(o, "is_basket", False)
        if is_big_fixture:
            dx = max(dx, 0.15); dy = max(dy, 0.15)
            clear = CLEARANCE + (0.10 if big else 0.06)
        else:
            clear = CLEARANCE
        return (cx - dx - clear, cx + dx + clear,
                cy - dy - clear, cy + dy + clear)

    for o in list(task.objects) + list(getattr(task, "fixtures", [])):
        r = task.regions.get(getattr(o, "region", None)) if hasattr(task, "regions") else None
        # Objects with an explicit pos and no region (welded mug_tree / plates /
        # shelf ...) still occupy physical space -- forbid it too (the old code
        # skipped them, so a distractor could spawn inside the mug tree).
        box = forbidden_box(o, r)
        if box is not None:
            forbidden.append(box)
    def overlaps_forbidden(x, y):
        return any(fx0 <= x <= fx1 and fy0 <= y <= fy1 for fx0, fx1, fy0, fy1 in forbidden)
    if big:
        xs = [0.22, 0.40, 0.58]
        ys = [0.45, -0.45, 0.70, -0.70]
    else:
        xs = [0.30, 0.15, 0.0, -0.10, -0.25]
        ys = [0.28, -0.28, 0.0]
    return [(x, y) for x in xs for y in ys
            if cxlo + DXY <= x <= cxhi - DXY and cylo + DXY <= y <= cyhi - DXY
            and xlo + DXY <= x <= xhi - DXY and ylo + DXY <= y <= yhi - DXY
            and not overlaps_forbidden(x, y)]


def add_distractors(task, rng: np.random.Generator, n_min=3, n_max=6,
                    eval_seed: Optional[int] = None):
    """Add fixed-count, seed-selected, geometry-safe, spread distractors."""
    from bidex.scene_builder import object_footprint
    from bidex.tasks.task_spec import ObjectInit, Region
    (xlo, xhi), (ylo, yhi), table_z = _table_world_for(task)
    (cxlo, cxhi), (cylo, cyhi) = _distractor_center_bounds(task)
    big = (yhi - ylo) > 1.0
    # The compact Franka table needs a smaller footprint pad; exact MJCF
    # extents still retain 3 cm total clearance plus the pair gap.
    dxy, edge_gap, pair_gap = (0.025 if big else 0.015), 0.01, 0.01
    scale_guard = 1.0

    def static_center(obj, region, seen=None):
        if region is None:
            p = getattr(obj, "pos", None)
            return (float(p[0]), float(p[1])) if p is not None else None
        cx, cy, _ = region.center
        if region.target == "world":
            return float(cx), float(cy)
        import re
        match = re.match(r"obj(\d+)_", region.target)
        if not match:
            return None
        target_id = int(match.group(1))
        seen = set() if seen is None else set(seen)
        if target_id in seen:
            return None
        seen.add(target_id)
        for target in list(task.objects) + list(getattr(task, "fixtures", [])):
            if target.instance_id == target_id:
                base = static_center(
                    target, task.regions.get(getattr(target, "region", None)), seen
                )
                return None if base is None else (base[0] + cx, base[1] + cy)
        return None

    def yaw_envelope(hx, hy, region):
        if region is None or region.yaw_range[1] - region.yaw_range[0] >= np.pi:
            radius = float(np.hypot(hx, hy))
            return radius, radius
        angles = np.linspace(*map(float, region.yaw_range), 17)
        return (
            float(max(abs(np.cos(a)) * hx + abs(np.sin(a)) * hy for a in angles)),
            float(max(abs(np.sin(a)) * hx + abs(np.cos(a)) * hy for a in angles)),
        )

    obstacles = []
    for obj in list(task.objects) + list(getattr(task, "fixtures", [])):
        # Freejoint objects are placed first for each episode; distractors are
        # then fitted around their actual sampled poses in resample_distractors.
        if getattr(obj, "freejoint", True):
            continue
        region = task.regions.get(getattr(obj, "region", None))
        center = static_center(obj, region)
        if center is None:
            continue
        fp = object_footprint(
            obj.name, float(getattr(obj, "scale", 1.0)) * scale_guard
        )
        ex, ey = (0.12, 0.12) if fp is None else yaw_envelope(fp[0], fp[1], region)
        rx, ry = (0.0, 0.0) if region is None else region.xy_range
        protect = _distractor_protection_margin(task, obj)
        obstacles.append((center[0], center[1],
                          float(rx) + ex + protect, float(ry) + ey + protect))

    desired = (int(n_min) if int(n_min) == int(n_max)
               else int(rng.integers(int(n_min), int(n_max) + 1)))
    legacy = _legacy_distractor_entry(task, eval_seed)
    preserved_names = list(legacy.get("preserve", []))
    task._ood_omitted_legacy_distractors = list(
        legacy.get("omitted", [])
    )
    if len(preserved_names) > desired:
        raise RuntimeError(
            f"{task.suite}/{task.name}: fixed distractor count {desired} is "
            f"smaller than legacy seeded set {len(preserved_names)}"
        )
    names = _collect_similar_objects(
        task.suite, task.name, {obj.name for obj in task.objects}, rng, 100000
    )
    metrics_by_name = {}
    for order, name in enumerate(names):
        fp = object_footprint(name, scale_guard)
        if fp is None:
            continue
        extent_x, extent_y = float(fp[0]) + dxy, float(fp[1]) + dxy
        volume = max(8.0 * fp[0] * fp[1] * max(fp[3] - fp[2], 0.005), 1e-9)
        metrics_by_name[name] = (name, extent_x, extent_y, volume, order)
    preserved = []
    for order, name in enumerate(preserved_names):
        fp = object_footprint(name, scale_guard)
        if fp is None:
            raise RuntimeError(
                f"{task.suite}/{task.name}: legacy distractor has no footprint: {name}"
            )
        extent_x, extent_y = float(fp[0]) + dxy, float(fp[1]) + dxy
        volume = max(8.0 * fp[0] * fp[1] * max(fp[3] - fp[2], 0.005), 1e-9)
        preserved.append((name, extent_x, extent_y, volume, order, True))
    # Pack the required legacy set largest-first; the returned name set is
    # unchanged, while avoiding a small object greedily consuming the only slot
    # that can accommodate a tray, cutting board, pan, or long utensil.
    preserved.sort(key=lambda item: (-max(item[1], item[2]), item[4]))
    excluded = set(preserved_names)
    extras = [m for name, m in metrics_by_name.items() if name not in excluded]
    if not preserved and not extras:
        raise RuntimeError(f"{task.suite}/{task.name}: no usable distractor assets")

    anchor_source = preserved if preserved else extras
    anchor = float(np.median(np.log([m[3] for m in anchor_source])))
    extras = sorted(extras, key=lambda m: (abs(np.log(m[3])-anchor), m[4]))
    pool = preserved + [(*m, False) for m in extras]

    step = 0.04 if big else 0.035
    candidates = [
        (round(float(x), 4), round(float(y), 4))
        for x in np.arange(cxlo + edge_gap, cxhi - edge_gap + 1e-9, step)
        for y in np.arange(cylo + edge_gap, cyhi - edge_gap + 1e-9, step)
    ]
    rng.shuffle(candidates)
    tcx, tcy = 0.5 * (cxlo + cxhi), 0.5 * (cylo + cyhi)

    def placement_options(item, placed, available):
        name, extent_x, extent_y, _volume, _order, _required = item
        valid = []
        for x, y in available:
            if not (xlo + extent_x + edge_gap <= x <= xhi - extent_x - edge_gap
                    and ylo + extent_y + edge_gap <= y <= yhi - extent_y - edge_gap):
                continue
            if any(not (abs(x-ox) >= extent_x+ex+pair_gap
                        or abs(y-oy) >= extent_y+ey+pair_gap)
                   for ox, oy, ex, ey in obstacles):
                continue
            if any(not (abs(x-px) >= extent_x+pex+pair_gap
                        or abs(y-py) >= extent_y+pey+pair_gap)
                   for px, py, pex, pey, _ in placed):
                continue
            peers = ([(o[0], o[1]) for o in obstacles]
                     + [(q[0], q[1]) for q in placed])
            sep = min([np.hypot(x-qx, y-qy) for qx, qy in peers] or [1.0])
            mx = float(np.mean([q[0] for q in placed] + [x]))
            my = float(np.mean([q[1] for q in placed] + [y]))
            opposite = any((py-tcy)*(y-tcy) < 0 for _, py, _, _, _ in placed)
            score = sep + 0.08*opposite - 0.35*(abs(mx-tcx)+abs(my-tcy))
            valid.append((score, x, y))
        return sorted(valid, reverse=True)

    required_items = [item for item in pool if item[-1]]
    def pack_required(index, placed, available, branch_limit):
        if index == len(required_items):
            return placed, available
        item = required_items[index]
        name, extent_x, extent_y, _, _, _ = item
        for _score, x, y in placement_options(item, placed, available)[:branch_limit]:
            added = (x, y, extent_x, extent_y, name)
            remaining = [xy for xy in available if xy != (x, y)]
            result = pack_required(
                index + 1, placed + [added], remaining, branch_limit
            )
            if result is not None:
                return result
        return None

    packed = None
    for branch_limit in (24, 64, 160):
        packed = pack_required(0, [], candidates, branch_limit)
        if packed is not None:
            break
    if packed is None:
        raise RuntimeError(
            f"{task.suite}/{task.name}: cannot safely retain complete legacy "
            f"distractor set {preserved_names}"
        )
    picked, candidates = packed
    for item in (entry for entry in pool if not entry[-1]):
        if len(picked) == desired:
            break
        name, extent_x, extent_y, _volume, _order, _required = item
        valid = placement_options(item, picked, candidates)
        if not valid:
            continue
        _score, x, y = valid[0]
        picked.append((x, y, extent_x, extent_y, name))
        candidates.remove((x, y))
    if len(picked) != desired:
        raise RuntimeError(
            f"{task.suite}/{task.name}: fixed distractor count {desired} requested, "
            f"only {len(picked)} safe placements fit"
        )

    ids = [o.instance_id for o in list(task.objects) + list(getattr(task, "fixtures", []))]
    next_id = max(ids) + 1 if ids else 1
    added = []
    for i, (x, y, _extent_x, _extent_y, name) in enumerate(picked):
        region_name = f"_ood_distractor_{i}"
        task.regions[region_name] = Region(
            # Compile-time/reset staging pose stays above all task objects.
            # resample_distractors moves it onto the table before any eval step.
            name=region_name, center=(x, y, table_z + 0.50),
            xy_range=(dxy, dxy), yaw_range=(0.0, 0.0),
            rest_on_surface=True,
        )
        # Explicit compile-time pose prevents every region-based freejoint
        # object from starting at the same placeholder (0, 0, 0.85), which can
        # overflow MuJoCo's contact stack before reset gets a chance to place it.
        obj = ObjectInit(
            name=name, region=region_name, freejoint=True, scale=1.0,
            pos=(x, y, table_z + 0.50), quat=(1.0, 0.0, 0.0, 0.0),
        )
        obj.instance_id = next_id + i
        task.objects.append(obj)
        added.append(name)
    positions = ["({:.2f},{:.2f})".format(x, y)
                 for x, y, _, _, _ in picked]
    print(
        f"[OOD] {task.suite}/{task.name}: placed fixed {len(added)} "
        f"geometry-safe distractors at {positions}", flush=True,
    )
    return added

def check_freejoint_overlap(env, obj_names: list[str], min_dist: float = 0.08,
                            task=None) -> bool:
    """Check if any pair of freejoint objects are closer than min_dist (overlap).

    When ``task`` is given, each pair's threshold is raised to the sum of the
    two objects' physical xy radii (from
    :func:`scene_builder.object_footprint`, circular bound) + 1 cm -- a bare
    8 cm centre distance lets a small distractor embed in a large body (chest
    freezer / cutting board) whose centre is far from its walls."""
    import mujoco
    m = env.model; d = env.data
    radii = {}
    if task is not None:
        from bidex.scene_builder import object_footprint
        for o in task.all_object_inits():
            fp = object_footprint(o.name, getattr(o, "scale", 1.0))
            if fp is not None:
                radii[o.name] = (fp[0] ** 2 + fp[1] ** 2) ** 0.5
    positions = {}
    for name in obj_names:
        # find the body by name pattern — objects are prefixed obj{i}_
        for bid in range(m.nbody):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bname and name in bname and "freejoint" not in bname:
                positions[name] = d.xpos[bid].copy()
                break
    for a in positions:
        for b in positions:
            if a < b:
                thresh = max(min_dist,
                             radii.get(a, 0.0) + radii.get(b, 0.0) + 0.01)
                if np.linalg.norm(positions[a] - positions[b]) < thresh:
                    return True
    return False


def resample_distractors(env, task, rng, max_tries=10):
    """Apply one deterministic episode placement for targets and distractors.

    Distractor slots are already filtered analytically by ``distractor_slots``.
    Do not let a platform-dependent MuJoCo overlap decision consume a different
    number of RNG draws on different computers; the live overlap check is only
    diagnostic after the single, canonical placement has been applied.
    """
    from bidex.tasks.domain import apply_task_initial_state
    import mujoco

    def object_root_body_id(instance_id: int) -> int:
        """Find the top-level model body for an objN_ namespace."""
        prefix = f"obj{instance_id}_"
        exact = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, f"obj{instance_id}_root"
        )
        if exact >= 0:
            return exact
        for body_id in range(1, env.model.nbody):
            name = mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if not name or not name.startswith(prefix):
                continue
            parent_id = int(env.model.body_parentid[body_id])
            parent_name = mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_BODY, parent_id
            )
            if not parent_name or not parent_name.startswith(prefix):
                return body_id
        return -1

    distractors = [o for o in task.objects
                   if o.region and o.region.startswith("_ood_distractor")]
    # Resample every task object, not only distractor-bearing tasks. This makes
    # OOD object placement depend solely on the named episode RNG stream rather
    # than on any unrelated random draws made inside env.reset().
    apply_task_initial_state(env.model, env.data, task, rng=rng)
    if distractors:
        from bidex.scene_builder import object_footprint
        (xlo, xhi), (ylo, yhi), _table_z = _table_world_for(task)
        (cxlo, cxhi), (cylo, cyhi) = _distractor_center_bounds(task)
        edge_gap, pair_gap = 0.01, 0.01
        distractor_ids = {o.instance_id for o in distractors}
        occupied = []
        focus_points = []
        task_cup_points = []
        object_poses = {}
        focus_names = set(getattr(task, "obj_of_interest", []) or [])
        for obj in list(task.objects) + list(getattr(task, "fixtures", [])):
            if obj.instance_id in distractor_ids:
                continue
            fp = object_footprint(obj.name, getattr(obj, "scale", 1.0))
            half_x, half_y = ((0.12, 0.12) if fp is None
                              else (float(fp[0]), float(fp[1])))
            protect = _distractor_protection_margin(task, obj)
            half_x += protect
            half_y += protect
            jid = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_JOINT,
                f"obj{obj.instance_id}_freejoint",
            )
            if jid >= 0:
                adr = env.model.jnt_qposadr[jid]
                quat = env.data.qpos[adr + 3:adr + 7]
                w, qx, qy, qz = map(float, quat)
                yaw = np.arctan2(2*(w*qz + qx*qy),
                                 1 - 2*(qy*qy + qz*qz))
                c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
                pose = (
                    float(env.data.qpos[adr]), float(env.data.qpos[adr + 1]),
                    c*half_x + s*half_y, s*half_x + c*half_y,
                )
                occupied.append(pose)
                object_poses[obj.name] = (
                    pose[0], pose[1], float(yaw), float(getattr(obj, "scale", 1.0))
                )
                if obj.name in focus_names:
                    focus_points.append(pose[:2])
                if (
                    task.name == "drop_two_spoons_from_tray_into_two_cups"
                    and ("mug" in obj.name.lower() or "cup" in obj.name.lower())
                ):
                    task_cup_points.append(pose[:2])
                continue
            bid = object_root_body_id(obj.instance_id)
            if bid >= 0:
                w, qx, qy, qz = map(float, env.data.xquat[bid])
                yaw = np.arctan2(2*(w*qz + qx*qy),
                                 1 - 2*(qy*qy + qz*qz))
                c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
                pose = (
                    float(env.data.xpos[bid, 0]), float(env.data.xpos[bid, 1]),
                    c*half_x + s*half_y, s*half_x + c*half_y,
                )
                occupied.append(pose)
                object_poses[obj.name] = (
                    pose[0], pose[1], float(yaw), float(getattr(obj, "scale", 1.0))
                )
                if obj.name in focus_names:
                    focus_points.append(pose[:2])
                if (
                    task.name == "drop_two_spoons_from_tray_into_two_cups"
                    and ("mug" in obj.name.lower() or "cup" in obj.name.lower())
                ):
                    task_cup_points.append(pose[:2])

        step = 0.035 if (yhi - ylo) <= 1.0 else 0.04
        # Original objects use their sampled-yaw AABB; distractors use
        # yaw=0 and their tighter physical AABB.
        occupied_xy = occupied
        candidates = [
            (round(float(x), 4), round(float(y), 4))
            for x in np.arange(cxlo + edge_gap, cxhi - edge_gap + 1e-9, step)
            for y in np.arange(cylo + edge_gap, cyhi - edge_gap + 1e-9, step)
        ]
        distractor_cfg = (
            (getattr(task, "ood", None) or {}).get("distractors") or {}
        )
        placement_mode = str(
            distractor_cfg.get("placement_mode", "spread")
        )
        if placement_mode not in {"spread", "random_table"}:
            raise ValueError(
                f"{task.suite}/{task.name}: unsupported distractor "
                f"placement_mode {placement_mode!r}"
            )
        if placement_mode == "random_table":
            # Random sequential packing can put rolling objects on less
            # symmetric poses than the spread scorer.  Keep enough free space
            # for their short post-reset settle without allowing contacts.
            pair_gap = max(pair_gap, 0.05)
            edge_gap = max(edge_gap, 0.06)

        # Optional task-level swept-volume keepout.  Coordinates are expressed
        # in the anchor object's unscaled local frame, then transformed using
        # its sampled episode pose and scale.  This keeps clutter out of a
        # moving door's path without hard-coding world coordinates.
        sweep_points = []
        sweep_clearance = 0.0
        sweep_cfg = distractor_cfg.get("articulation_sweep_keepout")
        if sweep_cfg:
            if not isinstance(sweep_cfg, dict):
                raise ValueError(
                    f"{task.suite}/{task.name}: articulation_sweep_keepout "
                    "must be a mapping"
                )
            anchor_name = str(sweep_cfg.get("anchor", ""))
            if anchor_name not in object_poses:
                raise ValueError(
                    f"{task.suite}/{task.name}: sweep anchor "
                    f"{anchor_name!r} has no runtime pose"
                )
            hinge = np.asarray(sweep_cfg.get("hinge", []), dtype=np.float64)
            tip = np.asarray(sweep_cfg.get("tip", []), dtype=np.float64)
            angle_range = np.asarray(
                sweep_cfg.get("angle_range", []), dtype=np.float64
            )
            if hinge.shape != (2,) or tip.shape != (2,) or angle_range.shape != (2,):
                raise ValueError(
                    f"{task.suite}/{task.name}: sweep hinge, tip, and "
                    "angle_range must each contain two numbers"
                )
            rotation_sign = float(sweep_cfg.get("rotation_sign", 1.0))
            sweep_clearance = float(sweep_cfg.get("clearance", 0.05))
            if rotation_sign == 0.0 or sweep_clearance < 0.0:
                raise ValueError(
                    f"{task.suite}/{task.name}: invalid sweep rotation_sign "
                    f"or clearance"
                )
            anchor_x, anchor_y, anchor_yaw, anchor_scale = object_poses[anchor_name]
            ca, sa = np.cos(anchor_yaw), np.sin(anchor_yaw)
            anchor_rotation = np.asarray(((ca, -sa), (sa, ca)))
            scaled_hinge = hinge * anchor_scale
            scaled_segment = (tip - hinge) * anchor_scale
            for angle in np.linspace(angle_range[0], angle_range[1], 25):
                theta = rotation_sign * float(angle)
                ct, st = np.cos(theta), np.sin(theta)
                door_rotation = np.asarray(((ct, -st), (st, ct)))
                for fraction in np.linspace(0.0, 1.0, 17):
                    local_point = (
                        scaled_hinge
                        + door_rotation @ (scaled_segment * float(fraction))
                    )
                    world_point = (
                        np.asarray((anchor_x, anchor_y))
                        + anchor_rotation @ local_point
                    )
                    sweep_points.append(tuple(map(float, world_point)))

        # An optional anchor-relative half-plane quota makes one side visibly
        # less cluttered while keeping the actual placements episode-random.
        limited_side_fn = None
        limited_side_max = 0
        side_cfg = distractor_cfg.get("half_plane_limit")
        if side_cfg:
            if not isinstance(side_cfg, dict):
                raise ValueError(
                    f"{task.suite}/{task.name}: half_plane_limit must be a mapping"
                )
            side_anchor = str(side_cfg.get("anchor", ""))
            if side_anchor not in object_poses:
                raise ValueError(
                    f"{task.suite}/{task.name}: side-limit anchor "
                    f"{side_anchor!r} has no runtime pose"
                )
            local_normal = np.asarray(
                side_cfg.get("local_normal", []), dtype=np.float64
            )
            if local_normal.shape != (2,) or np.linalg.norm(local_normal) == 0.0:
                raise ValueError(
                    f"{task.suite}/{task.name}: half-plane local_normal "
                    "must be a non-zero pair"
                )
            local_normal /= np.linalg.norm(local_normal)
            side_threshold = float(side_cfg.get("threshold", 0.0))
            limited_side_max = int(side_cfg.get("max_count", 0))
            if limited_side_max < 0:
                raise ValueError(
                    f"{task.suite}/{task.name}: half-plane max_count must be >= 0"
                )
            side_x, side_y, side_yaw, _side_scale = object_poses[side_anchor]
            cs, ss = np.cos(side_yaw), np.sin(side_yaw)
            world_to_anchor = np.asarray(((cs, ss), (-ss, cs)))

            def limited_side_fn(x, y):
                local_xy = world_to_anchor @ (
                    np.asarray((x, y)) - np.asarray((side_x, side_y))
                )
                return float(local_normal @ local_xy) < side_threshold
        focus_min_radius = float(
            distractor_cfg.get("focus_min_radius", 0.0)
        )
        default_focus_radius = (
            0.34
            if "astribot" in str(
                getattr(task, "robot_type", "") or ""
            ).lower()
            else None
        )
        focus_radius = distractor_cfg.get(
            "focus_radius", default_focus_radius
        )
        local_focus = bool(focus_points) and focus_radius is not None
        focus_target_radius = None
        if local_focus:
            # Keep clutter in a configurable annulus around the live sampled
            # task objects. A non-zero inner radius preserves manipulation
            # clearance, while the outer radius keeps clutter operationally
            # relevant rather than turning it into background decoration.
            focus_radius = float(focus_radius)
            if not 0.0 <= focus_min_radius <= focus_radius:
                raise ValueError(
                    f"{task.suite}/{task.name}: distractor focus radii must "
                    f"satisfy 0 <= focus_min_radius <= focus_radius, got "
                    f"{focus_min_radius} and {focus_radius}"
                )
            configured_target = distractor_cfg.get("focus_target_radius")
            focus_target_radius = (
                min(focus_radius, max(focus_min_radius, 0.24))
                if configured_target is None else float(configured_target)
            )
            if not focus_min_radius <= focus_target_radius <= focus_radius:
                raise ValueError(
                    f"{task.suite}/{task.name}: focus_target_radius "
                    f"{focus_target_radius} must be inside "
                    f"[{focus_min_radius}, {focus_radius}]"
                )
            candidates = [
                (x, y) for x, y in candidates
                if focus_min_radius
                <= min(np.hypot(x-fx, y-fy) for fx, fy in focus_points)
                <= focus_radius
            ]
        rng.shuffle(candidates)
        # Preserve the shuffled order as a deterministic random field.  Tasks
        # using random_table therefore sample the whole reachable tabletop
        # instead of being optimized into a conspicuous ring or grid.
        random_rank = {xy: rank for rank, xy in enumerate(candidates)}
        if local_focus:
            tcx = float(np.mean([p[0] for p in focus_points]))
            tcy = float(np.mean([p[1] for p in focus_points]))
        else:
            tcx, tcy = 0.5 * (cxlo + cxhi), 0.5 * (cylo + cyhi)
        items = []
        for obj in distractors:
            fp = object_footprint(obj.name, getattr(obj, "scale", 1.0))
            extent_x, extent_y = ((0.12, 0.12) if fp is None
                                  else (float(fp[0]), float(fp[1])))
            items.append((extent_x, extent_y, obj))
        items.sort(key=lambda item: -max(item[0], item[1]))
        spoon_task_cup_side = (
            task.name == "drop_two_spoons_from_tray_into_two_cups"
            and bool(task_cup_points)
            and any("glass_cup" in item[2].name.lower() for item in items)
        )
        table_mid_x = 0.5 * (xlo + xhi)
        task_cup_x = (
            float(np.mean([p[0] for p in task_cup_points]))
            if task_cup_points else table_mid_x
        )

        def options(item, placed, available):
            base_x, base_y, obj = item
            orientations = [(base_x, base_y, 0.0)]
            if not np.isclose(base_x, base_y):
                orientations.append((base_y, base_x, float(np.pi / 2)))
            valid = []
            for extent_x, extent_y, yaw in orientations:
                for x, y in available:
                    if not (
                        xlo+extent_x+edge_gap <= x <= xhi-extent_x-edge_gap
                        and ylo+extent_y+edge_gap <= y <= yhi-extent_y-edge_gap
                    ):
                        continue
                    if any(
                        abs(x-px) < extent_x+sweep_clearance
                        and abs(y-py) < extent_y+sweep_clearance
                        for px, py in sweep_points
                    ):
                        continue
                    if (
                        limited_side_fn is not None
                        and limited_side_fn(x, y)
                        and sum(
                            limited_side_fn(other[0], other[1])
                            for other in placed
                        ) >= limited_side_max
                    ):
                        continue
                    if any(not (
                        abs(x-other[0]) >= extent_x+other[2]+pair_gap
                        or abs(y-other[1]) >= extent_y+other[3]+pair_gap
                    ) for other in occupied_xy + placed):
                        continue
                    if (
                        spoon_task_cup_side
                        and "glass_cup" in obj.name.lower()
                        and (
                            (x-table_mid_x)*(task_cup_x-table_mid_x) < 0
                            or min(
                                np.hypot(x-cx, y-cy)
                                for cx, cy in task_cup_points
                            ) > 0.34
                        )
                    ):
                        continue
                    peers = occupied_xy + placed
                    separation = min(
                        [np.hypot(x-q[0], y-q[1]) for q in peers] or [1.0]
                    )
                    xs = [q[0] for q in placed] + [x]
                    ys = [q[1] for q in placed] + [y]
                    balance = (abs(float(np.mean(xs))-tcx)
                               + abs(float(np.mean(ys))-tcy))
                    opposite = any(
                        (q[1]-tcy)*(y-tcy) < 0 for q in placed
                    )
                    if placement_mode == "random_table":
                        score = -float(random_rank[(x, y)])
                    elif local_focus:
                        focus_distance = min(
                            np.hypot(x-fx, y-fy) for fx, fy in focus_points
                        )
                        # Prefer the configured part of the annulus while also
                        # maximizing clearance from every occupied footprint.
                        score = (
                            separation
                            - 1.8*abs(focus_distance-focus_target_radius)
                            - 0.12*balance
                        )
                    else:
                        score = separation + 0.08*opposite - 0.35*balance
                    valid.append(
                        (score, -yaw, x, y, extent_x, extent_y, yaw)
                    )
            return sorted(valid, reverse=True)

        def pack(index, placed, available, branch_limit):
            if index == len(items):
                return placed
            for (_score, _yaw_order, x, y, extent_x, extent_y, yaw
                 ) in options(items[index], placed, available)[:branch_limit]:
                result = pack(
                    index + 1,
                    placed + [(x, y, extent_x, extent_y, yaw,
                               items[index][2].name)],
                    [xy for xy in available if xy != (x, y)],
                    branch_limit,
                )
                if result is not None:
                    return result
            return None

        placed = None
        for branch_limit in (24, 64):
            placed = pack(0, [], candidates, branch_limit)
            if placed is not None:
                break
        if placed is None:
            if max_tries > 1:
                # Pure geometry + PCG64 only: retrying consumes the same named
                # episode stream on every machine and jointly resamples the
                # task objects until the fixed distractor set fits.
                return resample_distractors(
                    env, task, rng, max_tries=max_tries - 1
                )
            raise RuntimeError(
                f"{task.suite}/{task.name}: no safe episode packing for "
                f"{len(distractors)} fixed distractors after 10 deterministic tries"
            )
        for (_extent_x, _extent_y, obj), (
            x, y, _pex, _pey, yaw, _name
        ) in zip(items, placed):
            jid = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_JOINT,
                f"obj{obj.instance_id}_freejoint",
            )
            if jid < 0:
                raise RuntimeError(f"missing distractor freejoint: obj{obj.instance_id}")
            adr = env.model.jnt_qposadr[jid]
            env.data.qpos[adr] = x
            env.data.qpos[adr + 1] = y
            fp = object_footprint(obj.name, getattr(obj, "scale", 1.0))
            z_low = -0.03 if fp is None else float(fp[2])
            env.data.qpos[adr + 2] = _table_z - z_low
            env.data.qpos[adr + 3:adr + 7] = (
                np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
            )
            vadr = env.model.jnt_dofadr[jid]
            env.data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    # The canonical AABB packing above is deterministic and already proves
    # table containment plus pairwise separation. Avoid the legacy name-based
    # circular diagnostic, which aliases duplicate object names and rejects
    # valid long utensils by their circumscribed circles.
    return True


def grab_wrong_object(env, target_names: list[str], distractor_names: list[str],
                       tool_body_names=("right_tool_link", "left_tool_link"), grab_dist=0.05) -> bool:
    """Return True if ANY tool is closer to a distractor than to ALL targets
    AND within grab_dist of a distractor (i.e. grabbed the wrong object)."""
    import mujoco
    m = env.model; d = env.data
    tool_pos = []
    for tb in tool_body_names:
        full = [n for n in ["astribot_arm_right_tool_link", "astribot_arm_left_tool_link",
                            "right_fr3v2_link8", "left_fr3v2_link8"] if tb in n]
        for fn in full:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fn)
            if bid >= 0:
                tool_pos.append(d.xpos[bid].copy())
    target_pos = []
    for tn in target_names:
        for bid in range(m.nbody):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bn and tn in bn:
                target_pos.append(d.xpos[bid].copy()); break
    distractor_pos = []
    for dn in distractor_names:
        for bid in range(m.nbody):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bn and dn in bn:
                distractor_pos.append(d.xpos[bid].copy()); break
    for tp in tool_pos:
        d_to_distract = [np.linalg.norm(tp - dp) for dp in distractor_pos] if distractor_pos else [1e9]
        d_to_target = [np.linalg.norm(tp - tp2) for tp2 in target_pos] if target_pos else [1e9]
        if min(d_to_distract) < grab_dist and min(d_to_distract) < min(d_to_target):
            return True
    return False


def swap_articulated(task, rng: np.random.Generator, report_missing: bool = True):
    """For articraft tasks: find an alternate articraft object with the
    same local joint name(s). Swaps task.objects[0].name to the alternate.

    Returns (task, alternate_name or None)."""
    # find the articulated object + its joint
    if not task.articulations:
        return task, None
    art = task.articulations[0]
    obj_name = art.get("object") if isinstance(art, dict) else getattr(art, "object", None)
    joint_name = art.get("joint") if isinstance(art, dict) else getattr(art, "joint", None)
    if not joint_name:
        return task, None

    # enumerate articraft_objects with the same local joint name
    candidates = []
    for d in sorted(ARTICRAFT_DIR.iterdir()):
        if not d.is_dir():
            continue
        xmls = sorted(d.glob("*.xml"))
        if not xmls:
            continue
        try:
            tree = ET.parse(xmls[0])
            for j in tree.iter("joint"):
                if j.get("name") == joint_name and d.name != obj_name:
                    candidates.append(d.name)
                    break
        except Exception:
            continue
    if not candidates:
        if report_missing:
            print(f"[OOD] articulated swap: no alternate with joint '{joint_name}' for '{obj_name}' in {task.suite}/{task.name}")
        return task, None
    alt = str(rng.choice(candidates))
    for obj in task.objects:
        if obj.name == obj_name:
            obj.name = alt
            break
    # update articulations + goal refs (they reference the old name)
    for a in task.articulations:
        if isinstance(a, dict) and a.get("object") == obj_name:
            a["object"] = alt
    for g in task.goal:
        if isinstance(g.args, list) and obj_name in g.args:
            g.args[g.args.index(obj_name)] = alt
    if obj_name in task.obj_of_interest:
        task.obj_of_interest[task.obj_of_interest.index(obj_name)] = alt
    return task, alt


def object_offset(env, task, rng: np.random.Generator, offset_m=0.05):
    """Post-reset: offset freejoint objects' xy by ±offset_m."""
    import mujoco
    m = env.model; d = env.data
    applied = []
    for obj in task.objects:
        jname = f"obj{obj.instance_id}_freejoint"
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        adr = m.jnt_qposadr[jid]
        d.qpos[adr] += rng.uniform(-offset_m, offset_m)
        d.qpos[adr + 1] += rng.uniform(-offset_m, offset_m)
        applied.append(obj.name)
    mujoco.mj_forward(m, d)
    return applied
