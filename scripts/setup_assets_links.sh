#!/usr/bin/env bash
# Overwriteable asset linker for BiDex-100.
#
# After downloading the `assets/` component, run this script to validate and
# extract each pack in place, then symlink it into the repository. Idempotent:
# safe to re-run (skips unchanged archives, refreshes links, and leaves
# non-symlink repository directories untouched).
#
#   bash scripts/setup_assets_links.sh downloads/BiDex-100
#   BIDEX_ASSETS=downloads/BiDex-100/assets bash scripts/setup_assets_links.sh
#
# Each pack directory initially contains its license, README, and a same-named
# `.tar.gz`. This script extracts the archive into that directory.
#
# This script also RE-CREATES the within-pack "one model, many names" alias symlinks
# (assembly_objects/cube_a/cube_a.xml -> ../assembly_cube/assembly_cube.xml,
# dexjoco_objects/threading_rod_left -> ../threading_rod/threading_rod.xml, ...) so the
# env is reproducible regardless of OSS upload mode (--no-symlink drops them; a
# dereferencing upload materializes them as copies; ln -sf re-establishes the symlinks).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_INPUT="${1:-${BIDEX_ASSETS:-}}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [download-root-or-assets-directory]" >&2
  exit 2
fi

if [[ -z "$ASSETS_INPUT" ]]; then
  echo "ERROR: pass the download root/assets directory or set BIDEX_ASSETS." >&2
  echo "Example: $0 downloads/BiDex-100" >&2
  exit 2
fi

if [[ ! -d "$ASSETS_INPUT" ]]; then
  echo "ERROR: asset path not found: $ASSETS_INPUT" >&2
  exit 1
fi

# Accept either the Hugging Face checkout root or its assets/ subdirectory.
# SHA256SUMS is optional, so directory detection must not depend on it.
if [[ -d "$ASSETS_INPUT/assets" ]]; then
  ASSETS_INPUT="$ASSETS_INPUT/assets"
fi
ASSETS="$(cd "$ASSETS_INPUT" && pwd -P)"

CHECKSUM_MANIFEST="$ASSETS/SHA256SUMS"
if [[ -f "$CHECKSUM_MANIFEST" ]]; then
  echo "Checksum manifest found; downloaded archives will be verified."
else
  echo "Checksum manifest not present; using local archive digests for idempotency."
fi

for command_name in sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $command_name" >&2
    exit 1
  fi
done

pack_ready() {
  local pack="$1" pack_dir="$ASSETS/$1"
  case "$pack" in
    articraft_objects)
      [[ -n "$(find "$pack_dir" -mindepth 2 -maxdepth 2 -type f -name '*.xml' -print -quit)" ]]
      ;;
    assembly_objects)
      [[ -f "$pack_dir/assembly_cube/assembly_cube.xml" ]]
      ;;
    astribot_descriptions)
      [[ -d "$pack_dir/urdf/astribot_s1_urdf/meshes" ]]
      ;;
    libero)
      [[ -d "$pack_dir/objects" ]]
      ;;
    robocasa_assets)
      [[ -d "$pack_dir/objects" && -d "$pack_dir/fixtures" ]]
      ;;
  esac
}

extract_pack() {
  local pack="$1"
  local rel="$pack/$pack.tar.gz"
  local pack_dir="$ASSETS/$pack"
  local archive="$ASSETS/$rel"
  local marker="$pack_dir/.bidex-extracted.sha256"
  local expected="" actual recorded

  if [[ ! -f "$archive" ]]; then
    echo "ERROR: missing asset archive: $archive" >&2
    return 1
  fi

  if [[ -f "$CHECKSUM_MANIFEST" ]]; then
    expected="$(awk -v rel="$rel" '$2 == rel {print $1}' "$CHECKSUM_MANIFEST")"
    if [[ ! "$expected" =~ ^[0-9a-fA-F]{64}$ ]]; then
      echo "ERROR: no valid checksum for $rel in $CHECKSUM_MANIFEST" >&2
      return 1
    fi
  fi

  actual="$(sha256sum "$archive" | awk '{print $1}')"
  if [[ -n "$expected" && "$actual" != "$expected" ]]; then
    echo "ERROR: checksum mismatch for $archive" >&2
    echo "  expected: $expected" >&2
    echo "  actual  : $actual" >&2
    return 1
  fi

  recorded=""
  [[ -f "$marker" ]] && recorded="$(tr -d '[:space:]' < "$marker")"
  if [[ "$recorded" == "$actual" ]] && pack_ready "$pack"; then
    echo "  READY $pack (archive already extracted)"
    return 0
  fi

  if [[ -n "$expected" ]]; then
    echo "  VERIFIED $rel"
  else
    echo "  DIGEST $rel  $actual"
  fi

  echo "  EXTRACT $archive -> $pack_dir"
  tar --no-same-owner -xzf "$archive" -C "$pack_dir"
  if ! pack_ready "$pack"; then
    echo "ERROR: $pack archive extracted but required contents are missing" >&2
    return 1
  fi
  printf '%s\n' "$actual" > "$marker"
}

echo "Asset root : $ASSETS"
echo "Extracting asset packs:"
for pack in \
  articraft_objects \
  assembly_objects \
  astribot_descriptions \
  libero \
  robocasa_assets; do
  extract_pack "$pack"
done

err=0
# link <repo_path> <asset_target>  — create/refresh a symlink, refuse to clobber a real dir.
link() {
  local dst="$1" src="$2"
  if [[ -e "$dst" && ! -L "$dst" ]]; then
    echo "  SKIP  $dst  (exists and is not a symlink; leaving it)" >&2; err=1; return
  fi
  if [[ -L "$dst" || -e "$dst" ]]; then rm -f "$dst"; fi
  ln -s "$src" "$dst"
  echo "  LINK  $dst  ->  $src"
}

# alias_link <pack_dir> <alias_name> <canonical_name>  — INSIDE an asset pack, make
# <pack_dir>/<alias>/<alias>.xml a RELATIVE symlink to ../<canonical>/<canonical>.xml,
# so the object resolver finds <alias> at the canonical model. These "one physical
# model, many role names" aliases are the dedup design (cube_a..d -> assembly_cube,
# threading_rod_left/mid/right -> threading_rod, ...). An OSS re-upload in --no-symlink
# mode DROPS them (empty dirs); a dereferencing upload turns them into standalone
# copies. `ln -sf` re-establishes them as symlinks in every case (force-overwrites a
# copy or stale link; canonical real files are untouched). Only acts when the
# canonical exists, so a missing pack degrades to per-alias SKIPs.
alias_link() {
  local pack="$1" alias="$2" canon="$3"
  if [[ ! -f "$pack/$canon/$canon.xml" ]]; then
    echo "  SKIP  alias $alias (canonical $canon missing in $pack)" >&2; return 0
  fi
  mkdir -p "$pack/$alias"
  ln -sf "../$canon/$canon.xml" "$pack/$alias/$alias.xml"
  echo "  ALIAS $pack/$alias/$alias.xml  ->  ../$canon/$canon.xml"
}

echo "Repo       : $REPO"
echo "Creating symlinks:"

# --- repo -> assets ---
link "$REPO/bidex/libero"                            "$ASSETS/libero"
link "$REPO/bidex/robocasa/assets"                   "$ASSETS/robocasa_assets"
link "$REPO/bidex/robots/assembly/astribot_s1/meshes"        "$ASSETS/astribot_descriptions/urdf/astribot_s1_urdf/meshes"
link "$REPO/bidex/robots/assembly/astribot_s1/shared/meshes" "$ASSETS/astribot_descriptions/urdf/astribot_s1_urdf/meshes"

# --- asset-internal: top-level asset packs surfaced under libero/objects/ so the
#     object resolver (scene_builder.OBJECT_CATEGORIES, which scans
#     libero/objects/<category>/<name>/<name>.xml) can find them. ---
# articraft_objects: Articraft-10K MJCF pack.
link "$ASSETS/libero/objects/articraft_objects" "$ASSETS/articraft_objects"
# assembly_objects: hand-built MuJoCo primitive parts/fixtures for the assemble_*
# tabletop tasks. ONE canonical XML per part; many task roles alias to it via RELATIVE
# symlinks INSIDE the pack (cube_a..d -> ../assembly_cube/assembly_cube.xml,
# ball_a..c -> ../assembly_ball/..., tower_post_a/b & pier_a/b & stool_leg_a/b ->
# ../assembly_block_short/..., and in dexjoco_objects: threading_rod_left/mid/right
# -> ../threading_rod/threading_rod.xml). So a single physical file is reachable under
# several names — preserve those relative symlinks when populating the pack; only this
# one surface link is needed to expose the whole category to the resolver.
link "$ASSETS/libero/objects/assembly_objects" "$ASSETS/assembly_objects"

# --- within-pack alias symlinks: re-establish so the env is robust to EITHER OSS
#     upload mode (see alias_link above). Runs after the surface links. ---
echo "Recreating within-pack alias symlinks:"
# assembly_objects: <canonical_model> <alias1> <alias2> ... — one physical model
# shared by many role names; each alias <name>.xml -> ../<canonical>/<canonical>.xml.
while read -r canon aliases; do
  if [[ -z "$canon" ]]; then continue; fi
  for a in $aliases; do alias_link "$ASSETS/assembly_objects" "$a" "$canon"; done
done <<'ASSEMBLY_ALIASES'
assembly_ball              ball_a ball_b ball_c
assembly_base              base_a base_b base_c base_d base_e
assembly_block_long        block_long_l1 tower_post_a tower_post_b
assembly_block_long_grn    block_long_l3
assembly_block_short       block_short_l2a block_short_l2b pier_a pier_b stool_leg_a stool_leg_b
assembly_block_short_prp   block_short_l4
assembly_bowl              bowl_a bowl_b bowl_c bowl_d
assembly_cap               cap_a cap_b cap_c
assembly_cube              cube_a cube_b cube_c cube_d
assembly_cube_blu          cube_blu_a
assembly_cube_grn          cube_grn_a cube_grn_b
assembly_disk_flat         disk_a disk_b disk_c
assembly_domino            domino_a domino_b domino_c
assembly_post              post_a post_b
assembly_sleeve            sleeve_a sleeve_b
ASSEMBLY_ALIASES
# dexjoco_objects: the three side/middle rods alias the canonical threading_rod.
for a in threading_rod_left threading_rod_mid threading_rod_right; do
  alias_link "$ASSETS/libero/objects/dexjoco_objects" "$a" threading_rod
done

# robocasa lightwheel/tiered_basket: the cabinet-basket (shelf/tray) tasks use 4
# instance aliases of TieredBasket007 (S1..S4 shelf positions). RoboCasa objects
# are `<name>/model.xml` dirs, so the alias is a DIRECTORY symlink.
for a in TieredBasket007_b TieredBasket007_c TieredBasket007_d; do
  tb="$ASSETS/robocasa_assets/objects/lightwheel/tiered_basket"
  if [[ -d "$tb/TieredBasket007" ]]; then
    ln -sfn TieredBasket007 "$tb/$a"
    echo "  LINK  $tb/$a -> TieredBasket007"
  else
    echo "  SKIP  $a (TieredBasket007 missing in $tb)" >&2
  fi
done

# (sharpa_assets.xml / sharpa_contacts.xml now live as real files in the repo at
#  bidex/robots/assembly/astribot_s1/shared/ — no symlink needed.)

# --- verify targets exist ---
echo "Verifying targets:"
fail=0
check() { if [[ -e "$1" ]]; then echo "  OK    $1"; else echo "  MISS  $1" >&2; fail=1; fi; }
check "$ASSETS/libero"
check "$ASSETS/robocasa_assets"
check "$ASSETS/astribot_descriptions/urdf/astribot_s1_urdf/meshes"
check "$ASSETS/articraft_objects"
check "$ASSETS/assembly_objects"
check "$ASSETS/assembly_objects/cube_a/cube_a.xml"                              # alias symlink -> assembly_cube
check "$ASSETS/libero/objects/dexjoco_objects/threading_rod_left/threading_rod_left.xml"  # alias -> threading_rod

if [[ $fail -ne 0 ]]; then echo "ERROR: one or more asset targets missing (see MISS above)." >&2; exit 1; fi
if [[ $err -ne 0 ]]; then echo "WARNING: some links were skipped (real dirs in the way); review SKIP lines above." >&2; fi
echo "Done."
