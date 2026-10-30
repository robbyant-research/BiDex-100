#!/usr/bin/env bash
# Download selected parts of the BiDex-100 release from either dataset host.
set -euo pipefail

source="${BIDEX_DOWNLOAD_SOURCE:-huggingface}"
DEFAULT_DESTINATION="${BIDEX_DOWNLOAD_ROOT:-downloads/BiDex-100}"

usage() {
  cat <<EOF
Usage: $0 [--source huggingface|modelscope] <component> [destination]

Components:
  assets       3D assets and metadata
  lerobotv21   LeRobot v2.1 datasets
  lerobotv30   LeRobot v3.0 datasets
  datasets     both dataset formats
  all          assets and both dataset formats

The destination keeps the same repository layout for either source:
  <destination>/assets/
  <destination>/datasets/lerobotv21/
  <destination>/datasets/lerobotv30/

Environment variables:
  BIDEX_DOWNLOAD_ROOT  default destination (default: downloads/BiDex-100)
  BIDEX_DOWNLOAD_SOURCE  huggingface (default) or modelscope
  BIDEX_HF_REPO        Hugging Face repository (default: Robbyant-Research/BiDex-100)
  BIDEX_MODELSCOPE_REPO ModelScope repository (default: Robbyant-Research/BiDex-100)
  BIDEX_EXPECTED_TASKS expected task count per dataset format (default: 106)
EOF
}

if [[ "${1:-}" == "--source" ]]; then
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "ERROR: --source requires huggingface or modelscope" >&2
    exit 2
  fi
  source="$2"
  shift 2
fi

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -ge 1 ]] && exit 0
  exit 2
fi

component="$1"
destination="${2:-$DEFAULT_DESTINATION}"
if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

case "$component" in
  assets)
    includes=("assets/**")
    ;;
  lerobotv21|v21|2.1)
    component="lerobotv21"
    includes=("datasets/lerobotv21/**")
    ;;
  lerobotv30|v30|3.0)
    component="lerobotv30"
    includes=("datasets/lerobotv30/**")
    ;;
  datasets)
    includes=("datasets/lerobotv21/**" "datasets/lerobotv30/**")
    ;;
  all)
    includes=(
      "assets/**"
      "datasets/lerobotv21/**"
      "datasets/lerobotv30/**"
    )
    ;;
  *)
    echo "ERROR: unknown component: $component" >&2
    usage >&2
    exit 2
    ;;
esac

case "$source" in
  huggingface|hf)
    source="huggingface"
    REPO_ID="${BIDEX_HF_REPO:-Robbyant-Research/BiDex-100}"
    cli="hf"
    package="huggingface_hub"
    ;;
  modelscope)
    REPO_ID="${BIDEX_MODELSCOPE_REPO:-Robbyant-Research/BiDex-100}"
    cli="modelscope"
    package="modelscope"
    ;;
  *)
    echo "ERROR: unknown download source: $source (use huggingface or modelscope)" >&2
    exit 2
    ;;
esac

if ! command -v "$cli" >/dev/null 2>&1; then
  echo "ERROR: $cli CLI not found. Install it with:" >&2
  echo "  python -m pip install -U $package" >&2
  exit 1
fi

mkdir -p "$destination"
echo "Source     : $source"
echo "Repository : $REPO_ID"
echo "Component  : $component"
echo "Destination: $destination"
echo "Includes   : ${includes[*]}"

if [[ "$source" == "modelscope" ]]; then
  # ModelScope 1.38+ uses --local-dir; older SDKs use --local_dir.
  local_dir_option="--local_dir"
  if [[ "$(modelscope download --help)" == *"--local-dir"* ]]; then
    local_dir_option="--local-dir"
  fi
  modelscope download "$REPO_ID" --repo-type dataset \
    --include "${includes[@]}" \
    "$local_dir_option" "$destination"
else
  hf download "$REPO_ID" \
    --repo-type dataset \
    --include "${includes[@]}" \
    --local-dir "$destination"
fi

validate_assets() {
  local pack archive
  for pack in \
    articraft_objects \
    assembly_objects \
    astribot_descriptions \
    libero \
    robocasa_assets; do
    archive="$destination/assets/$pack/$pack.tar.gz"
    if [[ ! -s "$archive" ]]; then
      echo "ERROR: asset download is incomplete; missing or empty: $archive" >&2
      return 1
    fi
  done
}

validate_dataset() {
  local format="$1"
  local root="$destination/datasets/$format"
  local expected="${BIDEX_EXPECTED_TASKS:-106}"
  local found info task_root

  if [[ ! "$expected" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BIDEX_EXPECTED_TASKS must be a positive integer: $expected" >&2
    return 1
  fi
  if [[ -d "$root" ]]; then
    found="$(find "$root" -type f -path '*/meta/info.json' | wc -l)"
    found="${found//[[:space:]]/}"
  else
    found=0
  fi
  if [[ "$found" != "$expected" ]]; then
    echo "ERROR: $format download is incomplete: expected $expected task metadata files, found $found under $root" >&2
    return 1
  fi
  while IFS= read -r -d '' info; do
    task_root="${info%/meta/info.json}"
    if [[ -z "$(find "$task_root/data" -type f -name '*.parquet' -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: $format download is incomplete; no parquet data under $task_root" >&2
      return 1
    fi
    if [[ -z "$(find "$task_root/videos" -type f -name '*.mp4' -print -quit 2>/dev/null)" ]]; then
      echo "ERROR: $format download is incomplete; no video data under $task_root" >&2
      return 1
    fi
  done < <(find "$root" -type f -path '*/meta/info.json' -print0)
}

case "$component" in
  assets)
    validate_assets
    ;;
  lerobotv21|lerobotv30)
    validate_dataset "$component"
    ;;
  datasets)
    validate_dataset lerobotv21
    validate_dataset lerobotv30
    ;;
  all)
    validate_assets
    validate_dataset lerobotv21
    validate_dataset lerobotv30
    ;;
esac

echo "Download complete."
if [[ "$component" == "assets" || "$component" == "all" ]]; then
  echo "Prepare assets with:"
  echo "  bash scripts/setup_assets_links.sh '$destination'"
fi
