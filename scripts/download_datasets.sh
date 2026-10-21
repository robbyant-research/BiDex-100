#!/usr/bin/env bash
# Convenience wrapper for downloading one or both LeRobot dataset formats.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_args=()
if [[ "${1:-}" == "--source" ]]; then
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "ERROR: --source requires huggingface or modelscope" >&2
    exit 2
  fi
  source_args=(--source "$2")
  shift 2
fi

usage() {
  echo "Usage: $0 [--source huggingface|modelscope] [lerobotv21|lerobotv30|all] [destination]" >&2
}

component="${1:-all}"
case "$component" in
  lerobotv21|v21|2.1|lerobotv30|v30|3.0)
    shift || true
    ;;
  all)
    component="datasets"
    shift || true
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    # Backward compatibility: the old interface accepted only a destination.
    destination="$component"
    component="datasets"
    shift || true
    ;;
esac

destination="${destination:-${1:-${BIDEX_DATASET_ROOT:-${BIDEX_DOWNLOAD_ROOT:-downloads/BiDex-100}}}}"
if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

exec "$SCRIPT_DIR/download_release.sh" "${source_args[@]}" "$component" "$destination"
