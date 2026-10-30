#!/usr/bin/env bash
# Convenience wrapper for downloading only the asset component.
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
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [--source huggingface|modelscope] [destination]"
  exit 0
fi
destination="${1:-${BIDEX_DOWNLOAD_ROOT:-downloads/BiDex-100}}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--source huggingface|modelscope] [destination]" >&2
  exit 2
fi

exec "$SCRIPT_DIR/download_release.sh" "${source_args[@]}" assets "$destination"
