#!/usr/bin/env bash
# Convenience wrapper for downloading only the asset component.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${BIDEX_DOWNLOAD_ROOT:-downloads/BiDex-100}}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [destination]" >&2
  exit 2
fi

exec "$SCRIPT_DIR/download_release.sh" assets "$destination"
