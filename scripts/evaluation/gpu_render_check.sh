#!/usr/bin/env bash
# Render self-check for every GPU in a comma list (e.g. "0,1,2,3").
#
# Each GPU gets one check process (mujoco caches the EGL display per-process, so
# one process can only test one MUJOCO_EGL_DEVICE_ID) running
# scripts/evaluation/check_gpu_render.py: a static scene is rendered twice and byte-compared.
# A faulty card (Xid-31/109 dropping glReadPixels writes) returns stale /
# nondeterministic / black frames — which in evals surface as upside-down
# wrong-camera frames that pollute BOTH the mp4s and the policy inputs.
#
# Stdout: the HEALTHY subset of the input list (comma-joined, original order).
# Stderr: one PASS/FAIL line per GPU (FAIL lines include PCI id + reason).
# Exit: 0 if at least one GPU is healthy, 1 otherwise.
#
# Env knobs:
#   GPU_CHECK_PY_BIN  python with mujoco (default: lingbot eval venv)
#   GPU_CHECK_ITERS   double-renders per camera per GPU (default 120)
set -uo pipefail

GPUS_IN="${1:?usage: gpu_render_check.sh <gpu-list>  e.g. 0,1,2,3}"
PY_BIN="${GPU_CHECK_PY_BIN:-${MODEL_PY:-python3}}"
ITERS="${GPU_CHECK_ITERS:-120}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$PY_BIN" ]; then
  echo "[gpu-check] ERROR: python not found: $PY_BIN (set GPU_CHECK_PY_BIN)" >&2
  exit 1
fi

LOGDIR="$(mktemp -d /tmp/gpu_render_check.XXXXXX)"
pids=()
gpus=()
for g in ${GPUS_IN//,/ }; do
  CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$g" \
    "$PY_BIN" "$HERE/check_gpu_render.py" --gpu "$g" --iters "$ITERS" \
    > "$LOGDIR/gpu${g}.log" 2>&1 &
  pids+=($!)
  gpus+=("$g")
done

healthy=()
for i in "${!gpus[@]}"; do
  g="${gpus[$i]}"
  if wait "${pids[$i]}"; then
    healthy+=("$g")
    echo "[gpu-check] GPU$g PASS" >&2
  else
    pci="$(nvidia-smi -i "$g" --query-gpu=pci.bus_id,name --format=csv,noheader 2>/dev/null)"
    reason="$(tail -1 "$LOGDIR/gpu${g}.log" 2>/dev/null)"
    echo "[gpu-check] WARNING: GPU$g ($pci) FAILED render self-check -> EXCLUDED from this eval" >&2
    [ -n "$reason" ] && echo "[gpu-check]   $reason" >&2
  fi
done
rm -rf "$LOGDIR"

( IFS=','; echo "${healthy[*]}" )
[ "${#healthy[@]}" -ge 1 ]
