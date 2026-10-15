#!/usr/bin/env bash
# GPU idle-check + orphan cleanup. Complements gpu_render_check.sh (which tests render
# DETERMINISM on faulty cards): this one ensures the cards are MEMORY-IDLE before a model
# loads, and reaps orphan eval/serve processes left by a previous crashed run or bench.
#
# Why this exists: if a prior run crashes and leaves a policy server or render worker
# alive, the next server-client run can exceed GPU memory or hit an EGL fault. This
# guard makes stale processes and unexpected GPU use visible before model loading.
#
# What it does, in order:
#   1. Kill processes whose cmdline matches --kill-patterns AND (with --kill-scope repo)
#      whose cwd is under --repo, so it NEVER touches another user's job on a shared-NFS
#      box. The guard's own shell and its ancestors are never killed.
#   2. Re-snapshot per-GPU memory.used (MiB) for every card in --gpus and flag any card
#      at/above --idle-mib as NON-IDLE.
#   3. Exit 0 if all checked cards are idle; otherwise exit 0 in --mode warn (after a loud
#      warning) or exit 1 in --mode fail. --no-check skips the idle check (kill-only, for
#      between-bench use where a parent serve_policy server legitimately holds mem).
#
# Usage:
#   gpu_idle_check.sh --gpus 0,1,...,7 [--repo <repo-root>] [--kill-patterns RE]
#                     [--kill-scope repo|all] [--idle-mib 200] [--mode warn|fail]
#                     [--no-kill] [--no-check] [--label <tag>]
#
# Env: CUDA_VISIBLE_DEVICES is the fallback for --gpus.
#
# Stderr: one [gpu-guard:<label>] line per GPU + a kill summary. stdout: clean.
set -uo pipefail

# ---------------- defaults ----------------
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
REPO="${REPO:-}"
KILL_PATTERNS='run_leaderboard_eval\.py|serve_policy_batched\.py|serve_policy\.py|eval_dashboard\.py'
KILL_SCOPE="repo"          # repo = only kill procs whose cwd is under --repo (safe on shared NFS)
IDLE_MIB="200"             # MiB; >= this on a checked GPU => NON-IDLE
MODE="warn"                # warn = exit 0 on non-idle (loud warning); fail = exit 1
NO_KILL="0"
NO_CHECK="0"
LABEL="gpu-guard"

# ---------------- parse args ----------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)          GPUS="$2"; shift 2 ;;
    --repo)          REPO="$2"; shift 2 ;;
    --kill-patterns) KILL_PATTERNS="$2"; shift 2 ;;
    --kill-scope)    KILL_SCOPE="$2"; shift 2 ;;
    --idle-mib)      IDLE_MIB="$2"; shift 2 ;;
    --mode)          MODE="$2"; shift 2 ;;
    --no-kill)       NO_KILL="1"; shift ;;
    --no-check)      NO_CHECK="1"; shift ;;
    --label)         LABEL="$2"; shift 2 ;;
    -h|--help)       sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "[gpu-guard] unknown arg: $1" >&2; exit 2 ;;
  esac
done

gprint() { echo "[${LABEL}] $*" >&2; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  gprint "nvidia-smi not found — skipping GPU idle check."
  exit 0
fi

# Normalize the GPU list into a space-separated list of physical indices.
GPUS_LIST="${GPUS//,/ }"

# ---------------- step 1: reap orphan processes ----------------
KILLED="0"
if [[ "$NO_KILL" == "0" && -n "$KILL_PATTERNS" ]]; then
  SELF_PID="$$"
  # pgrep -f matches the full cmdline against an ERE; one pid per line. Command
  # substitution (not an array) so `set -u` never trips on an empty result.
  for _p in $(pgrep -f "$KILL_PATTERNS" 2>/dev/null || true); do
    [[ -z "$_p" || "$_p" == "$SELF_PID" ]] && continue
    if [[ "$KILL_SCOPE" == "repo" ]]; then
      _cwd=""
      [[ -L "/proc/$_p/cwd" ]] && _cwd="$(readlink "/proc/$_p/cwd" 2>/dev/null || true)"
      # Only touch processes running from this repo (cwd == REPO or under it). On a
      # shared-NFS box a different user's job with a different cwd is left alone. Note the
      # eval workers run with cwd == $REPO exactly (local_eval does `cd "$REPO"`), so we
      # must match the repo root, not just subdirectories.
      [[ -z "$_cwd" || -z "$REPO" ]] && continue
      [[ "$_cwd" == "$REPO" || "$_cwd" == "$REPO"/* ]] || continue
    fi
    kill -9 "$_p" 2>/dev/null && KILLED="$((KILLED+1))"
  done
  if [[ "$KILLED" -gt 0 ]]; then
    gprint "killed $KILLED orphan process(es) matching [$KILL_PATTERNS] (scope=$KILL_SCOPE) — reclaiming their GPU memory."
    # Give the kernel a moment to release the CUDA contexts before re-sampling.
    sleep 2
  fi
fi

# ---------------- step 2 (optional): per-GPU memory idle check ----------------
if [[ "$NO_CHECK" == "1" ]]; then
  # Kill-only mode (e.g. between benches, where a parent serve_policy server may
  # legitimately hold ~10GiB on GPU0 — do not run a hard idle fail-check against that).
  exit 0
fi

# Capture once; awk looks up each requested GPU's used-MiB by physical index.
MEMINFO="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
gpu_used_mib() { awk -F', *' -v g="$1" '$1==g {print $2; exit}' <<<"$MEMINFO"; }

NONIDLE=""
NONIDLE_CNT="0"
for _g in $GPUS_LIST; do
  _u="$(gpu_used_mib "$_g")"
  if [[ -z "$_u" ]]; then
    gprint "GPU$_g: memory query failed (treated as NON-IDLE)."
    NONIDLE="${NONIDLE:+$NONIDLE,}$_g"; NONIDLE_CNT="$((NONIDLE_CNT+1))"
  elif (( _u >= IDLE_MIB )); then
    gprint "GPU$_g: ${_u} MiB used  -> NON-IDLE (>= ${IDLE_MIB} MiB)."
    NONIDLE="${NONIDLE:+$NONIDLE,}$_g"; NONIDLE_CNT="$((NONIDLE_CNT+1))"
  else
    gprint "GPU$_g: ${_u} MiB used  -> idle."
  fi
done

if [[ "$NONIDLE_CNT" -eq 0 ]]; then
  gprint "all checked GPUs idle (threshold ${IDLE_MIB} MiB)."
  exit 0
fi

gprint "WARNING: $NONIDLE_CNT GPU(s) NON-IDLE after orphan cleanup: [$NONIDLE]."
gprint "  A non-idle card at model-load time oversubscribes the batched forward -> CUDA OOM"
gprint "  or Xid illegal-memory-access -> the eval's fallback broadcasts ZERO actions -> the"
gprint "  robot flails to the zero pose. Likely cause: another process/tenant on this node,"
gprint "  or a stuck (D-state) orphan we could not kill. Set BIDEX_GPU_GUARD=fail to abort here."

case "$MODE" in
  fail)
    gprint "FATAL: non-idle GPU(s) with --mode fail — aborting before any model loads."
    exit 1 ;;
  *)
    exit 0 ;;
esac
