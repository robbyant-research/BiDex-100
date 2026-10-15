#!/usr/bin/env bash
# Vectorized eval launcher — scalable per-model, with leaderboard URL hint.
#
# Each invocation evaluates one model with an explicit policy/render GPU split.
#
# Usage:
#   ./scripts/evaluation/run_vec_models_eval.sh MODEL [smoke]
#
#   MODEL  : pi05 | galaxea | gr00t | lingbot | xr1 | internvla | dm05
#            Exactly one model is evaluated per invocation.
#   smoke  : pour_ice_between_two_cups x 1 episode x 1 seed
#
# Examples:
#   ./scripts/evaluation/run_vec_models_eval.sh pi05              # only pi05
#   ./scripts/evaluation/run_vec_models_eval.sh pi05 smoke        # smoke pi05 only
#
# Environment overrides (per-run):
#   GPUS / SEEDS / EPS / ENVS_PER_GPU / MAX_RECORD_ENVS / BIDEX_MAX_ENVS_PER_GPU
#   MODEL_GPUS — comma list of MODEL (policy-server) GPUs. run_<model> starts ONE
#                serve_policy_batched instance per entry (ports base, base+1, ...);
#                render workers are spread across those servers (rank r -> server
#                r % N in launch_vector), so the sim-eval load is allocated by the
#                MODEL-GPU count. Default: 0.
#   RENDER_GPUS — comma list of MuJoCo render-worker GPUs; overlap is allowed.
#                Default: GPUS minus MODEL_GPUS (=> 1..7 on an 8-GPU host).
#                GPUS stays as a compatibility alias for the union.
#   SKIP_GPU_CHECK=1 — bypass the pre-eval GPU render self-check (faulty-card guard;
#                      GPUs in GPUS that fail it are otherwise excluded automatically)
#   ── Unified, UN-prefixed PATH params (REQUIRED; export each in the submit yml — this
#      script hardcodes NO path default; a missing one hard-fails the whole batch up front
#      via _check_model_required()). The yml uses ONE uniform name set; the per-adapter
#      adapter env names (GALAXEA_REPO / QWEN3VL_PATH / ...) are translated
#      from these by the script (the python adapters read the legacy names) ──
#   MODEL_PY   — interpreter of the selected model's dedicated venv.
#   MODEL_REPO — model SOURCE root; PYTHONPATH is DERIVED from it (+ this repo's $REPO),
#                so the yml should NOT export PYTHONPATH (no path duplication).
#   MODEL_CKPT — weights for the selected model.
#   MODEL_VLM_PATH  — local base-VLM dir (required for lingbot/xr1/internvla).
#   OPENPI_DATA_HOME — optional native OpenPI cache override.
#   BIDEX_LEADERBOARD_ROOT — results root (REQUIRED all). `OUT` is an internal alias.
#   DM05_DATA_PATH / OPENDM_DATA_PATH are derived from MODEL_REPO.
#   MID_OVERRIDE — model_id (leaderboard row label); export in the yml (run_<model> falls
#                back to a computed label when unset).
#
#   ── Generic run knobs (have script defaults; the yml may override) ──
#   BENCHES — benchmarks (default: all 5, in eval order IF→Memory→LH→LD→Gen). NO_DASHBOARD=1
#          — log-only. GPUS/MODEL_GPUS/
#          RENDER_GPUS, seeds/EPS, *_SERVER_PORT, BIDEX_VEC_*, DM05_ROBOT_TYPE,
#          INTERNVLA_NOISE/LOG_ACTIONS, XLA_PYTHON_CLIENT_MEM_FRACTION, BIDEX_EGL_DEVICE_ID
#          — operational knobs.
set -uo pipefail

# Repo root = two levels above scripts/evaluation (portable across machines).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"; ulimit -c 0

# Pin torch CPU thread pools to 1/worker. The forward is GPU-bound, so 1 thread/rank
# doesn't slow inference, but it prevents render-thread starvation: with OMP_NUM_THREADS
# unset, torch spawns ~128 threads/rank x 8 ranks and the MuJoCo EGL render thread is
# descheduled -> the render queue saturates -> the sim stalls for seconds per batch.
# Set the BLAS family (MKL/OpenBLAS/NumExpr) too. Override by pre-exporting your own value.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Reduce torch CUDA-allocator fragmentation on the policy-server GPUs. The galaxea
# vision-tower varlen flash-attn can fault (cudaErrorIllegalAddress) at step 0 only
# under full multi-render-worker contention; expandable_segments keeps large
# workspaces contiguous so the flash-attn workspace isn't scattered.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tests source this file to exercise path compatibility without running an eval.
if [[ "${BIDEX_EVAL_SOURCE_ONLY:-0}" == "1" ]]; then
  return 0
fi

# ────────────────────────── 1. Parse arguments ──────────────────────────

ALL_MODELS=(pi05 galaxea gr00t lingbot xr1 internvla dm05)
SMOKE=""
SELECTED=()

for arg in "$@"; do
  case "$arg" in
    smoke) SMOKE="smoke" ;;
    pi05|galaxea|gr00t|lingbot|xr1|internvla|dm05) SELECTED+=("$arg") ;;
    *) echo "Unknown argument: $arg (expected: ${ALL_MODELS[*]} or smoke)" >&2; exit 2 ;;
  esac
done

if [ ${#SELECTED[@]} -eq 0 ]; then
  echo "A model is required (expected: ${ALL_MODELS[*]})." >&2
  exit 2
fi
if [ ${#SELECTED[@]} -ne 1 ]; then
  echo "Exactly one model is supported per invocation; got: ${SELECTED[*]}" >&2
  exit 2
fi

# ────────────────────────── 1b. Required path params — fail fast ──────────────────────────
# Model-specific PATH params are no longer defaulted in this script — the submit yml must
# `export` each (see _req_vars). A missing one is a config bug (NOT "weights not trained"),
# so hard-fail the whole batch up front with a clear message, before any GPU/server work.
_req_vars() {
  case "$1" in
    pi05)      echo "MODEL_PY MODEL_REPO MODEL_CKPT" ;;
    lingbot)   echo "MODEL_PY MODEL_REPO MODEL_VLM_PATH MODEL_CKPT" ;;
    gr00t)     echo "MODEL_PY MODEL_REPO MODEL_CKPT" ;;
    galaxea)   echo "MODEL_PY MODEL_REPO MODEL_CKPT" ;;
    dm05)      echo "MODEL_PY MODEL_REPO MODEL_CKPT" ;;
    xr1)       echo "MODEL_PY MODEL_REPO MODEL_VLM_PATH MODEL_CKPT" ;;
    internvla) echo "MODEL_PY MODEL_REPO MODEL_VLM_PATH MODEL_CKPT" ;;
    *) echo "" ;;
  esac
}
_check_model_required() {
  local _m="$1" _v _missing=""
  for _v in $(_req_vars "$_m"); do
    [ -n "${!_v:-}" ] || _missing="${_missing:+$_missing }$_v"
  done
  if [ -n "$_missing" ]; then
    echo "FATAL: model '$_m' missing required path param(s):${_missing}" >&2
    echo "       export each of them in the submit yml — this script no longer hardcodes path defaults." >&2
    return 1
  fi
  [ -x "$MODEL_PY" ] || { echo "FATAL: MODEL_PY is not executable: $MODEL_PY" >&2; return 1; }
  [ -d "$MODEL_REPO" ] || { echo "FATAL: MODEL_REPO is not a directory: $MODEL_REPO" >&2; return 1; }
  [ -e "$MODEL_CKPT" ] || { echo "FATAL: MODEL_CKPT does not exist: $MODEL_CKPT" >&2; return 1; }
  case " $(_req_vars "$_m") " in
    *" MODEL_VLM_PATH "*)
      [ -d "$MODEL_VLM_PATH" ] || {
        echo "FATAL: MODEL_VLM_PATH is not a directory: $MODEL_VLM_PATH" >&2
        return 1
      } ;;
  esac
  return 0
}
for _m in "${SELECTED[@]}"; do
  _check_model_required "$_m" || exit 1
done
unset _m

# ────────────────────────── 2. Common env knobs ──────────────────────────

# NOTE: 50 envs/GPU can saturate render threads / risk Xid-31 on stacked hosts.
# It is the benchmark-protocol default; drop back via ENVS_PER_GPU=32 if a host
# shows frame drops/Xid.
export MAX_RECORD_ENVS=${MAX_RECORD_ENVS:-2}

# Every adapter receives weights through the same variable. Export it so
# local_eval and its workers inherit the selected checkpoint.
export MODEL_CKPT

if [[ "$SMOKE" == "smoke" ]]; then
  export GPUS=${GPUS:-0,1}
  export SEEDS=${SEEDS:-0}
  export ID_SEEDS=${ID_SEEDS:-0}
  export OOD_SEEDS=${OOD_SEEDS:-2}
  export EPS=${EPS:-1}
  export ENVS_PER_GPU=${ENVS_PER_GPU:-4}      # small for a quick smoke
  export BIDEX_MAX_ENVS_PER_GPU=${BIDEX_SMOKE_MAX_ENVS_PER_GPU:-${BIDEX_MAX_ENVS_PER_GPU:-4}}
  # Smoke: limit to one task for speed.  Only set BIDEX_TASKS when the user
  # hasn't narrowed BENCHES (e.g. BENCHES=Instruction_Following already constrains
  # the task set; BIDEX_TASKS would filter away non-Memory tasks and leave 0).
  if [ -z "${BIDEX_TASKS:-}" ] && [ "${BENCHES:-Instruction_Following Memory Long_Horizon Limited_Demonstrations Generalization}" = "Instruction_Following Memory Long_Horizon Limited_Demonstrations Generalization" ]; then
    export BIDEX_TASKS="Memory/pour_ice_between_two_cups"
  fi
  EFLAG=smoke
else
  export BIDEX_MAX_ENVS_PER_GPU=${BIDEX_MAX_ENVS_PER_GPU:-50}
  # Standard protocol: per task 100 rollouts = in-domain (seeds 0,1) x 25 eps
  # + out-of-domain (seeds 2,3) x 25 eps.
  export GPUS=${GPUS:-0,1,2,3,4,5,6,7}
  export ID_SEEDS=${ID_SEEDS:-0,1}
  export OOD_SEEDS=${OOD_SEEDS:-2,3}
  export SEEDS=${SEEDS:-$ID_SEEDS}
  export EPS=${EPS:-25}
  export ENVS_PER_GPU=${ENVS_PER_GPU:-50}
  EFLAG=""
fi

# ────────────────────────── 2b. GPU split: MODEL_GPUS / RENDER_GPUS ──────────────────────────
# GPUS is split into policy-server and render-worker roles:
#   MODEL_GPUS  — policy-server cards; run_<model> starts ONE serve_policy_batched
#                 instance per entry (ports base, base+1, ...) and launch_vector
#                 spreads render workers round-robin (rank r -> server r % N), so
#                 the sim-eval load is allocated by the MODEL-GPU count.
#   RENDER_GPUS — MuJoCo render-worker cards; overlap is allowed.
# Unset RENDER_GPUS derives from the legacy GPUS knob:
#   GPUS minus MODEL_GPUS → dedicated render cards.
#   When no dedicated render cards remain (every GPU is a model GPU — e.g. the user
#   set MODEL_GPUS=0,1,...,7), the model GPUs must also render, so RENDER_GPUS
#   falls back to MODEL_GPUS (complete overlap, still separate server/client processes).
# GPUS is then recomputed as the ordered union (model cards first).
#
# Every layout uses the same server-client transport. GPU overlap changes only
# placement: an overlapping card hosts one policy-server process and one
# render-worker process. It must never change policy inference semantics.
export MODEL_GPUS=${MODEL_GPUS:-0}
if [ -z "${RENDER_GPUS:-}" ]; then
  _render=""; IFS=',' read -ra _mgpa <<< "$MODEL_GPUS"
  IFS=',' read -ra _gpa <<< "$GPUS"
  for _g in "${_gpa[@]}"; do
    _is_model=0
    for _m in "${_mgpa[@]}"; do [ "$_g" = "$_m" ] && _is_model=1; done
    [ "$_is_model" = "1" ] && continue
    _render="${_render:+$_render,}$_g"
  done
  if [ -n "$_render" ]; then RENDER_GPUS="$_render"; else RENDER_GPUS="$MODEL_GPUS"; fi
fi
export RENDER_GPUS

# Compute GPUS as the ordered union of MODEL_GPUS ∪ RENDER_GPUS
_gp_union="$MODEL_GPUS"
IFS=',' read -ra _rgpa <<< "$RENDER_GPUS"
for _g in "${_rgpa[@]}"; do
  case ",$_gp_union," in *",$_g,"*) ;; *) _gp_union="$_gp_union,$_g" ;; esac
done
export GPUS="$_gp_union"

# ── Describe disjoint / partial overlap / complete overlap placement ──
# GPU overlap = any GPU that appears in BOTH MODEL_GPUS and RENDER_GPUS.
#   Disjoint  (MODEL_GPUS ∩ RENDER_GPUS = ∅):
#     server-client — model servers on MODEL_GPUS, render workers on RENDER_GPUS.
#   Complete  (MODEL_GPUS == RENDER_GPUS as sets):
#     every GPU hosts one server process and one render-worker process.
#   Partial   (overlap exists but sets differ):
#     hybrid server-client — servers on MODEL_GPUS, render workers on
#     RENDER_GPUS.  Overlapping GPUs naturally run BOTH a server process
#     and a render worker; model-only GPUs serve only; render-only GPUs
#     connect to the servers; render-only cards never load the model.
_gpu_overlap=""   # GPUs shared between model and render
IFS=',' read -ra _ck_mgpa <<< "$MODEL_GPUS"
IFS=',' read -ra _ck_rgpa <<< "$RENDER_GPUS"
_n_model=${#_ck_mgpa[@]}
_n_render=${#_ck_rgpa[@]}
_complete_gpu_overlap=0
for _m in "${_ck_mgpa[@]}"; do
  for _r in "${_ck_rgpa[@]}"; do
    [ "$_m" = "$_r" ] && _gpu_overlap="${_gpu_overlap:+$_gpu_overlap,}$_m"
  done
done

if [ -n "$_gpu_overlap" ]; then
  IFS=',' read -ra _ov_arr <<< "$_gpu_overlap"
  _n_overlap=${#_ov_arr[@]}
  if [ "$_n_model" -eq "$_n_overlap" ] && [ "$_n_render" -eq "$_n_overlap" ]; then
    _complete_gpu_overlap=1
  fi
fi
export GALAXEA_SERVER=1 LINGBOT_SERVER=1 GR00T_SERVER=1
export XR1_SERVER=1 INTERNVLA_SERVER=1 DM05_SERVER=1
echo "[gpu-layout] SERVER-CLIENT: MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS overlap=${_gpu_overlap:-none}"

# ── GPU idle-check + orphan cleanup (oversubscription guard) ──
# Before loading a model, detect non-idle cards and optionally reap evaluator
# processes left by an earlier run of this repository.
# Knobs: BIDEX_GPU_GUARD=warn (default; warn + proceed) | fail | off.
# Orphan termination is opt-in with BIDEX_ORPHAN_KILL=1 so concurrent evaluations
# are never stopped by default.
if [[ "${BIDEX_GPU_GUARD:-warn}" != "off" && "${SKIP_GPU_GUARD:-0}" != "1" ]]; then
  _guard_args=(--gpus "$GPUS" --repo "$REPO" --mode "${BIDEX_GPU_GUARD:-warn}" --label "pre-run")
  [[ -n "${BIDEX_GPU_IDLE_MIB:-}" ]] && _guard_args+=(--idle-mib "$BIDEX_GPU_IDLE_MIB")
  [[ "${BIDEX_ORPHAN_KILL:-0}" == "0" ]] && _guard_args+=(--no-kill)
  if ! "$REPO/scripts/evaluation/gpu_idle_check.sh" "${_guard_args[@]}"; then
    echo "[gpu-guard] FATAL: pre-run idle-check failed (BIDEX_GPU_GUARD=${BIDEX_GPU_GUARD:-warn});" >&2
    echo "[gpu-guard]        GPUS=$GPUS non-idle after orphan cleanup. Clean the node / stuck" >&2
    echo "[gpu-guard]        process, or set BIDEX_GPU_GUARD=warn|off to proceed at your own risk." >&2
    exit 1
  fi
fi

# ── GPU render self-check (faulty-card guard) ──
# A faulty render card (e.g. NVRM Xid-31/109 dropping glReadPixels writes on the
# graphics channel) silently injects stale/flipped/black frames into BOTH the eval
# mp4s and the policy inputs (the recorded frame and the infer obs are the same
# array) — scores get corrupted with no error raised. Check every GPU in $GPUS
# BEFORE launching anything; failed GPUs are dropped from the pool.
# Knobs: SKIP_GPU_CHECK=1 (bypass), GPU_CHECK_ITERS (default 120),
#        GPU_CHECK_ATTEMPTS (default 3), GPU_CHECK_MIN_PASS (default 2),
#        GPU_CHECK_MAX_PX_DIFF (default 15), GPU_CHECK_MAX_DIFF_FRAC (default 0.01),
#        GPU_CHECK_PY_BIN (python with mujoco).
if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  echo "[gpu-check] render determinism self-check on GPUS=$GPUS ..."
  if ! _healthy_gpus=$("$REPO/scripts/evaluation/gpu_render_check.sh" "$GPUS"); then
    echo "[gpu-check] FATAL: every GPU in GPUS=$GPUS failed the render self-check" >&2
    exit 1
  fi
  if [[ "$_healthy_gpus" != "$GPUS" ]]; then
    # Any MODEL_GPUS card that failed the check cannot be relocated (the policy
    # servers _srv_start sets CUDA_VISIBLE_DEVICES per model GPU) — abort loudly
    # rather than run the model on a faulty card.
    IFS=',' read -ra _mgpa2 <<< "$MODEL_GPUS"
    for _m in "${_mgpa2[@]}"; do
      case ",$_healthy_gpus," in
        *",$_m,"*) ;;
        *) echo "[gpu-check] FATAL: model GPU $_m (MODEL_GPUS=$MODEL_GPUS) failed the render self-check." >&2
           echo "[gpu-check] Fix/replace the card, set MODEL_GPUS= to exclude it, or rerun with SKIP_GPU_CHECK=1 at your own risk." >&2
           exit 1 ;;
      esac
    done
    _render_keep=""
    IFS=',' read -ra _rgpa2 <<< "$RENDER_GPUS"
    for _g in "${_rgpa2[@]}"; do
      case ",$_healthy_gpus," in *",$_g,"*) _render_keep="${_render_keep:+$_render_keep,}$_g" ;; esac
    done
    if [ -n "$_render_keep" ] && [ "$_render_keep" != "$RENDER_GPUS" ]; then
      echo "[gpu-check] RENDER_GPUS=$RENDER_GPUS -> dropping failed cards: $_render_keep"
      export RENDER_GPUS="$_render_keep"
    fi
    echo "[gpu-check] GPUS=$GPUS -> healthy subset: $_healthy_gpus"
  else
    echo "[gpu-check] all GPUs healthy (MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS)"
  fi
fi

OUT="${BIDEX_LEADERBOARD_ROOT:?export BIDEX_LEADERBOARD_ROOT in the submit yml}"   # results root is required from the yml (OUT is an internal alias)

# Vectorized batched-server eval knobs (shared by all server-client models:
# pi05/openpi + galaxea/lingbot/gr00t/xr1/internvla serve_policy_batched).
# The pi05-only toggles (VEC_OPENPI, OPENPI_PY, PI) live in run_pi05().
export BIDEX_VEC_CONCURRENT=1     # enable concurrent infer path (fire due envs in parallel)
export BIDEX_MAX_BATCH=${BIDEX_MAX_BATCH:-64}
# VEC_CONCURRENCY=50: 50 concurrent websocket connections per worker, so
# all due envs (up to ENVS_PER_GPU=50) can fire in parallel when a new task
# starts.  The batched server collects these into one forward of up to
# MAX_BATCH (padded to next power-of-2 bucket = 64 for batch ≤ 50).
export BIDEX_VEC_CONCURRENCY=${BIDEX_VEC_CONCURRENCY:-50}

# Benchmarks to evaluate
BENCHES=${BENCHES:-"Instruction_Following Memory Long_Horizon Limited_Demonstrations Generalization"}

mkdir -p "$OUT/runs" "$OUT/results" "$OUT/mp4s"

# ────────────────────────── 3. Print banner + leaderboard URL ──────────────────────────

# Guess the leaderboard cloudproxy URL from the QC workflow convention.
# The leaderboard server (baseline_leaderboard_server.py) runs on port 12006
# with prefix /baselines.  Override with WORKFLOW / PROXY_HOST env vars.
PROXY_HOST="${PROXY_HOST:-}"
WORKFLOW="${WORKFLOW:-61720624}"
LB_URL="${PROXY_HOST:+https://${PROXY_HOST}/proxy/workflow_${WORKFLOW}:12006/baselines/}"

echo ""
cat <<BANNER

╔══════════════════════════════════════════════════════════════════════════════════╗
║  BiDex-100 Vectorized Eval Launcher                                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Models:     ${SELECTED[*]}
║  Smoke:      ${SMOKE:-no}
║  GPUs:       ${GPUS}
║  GPU split:  MODEL=${MODEL_GPUS} (policy servers) RENDER=${RENDER_GPUS} (render workers)
║  EPS/seeds:  ${EPS} / ID=${ID_SEEDS} OOD=${OOD_SEEDS}
║  BENCHES:    ${BENCHES}
║  Results →  ${OUT}
╠══════════════════════════════════════════════════════════════════════════════════╣
║  🏆 Leaderboard:  ${LB_URL}
╚══════════════════════════════════════════════════════════════════════════════════╝
BANNER
echo ""

# ────────────────────────── 4. Per-model config & launch ──────────────────────────

# Kill a process and all its descendants (recursive pgrep -P).
_kill_tree() {
  local _pid=$1
  local _child
  for _child in $(pgrep -P "$_pid" 2>/dev/null); do _kill_tree "$_child"; done
  kill -9 "$_pid" 2>/dev/null
}

# Track PIDs we own so _cleanup_between only kills our processes, not unrelated
# serve_policy instances (e.g. another user's eval sharing the same machine via NFS).
_OUR_PIDS=()   # filled by each run_* function

# ── Multi-model-GPU policy server helpers ─────────────────────────────────
# One serve_policy_batched instance per MODEL_GPUS entry (ports base, base+1, ...);
# launch_vector spreads render workers across the servers (rank r -> server r % N.
# so the sim-eval load is allocated by the model-GPU count). All server PIDs are
# tracked in _OUR_PIDS so _cleanup_between reaps them between models.
_SRV_PIDS=(); _SRV_PORTS=(); _SRV_GPUS=(); _SRV_LOGPFX=""
_SRV_ENV=()   # per-model extra env for the server process (VAR=VAL entries)

# step label (60k/200k) for the MID: digits of the ckpt dir name, walking up
# through parents (handles checkpoint-200000 / global_step_60000 /
# epoch=0-step=60000.ckpt / <060000>/pretrained_model).
_stepk() {
  local d="$1" n i
  for i in 1 2 3; do
    n=$(basename "$d" | grep -oE '[0-9]+' | tail -1)
    if [ -n "$n" ]; then n=$((10#$n)); echo "$(( (n+500)/1000 ))k"; return; fi
    d=$(dirname "$d")
  done
  echo "unk"
}

# Ensure the venv python is executable and has the server's
# websocket/msgpack deps (idempotent pip bootstrap).
_ensure_srv_py() {
  local _py="$1"
  [ -x "$_py" ] || chmod +x "$_py" 2>/dev/null || true
  [[ "${DRY:-0}" == "1" ]] && return 0
  if ! "$_py" -c "import msgpack, websockets" >/dev/null 2>&1; then
    echo "[$(date '+%T')] bootstrapping server deps (msgpack/websockets) into the venv of $_py ..."
    "$_py" -m pip install -q msgpack websockets || \
      echo "  WARN: pip bootstrap failed — the server may not start" >&2
  fi
}

# _srv_start <port_base> <log_prefix> -- <cmd...>  (cmd must NOT pass --port: it
# is appended as base, base+1, ... per MODEL_GPUS entry).
_srv_start() {
  local _base="$1"; _SRV_LOGPFX="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  mkdir -p "$(dirname "$_SRV_LOGPFX")"
  _SRV_PIDS=(); _SRV_PORTS=(); _SRV_GPUS=()
  local _i=0 _mg _p _port _log
  IFS=',' read -ra _sgp_arr <<< "$MODEL_GPUS"
  for _mg in "${_sgp_arr[@]}"; do
    _port=$(( _base + _i ))
    _log="${_SRV_LOGPFX}.${_i}.log"; : > "$_log"
    echo "[$(date '+%T')] starting policy server #${_i} on GPU${_mg} (port ${_port}) log=${_log}"
    _SRV_PORTS+=("$_port"); _SRV_GPUS+=("$_mg")
    if [[ "${DRY:-0}" == "1" ]]; then
      printf 'DRY server: CUDA_VISIBLE_DEVICES=%s' "$_mg"
      printf ' %q' "$@" --port "$_port"
      printf '\n'
    else
      env "${_SRV_ENV[@]}" \
        CUDA_VISIBLE_DEVICES="$_mg" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
        "$@" --port "$_port" >> "$_log" 2>&1 &
      _p=$!
      _SRV_PIDS+=("$_p"); _OUR_PIDS+=("$_p")
    fi
    _i=$(( _i + 1 ))
  done
}

# _srv_wait <grep-pattern>: every server must log the pattern. Returns 1 (and
# prints the dead server's log tail) if any server dies. Cold model loads from
# shared storage can take well over 12 minutes, so keep the timeout configurable
# and use a conservative default; a dead process still fails immediately.
_srv_wait() {
  local _pat="$1" _j _all
  [[ "${DRY:-0}" == "1" ]] && return 0
  local _i _timeout_s="${BIDEX_SERVER_READY_TIMEOUT_S:-1800}" _poll_s=2 _tries
  if ! [[ "$_timeout_s" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid BIDEX_SERVER_READY_TIMEOUT_S=$_timeout_s (expected positive integer)" >&2
    return 2
  fi
  _tries=$(( (_timeout_s + _poll_s - 1) / _poll_s ))
  for _i in $(seq 1 "$_tries"); do
    _all=1
    for _j in "${!_SRV_PIDS[@]}"; do
      grep -q "$_pat" "${_SRV_LOGPFX}.${_j}.log" 2>/dev/null || _all=0
      if ! kill -0 "${_SRV_PIDS[$_j]}" 2>/dev/null; then
        echo "[$(date '+%T')] policy server #${_j} (pid ${_SRV_PIDS[$_j]}) DIED — tail of its log:" >&2
        tail -20 "${_SRV_LOGPFX}.${_j}.log" >&2
        return 1
      fi
    done
    [ "$_all" = "1" ] && return 0
    sleep "$_poll_s"
  done
  echo "[$(date '+%T')] policy server readiness TIMEOUT after ${_timeout_s}s (pattern: ${_pat})" >&2
  return 1
}

_srv_stop() {
  local _p
  for _p in "${_SRV_PIDS[@]}"; do kill -9 "$_p" 2>/dev/null; done
  _SRV_PIDS=()
}

_cleanup_between() {
  [[ "${DRY:-0}" == "1" ]] && { _OUR_PIDS=(); return 0; }
  for _p in "${_OUR_PIDS[@]}"; do _kill_tree "$_p" 2>/dev/null; done
  _OUR_PIDS=()
}

run_pi05() {
  echo "############ ($(date '+%T')) pi05 (vec, joint, batched server on MODEL_GPUS=$MODEL_GPUS + render cards=$RENDER_GPUS, N=${ENVS_PER_GPU}/gpu) ############"
  # pi05-only knobs, scoped to this function (were section-2 globals). VEC_OPENPI
  # and OPENPI_PY are exported so local_eval.sh / workers inherit them; PI stays
  # non-exported and is passed via the env-prefix below (PI="$_pi").
  export VEC_OPENPI=1
  # Unified path params come from the launcher (MODEL_PY/MODEL_REPO/MODEL_CKPT;
  # see _check_model_required). Translate to the openpi legacy env names the server + the
  # openpi python lib read; export so local_eval.sh / workers / the server inherit them.
  export OPENPI_PY="$MODEL_PY" OPENPI_REPO="$MODEL_REPO"
  # The published benchmark is BiDex-100, while the pinned OpenPI training repo
  # still registers this checkpoint under its legacy internal config identifier.
  local _batched_config="${BATCHED_CONFIG:-DexSim-All-joint-1}"
  local _batched_ckpt="$MODEL_CKPT"
  if [ -z "$_batched_ckpt" ] || [ ! -d "$_batched_ckpt" ]; then
    echo "  SKIP pi05 (ckpt not found: ${_batched_ckpt:-<unset>}; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  local _mid="${MID_OVERRIDE:-pi05_alljoint_$(_stepk "$_batched_ckpt")}"

  # ── Start the batched policy servers (one per MODEL_GPUS entry) ──
  # Start one server per model GPU, warm it, then let workers select ports
  # round-robin. Build PYTHONPATH only from the two repository roots.
  export PYTHONPATH="$MODEL_REPO/packages/openpi-client/src:$MODEL_REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
  # Leave room for MuJoCo when policy and render roles share a card. In OOD,
  # 25 episodes intentionally own 25 distinct models/renderers; JAX's usual 90%
  # preallocation would reduce auto-concurrency and can exhaust the card.
  local _xla_mem_fraction="${XLA_PYTHON_CLIENT_MEM_FRACTION:-}"
  if [ -z "$_xla_mem_fraction" ]; then
    if [ "$_complete_gpu_overlap" = "1" ]; then
      _xla_mem_fraction=0.70
    else
      _xla_mem_fraction=0.90
    fi
  fi
  _SRV_ENV=("XLA_PYTHON_CLIENT_MEM_FRACTION=$_xla_mem_fraction")
  echo "[$(date '+%T')] OpenPI XLA memory fraction=$_xla_mem_fraction"
  echo "[$(date '+%T')] starting batched policy servers (MODEL_GPUS=$MODEL_GPUS, ports ${OPENPI_SERVER_PORT:-8765}+; config=$_batched_config ckpt=$_batched_ckpt)"
  _srv_start "${OPENPI_SERVER_PORT:-8765}" "$OUT/runs/$_mid/serve_policy_batched" -- \
    "$OPENPI_PY" "$REPO/scripts/evaluation/serve_policy_batched.py" \
      --policy.config "$_batched_config" \
      --policy.dir "$_batched_ckpt" \
      --batch-wait-ms "${BIDEX_BATCH_WAIT_MS:-10}" \
      --max-batch "${BIDEX_MAX_BATCH:-32}"

  # Wait for every server to listen (pre-warm JIT compiles ~2-3 min)
  _srv_wait "batched server listening" || { _srv_stop; return 1; }
  echo "[$(date '+%T')] batched server(s) listening (${_SRV_PORTS[*]})"

  # Warmup: send a dummy infer to trigger JAX JIT compile before workers connect
  # (per server port — each is its own JAX process)
  local _warm_log="$OUT/runs/$_mid/serve_policy_batched.warm.log"; : > "$_warm_log"
  for _i in "${!_SRV_PORTS[@]}"; do
    if [[ "${DRY:-0}" == "1" ]]; then
      echo "DRY warmup: port ${_SRV_PORTS[$_i]}"
      continue
    fi
    echo "[$(date '+%T')] warming server port ${_SRV_PORTS[$_i]} (JIT compile, ~1-2min)..."
    if CUDA_VISIBLE_DEVICES="" "$OPENPI_PY" "$REPO/scripts/evaluation/warm_openpi_server.py" \
         "${OPENPI_SERVER_HOST:-127.0.0.1}" "${_SRV_PORTS[$_i]}" >> "$_warm_log" 2>&1; then
      echo "[$(date '+%T')] server ${_SRV_PORTS[$_i]} warm"
    else
      echo "[$(date '+%T')] WARN: warmup failed on port ${_SRV_PORTS[$_i]} — workers may deadlock on first infer" >&2
    fi
  done

  # Multi-model-GPU assignment for launch_vector (OPENPI_SERVER_PORTS is the
  # contract the openpi worker clients read; PORT stays the first for compat).
  export OPENPI_SERVER_PORTS="$(IFS=,; echo "${_SRV_PORTS[*]}")"
  export OPENPI_SERVER_PORT="${_SRV_PORTS[0]}"

  # ── Run local_eval.sh with the external servers ──
  # BENCHES must be exported for local_eval.sh to read it from the env
  # (it uses ${BENCHES:-$DEFAULT_BENCHES}, not a positional arg).
  OPENPI_SERVER_EXTERNAL=1 \
  MID_OVERRIDE="$_mid" \
  NO_DASHBOARD="${NO_DASHBOARD:-0}" \
  RENDER_GPUS_OVERRIDE="$RENDER_GPUS" \
  BENCHES="$BENCHES" \
    ./scripts/evaluation/local_eval.sh pi05 joint allmixed $EFLAG
  local _eval_rc=$?

  # ── Stop the batched servers (PID-scoped: never pkill other users' servers) ──
  echo "[$(date '+%T')] stopping batched policy server (pids ${_SRV_PIDS[*]})..."
  _srv_stop

  return $_eval_rc
}

# Run one mandatory policy-server backend. A server startup failure is a hard
# evaluation failure: silently switching to an in-process adapter would change
# the policy's action-chunk semantics and invalidate comparisons.
_run_policy_eval() {
  local _backend="$1" _model="$2" _prefix="$3" _py="$4" _ckpt="$5"
  local _mid="$6" _base_port="$7" _ready="$8" _rc=0
  _srv_start "$_base_port" "$OUT/runs/$_mid/${_backend}_server" -- \
    "$_py" "$REPO/scripts/evaluation/serve_policy_batched.py" \
      --backend "$_backend" --ckpt "$_ckpt"
  if ! _srv_wait "$_ready"; then
    echo "[$(date '+%T')] ${_backend} policy server failed to become ready" >&2
    _srv_stop
    return 1
  fi

  local _ports; _ports="$(IFS=,; echo "${_SRV_PORTS[*]}")"
  printf -v "BIDEX_${_prefix}_SERVER_PORTS" '%s' "$_ports"
  printf -v "BIDEX_${_prefix}_SERVER_PORT" '%s' "${_SRV_PORTS[0]}"
  printf -v "BIDEX_${_prefix}_SERVER_HOST" '%s' "127.0.0.1"
  export "BIDEX_${_prefix}_SERVER_PORTS" "BIDEX_${_prefix}_SERVER_PORT" \
         "BIDEX_${_prefix}_SERVER_HOST"
  export BIDEX_VEC_INLINE_RENDER="${BIDEX_VEC_INLINE_RENDER:-0}"
  echo "[$(date '+%T')] ${_backend} server(s) listening (ports $_ports)"

  BENCHES="$BENCHES" MID_OVERRIDE="$_mid" \
    RENDER_GPUS_OVERRIDE="$RENDER_GPUS" \
      ./scripts/evaluation/local_eval.sh "$_model" joint allmixed $EFLAG || _rc=$?
  echo "[$(date '+%T')] stopping ${_backend} policy server (pids ${_SRV_PIDS[*]})..."
  _srv_stop
  return "$_rc"
}

run_galaxea() {
  echo "############ ($(date '+%T')) galaxea (vec, joint, server-client, MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS) ############"
  # Model processes stay on MODEL_GPUS; render-only workers stay on RENDER_GPUS.
  local PY="$MODEL_PY"
  [ -x "$PY" ] || { echo "  SKIP galaxea (python env missing: $PY; set MODEL_PY= in the yml)" >&2; return 0; }
  _ensure_srv_py "$PY"
  export GALAXEA_REPO="$MODEL_REPO"
  export PYTHONPATH="$MODEL_REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
  # g05's vision tower imports flash_attn.cute (FA4), whose own comment limits it to
  # Hopper/Blackwell (Ampere/Ada should use FA2). The galaxea venv ships cute only, so on
  # L20 (Ada) the FA4 path faults (cudaErrorIllegalAddress) at step 0 under full harness
  # load; the forward is robust in isolation. Forcing the torch SDPA fallback g05 already
  # implements makes it robust (default SDPA here). On a true Hopper node set
  # BIDEX_G05_VISION_ATTN=fa4 to keep the fast path.
  export BIDEX_G05_VISION_ATTN="${BIDEX_G05_VISION_ATTN:-sdpa}"
  local GALRUN="$MODEL_CKPT"
  if [[ -z "$GALRUN" ]]; then
    echo "  SKIP galaxea (ckpt not set; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  # Derive the checkpoint step when it is present in the path.
  local _sk; _sk=$(_stepk "$(readlink -f "$GALRUN/last.pt" 2>/dev/null || echo "$GALRUN")")
  local _mid="${MID_OVERRIDE:-galaxea_g05_allmixed_${_sk}}"

  _SRV_ENV=()
  _run_policy_eval galaxea galaxea GALAXEA "$PY" "$GALRUN" "$_mid" \
    "${GALAXEA_SERVER_PORT:-8766}" "galaxea batched server listening"
}

run_lingbot() {
  echo "############ ($(date '+%T')) lingbot (vec, joint, server-client, MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS) ############"
  # Unified path params from the yml (MODEL_PY/MODEL_REPO/MODEL_VLM_PATH/MODEL_CKPT; see
  # _check_model_required). LINGBOT_REPO = MODEL_REPO; QWEN3VL_PATH = MODEL_VLM_PATH (lingbot's
  # deploy loader reads it as the base Qwen3-VL-4B dir); PYTHONPATH uses $MODEL_REPO + $REPO.
  # Policy servers load on MODEL_GPUS; workers only render.
  local PY="$MODEL_PY"
  [ -x "$PY" ] || { echo "  SKIP lingbot (python env missing: $PY; set MODEL_PY= in the yml)" >&2; return 0; }
  _ensure_srv_py "$PY"
  export LINGBOT_REPO="$MODEL_REPO"
  export QWEN3VL_PATH="$MODEL_VLM_PATH"
  export PYTHONPATH="$MODEL_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
  local LINGBOT_RUN="$MODEL_CKPT"
  if [[ -z "$LINGBOT_RUN" ]]; then
    echo "  SKIP lingbot (ckpt not set; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  [[ -e "$LINGBOT_RUN" ]] || { echo "  SKIP lingbot (ckpt missing: $LINGBOT_RUN)" >&2; return 0; }
  local _sk; _sk=$(_stepk "$LINGBOT_RUN")
  local _mid="${MID_OVERRIDE:-lingbot_alljoint_${_sk}}"

  _SRV_ENV=()
  _run_policy_eval lingbot lingbot LINGBOT "$PY" "$LINGBOT_RUN" "$_mid" \
    "${LINGBOT_SERVER_PORT:-8767}" "lingbot batched server listening"
}

run_gr00t() {
  echo "############ ($(date '+%T')) gr00t (vec, joint, server-client, MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS) ############"
  # Unified path params from the yml (MODEL_PY/MODEL_REPO/MODEL_CKPT; see _check_model_required).
  # GR00T_REPO = MODEL_REPO. Each server preloads Franka and Astribot variants.
  local PY="$MODEL_PY"
  [ -x "$PY" ] || { echo "  SKIP gr00t (python env missing: $PY; set MODEL_PY= in the yml)" >&2; return 0; }
  _ensure_srv_py "$PY"
  export GR00T_REPO="$MODEL_REPO"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  local GR00T_RUN="$MODEL_CKPT"
  if [[ -z "$GR00T_RUN" || ! -e "$GR00T_RUN" ]]; then
    echo "  SKIP gr00t (ckpt not set or missing; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  local _sk; _sk=$(_stepk "$GR00T_RUN")
  local _mid="${MID_OVERRIDE:-gr00t_n1_7_allmixed_${_sk}}"

  _SRV_ENV=()
  _run_policy_eval gr00t gr00t GR00T "$PY" "$GR00T_RUN" "$_mid" \
    "${GR00T_SERVER_PORT:-8768}" "gr00t batched server listening"
}
run_dm05() {
  echo "############ ($(date '+%T')) dm05 (vec, joint, server-client, MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS) ############"
  # Unified path params from the yml (MODEL_PY/MODEL_REPO/MODEL_CKPT; see _check_model_required).
  # DM05_REPO = MODEL_REPO; DM05_DATA_PATH + OPENDM_DATA_PATH are DERIVED from MODEL_REPO
  # ($MODEL_REPO/opendm/dataset — the opendm.dataset registration module) — NOT yml params.
  # PYTHONPATH uses $REPO; OpenDM self-inserts via DM05_REPO/_DATA_PATH.
  local PY="$MODEL_PY"
  [ -x "$PY" ] || { echo "  SKIP dm05 (python env missing: $PY; set MODEL_PY= in the yml)" >&2; return 0; }
  _ensure_srv_py "$PY"
  export DM05_REPO="$MODEL_REPO"
  export DM05_DATA_PATH="$MODEL_REPO/opendm/dataset"
  export OPENDM_DATA_PATH="$MODEL_REPO/opendm/dataset"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  local DM05_RUN="$MODEL_CKPT"
  if [[ -z "$DM05_RUN" || ! -e "$DM05_RUN/model.safetensors" ]]; then
    echo "  SKIP dm05 (ckpt not set or no model.safetensors; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  local _sk; _sk=$(_stepk "$DM05_RUN")
  local _mid="${MID_OVERRIDE:-dm05_alljoint_flash_${_sk}}"

  _SRV_ENV=()
  _run_policy_eval dm05 dm05 DM05 "$PY" "$DM05_RUN" "$_mid" \
    "${DM05_SERVER_PORT:-8771}" "dm05 batched server listening"
}

run_xr1() {
  echo "############ ($(date '+%T')) xr1 (vec, joint, server-client, MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS) ############"
  # Unified path params from the yml (MODEL_PY/MODEL_REPO/MODEL_VLM_PATH/MODEL_CKPT; see
  # _check_model_required). XR1_REPO = MODEL_REPO; XR1_VLM_PATH = MODEL_VLM_PATH (the xr1
  # deploy loader reads it for the Qwen3-VL-4B processor).
  local PY="$MODEL_PY"
  [ -x "$PY" ] || { echo "  SKIP xr1 (python env missing: $PY; set MODEL_PY= in the yml)" >&2; return 0; }
  _ensure_srv_py "$PY"
  export XR1_REPO="$MODEL_REPO"
  export XR1_VLM_PATH="$MODEL_VLM_PATH"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  local XR1_RUN="$MODEL_CKPT"
  if [[ -z "$XR1_RUN" || ! -e "$XR1_RUN" ]]; then
    echo "  SKIP xr1 (ckpt not set or missing; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  local _sk; _sk=$(_stepk "$XR1_RUN")
  local _mid="${MID_OVERRIDE:-xr1_alljoint_${_sk}}"

  _SRV_ENV=()
  _run_policy_eval xr1 xr1 XR1 "$PY" "$XR1_RUN" "$_mid" \
    "${XR1_SERVER_PORT:-8769}" "xr1 batched server listening"
}

run_internvla() {
  echo "############ ($(date '+%T')) internvla (vec, joint, server-client, MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS) ############"
  # Unified path params from the yml (MODEL_PY/MODEL_REPO/MODEL_VLM_PATH/MODEL_CKPT; see
  # _check_model_required). INTERNVLA_REPO = MODEL_REPO; INTERNVLA_VLM_PATH = MODEL_VLM_PATH
  # (local Qwen3.5-2B weights for InternVLAA15Backend).
  local PY="$MODEL_PY"
  [ -x "$PY" ] || { echo "  SKIP internvla (python env missing: $PY; set MODEL_PY= in the yml)" >&2; return 0; }
  _ensure_srv_py "$PY"
  export INTERNVLA_REPO="$MODEL_REPO"
  export INTERNVLA_VLM_PATH="$MODEL_VLM_PATH"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  local INTERNVLA_RUN="$MODEL_CKPT"
  if [[ -z "$INTERNVLA_RUN" || ! -e "$INTERNVLA_RUN/model.safetensors" ]]; then
    echo "  SKIP internvla (ckpt not set or no model.safetensors; set MODEL_CKPT= in the yml)" >&2
    return 0
  fi
  local _sk; _sk=$(_stepk "$INTERNVLA_RUN")
  local _mid="${MID_OVERRIDE:-internvla_alljoint_${_sk}}"

  _SRV_ENV=("INTERNVLA_NOISE=${INTERNVLA_NOISE:-}"
            "INTERNVLA_LOG_ACTIONS=${INTERNVLA_LOG_ACTIONS:-0}")
  _run_policy_eval internvla internvla INTERNVLA "$PY" "$INTERNVLA_RUN" "$_mid" \
    "${INTERNVLA_SERVER_PORT:-8770}" "internvla batched server listening"
}

# ────────────────────────── 5. Run selected models sequentially ──────────────────────────

echo "[run_vec] WHICH=(${SELECTED[*]}) GPUS=$GPUS MODEL_GPUS=$MODEL_GPUS RENDER_GPUS=$RENDER_GPUS ID_SEEDS=${ID_SEEDS:-} OOD_SEEDS=${OOD_SEEDS:-} EPS=$EPS ENVS_PER_GPU=$ENVS_PER_GPU"
echo "[run_vec] results -> $OUT/results/<MID>/"
echo "[run_vec] tail live:  tail -f \$OUT/runs/*/*.log"
echo ""

TOTAL_START=$(date +%s)

for model in "${SELECTED[@]}"; do
  MODEL_START=$(date +%s)
  echo ""
  echo "━━━━━━━━━━  $(date '+%F %T')  Starting $model  ━━━━━━━━━━"

  case "$model" in
    pi05)     run_pi05 ;;
    galaxea)  run_galaxea ;;
    gr00t)    run_gr00t ;;
    lingbot)  run_lingbot ;;
    xr1)      run_xr1 ;;
    internvla) run_internvla ;;
    dm05)     run_dm05 ;;
  esac
  RC=$?

  MODEL_ELAPSED=$(( $(date +%s) - MODEL_START ))
  echo "━━━━━━━━━━  $(date '+%F %T')  Finished $model  (rc=$RC, ${MODEL_ELAPSED}s)  ━━━━━━━━━━"

  # Belt-and-braces: kill any lingering eval procs + wait for GPUs to free
  _cleanup_between

  echo ""
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "############ ($(date '+%T')) DONE ${SELECTED[*]} (total ${TOTAL_ELAPSED}s; results under $OUT/results/<MID>/) ############"
echo ""
echo "🏆 View leaderboard:  ${LB_URL}"
echo ""
