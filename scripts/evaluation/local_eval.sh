#!/usr/bin/env bash
# Single-host leaderboard driver for one model checkpoint. Results are grouped
# by model_id so checkpoints/configurations can be compared without overwrites.
#
# Usage:
#   ./local_eval.sh <model> <mode> <scheme> [smoke]
#     model  : pi05 | gr00t | lingbot | galaxea | dm05 | xr1 | internvla
#              (server-client mode is also supported by galaxea/lingbot/gr00t/xr1/internvla;
#               the adapter layer dispatches on the --model name)
#     mode   : joint | camera_ee | robot_ee
#     scheme : allmixed (retained as a positional label for launcher compatibility)
#     smoke  : optional; "smoke" = 1 seed x 1 episode x 1 GPU x Memory only
#
#   Environment overrides: GPUS / SEEDS / EPS / BENCHES / DRY
#
set -uo pipefail

# Pin torch CPU thread pools to 1/worker (also set in run_vec_models_eval.sh; reused
# here so DIRECT callers — e.g. the dm05/lingbot/xr1/internvla eval YAMLs that invoke
# local_eval.sh without going through run_vec — are protected too). See run_vec for the
# full rationale: with OMP_NUM_THREADS unset, torch spawns ~128 CPU threads/rank x 8
# ranks = ~1000 threads on ~132 cores -> the shared MuJoCo EGL render thread is
# descheduled -> render-queue saturates -> sim stalls ~193s/batch. The forward is
# GPU-bound, so 1 thread does not slow inference. Override by pre-exporting your own.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Tests source this file to exercise the final path-selection boundary only.
if [[ "${BIDEX_LOCAL_EVAL_SOURCE_ONLY:-0}" == "1" ]]; then
  return 0
fi

# ---------- basic params ----------
MODEL="${1:-}"; MODE="${2:-}"; SCHEME="${3:-}"; SMOKE="${4:-}"
if [[ -z "$MODEL" || -z "$MODE" || -z "$SCHEME" ]]; then
  echo "usage: $0 <pi05|gr00t|lingbot|galaxea|dm05|xr1|internvla> <joint|camera_ee|robot_ee> allmixed [smoke]" >&2
  exit 2
fi
case "$MODEL" in pi05|gr00t|lingbot|galaxea|dm05|xr1|internvla) ;; *) echo "unknown model: $MODEL" >&2; exit 2 ;; esac
case "$MODE"  in joint|camera_ee|robot_ee) ;; *) echo "unknown mode: $MODE" >&2; exit 2 ;; esac
[[ "$SCHEME" == "allmixed" ]] || { echo "unsupported scheme: $SCHEME (expected allmixed)" >&2; exit 2; }

# ---------- tunable env knobs ----------
# EPS       rollouts per seed (full protocol 25: ID 2 seeds x 25 + OOD 2 seeds x 25 = 100 per task)
# ID_SEEDS  in-domain seeds (default 0,1); OOD_SEEDS  out-of-domain seeds (default 2,3)
# SEEDS     fallback seed list for single-domain runs (overridden by ID/OOD_SEEDS when --domain both)
DEFAULT_BENCHES="Instruction_Following Memory Long_Horizon Limited_Demonstrations Generalization"
if [[ "$SMOKE" == "smoke" ]]; then
  GPUS="${GPUS:-0}"; SEEDS="${SEEDS:-0}"; EPS="${EPS:-1}"; BENCHES="${BENCHES:-Memory}"
  ID_SEEDS="${ID_SEEDS:-0}"; OOD_SEEDS="${OOD_SEEDS:-2}"
else
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; SEEDS="${SEEDS:-0,1}"; EPS="${EPS:-25}"; BENCHES="${BENCHES:-$DEFAULT_BENCHES}"
  ID_SEEDS="${ID_SEEDS:-0,1}"; OOD_SEEDS="${OOD_SEEDS:-2,3}"
fi
MID_OVERRIDE="${MID_OVERRIDE:-}"  # non-empty overrides model_id (separate chunk/config variant row)
DRY="${DRY:-0}"
# ---- vectorized (batched) eval knobs ----
# Policy servers live on MODEL_GPUS; rendering fans out across RENDER_GPUS
# (comma list). Each render card runs ENVS_PER_GPU envs in one batched forward; MAX_RECORD_ENVS
# caps how many envs are recorded (-1 = all).
# OpenPI uses a policy server on MODEL_GPU; render workers use RENDER_GPUS.
# A single shared card falls back to osmesa rendering.
# MODEL_GPUS is set by run_vec_models_eval.sh; worker-port assignment lives in
# that parent launcher. MODEL_GPU remains the first entry for single-card osmesa.
MODEL_GPU="${MODEL_GPU:-0}"
if [[ -n "${MODEL_GPUS:-}" ]]; then MODEL_GPU="${MODEL_GPUS%%,*}"; fi
RENDER_GPUS="${RENDER_GPUS:-1}"
ENVS_PER_GPU="${ENVS_PER_GPU:-8}"
MAX_RECORD_ENVS="${MAX_RECORD_ENVS:--1}"
VEC_DOMAIN="${VEC_DOMAIN:-both}"        # vectorized now supports in_domain + out_of_domain
# Empty/zero means use the full chunk returned by the model. A positive value is
# an explicit debugging cap and may only shorten that model-native horizon.
BIDEX_VEC_CHUNK_REUSE="${BIDEX_VEC_CHUNK_REUSE:-}"
export BIDEX_VEC_CHUNK_REUSE
export BIDEX_MAX_ENVS_PER_GPU="${BIDEX_MAX_ENVS_PER_GPU:-4}"   # per-render-card env cap (prevents frame drops/Xid; render is serial so this doesn't limit throughput)
# (pi05 openpi policy-server knobs — repo / host / port — live in the pi05 branch below; they
# are pi05-only, not shared.)

OUT="${BIDEX_LEADERBOARD_ROOT:?export BIDEX_LEADERBOARD_ROOT in the submit yml}"   # results root is required from the yml (OUT is an internal alias)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # self-resolved from scripts/evaluation (matches run_vec); no REPO env / hardcode needed

# NOTE: cannot use `${X:-...{robot}/sim_dataset}` here — bash treats the `}`
mkdir -p "$OUT/runs" "$OUT/results" "$OUT/mp4s"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$GPUS"

# ---------- per-model config: Python env + adapter ID ----------
case "$MODEL" in
  gr00t)   PY="${MODEL_PY:?}"            # required from the yml (see _check_model_required in run_vec)
           export HF_HUB_OFFLINE=1
           export GR00T_REPO="$MODEL_REPO"
           export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   # gr00t pip-installed; adapter sys.path.inserts GR00T_REPO
           CLI_MODEL=groot;  BASE_ID=gr00t_n1_7 ;;
  pi05)    PY="${MODEL_PY:?}"
           # Serve OpenPI from MODEL_REPO. OPENPI_DATA_HOME remains an optional
           # native OpenPI cache override; otherwise OpenPI uses its default cache.
           export OPENPI_REPO="$MODEL_REPO"
           OPENPI_SERVER_HOST="${OPENPI_SERVER_HOST:-127.0.0.1}"
           OPENPI_SERVER_PORT="${OPENPI_SERVER_PORT:-8765}"
           export OPENPI_SERVER_HOST OPENPI_SERVER_PORT
           export PYTHONPATH="$MODEL_REPO/packages/openpi-client/src:$MODEL_REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
           export HF_HUB_OFFLINE=1
           CLI_MODEL=openpi; BASE_ID=pi05 ;;
  lingbot) PY="${MODEL_PY:?}"
           export HF_HUB_OFFLINE=1
           export LINGBOT_REPO="$MODEL_REPO"
           export QWEN3VL_PATH="$MODEL_VLM_PATH"
           export PYTHONPATH="$MODEL_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
           CLI_MODEL=lingbot; BASE_ID=lingbot_v2 ;;
  galaxea) PY="${MODEL_PY:?}"
           # GalaxeaVLA G0.5: needs g05 on PYTHONPATH ($MODEL_REPO/src) + mujoco (in the venv).
           export HF_HUB_OFFLINE=1
           export GALAXEA_REPO="$MODEL_REPO"
           export PYTHONPATH="$MODEL_REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
           # MuJoCo render (Xid-31-safe): by default do NOT set BIDEX_EGL_DEVICE_ID -> each
           # rank renders on its OWN --gpu (1 EGL context/GPU). Setting =1 for ALL ranks puts N
           # concurrent EGL contexts on one GPU -> NVRM Xid-31.
           CLI_MODEL=galaxea; BASE_ID=galaxea_g05 ;;
  dm05)     PY="${MODEL_PY:?}"
            # OpenDM DM05: needs opendm on sys.path (via DM05_REPO) + the bidex dataset
            # registration under DM05_DATA_PATH/OPENDM_DATA_PATH (DERIVED from MODEL_REPO).
            export HF_HUB_OFFLINE=1
            export DM05_REPO="$MODEL_REPO"
            export DM05_DATA_PATH="$MODEL_REPO/opendm/dataset"
            export OPENDM_DATA_PATH="$MODEL_REPO/opendm/dataset"
            export BIDEX_EGL_DEVICE_ID="${BIDEX_EGL_DEVICE_ID:-1}"
            CLI_MODEL=dm05;    BASE_ID=dm05 ;;
  xr1)      PY="${MODEL_PY:?}"
            # Xiaomi-Robotics-1 (xr1): Qwen3-VL-4B VLM + 60-D DiT. loads via
            # mibot/server/deploy.py in-process + needs the processor (XR1_VLM_PATH = MODEL_VLM_PATH).
            export HF_HUB_OFFLINE=1
            export XR1_REPO="$MODEL_REPO"
            export XR1_VLM_PATH="$MODEL_VLM_PATH"
            export BIDEX_EGL_DEVICE_ID="${BIDEX_EGL_DEVICE_ID:-1}"
            CLI_MODEL=xr1;     BASE_ID=xr1 ;;
  internvla) PY="${MODEL_PY:?}"
            # InternVLA-A1.5 (Qwen3.5-2B VLM + flow-matching action expert); needs the repo
            # src/lerobot (self-inserted via INTERNVLA_REPO) + local Qwen3.5-2B (INTERNVLA_VLM_PATH).
            export HF_HUB_OFFLINE=1
            export INTERNVLA_REPO="$MODEL_REPO"
            export INTERNVLA_VLM_PATH="$MODEL_VLM_PATH"
            export BIDEX_EGL_DEVICE_ID="${BIDEX_EGL_DEVICE_ID:-1}"
            CLI_MODEL=internvla; BASE_ID=internvla ;;
esac
[ -x "$PY" ] || { echo "MISSING python env: $PY" >&2; exit 1; }

# Convert a checkpoint step to the model-id suffix used by the leaderboard.
stepk() {
  local n; n=$(basename "$1" | grep -oE '[0-9]+' | tail -1)
  [ -z "$n" ] && { echo "unk"; return; }
  n=$((10#$n)); echo "$(( (n+500)/1000 ))k"
}

# MODEL_CKPT is validated by the parent launcher and is the single checkpoint
# source for every model. OpenPI additionally needs its control-mode config.
resolve() {
  local cfg=""
  [[ "$MODEL" == "pi05" ]] && cfg="BiDex-All-${MODE}-1"
  printf '%s\n%s\n%s\n' "$MODEL_CKPT" "dual" "$cfg"
}

# ---------- per-(B,R) eval loop ----------
TAG="${MODEL}_${MODE}_${SCHEME}"
RUNS_KEY="${MID_OVERRIDE:-$TAG}"   # nest runs/ per MID_OVERRIDE (like mp4s/<model_id>/) so concurrent runs of the same TAG do not overwrite
RUNS_ROOT="$OUT/runs/$RUNS_KEY"
mkdir -p "$RUNS_ROOT"
FAILS="$RUNS_ROOT/eval_failed.txt"
: > "$FAILS"
echo "================ $(date '+%F %T') START local_eval $TAG  gpus=$GPUS seeds=$SEEDS eps=$EPS benches=[$BENCHES] ================"
cd "$REPO"

# ---------- openpi policy-server lifecycle (pi05 vectorized server-client) ----------
# A single model copy serves vectorized workers. Reuse is keyed by
# (config, checkpoint), so the server remains live across benchmark categories.
OPENPI_SERVER_PID=""
OPENPI_SERVER_KEY=""
EVAL_PID=""
OPENPI_SERVER_LOG="$RUNS_ROOT/openpi_server.log"
start_openpi_server() {
  local cfg="$1" ckpt="$2" key="${1}::${2}"
  [ "$MODEL" = "pi05" ] || return 0
  if [[ "$DRY" == "1" ]]; then
    echo "DRY server: gpu=$MODEL_GPU port=$OPENPI_SERVER_PORT config=$cfg ckpt=$ckpt"
    return 0
  fi
  # Both OpenPI rollout paths use the selected policy server. An external server
  # can be supplied with OPENPI_SERVER_EXTERNAL=1.
  [ "${OPENPI_SERVER_EXTERNAL:-0}" = "1" ] && { echo "[$(date '+%T')] using EXTERNAL openpi server $OPENPI_SERVER_HOST:$OPENPI_SERVER_PORT"; return 0; }
  [ "$key" = "$OPENPI_SERVER_KEY" ] && [ -n "$OPENPI_SERVER_PID" ] && kill -0 "$OPENPI_SERVER_PID" 2>/dev/null && return 0
  stop_openpi_server
  : > "$OPENPI_SERVER_LOG"   # truncate old log so the "listening on" grep below doesn't match a stale line
  echo "[$(date '+%T')] starting openpi server cfg=$cfg ckpt=$ckpt on gpu$MODEL_GPU port $OPENPI_SERVER_PORT" | tee -a "$OPENPI_SERVER_LOG"
  local cfgarg=(); [ -n "$cfg" ] && cfgarg=(--policy.config "$cfg")
  CUDA_VISIBLE_DEVICES="$MODEL_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    HF_HUB_OFFLINE=1 "$PY" "$OPENPI_REPO/scripts/serve_policy.py" --port "$OPENPI_SERVER_PORT" \
    policy:checkpoint "${cfgarg[@]}" --policy.dir "$ckpt" >> "$OPENPI_SERVER_LOG" 2>&1 &
  OPENPI_SERVER_PID=$!
  OPENPI_SERVER_KEY="$key"
  # wait for server to start listening (model load ~1-2min)
  for i in $(seq 1 120); do
    if grep -q "listening on" "$OPENPI_SERVER_LOG" 2>/dev/null; then echo "[$(date '+%T')] server ready (pid $OPENPI_SERVER_PID)"; break; fi
    kill -0 "$OPENPI_SERVER_PID" 2>/dev/null || { echo "[$(date '+%T')] server DIED early — see $OPENPI_SERVER_LOG" >&2; OPENPI_SERVER_PID=""; return 1; }
    sleep 2
  done
  grep -q "listening on" "$OPENPI_SERVER_LOG" 2>/dev/null || { echo "[$(date '+%T')] server readiness TIMEOUT" >&2; OPENPI_SERVER_PID=""; return 1; }
  # Warm JAX JIT compile BEFORE workers connect: the 7-worker x N-env first-infer storm would
  # make the server compile on first infer (blocking asyncio), during which websocket pings time
  # out and disconnect -> worker deadlock. Use a ping_interval=None raw connection to send one
  # dummy infer to trigger the compile, then let workers connect.
  echo "[$(date '+%T')] warming server (JIT compile, 1-2min)..." | tee -a "$OPENPI_SERVER_LOG"
  if CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/evaluation/warm_openpi_server.py" "$OPENPI_SERVER_HOST" "$OPENPI_SERVER_PORT" >> "$OPENPI_SERVER_LOG" 2>&1; then
    echo "[$(date '+%T')] server warm"
  else
    echo "[$(date '+%T')] WARN: warmup failed — workers may deadlock on first infer" >&2
  fi
  return 0
}
stop_openpi_server() {
  [ -n "$OPENPI_SERVER_PID" ] || return 0
  kill "$OPENPI_SERVER_PID" 2>/dev/null; sleep 1
  kill -9 "$OPENPI_SERVER_PID" 2>/dev/null
  OPENPI_SERVER_PID=""; OPENPI_SERVER_KEY=""
}
# Kill a process and all its descendants (recursive pgrep -P).
_kill_tree() {
  local _pid=$1
  local _child
  for _child in $(pgrep -P "$_pid" 2>/dev/null); do _kill_tree "$_child"; done
  kill -9 "$_pid" 2>/dev/null
}
# Stop only processes descended from this launch. Other evaluations may use the
# same repository and must never be selected by a process-name pattern.
_cleanup_all() {
  echo "[$(date '+%T')] cleanup: tearing down this eval tree..." >&2
  if [ -n "$EVAL_PID" ]; then _kill_tree "$EVAL_PID" 2>/dev/null; fi
  stop_openpi_server
}
trap '_cleanup_all; exit 130' INT
trap '_cleanup_all; exit 143' TERM
trap 'stop_openpi_server' EXIT

# OpenPI uses the same vectorized server-client transport as every other model.
if [ "$MODEL" = "pi05" ]; then
  VEC_OPENPI=1
  # RENDER_GPUS = GPUS minus MODEL_GPU (run_pi05 passes RENDER_GPUS_OVERRIDE directly in the
  # server-client layout; with multi-model-GPUs it already excludes all model cards, so it wins)
  if [ -n "${RENDER_GPUS_OVERRIDE:-}" ]; then
    RENDER_GPUS="$RENDER_GPUS_OVERRIDE"
  else
  _rg=""
  IFS=',' read -ra _gpa <<< "$GPUS"
  for _g in "${_gpa[@]}"; do [ "$_g" = "$MODEL_GPU" ] && continue; _rg="${_rg:+$_rg,}$_g"; done
  if [ -z "$_rg" ]; then
    RENDER_GPUS=""        # single card -> launch_vector uses --single-card osmesa
  else
    RENDER_GPUS="$_rg"
  fi
  fi
fi

# The parent launcher starts all policy servers. Workers are always render-only;
# overlap simply means the two separate processes share a physical card.
if [[ "$MODEL" != "pi05" ]]; then
  RENDER_GPUS="${RENDER_GPUS_OVERRIDE:-$GPUS}"
fi

# ---------- eval main loop ----------
# By default the whole (B,R) combo loop runs in a background subshell (logging to the master log)
# while the foreground runs a rich live dashboard — one command and the terminal immediately
# shows per-task / per-env progress bars, no second terminal needed.
# NO_DASHBOARD=1 -> legacy behavior: run sequentially in the foreground, no dashboard (CI/smoke/just-logging).
# DRY=1          -> print commands only, don't instantiate the dashboard.
DASH_PY="$REPO/scripts/evaluation/eval_dashboard.py"
# Render-worker count is the number of entries in RENDER_GPUS.
_nw_gpus="$RENDER_GPUS"; [[ -z "$_nw_gpus" ]] && _nw_gpus="$GPUS"
_nw=$(IFS=','; set -- $_nw_gpus; echo $#)
[[ -z "$_nw" || "$_nw" -lt 1 ]] && _nw=1
# expected settings (the dashboard overall-progress-bar endpoint): bench_robot_mode_{in,out}_domain.
EXPECTED_SETTINGS=""
for _B in $BENCHES; do
  if [[ "$_B" == "Generalization" ]]; then _ROBOTS="astribot franka"; else _ROBOTS="franka"; fi
  for _R in $_ROBOTS; do
    for _DOM in in_domain out_of_domain; do
      s="${_B}_${_R}_${MODE}_${_DOM}"
      EXPECTED_SETTINGS="${EXPECTED_SETTINGS:+$EXPECTED_SETTINGS,}$s"
    done
  done
done
export EXPECTED_SETTINGS BIDEX_CKPT="${BIDEX_CKPT:-}" BIDEX_DASH_TAG="$TAG"
export BIDEX_DASH_MAX_WORKERS="$_nw"
DASH_MASTER_LOG="$RUNS_ROOT/${TAG}.dashboard.master.log"
: > "$DASH_MASTER_LOG"

if [[ "$DRY" == "1" || "${NO_DASHBOARD:-0}" == "1" ]]; then
  _DASH_MODE=foreground
else
  _DASH_MODE=background
fi

_run_all_combos() {
  # Main loop: (B,R) combo -> resolve ckpt/arm/cfg -> start/reuse openpi server -> run_leaderboard_eval.
  # Used by both the foreground path (NO_DASHBOARD) and the background path (default, paired with
  # the foreground dashboard), hence factored into a function.
  # Check GPU availability before loading a model. Orphan termination is opt-in;
  # the default cannot stop another evaluation from the same checkout.
  if [[ "${BIDEX_GPU_GUARD:-warn}" != "off" && "${SKIP_GPU_GUARD:-0}" != "1" ]]; then
    _guard_kill=()
    [[ "${BIDEX_ORPHAN_KILL:-0}" == "1" ]] || _guard_kill+=(--no-kill)
    "$REPO/scripts/evaluation/gpu_idle_check.sh" --gpus "$GPUS" --repo "$REPO" \
      --kill-patterns 'run_leaderboard_eval\.py' --no-check "${_guard_kill[@]}" \
      --label "local-eval-pre" || true
  fi
for B in $BENCHES; do
  if [[ "$B" == "Generalization" ]]; then ROBOTS="astribot franka"; else ROBOTS="franka"; fi
  for R in $ROBOTS; do
    # Recheck between benchmark/robot pairs without terminating other runs.
    if [[ "${BIDEX_GPU_GUARD:-warn}" != "off" && "${SKIP_GPU_GUARD:-0}" != "1" ]]; then
      "$REPO/scripts/evaluation/gpu_idle_check.sh" --gpus "$GPUS" --repo "$REPO" \
        --kill-patterns 'run_leaderboard_eval\.py' --no-check "${_guard_kill[@]}" \
        --label "bench-pre" || true
    fi
    ckpt_arm_cfg=$(resolve) || { echo "SKIP $B/$R (checkpoint resolution failed)" | tee -a "$OUT/eval_skipped.txt"; continue; }
    { read -r CKPT; read -r ARM; read -r CFG; } <<< "$ckpt_arm_cfg"
    [ -e "$CKPT" ] || { echo "SKIP $B/$R (ckpt missing: $CKPT)" | tee -a "$OUT/eval_skipped.txt"; continue; }
    # model_id encodes the step so different steps don't end up on one row (10k/7k/30k each get their own row)
    MID="${BASE_ID}_${SCHEME}_$(stepk "$CKPT")"
    [[ -n "$MID_OVERRIDE" ]] && MID="$MID_OVERRIDE"   # config variants (e.g. chunk50) get their own row
    # Every model uses one vectorized render-only worker per RENDER_GPUS entry.
    VEC_ARG=(--model-gpu "$MODEL_GPU" --render-gpus "$RENDER_GPUS"
             --envs-per-gpu "$ENVS_PER_GPU" --max-record-envs "$MAX_RECORD_ENVS" --auto-envs)
    EVAL_DOMAIN="$VEC_DOMAIN"
    if [[ "$MODEL" == "pi05" ]]; then
      start_openpi_server "$CFG" "$CKPT" || { echo "FAIL $B/$R (openpi server not ready)" | tee -a "$FAILS"; continue; }
      if [ -z "$RENDER_GPUS" ]; then
        VEC_ARG=(--model-gpu "$MODEL_GPU" --single-card --auto-envs
                 --envs-per-gpu "$ENVS_PER_GPU" --max-record-envs "$MAX_RECORD_ENVS")
      fi
    fi
    RUN_NAME="${TAG}_${B}_${R}"
    LOG="$RUNS_ROOT/${RUN_NAME}.log"
    echo "---------------- $(date '+%T') $TAG $B/$R arm=$ARM mid=$MID ckpt=$CKPT vec_envs=$ENVS_PER_GPU render_gpus=$RENDER_GPUS ----------------"
    CMD=("$PY" bidex/eval/run_leaderboard_eval.py
         --model "$CLI_MODEL" --ckpt "$CKPT"
         --benchmark "$B" --robot "$R" --control-mode "$MODE" --domain "$EVAL_DOMAIN"
         --seeds "$SEEDS" --id-seeds "$ID_SEEDS" --ood-seeds "$OOD_SEEDS"
         --episodes-per-seed "$EPS" --arm "$ARM"
         "${VEC_ARG[@]}"
         --model-id "$MID"
         --results-dir "$OUT/results" --mp4-root "$OUT/mp4s"
         --out-dir "$RUNS_ROOT/${RUN_NAME}" --log-file "$LOG")
    if [[ "$DRY" == "1" ]]; then printf 'DRY: %s\n' "${CMD[*]}"; continue; fi
    cmd_rc=0
    "${CMD[@]}" >> "$LOG" 2>&1 || cmd_rc=$?
    # Validate artifacts even when the Python parent returned zero: worker
    # failures used to be swallowed and produced an empty, falsely successful
    # platform job. Both domain JSONs are mandatory for every setting.
    n_json=0
    for dom in in_domain out_of_domain; do
      [ -s "$OUT/results/$MID/${B}_${R}_${MODE}_${dom}.json" ] && n_json=$((n_json+1))
    done
    if [ "$n_json" -ne 2 ]; then
      echo "FAIL $B/$R (exit=$cmd_rc; only $n_json/2 domain json written)" | tee -a "$FAILS"
    elif [ "$cmd_rc" -ne 0 ]; then
      echo "WARN $B/$R (python exit=$cmd_rc but both domain results are present)" | tee -a "$OUT/eval_warn.txt"
    fi
  done
done
echo "================ $(date '+%F %T') DONE local_eval $TAG  (fails see $FAILS) ================"
# ---------- eval result aggregation ----------
# After all (B,R) runs, roll up all result JSONs into one summary.json for quick inspection.
summary_rc=0
if [ -n "$MID" ] && [ -d "$OUT/results/$MID" ]; then
  echo "[$(date '+%T')] summarizing eval results for $MID ..." >&2
  "$REPO/scripts/evaluation/summarize_eval_results.py" --mid "$MID" --results-dir "$OUT/results" || summary_rc=$?
else
  echo "FAIL summary (missing results directory for model_id=${MID:-unset})" | tee -a "$FAILS"
  summary_rc=1
fi
if [ -s "$FAILS" ] || [ "$summary_rc" -ne 0 ]; then
  echo "[eval] evaluation failed; see $FAILS" >&2
  return 1
fi
}  # end _run_all_combos

# ---------- dispatch: background eval + foreground rich dashboard (one command shows live progress) ----------
if [[ "$_DASH_MODE" == "foreground" ]]; then
  _run_all_combos
else
  # A background subshell runs the full combo loop (with the openpi server lifecycle + trap),
  # logging into the master log; the foreground starts the dashboard immediately, reading
  # vec_rank*.log / .progress.json to render per-task/env progress bars.
  _run_all_combos > "$DASH_MASTER_LOG" 2>&1 &
  EVAL_PID=$!
  echo "[$(date '+%T')] eval running in background pid=$EVAL_PID; launching live dashboard..." >&2
  # The dashboard exits after the eval PID is gone and its final refresh completes.
  BIDEX_EVAL_PID="$EVAL_PID" "$PY" "$DASH_PY" \
      --out "$OUT" --tag "$TAG" --model-id "${MID_OVERRIDE:-}" \
      --settings "$EXPECTED_SETTINGS" --max-workers "$_nw" \
      --refresh "${BIDEX_DASH_REFRESH:-1.0}" || true
  # On interruption, stop the background tree owned by this launch.
  _cleanup_all
  wait "$EVAL_PID" 2>/dev/null; EVAL_RC=$?
  echo "[$(date '+%T')] eval finished (rc=$EVAL_RC). master log: $DASH_MASTER_LOG" >&2
  exit "$EVAL_RC"
fi
