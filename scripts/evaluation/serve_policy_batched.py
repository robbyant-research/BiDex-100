#!/usr/bin/env python3
"""Dynamic-batching policy server: openpi (π0.5), galaxea (G0.5), lingbot (v2),
gr00t (N1.7), xr1 (Xiaomi-Robotics-1) and internvla (A1.5) backends.

All backends use the same websocket+msgpack protocol. In disjoint layouts, model
servers run on MODEL_GPUS and MuJoCo workers run on RENDER_GPUS; partial and
complete overlap layouts are also supported explicitly by the launcher.

Architecture per backend:
  openpi  (ffmpeg/`serve_policy_batched` original): dynamic batching across N
          connections -> collects concurrent requests over a short window ->
          ONE batched `_sample_actions`, returns each position's action chunk.
          Uses openpi_client.msgpack_numpy (openpi venv has it) + jax.
  galaxea (G0.5 PolicyInferencer): each connection is one eval worker; one
          message = the worker's DUE obs subset (list[galaxea_obs_dict]) -> the
          server runs inferencer.infer(list) ONCE -> returns {arm:[n,H,14],
          dex:[n,H,44]} right-first. Uses a vendored numpy-aware msgpack bridge
          (no openpi_client dep; the GalaxeaVLA venv has msgpack+numpy+websockets
          but NOT openpi_client).

The openpi imports are DEFERRED to the openpi branch so THIS one file runs in
BOTH venvs (openpi venv for pi05, GalaxeaVLA venv for galaxea).

Env knobs:
  openpi   : --policy.config/--policy.dir/--batch-wait-ms/--max-batch, BIDEX_*
  galaxea  : --ckpt <run_dir>; GALAXEA_REPO, GALAXEA_STATS_PATH; the chunk
             horizon H defaults to the model's action_size (pass --chunk-reuse to
             cap, mirroring the worker's BIDEX_VEC_CHUNK_REUSE).
  lingbot  : --ckpt <hf_ckpt or global_step dir>; LINGBOT_REPO.
  gr00t    : --ckpt <checkpoint-*>; GR00T_REPO (modality configs). One unified
             joint Gr00tPolicy is preloaded.
  xr1      : --ckpt <run dir or epoch=N-step=M.ckpt dir>; XR1_REPO, XR1_VLM_PATH.
  internvla: --ckpt <pretrained_model dir>; INTERNVLA_REPO, INTERNVLA_VLM_PATH.
             One InternVLAA15Backend per robot_type schema (both preloaded).

Usage:
  openpi (OpenPI environment, policy-server GPU):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      $openpi_py scripts/evaluation/serve_policy_batched.py --port 8765 \
        --backend openpi --policy.config <config> --policy.dir <ckpt>
  galaxea (GalaxeaVLA environment, policy-server GPU):
    CUDA_VISIBLE_DEVICES=0 \
      $galaxea_py scripts/evaluation/serve_policy_batched.py --port 8766 \
        --backend galaxea --ckpt "$MODEL_CKPT"
"""
from __future__ import annotations
import argparse, asyncio, copy, logging, os, socket, sys, time
from websockets.exceptions import ConnectionClosed
import numpy as np

try:
    from scripts.evaluation.policy_protocol import packb, unpackb
except ModuleNotFoundError:  # direct execution without the repository on PYTHONPATH
    from policy_protocol import packb, unpackb

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("serve_policy_batched")


def _resolve_benchmark_processor(processor):
    """Return the checkpoint embodiment key and its BiDex data processor.

    Public BiDex-100 checkpoints should use the ``bidex`` embodiment name. Older
    internal checkpoints were trained before the rename and persist ``dexsim``
    in their Hydra config, so accepting that key is required for evaluation.
    """
    processors = getattr(processor, "processors", None)
    if processors is None:
        return "bidex", processor
    for name in ("bidex", "dexsim"):
        if name in processors:
            if name == "dexsim":
                logger.warning(
                    "checkpoint uses legacy processor key 'dexsim'; treating it as BiDex-100"
                )
            return name, processor[name]
    available = list(processors.keys())
    raise KeyError(
        "BiDex-100 processor not found; expected 'bidex' or legacy 'dexsim'. "
        f"Available: {available}"
    )


# --------------------------------------------------------------------------- #
# galaxea backend: load G0.5 PolicyInferencer + a numpy/numpy-aware batched
# Galaxea policy-server inference adapter.
# (the canonical server-side inference path).
# --------------------------------------------------------------------------- #
def _load_galaxea_inferencer(ckpt, cap_h):
    import glob, os, sys
    import torch
    repo = os.environ.get("GALAXEA_REPO", "")
    if repo not in sys.path:
        sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, "src"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from omegaconf import OmegaConf as _OC
    from hydra.utils import instantiate as _instantiate
    from g05.utils.config.config_resolvers import register_default_resolvers as _reg
    _reg()
    from g05.models.g05.inferencer import PolicyInferencer as _Infer
    from g05.utils.data.processor_utils import build_processors as _build_proc
    from g05.utils.data.normalizer import load_dataset_stats_from_json as _load_stats
    from g05.utils.checkpoint.checkpoint_utils import load_state_dict_safely as _load_safe

    ckpt_p = os.path.abspath(ckpt)
    if os.path.isdir(ckpt_p):
        pts = sorted(glob.glob(os.path.join(ckpt_p, "checkpoints", "step_*.pt")),
                     key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        if not pts:
            raise FileNotFoundError(f"no step_*.pt under {ckpt_p}/checkpoints")
        ckpt_file = pts[-1]; run_dir = ckpt_p
    else:
        ckpt_file = ckpt_p; run_dir = os.path.dirname(ckpt_p)
        for _ in range(5):
            if os.path.isfile(os.path.join(run_dir, ".hydra", "config.yaml")):
                break
            run_dir = os.path.dirname(run_dir)
    cfg = _OC.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    _saved = os.getcwd(); os.chdir(repo)
    try:
        policy = _instantiate(cfg.model.model_arch)
        _ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        policy = _load_safe(policy, _ck["model_state_dict"], extra_prefixes=["normalizer."])
        del _ck
        policy = policy.to("cuda").eval()
        if hasattr(policy, "action_tokenizer"):
            policy.action_tokenizer.to("cuda")
        processor = _build_proc(cfg)
        stats = os.environ.get("GALAXEA_STATS_PATH", "") or os.path.join(run_dir, "dataset_stats.json")
        processor.set_normalizer_from_stats(_load_stats(stats))
        processor.eval()
        inferencer = _Infer(policy, processor, device="cuda")
    finally:
        os.chdir(_saved)

    embodiment, sub = _resolve_benchmark_processor(processor)
    action_meta = sub.shape_meta["action"]
    image_keys = [str(m["key"]) for m in sub.shape_meta["images"]]
    num_obs = int(sub.num_obs_steps)
    horizon = int(cfg.data.action_size)
    GALAXEA_CAM = {"head": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}

    def _raw_dim(m):
        rs = m["raw_shape"]
        return int(rs) if isinstance(rs, int) else int(rs[0])

    FLAT = {"right_arm": 7, "left_arm": 7, "right_dex": 22, "left_dex": 22}
    H = min(cap_h, horizon) if cap_h > 0 else horizon

    def flat_chunk(pred, key):
        v = pred[key]
        arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        return arr.reshape(horizon, -1)[:H, :FLAT[key]].astype(np.float32)

    def infer(obs_list):
        """obs_list: list of {images:{head,left_wrist,right_wrist}: HWC uint8,
        arm_joint_position[14], dex_joint_position[44], language:str}. Returns
        list of {'arm':(H,14) f32, 'dex':(H,44) f32} (chunk horizon = H)."""
        obs_dicts = []
        for o in obs_list:
            img = o["images"]
            images = {}
            for env_cam, model_cam in GALAXEA_CAM.items():
                if model_cam not in image_keys:
                    continue
                chw = np.ascontiguousarray(np.transpose(np.asarray(img[env_cam], dtype=np.uint8), (2, 0, 1)))
                images[model_cam] = torch.from_numpy(chw).unsqueeze(0).expand(num_obs, -1, -1, -1).contiguous()
            arm = np.asarray(o["arm_joint_position"], dtype=np.float32).reshape(-1)
            dex = np.asarray(o["dex_joint_position"], dtype=np.float32).reshape(-1)
            state = {
                "right_arm": torch.from_numpy(arm[0:7]).unsqueeze(0).expand(num_obs, -1).contiguous(),
                "left_arm": torch.from_numpy(arm[7:14]).unsqueeze(0).expand(num_obs, -1).contiguous(),
                "right_dex": torch.from_numpy(dex[0:22]).unsqueeze(0).expand(num_obs, -1).contiguous(),
                "left_dex": torch.from_numpy(dex[22:44]).unsqueeze(0).expand(num_obs, -1).contiguous(),
            }
            action = {str(m["key"]): torch.zeros(horizon, _raw_dim(m), dtype=torch.float32) for m in action_meta}
            obs_dicts.append({"images": images, "state": state, "task": str(o.get("language", "")),
                              "action": action, "action_is_pad": torch.ones(horizon, dtype=torch.bool),
                              "state_is_pad": torch.zeros(num_obs, dtype=torch.bool),
                              "image_is_pad": torch.zeros(num_obs, dtype=torch.bool),
                              "idx": 0, "frequency": 25.0, "embodiment": embodiment})
        preds = inferencer.infer(obs_dicts)
        out = []
        for p in preds:
            arm_chunk = np.concatenate([flat_chunk(p, "right_arm"), flat_chunk(p, "left_arm")], axis=1)
            dex_chunk = np.concatenate([flat_chunk(p, "right_dex"), flat_chunk(p, "left_dex")], axis=1)
            out.append({"arm": arm_chunk, "dex": dex_chunk})
        del obs_dicts
        torch.cuda.empty_cache()
        return out

    return infer, H


class GalaxeaBatchedServer:
    """One connection == one render-only eval worker. One message = the worker's
    DUE obs subset (galaxea obs dicts). Server runs ONE inferencer.infer over it
    and replies {arm:[n,H,14], dex:[n,H,44]} right-first."""

    def __init__(self, infer_fn, host, port):
        self.infer_fn = infer_fn
        self.host = host
        self.port = port
        self._loop = None
        self._lock = None   # asyncio.Lock — serializes the VLM forward across workers (see run())
        self._logged_full = False  # log only the FIRST failure's full traceback once
        self._n = 0; self._sum_bs = 0.0
        self._pack = packb; self._unpack = unpackb

    async def _handler(self, ws):
        await ws.send(self._pack({}))  # metadata handshake
        while True:
            try:
                req = self._unpack(await ws.recv())
                obs_list = req["obs"]
                # Serialize the model forward across ALL worker connections: the g05
                # Qwen3.5 backbone stores mutable instance state (MaskHelper.
                # self._mrope_position_deltas — written every prefill, read every
                # decode), the same Qwen3-VL hazard gr00t/xr1 serialize against.
                # Concurrent threads race on it → corrupted attention/position_ids →
                # a CUDA illegal-memory-access that poisons the only CUDA context
                # (a failed run crashed exactly this way: 'NoneType is not a mapping'
                #  -> a CUDA-illegal-access cascade -> vector_rollout broadcasting zero
                #  actions -> the robot "flew away").
                async with self._lock:
                    res = await self._loop.run_in_executor(None, self.infer_fn, obs_list)
                arms = np.stack([r["arm"] for r in res], axis=0) if res else np.zeros((0, 0, 14), np.float32)
                dexs = np.stack([r["dex"] for r in res], axis=0) if res else np.zeros((0, 0, 44), np.float32)
                self._n += 1; self._sum_bs += len(res)
                if self._n % 50 == 0:
                    logger.info("[galaxea stats] batches=%d avg_bs=%.1f", self._n, self._sum_bs/max(1,self._n))
                await ws.send(self._pack({"arm": arms, "dex": dexs}))
            except ConnectionClosed:
                break
            except Exception as e:
                import traceback as _tb
                _full = _tb.format_exc()
                logger.warning("galaxea handler error: %s", e)
                # Log the FIRST failure's full traceback to the SERVER log ONCE.
                # str(e) hides the op ("CUDA error: an illegal memory access"); the
                # full traceback names the real frame — esp. under
                # CUDA_LAUNCH_BLOCKING=1. (The worker client truncates the returned
                # traceback to ~200 chars, so without this the real op is lost.)
                if not self._logged_full:
                    self._logged_full = True
                    logger.error("galaxea handler FIRST-failure full traceback:\n%s", _full)
                try:
                    await ws.send(_full)
                except Exception:
                    pass
                break

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        import websockets.asyncio.server as _server
        logger.info("galaxea batched server listening on 0.0.0.0:%d (ip %s)",
                    self.port, socket.gethostbyname(socket.gethostname()))
        async with _server.serve(self._handler, self.host, self.port,
                                 compression=None, max_size=None) as server:
            await server.serve_forever()


# --------------------------------------------------------------------------- #
# lingbot backend: LingbotVLAv2Server, closed-loop single-step (joint only).   #
# Mirrors the galaxea backend: same numpy-aware msgpack + websocket protocol;  #
# runs in the lingbot (lingbot-vla-v2) venv.                                   #
# --------------------------------------------------------------------------- #
def _resolve_lingbot_robot_config(repo):
    """Return the first available public or legacy joint robot config path."""
    config_dir = os.path.join(str(repo), "configs", "robot_configs")
    candidates = [os.path.join(config_dir, f"{name}.yaml")
                  for name in ("bidex", "dexsim")]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "LingBot joint robot config was not found; tried:\n  - "
        + "\n  - ".join(candidates)
    )


def _load_lingbot_inferencer(ckpt):
    """Load LingbotVLAv2Server ONCE and return
    infer(obs_list)->(arms[N,14], dexs[N,44]) right-first (joint, closed-loop).
    Loads Lingbot once and converts each worker request into a batched forward;
    joint-only (the run-vec launcher uses joint, like galaxea server-client)."""
    import sys as _sys, os as _os, glob as _glob
    import yaml as _yaml
    repo = _os.environ.get("LINGBOT_REPO", "")
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server  # noqa: E402
    from lingbotvla.data.vla_data.utils import FeatureTransform  # noqa: E402

    robot_config_path = _resolve_lingbot_robot_config(repo)
    # Lingbot saves *.safetensors in an `hf_ckpt/` subdir of the step dir; accept
    # either the step dir OR the hf_ckpt dir as --ckpt.
    if not _glob.glob(_os.path.join(ckpt, "*.safetensors")):
        _hf = _os.path.join(ckpt, "hf_ckpt")
        if _os.path.isdir(_hf) and _glob.glob(_os.path.join(_hf, "*.safetensors")):
            ckpt = _hf
    with open(robot_config_path) as _f:
        _rc = _yaml.safe_load(_f)
    _env_norm = _os.environ.get("LINGBOT_NORM_PATH", "").strip()
    if _env_norm:
        norm_path = _env_norm if _os.path.isabs(_env_norm) else _os.path.join(repo, _env_norm)
    else:
        norm_path = _os.path.join(repo, _rc["norm_stats"])

    server = LingbotVLAv2Server(
        path_to_pi_model=ckpt, robot_norm_path=norm_path,
        use_length=50, chunk_ret=False, use_bf16=True, use_compile=False)
    # The constructor loads the checkpoint config; make that checkpoint-native
    # maximum authoritative instead of keeping any transport-side constant.
    server.use_length = int(server.config.chunk_size)
    ft = FeatureTransform(robot_config_path, server.data_config, server.config, server.processor,
                          chunk_size=server.config.chunk_size, norm_stats_path=norm_path)
    server.vla.feature_transform = ft
    server.action_key = ft.org_features["actions"]
    # joint-only: right-first [R7,L7] arm / [R22,L22] dex, single-step (closed-loop).
    arm_key = "actions.arm_joint_position"
    dex_key = "actions.dex_joint_position"

    def infer(obs_list):
        """obs_list: list of {images:{head,left_wrist,right_wrist}: HWC uint8,
        arm_joint_position[14], dex_joint_position[44], language:str}. Returns
        (arms[N, chunk, 14], dexs[N, chunk, 44]) right-first — the FULL action chunk
        (chunk_size=50, the max), NOT truncated to a single step. The client replays
        each chunk for `chunk` control steps via _ChunkReusedVecInfer (galaxea-style
        open-loop reuse -> server queried ~once per 50 steps).
        NB: call server._infer_batch directly so the server always returns the full
        (B, chunk_size, dim) tensor and the client owns per-episode chunk caching."""
        batch = []
        for o in obs_list:
            img = o["images"]
            batch.append({
                "observation.images.head_color": np.asarray(img["head"], dtype=np.uint8),
                "observation.images.left_wrist_color": np.asarray(img["left_wrist"], dtype=np.uint8),
                "observation.images.right_wrist_color": np.asarray(img["right_wrist"], dtype=np.uint8),
                "observation.states.arm_joint_position": np.asarray(o["arm_joint_position"], dtype=np.float32),
                "observation.states.dex_joint_position": np.asarray(o["dex_joint_position"], dtype=np.float32),
                "task": str(o.get("language", "")),
            })
        preds = server._infer_batch(batch)   # dict[action_key] -> (N, chunk, dim)
        arms = np.asarray(preds[arm_key], dtype=np.float32)   # (N, chunk, 14)
        dexs = np.asarray(preds[dex_key], dtype=np.float32)   # (N, chunk, 44)
        if arms.ndim == 2:      # safety: single env -> ensure 3D (N, chunk, dim)
            arms = arms[None, ...]
        if dexs.ndim == 2:
            dexs = dexs[None, ...]
        return arms, dexs

    infer.chunk_size = int(server.config.chunk_size)
    return infer


class LingbotBatchedServer:
    """One connection == one render-only eval worker. One message = the worker's
    obs list (lingbot obs dicts). Server runs ONE LingbotVLAv2Server.infer over it
    and replies {arm:[N,14], dex:[N,44]} right-first (closed-loop single-step).
    Same msgpack+websocket protocol as GalaxeaBatchedServer; infer() returns the
    stacked (arms, dexs) tuple directly (no per-env chunk list)."""

    def __init__(self, infer_fn, host, port):
        self.infer_fn = infer_fn
        self.host = host; self.port = port
        self._loop = None
        self._lock = None   # asyncio.Lock — serializes infer across workers (see run())
        self._n = 0; self._sum_bs = 0.0
        self._pack = packb; self._unpack = unpackb

    async def _handler(self, ws):
        await ws.send(self._pack({}))  # metadata handshake
        while True:
            try:
                req = self._unpack(await ws.recv())
                obs_list = req["obs"]
                # Serialize infer across ALL worker connections: LingbotVLAv2Server
                # is NOT thread-safe, and its first forward one-time-caches image
                # split sizes for that batch — concurrent first-forwards race on
                # the init guard and corrupt the cache. Holding the lock across the
                # executor call guarantees exactly one infer at a time.
                async with self._lock:
                    arms, dexs = await self._loop.run_in_executor(None, self.infer_fn, obs_list)
                self._n += 1; self._sum_bs += len(obs_list)
                if self._n % 50 == 0:
                    logger.info("[lingbot stats] batches=%d avg_bs=%.1f", self._n, self._sum_bs / max(1, self._n))
                await ws.send(self._pack({"arm": arms, "dex": dexs}))
            except ConnectionClosed:
                break
            except Exception as e:
                logger.warning("lingbot handler error: %s", e)
                try:
                    import traceback as _tb
                    await ws.send(_tb.format_exc())
                except Exception:
                    pass
                break

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        import websockets.asyncio.server as _server
        logger.info("lingbot batched server listening on 0.0.0.0:%d (ip %s)",
                    self.port, socket.gethostbyname(socket.gethostname()))
        async with _server.serve(self._handler, self.host, self.port,
                                 compression=None, max_size=None) as server:
            await server.serve_forever()


# --------------------------------------------------------------------------- #
# Generic websocket server for the torch backends (gr00t / xr1 / internvla).   #
# Same numpy-aware msgpack protocol as Galaxea/LingbotBatchedServer: one      #
# connection == one render-only eval worker, one message = the worker's obs   #
# subset (+ request-level `robot` for backends that need the embodiment),     #
# ONE batched infer via the executor, reply {arm:(n,H,14), dex:(n,H,44)}.     #
# All three servers (lingbot, galaxea, gr00t/xr1/internvla) serialize their   #
# forwards behind an asyncio.Lock: each server's _handler holds _lock.        #
# --------------------------------------------------------------------------- #
class SimpleBatchedServer:
    """Generic websocket server for torch backends (gr00t / xr1 / internvla).

    Same numpy-aware msgpack protocol as Galaxea/LingbotBatchedServer: one
    connection == one render-only eval worker, one message = the worker's obs
    subset (+ request-level `robot` for backends that need the embodiment),
    ONE batched infer via the executor, reply {arm:(n,H,14), dex:(n,H,44)}.

    Inference is serialized behind an asyncio.Lock because Qwen3-VL models
    (used by gr00t and xr1) store mutable per-instance state
    (self.rope_deltas) that is NOT thread-safe — concurrent forwards from
    different worker connections race on this state and corrupt each other's
    attention masks, producing the "expanded size must match existing size"
    expand crash.  LingbotBatchedServer and GalaxeaBatchedServer serialize for
    the same reason — g05's Qwen3.5 backbone shares the mutable-instance-state
    hazard (MaskHelper._mrope_position_deltas), so galaxea must serialize too."""
    def __init__(self, infer_fn, host, port, name):
        self.infer_fn = infer_fn
        self.host = host; self.port = port; self.name = name
        self._loop = None
        self._lock = None   # asyncio.Lock — serializes infer across workers
        self._n = 0; self._sum_bs = 0.0
        self._pack = packb; self._unpack = unpackb

    async def _handler(self, ws):
        await ws.send(self._pack({}))  # metadata handshake
        while True:
            try:
                req = self._unpack(await ws.recv())
                obs_list = req["obs"]
                robot = str(req.get("robot", "franka"))
                # Serialize infer across ALL worker connections: Qwen3-VL models
                # (gr00t, xr1) store self.rope_deltas as mutable instance state;
                # concurrent threads race on it and corrupt attention masks.
                async with self._lock:
                    res = await self._loop.run_in_executor(None, self.infer_fn, obs_list, robot)
                arms = np.stack([r["arm"] for r in res], axis=0) if res else np.zeros((0, 0, 14), np.float32)
                dexs = np.stack([r["dex"] for r in res], axis=0) if res else np.zeros((0, 0, 44), np.float32)
                self._n += 1; self._sum_bs += len(res)
                if self._n % 50 == 0:
                    logger.info("[%s stats] batches=%d avg_bs=%.1f", self.name, self._n,
                                self._sum_bs / max(1, self._n))
                await ws.send(self._pack({"arm": arms, "dex": dexs}))
            except ConnectionClosed:
                break
            except Exception as e:
                logger.warning("%s handler error: %s", self.name, e)
                try:
                    import traceback as _tb
                    await ws.send(_tb.format_exc())
                except Exception:
                    pass
                break

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._lock = asyncio.Lock()
        import websockets.asyncio.server as _server
        logger.info("%s batched server listening on 0.0.0.0:%d (ip %s)",
                    self.name, self.port, socket.gethostbyname(socket.gethostname()))
        async with _server.serve(self._handler, self.host, self.port,
                                 compression=None, max_size=None) as server:
            await server.serve_forever()


def _warm_obs():
    """B=1 zero obs at the eval render resolution (360x480, camera_render_scale=0.75)."""
    return {"images": {"head": np.zeros((360, 480, 3), np.uint8),
                       "left_wrist": np.zeros((360, 480, 3), np.uint8),
                       "right_wrist": np.zeros((360, 480, 3), np.uint8)},
            "arm_joint_position": np.zeros(14, np.float32),
            "dex_joint_position": np.zeros(44, np.float32), "language": "warmup"}


# --------------------------------------------------------------------------- #
# gr00t backend: Gr00tPolicy (Isaac-GR00T), one policy per embodiment tag.    #
# GR00T server-side adapter.                                                  #
# due subset stacked along the batch dimension:                              #
# B, policy.get_action runs ONE batched forward, (B, horizon=16, D) decoded.  #
# --------------------------------------------------------------------------- #
_GR00T_CAM = {"head": "head_color", "left_wrist": "left_wrist_color", "right_wrist": "right_wrist_color"}
_GR00T_LANG_KEY = "annotation.human.task_description"


def _load_gr00t_inferencer(ckpt):
    """Load the ONE unified Gr00tPolicy (EmbodimentTag.DEX_SIM_JOINT; franka +
    astribot share the identical 58D schema and train under a single embodiment
    -> one norm-stats entry in statistics.json), PRELOADED so no worker ever
    stalls on a lazy mid-eval model load, and return infer(obs_list, robot) ->
    list of {'arm':(16,14),'dex':(16,44)}. Joint mode only (the launcher submits
    only compatible GR00T settings). Both robots route to this single
    policy regardless of the `robot` arg (kept for API compatibility)."""
    import importlib.util as _ilu
    import torch
    repo = os.environ.get("GR00T_REPO", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    _policies = {}   # tag_name -> Gr00tPolicy (lazily/cached; ~7GB bf16 each on H200)

    def _policy(tag_name):
        if tag_name in _policies:
            return _policies[tag_name]
        if tag_name not in MODALITY_CONFIGS:
            # register the BiDex-100 modality config (idempotent: skip when registered)
            cfg_candidates = [
                os.path.join(repo, "examples", name, f"{tag_name}_config.py")
                for name in ("BiDex-100", "DexSimBench")
            ]
            cfg = next((path for path in cfg_candidates if os.path.isfile(path)), cfg_candidates[0])
            spec = _ilu.spec_from_file_location(f"{tag_name}_config", cfg)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
        pol = Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(tag_name),
                          model_path=ckpt, device="cuda", strict=True)
        _policies[tag_name] = pol
        return pol

    for _tag in ("dex_sim_joint",):   # unified embodiment (franka+astribot) — single policy
        _policy(_tag)
        logger.info("gr00t policy loaded: %s", _tag)

    act_key = "arm_joint_position"
    horizon = 16   # gr00t action_horizon (delta_indices 0-15)

    def infer(obs_list, robot):
        policy = _policy("dex_sim_joint")   # both robots share the unified dex_sim_joint policy
        n = len(obs_list)
        video = {}
        for env_cam, g_cam in _GR00T_CAM.items():
            video[g_cam] = np.stack([np.asarray(o["images"][env_cam], dtype=np.uint8)[None, ...]
                                     for o in obs_list], axis=0)          # (N,1,H,W,3)
        state = {
            "arm_joint_position": np.stack([np.asarray(o["arm_joint_position"], dtype=np.float32)[None, :]
                                            for o in obs_list], axis=0),       # (N,1,14)
            "dex_joint_position": np.stack([np.asarray(o["dex_joint_position"], dtype=np.float32)[None, :]
                                            for o in obs_list], axis=0),       # (N,1,44)
        }
        obs_gr = {"video": video, "state": state,
                  "language": {_GR00T_LANG_KEY: [[str(o.get("language", ""))] for o in obs_list]}}
        chunk, _info = policy.get_action(obs_gr)   # chunk[key] -> (N, 16, D) decoded
        a = np.asarray(chunk[act_key], dtype=np.float32)[:, :horizon, :14]
        d = np.asarray(chunk["dex_joint_position"], dtype=np.float32)[:, :horizon, :44]
        del obs_gr
        torch.cuda.empty_cache()
        return [{"arm": a[i], "dex": d[i]} for i in range(n)]

    return infer


# --------------------------------------------------------------------------- #
# xr1 backend: Xiaomi-Robotics-1 (Qwen3-VL-4B + 60-D flow-matching DiT).      #
# Mirrors run_leaderboard_eval._build_xr1_infer's loader + per-env repack,    #
# but BATCHED over the due subset: per-env chat texts + images through the    #
# Qwen3-VL processor (padding=True) -> ONE model.generate -> (B, 30, 60).     #
# Accepts --ckpt as either the Lightning run dir (config.py + last.ckpt/) or  #
# a specific epoch=N-step=M.ckpt snapshot dir (weights under                 #
# <dir>/checkpoint/mp_rank_00_model_states.pt, config.py from the run dir).   #
# --------------------------------------------------------------------------- #
_XR1_ACT_PARTS = (("right_arm", slice(0, 7)), ("right_dex", slice(7, 29)),
                  ("left_arm", slice(29, 36)), ("left_dex", slice(36, 58)))


def _load_xr1_inferencer(ckpt):
    import sys as _sys, os as _os, glob as _glob
    import torch as _torch
    from PIL import Image as _Image
    repo = _os.environ.get("XR1_REPO", "")
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from mmengine import Config as _MMConfig
    from mibot.models import MIMODEL
    from transformers import AutoProcessor as _AutoProc
    import mibot.utils.io as _io
    from mibot.utils.io import normalize_quantile as _nq, denormalize_action as _denorm

    # ACTION_PARTS at runtime must equal the training override (dims 0..57 active);
    # monkeypatch mibot.utils.io so build_action_mask covers exactly those slots.
    _io.ACTION_PARTS = _XR1_ACT_PARTS

    # ckpt: run dir (config.py + last.ckpt/) or an epoch=N-step=M.ckpt snapshot dir.
    ckpt_dir = str(ckpt).rstrip("/")
    if _glob.glob(_os.path.join(ckpt_dir, "checkpoint", "mp_rank_*_model_states.pt")):
        weights = sorted(_glob.glob(_os.path.join(ckpt_dir, "checkpoint", "mp_rank_00_model_states.pt")))[0]
        run_dir = _os.path.dirname(ckpt_dir)          # config.py lives in the parent run dir
    else:
        run_dir = ckpt_dir
        weights = _os.path.join(run_dir, "last.ckpt", "checkpoint", "mp_rank_00_model_states.pt")
    # mibot.server.deploy.load_model hardcodes <model_dir>/last.ckpt/...; replay it
    # with the weights path above so epoch-snapshot ckpts load faithfully.
    cfg = _MMConfig.fromfile(_os.path.join(run_dir, "config.py"))
    model = MIMODEL.build(cfg.model.params.model).to(_torch.bfloat16)
    _sd = _torch.load(weights, map_location="cpu", mmap=True, weights_only=False)["module"]
    _strip = lambda sd, p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    print(model.load_state_dict(_strip(_sd, "model."), strict=True))
    model = model.eval().to("cuda")

    data = cfg.data.params.train_datasets
    _alen = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
    mean, std = _io.validate_stats(data.mean, data.std, _alen)         # (30,60)
    q01, q99 = _io.validate_quantiles(data.q01, data.q99)               # (1,60)
    action_mask = _io.build_action_mask(_alen)                          # (30,60)
    mean_np, std_np = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    q01_np, q99_np = np.asarray(q01, np.float64), np.asarray(q99, np.float64)
    amask_t = _torch.from_numpy(action_mask).to("cuda").float()
    _mdt = next(model.parameters()).dtype                              # bf16
    device = "cuda"

    vlm_id = _os.environ.get("XR1_VLM_PATH", "Qwen/Qwen3-VL-4B-Instruct")
    processor = _AutoProc.from_pretrained(vlm_id, use_fast=True)

    # Exact client.py prompt template (no <state> token at inference: state is in
    # the payload, not the chat). One conversation per env; padding=True batches.
    def _messages(instr):
        return [{"role": "user", "content": [
            {"type": "text", "text":
             "The following observations are captured from multiple views.\n"
             "# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
             "Generate robot actions for the task:\n" + str(instr) + " /no_cot"},
            {"type": "image"}, {"type": "image"}, {"type": "image"},
        ]}, {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]}]

    _batched_ok = [True]   # flip to serial fallback if generate rejects B>1

    def _infer_serial(obs_list):
        out = []
        for o in obs_list:
            out.extend(_infer_one([o]))
        return out

    def _infer_one(obs_list):
        n = len(obs_list)
        texts = [processor.apply_chat_template(_messages(o.get("language", "")), tokenize=False)
                 for o in obs_list]
        # Align each frame to Qwen3-VL's patch/merge grid BEFORE the processor, exactly
        # like xr1's official client.py / json_dataset.py: resize_image(factor=32,
        # max_pixels=160000) snaps H,W to multiples of 32, then do_resize=False keeps
        # that layout. Without this, raw eval frames (camera_render_scale=0.75 ->
        # 360x480; 360 % 32 != 0) crash the Fast image processor's patches.view().
        images = [[_io.resize_image(
                       _Image.fromarray(np.ascontiguousarray(o["images"][c].astype(np.uint8))),
                       factor=32, max_pixels=160000)
                   for c in ("head", "left_wrist", "right_wrist")] for o in obs_list]
        proc = processor(text=texts, images=images, padding=True, return_dict=True,
                         return_tensors="pt", images_kwargs={"do_resize": False})
        ip = {k: v.to(device) for k, v in proc.items() if _torch.is_tensor(v)}
        # 60-D state from BiDex 58D (right-first) -> right-first 60 layout.
        state60 = np.zeros((n, 60), dtype=np.float32)
        for i, o in enumerate(obs_list):
            arm = np.asarray(o["arm_joint_position"], dtype=np.float32)
            dex = np.asarray(o["dex_joint_position"], dtype=np.float32)
            state60[i, 0:7] = arm[0:7];    state60[i, 7:29] = dex[0:22]
            state60[i, 29:36] = arm[7:14]; state60[i, 36:58] = dex[22:44]
        # normalize_quantile requires value.shape == q01.shape (1,60): per-env rows.
        st_q = np.stack([_nq(state60[i][None, :], q01_np, q99_np) for i in range(n)], axis=0)
        batch = {**ip,
                 "state": _torch.from_numpy(st_q.astype(np.float32)).to(device).view(n, 1, 60).to(_mdt),
                 "action": _torch.zeros((n, _alen, 60), dtype=_mdt, device=device),
                 "prefix_length": 0,   # int, NOT an (n,) tensor: XR1.forward slices action[:, :prefix_length]; a multi-element tensor throws "only integer tensors of a single element can be converted to an index" for B>1. 0 = no prefix at inference.
                 "action_mask": amask_t.unsqueeze(0).expand(n, -1, -1).contiguous().to(_mdt)}
        with _torch.no_grad():
            act = model.generate(batch)                 # (n, 30, 60) normalized
        act = act.float().cpu().numpy()
        act = _denorm(act, mean_np, std_np)             # (n,30,60) RELATIVE-delta (train: action - state_anchor)
        # Training target (bidex_parquet_dataset.py:175) is delta vs the chunk-start
        # state anchor. Recover ABSOLUTE joint targets by adding back the raw per-env
        # state60 captured above (the anchor at chunk-query time). Without this the env
        # receives ~0-rad deltas as absolute ctrl targets -> arms slew to ~0 -> fly away.
        act = act + state60[:, None, :]                 # (n,30,60) absolute
        act = act.astype(np.float32)
        out = []
        for i in range(n):
            a = act[i]                                  # (30, 60)
            arm14 = np.concatenate([a[:, 0:7], a[:, 29:36]], axis=1)    # (30,14) [R7,L7]
            dex44 = np.concatenate([a[:, 7:29], a[:, 36:58]], axis=1)   # (30,44) [R22,L22]
            out.append({"arm": arm14, "dex": dex44})
        return out

    def infer(obs_list, robot):    # robot unused: xr1's 60-D layout is robot-agnostic
        if _batched_ok[0]:
            try:
                return _infer_one(obs_list)
            except Exception:
                import traceback as _tb
                logger.warning("xr1 batched generate failed; falling back to serial "
                               "per-env inference:\n%s", _tb.format_exc())
                _batched_ok[0] = False
        return _infer_serial(obs_list)

    return infer


# --------------------------------------------------------------------------- #
# internvla backend: InternVLA-A1.5 (Qwen3.5-2B VLM + flow-matching expert).  #
# Mirrors run_leaderboard_eval._build_internvla_infer's loader; the repo's    #
# InternVLAA15Backend.infer({"examples":[...]}) natively takes an example     #
# list, so one message == one backend.infer over the worker's due subset.     #
# One backend per robot_type schema (franka->dual_fr3v2_sharpa,               #
# astribot->astribot_s1_fixed_sharpa), both PRELOADED.                        #
# --------------------------------------------------------------------------- #
def _load_internvla_inferencer(ckpt):
    import sys as _sys, os as _os, json as _json
    repo = _os.environ.get("INTERNVLA_REPO", "")
    # InternVLA-A1.5 is a LeRobot fork: its own `src/lerobot` must shadow any
    # installed lerobot, so prepend the repo's src/ (and repo root for evaluation/).
    for _p in (_os.path.join(repo, "src"), repo):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from evaluation.LIBERO.policy_server.backends.policy_backend_internvla_a1_5 import (
        InternVLAA15Backend as _Backend)
    # The trainer's save DROPS the "type" key from config.json; restore it
    # (idempotent) so InternVLAA15Policy.from_pretrained (draccus) loads cleanly.
    _ck_json = _os.path.join(str(ckpt), "config.json")
    if _os.path.isfile(_ck_json):
        _d = _json.load(open(_ck_json))
        if "type" not in _d:
            _d = {"type": "internvla_a1_5", **_d}
            _json.dump(_d, open(_ck_json, "w"), indent=2)

    _ROBO = {"franka": "dual_fr3v2_sharpa", "astribot": "astribot_s1_fixed_sharpa"}
    _backends = {}
    vlm_path = _os.environ.get("INTERNVLA_VLM_PATH", "").strip() or None
    stats_key = _os.environ.get("INTERNVLA_STATS_KEY", "").strip() or None

    def _backend(robot):
        rt = _ROBO.get(robot, "dual_fr3v2_sharpa")
        if rt not in _backends:
            _backends[rt] = _Backend(ckpt_path=str(ckpt), device="cuda", stats_key=stats_key,
                                     robot_type=rt, vlm_model_path=vlm_path,
                                     action_loss_only=True, inference_backend="standard")
        return _backends[rt]

    for _r in ("franka", "astribot"):
        _backend(_r)
        logger.info("internvla backend loaded: robot_type=%s", _ROBO[_r])

    def infer(obs_list, robot):
        be = _backend(robot)
        examples = []
        for o in obs_list:
            img = o["images"]
            examples.append({
                "state": np.concatenate([
                    np.asarray(o["arm_joint_position"], dtype=np.float32),
                    np.asarray(o["dex_joint_position"], dtype=np.float32),
                ]).astype(np.float32),                       # (58,) right-first
                "image": [np.ascontiguousarray(img[c].astype(np.uint8))
                          for c in ("head", "left_wrist", "right_wrist")],  # HWC uint8
                "lang": str(o.get("language", "")),
            })
        out = be.infer({"examples": examples})    # {"actions": (n, chunk_size, 58)}
        acts = np.asarray(out["actions"], dtype=np.float32)   # (n, H, 58) abs denorm
        # Boundary-action probe (debug): log each chunk's first-step action and
        # its jump from the previous chunk's last step, so chunk-boundary
        # discontinuity is visible in the server log. Gated by
        # INTERNVLA_LOG_ACTIONS=1. Records a global running list of boundary
        # jumps to statistically compare pi05 (queues below) vs internvla.
        if os.environ.get("INTERNVLA_LOG_ACTIONS", "0") == "1":
            if not hasattr(infer, "_bnd"):
                infer._bnd = []   # list of boundary_jump_L2
                infer._noise_tag = os.environ.get("INTERNVLA_NOISE", "random")
            cur_last = acts[0, -1, :14] if acts.shape[1] else acts[0, -1, :14]
            if getattr(infer, "_prev_last", None) is not None:
                bjump = float(np.linalg.norm(acts[0, 0, :14] - infer._prev_last))
                infer._bnd.append(bjump)
                logger.info("[iv-action-bnd noise=%s] boundary_jump(arm14 L2)=%.4f n=%d",
                            infer._noise_tag, bjump, len(infer._bnd))
            infer._prev_last = cur_last.copy()
        return [{"arm": a[:, :14], "dex": a[:, 14:58]} for a in acts]

    return infer


# --------------------------------------------------------------------------- #
# openpi backend (original BatchedPolicyServer, openpi imports deferred so this
# file also runs in the galaxea venv without openpi installed).
# --------------------------------------------------------------------------- #
def _buckets(max_batch):
    bs = []
    b = 1
    while b <= max_batch:
        bs.append(b); b <<= 1
    return bs


def _dummy_obs(n):
    z = np.zeros((3, 480, 640), dtype=np.uint8)
    return [{"images": {"cam_high": z, "cam_left_wrist": z, "cam_right_wrist": z},
             "state_arm": np.zeros(14, dtype=np.float32),
             "state_dex": np.zeros(44, dtype=np.float32), "prompt": "warmup"} for _ in range(n)]


class BatchedPolicyServer:  # openpi dynamic-batching server (unchanged behaviour)
    def __init__(self, policy, host, port, batch_wait_ms, max_batch):
        self.policy = policy
        self.host = host; self.port = port
        self.batch_wait_ms = batch_wait_ms
        self.max_batch = max_batch
        self._bucket_sizes = _buckets(max_batch)
        self._q = asyncio.Queue()           # (obs, future)
        self._loop = None
        self._n_batches = 0; self._n_infers = 0
        self._sum_bs = 0; self._sum_fwd_ms = 0.0
        self._checked = False

    def _batch_forward(self, obs_list):
        import jax, jax.numpy as jnp
        from openpi.policies import policy_config as _policy_config  # noqa
        from openpi.models import model as _op_model  # noqa
        t0 = time.monotonic()
        bsz = len(obs_list)
        bucket = min(next((b for b in self._bucket_sizes if b >= bsz), self._bucket_sizes[-1]), self._bucket_sizes[-1])
        padded = list(obs_list)
        while len(padded) < bucket:
            padded.append(copy.deepcopy(obs_list[-1]))
        transformed = [self.policy._input_transform(copy.deepcopy(o)) for o in padded]
        batched = jax.tree.map(lambda *xs: np.stack(list(xs), axis=0), *transformed)
        batched = jax.tree.map(lambda x: jnp.asarray(x), batched)
        self.policy._rng, rng = jax.random.split(self.policy._rng)
        raw = np.asarray(self.policy._sample_actions(
            rng, _op_model.Observation.from_dict(batched), **self.policy._sample_kwargs))
        st = batched.get("state")
        actions = []
        for i in range(bsz):
            sing = {"actions": raw[i]}
            if st is not None:
                sing["state"] = np.asarray(st[i])
            actions.append(self.policy._output_transform(sing))
        fwd_ms = (time.monotonic() - t0) * 1000
        return actions, fwd_ms, bucket

    def _single_infer(self, obs):
        return self.policy.infer(copy.deepcopy(obs))

    def _check_1to1(self, obs_list, actions):
        try:
            errs = []
            for i, o in enumerate(obs_list):
                ref = self._single_infer(o)
                k = "actions_arm" if "actions_arm" in actions[i] else ("actions_ee" if "actions_ee" in actions[i] else None)
                if k is None:
                    continue
                errs.append(float(np.abs(np.asarray(actions[i][k])[0] - np.asarray(ref[k])[0]).max()))
            mx = max(errs) if errs else float("nan")
            logger.info("[1-to-1] max_i |batched[i]-serial_infer(obs_i)| arm[0] = %.5f  (expect ~0.01-0.03 RNG noise; large => crossed)", mx)
        except Exception as e:
            logger.info("[1-to-1] self-check skipped (%s: %s)", type(e).__name__, e)

    async def _batcher(self):
        while True:
            obs_list, futs = [], []
            obs0, fut0 = await self._q.get()
            obs_list.append(obs0); futs.append(fut0)
            deadline = time.monotonic() + self.batch_wait_ms / 1000.0
            while len(obs_list) < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0: break
                try:
                    obs_i, fut_i = await asyncio.wait_for(self._q.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                obs_list.append(obs_i); futs.append(fut_i)
            bsz = len(obs_list)
            actions, fwd_ms, bucket = await self._loop.run_in_executor(None, self._batch_forward, obs_list)
            if not self._checked and bsz >= 2:
                self._checked = True
                self._loop.run_in_executor(None, self._check_1to1, obs_list, actions)
            for fut, act in zip(futs, actions):
                if not fut.done():
                    fut.set_result(act)
            self._n_batches += 1; self._n_infers += bsz; self._sum_bs += bsz; self._sum_fwd_ms += fwd_ms
            if self._n_batches % 20 == 0:
                logger.info("[stats] batches=%d infers=%d avg_bs=%.1f avg_fwd_ms=%.1f",
                            self._n_batches, self._n_infers,
                            self._sum_bs / max(1, self._n_batches),
                            self._sum_fwd_ms / max(1, self._n_batches))

    async def _handler(self, ws):
        from openpi_client import msgpack_numpy as _mn
        packer = _mn.Packer(); unpack = _mn.unpackb
        await ws.send(packer.pack(self.policy.metadata))
        while True:
            try:
                obs = unpack(await ws.recv())
                fut = self._loop.create_future()
                await self._q.put((obs, fut))
                action = await fut
                action = dict(action)
                action["server_timing"] = {"batch_size": 0}
                await ws.send(packer.pack(action))
            except ConnectionClosed:
                break
            except Exception as e:
                logger.warning("handler error: %s", e)
                try:
                    import traceback as _tb
                    await ws.send(_tb.format_exc())
                except Exception:
                    pass
                break

    async def run(self):
        self._loop = asyncio.get_running_loop()
        logger.info("pre-warming JIT for bucket sizes %s (may take a few min)...", self._bucket_sizes)
        for b in self._bucket_sizes:
            await self._loop.run_in_executor(None, self._batch_forward, _dummy_obs(b))
        logger.info("pre-warm done (all bucket shapes compiled)")
        asyncio.create_task(self._batcher())
        import websockets.asyncio.server as _server
        async with _server.serve(self._handler, self.host, self.port,
                                 compression=None, max_size=None) as server:
            logger.info("batched server listening on 0.0.0.0:%s (wait_ms=%s max_batch=%s buckets=%s)",
                        self.port, self.batch_wait_ms, self.max_batch, self._bucket_sizes)
            await server.serve_forever()


def _load_openpi_policy(config_name, ckpt_dir):
    # openpi repo src; prefer OPENPI_REPO (set by the launcher) and fall back to
    # the patched copy on this machine.
    _REPO = os.path.join(os.environ.get("OPENPI_REPO", ""), "src")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    return _policy_config.create_trained_policy(_config.get_config(config_name), ckpt_dir)


# --------------------------------------------------------------------------- #
# dm05 backend: OpenDM DM0.5 (Gemma3+SigLIP VLA + flow-matching action expert,  #
# 58D dual-joint; ckpt trained RELATIVE/delta, opendm restores ABSOLUTE via       #
# loader + per-env data dict: builds the same DM05Exp + _initialize_inference_  #
# runtime and calls inf_cfg._predict(data) ONCE per env (DM05 _predict is       #
# single-example — action_mask=(1,1,D), output [0,:,:], single-sample           #
# input_transform), then stacks list[{'arm':(50,14),'dex':(50,44)}].           #
# SimpleBatchedServer stacks that to (N,H,14)/(N,H,44), right-first. Robot is  #
# ignored: dm05 has a single checkpoint norm-stats profile. BIDEX_DM05_SERVER_   #
# PORT(S)/HOST (default 8771) selects the server; the launcher's run_dm05          #
# passes DM05_REPO / DM05_DATA_PATH (auto-registers the opendm.dataset           #
# bidex_all_joint schema on import) via _SRV_ENV.                            #
# --------------------------------------------------------------------------- #
_DM05_STATE_DESC = ("JOINT",) * 58   # 58D all-joint (arm14 right-first + dex44)


def _resolve_dm05_profile(registry, dataset_override="", robot_override=""):
    """Prefer BiDex registration names while accepting legacy OpenDM entries."""
    requested = dataset_override.strip()
    if requested:
        if requested not in registry:
            raise KeyError(
                f"DM05 dataset {requested!r} is not registered; available={sorted(registry)}")
        dataset_name = requested
    else:
        dataset_name = next(
            (name for name in ("bidex_all_joint", "dexsim_all_joint")
             if name in registry),
            None,
        )
        if dataset_name is None:
            raise RuntimeError(
                "No BiDex DM05 dataset is registered; expected bidex_all_joint "
                "or legacy dexsim_all_joint")
    robot_type = robot_override.strip() or registry[dataset_name].get("robot_type", "")
    if not robot_type:
        raise RuntimeError(f"DM05 dataset {dataset_name!r} has no robot_type")
    return dataset_name, robot_type


def _dm05_pil(arr):
    """HWC uint8 array -> PIL.Image (DM05 PixelTransform expects PIL, no-aug path)."""
    from PIL import Image as _Image
    return _Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))


def _load_dm05_inferencer(ckpt):
    """Load a DM05 BiDex ckpt ONCE and return
    infer(obs_list, robot) -> list of {'arm':(50,14),'dex':(50,44)} right-first
    (joint ABSOLUTE: ckpt trained RELATIVE/delta, opendm restores absolute via
    Denormalize + ActionAbsolute(state+delta); chunk_size=50). The server loops
    inf_cfg._predict once per env because DM05 _predict is single-example.
    robot is ignored (single
    'BiDex' norm-stats profile)."""
    import sys as _sys
    repo = os.environ.get("DM05_REPO", "")
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    # The bidex dataset registration (bidex_all_joint, state_desc=58 joint, 3 cams)
    # must be importable BEFORE opendm.dataset is imported (register runs on import).
    _data_path = os.environ.get("DM05_DATA_PATH", "").strip()
    if _data_path:
        _sys.path.insert(0, _data_path)
        os.environ.setdefault("OPENDM_DATA_PATH", _data_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from opendm.constants.robot import ActionMode as _ActionMode
    from opendm.dataset.register import CONVERSATION_DATA
    from opendm.exp.dm05_exp import (
        DM05Exp as _DM05Exp, DM05ModelConfig as _ModCfg,
        DM05DataConfig as _DataCfg, DM05InferenceConfig as _InfCfg,
        DM05TrainerConfig as _TrCfg,
    )

    ds_name, robot_type = _resolve_dm05_profile(
        CONVERSATION_DATA, os.environ.get("DM05_DATASET", ""),
        os.environ.get("DM05_ROBOT_TYPE", ""))
    out_act_dim = int(os.environ.get("DM05_OUTPUT_ACTION_DIM", "58"))
    _state_desc = list(_DM05_STATE_DESC)

    exp = _DM05Exp(
        task="inference",
        use_lora=False,
        model_config=_ModCfg(model_name_or_path=str(ckpt), chunk_size=50,
                             llm_attn_implementation="eager",
                             vision_attn_implementation="eager",
                             action_attn_implementation="eager"),
        data_config=_DataCfg(dataset_name=ds_name, action_mode=_ActionMode.RELATIVE,
                             norm_stats_default_robot_type=robot_type),
        inference_config=_InfCfg(output_action_dim=out_act_dim,
                                 image_prompts=["Head", "Left wrist", "Right wrist"]),
        trainer_config=_TrCfg(model_max_length=1024),
    )
    exp._initialize_inference_runtime()         # builds model + input/output_transform
    inf_cfg = exp.inference_config

    def infer(obs_list, robot):   # robot ignored: dm05 single checkpoint profile
        out = []
        for o in obs_list:
            state = np.concatenate([
                np.asarray(o["arm_joint_position"], dtype=np.float32),
                np.asarray(o["dex_joint_position"], dtype=np.float32),
            ]).astype(np.float32)                  # (58,) right-first arm14 + dex44
            imgs = [_dm05_pil(np.asarray(o["images"][c], dtype=np.uint8))
                    for c in ("head", "left_wrist", "right_wrist")]
            data = {"images": list(imgs), "state": state,
                    "prompt": str(o.get("language", "")),
                    "meta_data": {"robot_type": robot_type,
                                  "state_desc": _state_desc}}
            chunk = inf_cfg._predict(data)       # np[chunk_size, 58] abs + denorm
            chunk = np.asarray(chunk, dtype=np.float32)
            out.append({"arm": chunk[:, :14], "dex": chunk[:, 14:58]})
        return out

    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", choices=["auto", "openpi", "galaxea", "lingbot",
                                           "gr00t", "xr1", "internvla", "dm05"], default="auto",
                    help="policy server backend. auto=openpi if --policy.config is set, else galaxea (when --ckpt).")
    ap.add_argument("--policy.config", dest="config", default="")
    # Accept BOTH `--ckpt` (galaxea run dir; used by run_galaxea + the docstring) and
    # `--policy.dir` (openpi ckpt dir; used by run_pi05) into the same `args.ckpt`.
    # Keep the two public spellings because OpenPI and the other backends use
    # different launcher conventions.
    ap.add_argument("--ckpt", "--policy.dir", dest="ckpt", default="",
                    help="checkpoint dir: galaxea run dir (--ckpt) or openpi ckpt dir (--policy.dir).")
    ap.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("BIDEX_BATCH_WAIT_MS", "10")))
    ap.add_argument("--max-batch", type=int, default=int(os.environ.get("BIDEX_MAX_BATCH", "64")))
    ap.add_argument("--chunk-reuse", type=int, default=int(os.environ.get("BIDEX_GALAXEA_SERVER_H", "0")),
                    help="galaxea: cap action-chunk horizon H (0 = full model horizon).")
    args = ap.parse_args()
    backend = args.backend
    if backend == "auto":
        backend = "galaxea" if (args.config == "" and args.ckpt != "") else "openpi"
    if backend == "galaxea":
        if not args.ckpt:
            ap.error("galaxea backend needs --ckpt <run_dir>")
        infer_fn, H = _load_galaxea_inferencer(args.ckpt, args.chunk_reuse)
        logger.info("galaxea model loaded; warming ...")
        # Override the g05 vision-tower attention backend (env-gated, default off).
        # g05 vision.py imports flash_attn.cute (FA4, Hopper/Blackwell-only) FIRST
        # and falls back only if it's unimportable. On NON-Hopper GPUs (here L20 /
        # cc 8.9, Ada) the cute backend is unsupported and is unstable under load:
        # its varlen spatial kernel faults cudaErrorIllegalAddress at step 0 of the
        # vectorized eval under multi-render-worker contention (the forward
        # is robust in isolation). BIDEX_G05_VISION_ATTN=sdpa forces the torch SDPA
        # fallback vision.py already implements for exactly this case; =fa2 forces
        # FA2 (needs flash_attn.flash_attn_varlen_func importable; not in this venv).
        _vfa = os.environ.get("BIDEX_G05_VISION_ATTN", "").strip().lower()
        if _vfa:
            try:
                import g05.models.g05.qwen35.vision as _g05vis
                if _vfa == "sdpa":
                    _g05vis._flash_attn_varlen = None
                    _g05vis._flash_attn_backend = None
                    logger.info("g05 vision-tower attention -> torch SDPA fallback (BIDEX_G05_VISION_ATTN=sdpa)")
                elif _vfa == "fa2":
                    from flash_attn import flash_attn_varlen_func as _fa2_varlen
                    _g05vis._flash_attn_varlen = _fa2_varlen
                    _g05vis._flash_attn_backend = "fa2"
                    logger.info("g05 vision-tower attention -> FA2 (BIDEX_G05_VISION_ATTN=fa2)")
            except Exception as _e:
                logger.warning("could not override g05 vision attn backend (%s): %s", _vfa, _e)
        # Warm at B=1 (fast load sanity) AND at the production first-batch size
        # (BIDEX_GALAXEA_WARM_BS, default = ENVS_PER_GPU=50). The eval's STEP-0
        # batch is B≈ENVS_PER_GPU (all envs due at once); a server warmed ONLY at
        # B=1 then hit by that B=50 forward while 7 render MUJoCo workers are
        # active sees the vision-tower varlen flash-attn (g05/qwen35/vision.py
        # _attend_spatial -> _flash_attn_varlen) fault with cudaErrorIllegalAddress
        # on the first real forward (the forward is fully
        # robust in isolation across B/concurrency/content/cold-jump — this is a
        # large-vision-attn workspace ALLOCATED FRESH UNDER GPU contention at
        # step 0). Pre-sizing that workspace at warm — which runs BEFORE the
        # render workers start (server warm/listen precedes local_eval) → no
        # contention → a clean large allocation — makes step 0 REUSE it instead
        # of allocating fresh under load. asyncio Lock (already in place) removed
        # the earlier mrope race; this is the residual harness-only failure.
        _warm_bs = int(os.environ.get("BIDEX_GALAXEA_WARM_BS",
                                      os.environ.get("ENVS_PER_GPU", "50")) or 50)

        def _warm_entry():
            return {"images": {"head": np.zeros((360, 480, 3), np.uint8),        # 360x480 = eval/dataset res (camera_render_scale=0.75)
                               "left_wrist": np.zeros((360, 480, 3), np.uint8),
                               "right_wrist": np.zeros((360, 480, 3), np.uint8)},
                    "arm_joint_position": np.zeros(14, np.float32),
                    "dex_joint_position": np.zeros(44, np.float32), "language": "warmup"}
        infer_fn([_warm_entry()])                              # B=1 fast load sanity
        if _warm_bs > 1:
            infer_fn([_warm_entry() for _ in range(_warm_bs)])  # production spike -> pre-size the vision flash-attn workspace
        logger.info("galaxea warm done (chunk horizon H=%s; warmed B=[1,%d])",
                    H if args.chunk_reuse > 0 else "full", _warm_bs)
        server = GalaxeaBatchedServer(infer_fn, "0.0.0.0", args.port)
        asyncio.run(server.run())
        return
    elif backend == "lingbot":
        if not args.ckpt:
            ap.error("lingbot backend needs --ckpt <hf_ckpt or step dir>")
        infer_fn = _load_lingbot_inferencer(args.ckpt)
        logger.info("lingbot model loaded; full action chunk=%d", infer_fn.chunk_size)
        # NOTE: deliberately NO N=1 warmup forward. LingbotVlaV2 caches its image
        # split_sizes / position_embeddings / cu_seqlens from the FIRST forward's
        # batch dimension (precompute_grid_thw) and reuses them verbatim for every
        # later forward (it never recomputes — verified by probe). A warmup with
        # B=1 would POISON the cache to single-sample split_sizes=[64,64,64]; the
        # first real B=N_envs worker forward then crashes with torch.split_with_sizes
        # ("expects split_sizes to sum exactly to 9600 ... got [64,64,64]"). Instead,
        # the first worker forward sets the cache for the production batch B, and
        # the client ALWAYS sends ENVS_PER_GPU envs (incl. idle dummy obs) -> image
        # grid shape is constant -> the cached split is valid for every forward.
        # LingbotBatchedServer serializes infer (asyncio.Lock) so the cache is
        # initialised by exactly one sequenced forward (no thread-safety / init race).
        server = LingbotBatchedServer(infer_fn, "0.0.0.0", args.port)
        asyncio.run(server.run())
        return
    elif backend == "gr00t":
        if not args.ckpt:
            ap.error("gr00t backend needs --ckpt <checkpoint-*> dir")
        infer_fn = _load_gr00t_inferencer(args.ckpt)
        logger.info("gr00t model loaded; warming (one forward per robot)...")
        infer_fn([_warm_obs()], "franka")
        infer_fn([_warm_obs()], "astribot")
        logger.info("gr00t warm done (chunk horizon H=16)")
        server = SimpleBatchedServer(infer_fn, "0.0.0.0", args.port, "gr00t")
        asyncio.run(server.run())
        return
    elif backend == "xr1":
        if not args.ckpt:
            ap.error("xr1 backend needs --ckpt <run dir or epoch=N-step=M.ckpt dir>")
        infer_fn = _load_xr1_inferencer(args.ckpt)
        logger.info("xr1 model loaded; warming (one B=1 forward)...")
        infer_fn([_warm_obs()], "franka")
        logger.info("xr1 warm done (chunk horizon H=30)")
        server = SimpleBatchedServer(infer_fn, "0.0.0.0", args.port, "xr1")
        asyncio.run(server.run())
        return
    elif backend == "internvla":
        if not args.ckpt:
            ap.error("internvla backend needs --ckpt <pretrained_model dir>")
        infer_fn = _load_internvla_inferencer(args.ckpt)
        logger.info("internvla model loaded; warming (one forward per robot_type)...")
        infer_fn([_warm_obs()], "franka")
        infer_fn([_warm_obs()], "astribot")
        logger.info("internvla warm done (chunk horizon H=chunk_size)")
        server = SimpleBatchedServer(infer_fn, "0.0.0.0", args.port, "internvla")
        asyncio.run(server.run())
        return
    elif backend == "dm05":
        if not args.ckpt:
            ap.error("dm05 backend needs --ckpt <HF checkpoint-*> dir")
        infer_fn = _load_dm05_inferencer(args.ckpt)
        # DM05 _predict is single-example (batch dim hardcoded 1), so EVERY forward
        # is B=1 — a B=1 warmup matches production exactly (no lingbot-style image-grid
        # cache-poisoning risk) and primes the flash_attn/cuda JIT once on the server
        # (NOT on the render workers, which are torch-free in server-client mode).
        logger.info("dm05 model loaded; warming (one B=1 forward, primes JIT)...")
        infer_fn([_warm_obs()], "franka")
        logger.info("dm05 warm done (chunk horizon H=chunk_size=50)")
        server = SimpleBatchedServer(infer_fn, "0.0.0.0", args.port, "dm05")
        asyncio.run(server.run())
        return
    # openpi backend (default, unchanged)
    if not args.config:
        ap.error("openpi backend needs --policy.config <name>")
    policy = _load_openpi_policy(args.config, args.ckpt)
    host_ip = socket.gethostbyname(socket.gethostname())
    logger.info("loaded openpi policy; serving on 0.0.0.0:%d (ip %s)", args.port, host_ip)
    server = BatchedPolicyServer(policy, "0.0.0.0", args.port, args.batch_wait_ms, args.max_batch)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
