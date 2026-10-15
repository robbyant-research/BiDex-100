"""Vectorized (batched) rollout engine for BiDex-100 eval.

Runs N sim envs concurrently and batches their observations into one policy
forward per control step. Normally episodes are pulled from a shared queue. OOD
waves may instead pin one episode to each env because every episode has its own
compiled appearance.

== Single-owner renderer architecture (Xid-31 safe, N envs per render GPU) ==

MuJoCo's ``render()`` calls ``make_current()`` but does not release the context, so
sharing a Renderer across env threads (even under a lock) is EGL-unsafe. All
renderers are therefore created, used, and closed by one dedicated thread.

So this manager runs:
  - ONE dedicated RENDER THREAD that owns all ``mujoco.Renderer`` objects;
    it is the ONLY thread that ever calls GL. Each env's ``_render_hook`` routes
    its ``get_obs()->_render_camera`` to the render thread via a request queue
    (submit (env_idx, cam) -> wait on event -> frame). The hook short-circuits
    each env's ``_init_renderer``.
    Ordinary waves share one renderer per camera. Heterogeneous OOD waves use
    one renderer per env, since a renderer is bound to one MjModel. MuJoCo
    supports switching cameras on that renderer.
  - N ENV-THREADS that do PHYSICS only in parallel (``mj_step`` releases the GIL
    -> real parallel physics across envs). Physics runs under each env's own
    ``env.lock`` (per-env, uncontended); the render thread reads ``env.data``
    WITHOUT ``env.lock`` (safe: the submitting env-thread is the sole mutator and
    is blocked on the render-result event while the render runs).

render-skip (open-loop chunk_reuse=R): the policy only re-queries every R steps,
so rendered obs is only CONSUMED on query steps. We skip rendering (physics-only
``step_physics_only``) on the R-1 intermediate steps. Scoring is physics-based
(SubtaskTracker reads MjData) so success/score are unchanged; only non-queried
video frames are dropped. Recorded episodes (those in the per-episode
``record_episodes`` plan, evenly across seeds) render every step for full-fps
video. The render decision per env per step is
``render_this = query_due(i) or self._recording[i]`` where ``query_due(i)``
(predicts the NEXT infer will re-query) is exposed by the infer backend, and
``_recording[i]`` is the per-env flag for the current (seed, ep) in the plan.

Model-agnostic: the caller supplies ``infer_batch_fn(obs_list, adapter_list) ->
(arms[N,arm_dim], dexs[N,dex_dim])`` (optionally with a ``query_due(i)`` method).
ControlAdapter / Mp4Writer / init-state restore are reused (lazily imported)
from run_leaderboard_eval; scoring comes from its canonical scoring module.
"""
import os
import queue
import threading
import time
import numpy as np


# Lazily resolved to avoid a circular import (run_leaderboard_eval imports this module).
def _helpers():
    from bidex.eval.run_leaderboard_eval import (
        restore_state_from_collection, Mp4Writer,
    )
    from bidex.eval.scoring import episode_score
    return restore_state_from_collection, episode_score, Mp4Writer


class _RenderReq:
    """One render request from an env-thread to the render thread."""
    __slots__ = ("env_idx", "cam", "need_depth", "event", "result")
    def __init__(self, env_idx, cam, need_depth):
        self.env_idx, self.cam, self.need_depth = env_idx, cam, need_depth
        self.event = threading.Event()
        self.result = None  # (color, depth) filled by the render thread


class VectorRolloutManager:
    """Run all episodes of a single task across N concurrent envs with batched
    policy inference. Returns the list of episode results in collection order.

    Args:
        envs: list[N] BiDexEnv (already built for THIS task; reset() reuses them).
        adapters: list[N] ControlAdapter (one per env; warm-start state per env).
        infer_batch_fn: callable(obs_list, adapter_list) -> (arms[N,A], dexs[N,D]).
            Called once per control step from the main thread. May expose
            ``query_due(i) -> bool`` (True iff the NEXT infer will re-query env i,
            used to decide which steps to render); if absent, every step renders.
        episode_queue: queue.Queue of (seed, ep) tuples for this task.
        task_ctx: dict with: suite, task, thash, init_states (np.ndarray|None),
            weights, max_steps, control_mode, robot, arm, t_arm, record_video,
            mp4_path_fn(seed,ep)->str|None, language (str), rank,
            record_episodes (set[(seed, ep)] — per-episode recording plan, evenly
            across seeds), settle_s (float, ~1s physics settle on each reset/episode).
            ``episode_assignments`` optionally pins one (seed, ep) to each env;
            ``heterogeneous_models`` selects one renderer per env.
    """

    def __init__(self, envs, adapters, infer_batch_fn, episode_queue, task_ctx):
        assert len(envs) == len(adapters)
        self.N = len(envs)
        self.envs = envs
        self.adapters = adapters
        self.infer_batch_fn = infer_batch_fn
        self.episode_queue = episode_queue
        self.ctx = task_ctx
        assignments = task_ctx.get("episode_assignments")
        if assignments is not None:
            if len(assignments) != self.N:
                raise ValueError(
                    f"episode_assignments has {len(assignments)} entries for {self.N} envs"
                )
            self._episode_queues = []
            for assignment in assignments:
                q = queue.Queue()
                q.put((int(assignment[0]), int(assignment[1])))
                self._episode_queues.append(q)
        else:
            self._episode_queues = None
        self._heterogeneous_models = bool(task_ctx.get("heterogeneous_models", False))
        # Which (seed, ep) to record — a per-EPISODE plan (built in
        # run_leaderboard_eval._build_record_episodes) replacing the old
        # "first R env indices" record_envs set that biased every video onto
        # seed0. Empty set -> record nothing.
        self._record_episodes = set(task_ctx.get("record_episodes", set()))
        # Per-env "is the CURRENT episode being recorded" flag; toggled in
        # _start_next_episode from the (seed, ep) plan. Drives both writer open
        # AND render-this-step (full-fps video for recorded episodes only).
        self._recording = [False] * self.N

        self.action_event = [threading.Event() for _ in range(self.N)]
        self.obs_event = [threading.Event() for _ in range(self.N)]
        self.action_slot = [None] * self.N        # (arm, dex, is_real, render_this)
        self.obs_slot = [None] * self.N
        self.idle = [True] * self.N
        self.stop = [False] * self.N
        self.threads = []
        self._results = []
        self._results_lock = threading.Lock()
        # Exceptions raised in daemon worker threads must reach the main thread.
        self._thread_error = None
        self._thread_error_lock = threading.Lock()

        # ---- Render plumbing (one owner thread for every GL context) ----
        self._render_queue = queue.Queue()
        self._render_ready = threading.Event()
        self._render_stop = False
        self._shared_renderer = None
        self._shared_so = None
        self._render_thread = None
        # Smoothness metric: peak render-request backlog (envs blocked waiting for
        # the single render thread). A high peak (→ N) means the render thread is
        # the per-card bottleneck (stutter); a small peak means it keeps up. Used
        # by the N-sweep (docs/RENDERING_XID_MAX_ENVS.md) to size max envs/card.
        self._max_qdepth = 0

        # ---- render-thread timing probe (gated by BIDEX_VEC_DEBUG_RENDER, default on) ----
        # Pinpoint where the rollout per-rank wall time goes: whether the per-frame
        # MuJoCo render is starved (CPU/GIL update_scene vs GPU r.render), and how
        # long env-threads block waiting on the shared render thread. Cheap: only
        # perf_counter per frame and one heartbeat print per window. Set
        # BIDEX_VEC_DEBUG_RENDER=0 to silence.
        self._dbg = os.environ.get("BIDEX_VEC_DEBUG_RENDER", "1") == "1"
        # cumulative, render-thread-only writes:
        self._rt_frames = 0           # number of camera frames rendered
        self._rt_ms = 0.0             # total render-thread wall (update_scene + render + depth)
        self._rt_us_ms = 0.0          # update_scene (CPU scene graph)
        self._rt_rd_ms = 0.0          # r.render() (GPU GL)
        self._rt_first_ms = None     # first frame cost; the GL boot probe fires here
        # cumulative env render-wait, env-thread writes under lock:
        self._wait_n = 0
        self._wait_ms = 0.0
        self._wait_lock = threading.Lock()
        # last heartbeat snapshot (to report per-window deltas for cumulative counters):
        self._snap = None
        # Install hooks before any env thread calls get_obs. Each env therefore
        # creates no renderer of its own; the render thread owns and closes the
        # cache selected for this wave.
        for i, e in enumerate(self.envs):
            e._render_hook = self._make_hook(i)
            e._owns_renderer = False

    # ------------------------------------------------------------------ render thread
    def _make_hook(self, i):
        def _hook(cam_name, need_depth):
            req = _RenderReq(i, cam_name, need_depth)
            t0 = time.perf_counter()
            self._render_queue.put(req)
            while not req.event.wait(timeout=1.0):
                # Do not wait forever if the shared render thread has crashed.
                self._raise_thread_error()
            if self._dbg:
                wms = (time.perf_counter() - t0) * 1000.0
                with self._wait_lock:
                    self._wait_n += 1
                    self._wait_ms += wms
            return req.result
        return _hook

    def _make_inline_hook(self, i):
        """Render SYNCHRONOUSLY in the CALLER's thread (no queue, no render thread).

        The single-thread inline path (run_inline) touches GL from exactly one
        thread: there is no shared consumer render-thread that one transient EGL
        fault can kill (which in the threaded path wedges all envs blocked on
        render events), and no N env-threads starve the GL pipeline under the GIL
        (the source of the partial vertical-flip frames). Verified flip-free on
        L20X (the threaded path wedged/flipped at ~step 200 under colocate load).
        """
        import mujoco
        def _hook(cam_name, need_depth):
            env = self.envs[i]
            cid = env.camera_ids[cam_name]
            # OOD envs may have different compiled texture/material models. A
            # Renderer cannot be reused across distinct MjModel instances.
            key = i if self._heterogeneous_models else cam_name
            r = self._cam_renderers.get(key)
            if r is None:
                r = mujoco.Renderer(env.model,
                                    height=env.img_height,
                                    width=env.img_width)
                self._cam_renderers[key] = r
            r.update_scene(env.data, camera=cid, scene_option=self._shared_so)
            color = np.asarray(r.render())
            depth = None
            if need_depth:
                r.enable_depth_rendering()
                try:
                    depth = (np.asarray(r.render()) * 1000).astype(np.uint16)
                finally:
                    r.disable_depth_rendering()
            return color, depth
        return _hook

    def _render_thread_main(self):
        """Own the renderer cache in one thread.

        Homogeneous waves keep one framebuffer per camera. OOD waves keep one
        renderer per env because every appearance is a distinct MjModel; camera
        switching remains local to that env and avoids tripling GPU memory.

        Reads ``env.data`` WITHOUT ``env.lock``: safe because the submitting
        env-thread (sole mutator of that env's data) is blocked on the render
        result while this runs. Acquiring env.lock here would deadlock.
        """
        import mujoco
        # Ordinary waves share a per-camera renderer because every env has the
        # same MjModel. OOD waves use env_idx: their texture/material variants
        # are compiled into different models and must not be mixed.
        cam_renderers = {}

        def renderer_for(env_idx, cam_name):
            env = self.envs[env_idx]
            key = env_idx if self._heterogeneous_models else cam_name
            r = cam_renderers.get(key)
            if r is None:
                r = mujoco.Renderer(env.model, height=env.img_height, width=env.img_width)
                cam_renderers[key] = r
            return r

        so = mujoco.MjvOption()
        so.geomgroup[0] = 0  # hide collision proxies
        so.geomgroup[1] = 1
        so.geomgroup[2] = 1
        self._shared_so = so
        self._cam_renderers = cam_renderers
        self._render_ready.set()
        while True:
            req = self._render_queue.get()
            if req is None:
                break
            # peak backlog = renders queued behind this one (lower bound: qsize
            # after the get under-counts; sample before would over-count under the
            # GIL — use qsize() after get as a cheap lower-bound on backlog).
            _q = self._render_queue.qsize()
            if _q > self._max_qdepth:
                self._max_qdepth = _q
            env = self.envs[req.env_idx]
            cam_id = env.camera_ids[req.cam]
            r = renderer_for(req.env_idx, req.cam)
            if self._dbg:
                _ts0 = time.perf_counter()
            r.update_scene(env.data, camera=cam_id, scene_option=so)
            if self._dbg:
                _tus = time.perf_counter()
            color = np.asarray(r.render())
            depth = None
            if req.need_depth:
                r.enable_depth_rendering()
                try:
                    depth_raw = r.render()
                finally:
                    r.disable_depth_rendering()
                depth = (depth_raw * 1000).astype(np.uint16)
            if self._dbg:
                _tr = time.perf_counter()
                _dt_us = (_tus - _ts0) * 1000.0
                _dt_rd = (_tr - _tus) * 1000.0
                self._rt_frames += 1
                self._rt_us_ms += _dt_us
                self._rt_rd_ms += _dt_rd
                self._rt_ms += _dt_us + _dt_rd
                if self._rt_first_ms is None:
                    # One-time boot probe fires on the FIRST frame, when the EGL
                    # context is current on THIS render thread (mujoco render()
                    # make_current's and does not release). Identifies HW NVIDIA
                    # EGL vs a CPU/software rasterizer fallback (llvmpipe/swrast/
                    # osmesa) — the suspected remote-pod ~40x render slowdown.
                    self._rt_first_ms = _dt_us + _dt_rd
                    rk = self.ctx.get("rank", 0)
                    print(f"[vec rank {rk} | {self.ctx.get('suite','')}/{self.ctx.get('task','')}] "
                          f"render-thread boot: first_frame={self._rt_first_ms:.1f}ms | "
                          f"{self._gl_probe_here()}", flush=True)
            req.result = (color, depth)
            req.event.set()
        for r in cam_renderers.values():
            try:
                r.close()
            except Exception:
                pass
        self._cam_renderers = None

    def _gl_probe_here(self):
        """One-time render-backend probe (called from the render thread on the
        first frame, with the mujoco EGL context current). Strings together to a
        single line. Everything is try/except-guarded so a missing tool never
        breaks a rollout."""
        import glob
        import subprocess
        parts = []
        for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
                  "BIDEX_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES", "EGL_VENDOR_PATH"):
            v = os.environ.get(k)
            parts.append(f"{k}={v if v is not None else 'unset'}")
        # EGL ICD presence (NVIDIA vs software)
        try:
            icds = (glob.glob("/usr/share/glvnd/egl_vendor.d/*.json")
                    + glob.glob("/etc/glvnd/egl_vendor.d/*.json"))
            nv = []
            for p in icds:
                try:
                    txt = open(p).read().lower()
                except Exception:
                    txt = ""
                if "nvidia" in txt or "libegl_nvidia" in txt:
                    nv.append(p)
            parts.append(f"egl_icds={len(icds)} nvidia_icd={'yes' if nv else 'NO'}")
        except Exception:
            parts.append("egl_icds=err")
        # Actual GL driver string (authoritative: 'NVIDIA' vs 'llvmpipe'/'softpipe'/'osmesa')
        try:
            from OpenGL import GL as _gl
            parts.append(f"GL_VENDOR={_gl.glGetString(_gl.GL_VENDOR).decode() if _gl.glGetString(_gl.GL_VENDOR) else None}")
            parts.append(f"GL_RENDERER={_gl.glGetString(_gl.GL_RENDERER).decode() if _gl.glGetString(_gl.GL_RENDERER) else None}")
        except Exception as e:
            parts.append(f"GL_VENDOR=err:{type(e).__name__}:{e}")
        # GPU clocks/power/util (helps catch throttle/low-clock; one shot per rank)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,clocks.gr,clocks.mem,power.draw,utilization.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5).stdout.strip().replace("\n", " | ")
            parts.append(f"nvidia-smi={out[:280]}")
        except Exception:
            parts.append("nvidia-smi=err")
        return " | ".join(str(p) for p in parts)

    def _record_thread_error(self, role, index, exc):
        """Record the first worker failure and wake every possible waiter."""
        with self._thread_error_lock:
            if self._thread_error is None:
                self._thread_error = (role, index, exc)
        for i in range(self.N):
            self.stop[i] = True
            self.action_event[i].set()
            self.obs_event[i].set()
        # Covers failure while the render thread is constructing its context.
        self._render_ready.set()

    def _raise_thread_error(self):
        with self._thread_error_lock:
            error = self._thread_error
        if error is None:
            return
        role, index, exc = error
        where = role if index is None else f"{role} {index}"
        raise RuntimeError(f"vector rollout {where} failed: {exc}") from exc

    def _env_thread_guarded(self, i):
        try:
            self._env_thread(i)
        except BaseException as exc:
            self._record_thread_error("environment thread", i, exc)

    def _render_thread_guarded(self):
        try:
            self._render_thread_main()
        except BaseException as exc:
            self._record_thread_error("render thread", None, exc)

    # ------------------------------------------------------------------ helpers
    def _dummy_obs(self):
        # Match the envs' ACTUAL render res (env.img_height/width). Hardcoding 480x640
        # here while the envs render 360x480 (camera_render_scale=0.75) would feed
        # 480x640 dummies for idle envs alongside 360x480 active frames in the same
        # always_full B=N batch -> a frozen image_grid cache (lingbot) sees a mixed
        # grid and crashes split_with_sizes / produces a res-mismatched forward.
        e0 = self.envs[0] if getattr(self, "envs", None) else None
        h = int(getattr(e0, "img_height", 360))
        w = int(getattr(e0, "img_width", 480))
        zimg = np.zeros((h, w, 3), dtype=np.uint8)
        return {
            "images": {"head": zimg, "left_wrist": zimg, "right_wrist": zimg},
            "arm_joint_position": np.zeros(14, dtype=np.float32),
            "dex_joint_position": np.zeros(44, dtype=np.float32),
            "abs_end_effector": np.zeros(14, dtype=np.float32),
            "language": self.ctx.get("language", ""),
        }

    def _start_next_episode(self, i):
        """Pop next (seed, ep); reset env i; return fresh obs or None if drained.

        Renders route through the shared renderer (env._render_hook is set), so no
        per-env GL context is created and no render_lock is needed here.
        """
        restore, _score, _writer = _helpers()
        ctx = self.ctx
        source_queue = (self._episode_queues[i]
                        if self._episode_queues is not None else self.episode_queue)
        try:
            seed, ep = source_queue.get_nowait()
        except queue.Empty:
            return None
        env = self.envs[i]
        from bidex.eval import ood as oodmod
        ep_seed = oodmod.episode_seed(
            ctx["suite"], ctx["task"], int(seed), ep, "environment")
        obs = env.reset(seed=ep_seed)
        domain = ctx.get("domain", "in_domain")
        if domain == "out_of_domain":
            # OOD per-episode: resample distractors + object offset (deterministic
            # per (seed, ep, task)); no collection init-state restore (mirrors eval_shard).
            placement_rng = oodmod.episode_rng(
                ctx["suite"], ctx["task"], int(seed), ep, "placement")
            # Exactly one canonical sampling pass. Rendering/physics diagnostics
            # must never decide how many RNG draws are consumed across machines.
            # Use this env's private Task instance. Vector envs must not share
            # mutable task bookkeeping (button/order/color-swap state).
            episode_task = env.task
            oodmod.resample_distractors(env, episode_task, placement_rng)
            if ctx.get("ood_offset", 0.0):
                offset_rng = oodmod.episode_rng(
                    ctx["suite"], ctx["task"], int(seed), ep, "object_offset")
                oodmod.object_offset(
                    env, episode_task, offset_rng, ctx["ood_offset"])
            obs = env.get_obs()
        else:
            init_states = ctx["init_states"]
            if init_states is not None and init_states.size:
                rng = oodmod.episode_rng(
                    ctx["suite"], ctx["task"], int(seed), ep,
                    "id.collection_init")
                # Draw exactly once. A previous implementation retried when a
                # rendered head frame looked too dark; that GPU/driver-dependent
                # diagnostic could consume a different number of RNG draws on
                # different computers and select a different initial state.
                idx = int(rng.integers(0, init_states.shape[0]))
                restore(env, init_states[idx])
                obs = env.get_obs()
        # New episode -> drop any cached action chunk so a stale chunk is never
        # carried into the new episode (openpi vectorized chunk-reuse path).
        if hasattr(self.infer_batch_fn, "reset_env"):
            self.infer_batch_fn.reset_env(i)
        self.adapters[i].reset_warmstart(env)
        if ctx["t_arm"] in ("left", "right"):
            o0 = env.get_obs()
            self.adapters[i].hold = {
                "active": ctx["t_arm"],
                "arm14": np.asarray(o0["arm_joint_position"], dtype=np.float64),
                "dex44": np.asarray(o0["dex_joint_position"], dtype=np.float64),
                "ee14": np.asarray(o0.get("abs_end_effector", np.zeros(14)), dtype=np.float64),
            }
        else:
            self.adapters[i].hold = None
        if obs.get("language") and not ctx.get("language"):
            ctx["language"] = obs["language"]   # same task -> same language across all envs
        # ~1s physics settle: let freshly placed freejoint objects relax under
        # gravity (robot pinned at its reset/restore pose). Runs AFTER the OOD
        # black-camera check above already validated a non-black scene and BEFORE
        # the writer opens, so the settle period is excluded from the saved video;
        # the policy's first action is only dispatched once this returns to the
        # caller's lockstep loop. BIDEX_SETTLE_S=0 disables (no-op).
        env.settle(ctx.get("settle_s", 1.0))
        obs = env.get_obs()  # stabilized first frame (rendered via shared-renderer hook)
        rec = (int(seed), ep) in self._record_episodes
        self._recording[i] = rec
        writer = None
        if (ctx["record_video"] and ctx["mp4_path_fn"] is not None and rec):
            writer = _writer(ctx["mp4_path_fn"](int(seed), ep))
            writer.write(obs["images"]["head"])
        self._seed[i] = int(seed)
        self._ep[i] = int(ep)
        self._n_steps[i] = 0
        self._writer[i] = writer
        self._ever_success[i] = False
        return obs

    def _finish_episode(self, i, info):
        _restore, episode_score, _writer = _helpers()
        ctx = self.ctx
        ep_score = episode_score(info.get("subtasks", []), ctx["weights"])
        # success = FINAL-frame env status (all subtasks done AND every must_hold
        # holds_now at the episode end). Recorded envs run to max_steps, so `info`
        # is the last frame -- a stack that held briefly then slid off reads
        # success=False here. The old `or self._ever_success[i]` counted any
        # momentary all-ok frame (e.g. a book stacked then slid off), inflating
        # success_rate for must_hold tasks. Mirrors the run_leaderboard_eval fix.
        ep_score["success"] = bool(info.get("success", False))
        ep_score["timed_out"] = bool(info.get("timed_out", False)) and not ep_score["success"]
        ep_score["grabbed_wrong"] = False
        ep_score["seed"] = self._seed[i]
        ep_score["ep"] = self._ep[i]
        ep_score["time_s"] = float(info.get("time", 0.0))
        ep_score["n_steps"] = self._n_steps[i]
        ep_score["mp4"] = (ctx["mp4_path_fn"](self._seed[i], self._ep[i])
                          if (ctx["record_video"] and ctx["mp4_path_fn"]
                              and (int(self._seed[i]), int(self._ep[i])) in self._record_episodes)
                          else None)
        with self._results_lock:
            self._results.append(ep_score)
        # Per-episode completion line (live per-task progress).
        print(f"[vec rank {ctx.get('rank',0)} | {ctx.get('suite','')}/{ctx.get('task','')}] "
              f"EP done s{self._seed[i]}_ep{self._ep[i]} success={ep_score['success']} "
              f"score={ep_score['score']}/{ep_score.get('max_score','?')} steps={self._n_steps[i]}/{ctx.get('max_steps','?')} "
              f"t={ep_score['time_s']:.0f}s  [done {len(self._results)}/{ctx.get('total_eps','?')}]", flush=True)

    def _env_thread(self, i):
        env = self.envs[i]
        adapter = self.adapters[i]
        ctx = self.ctx
        # Init: start the first episode IN THIS THREAD. get_obs() lazily renders
        # via the shared renderer (hook) -> the GL context lives only in the
        # render thread, never in this env-thread.
        obs = self._start_next_episode(i)
        if obs is None:
            self.idle[i] = True
            self.obs_slot[i] = self._dummy_obs()
        else:
            self.idle[i] = False
            self.obs_slot[i] = obs
        self.obs_event[i].set()

        while not self.stop[i]:
            self.action_event[i].wait()
            self.action_event[i].clear()
            if self.stop[i]:
                break
            arm, dex, is_real, render_this = self.action_slot[i]
            if is_real and not self.idle[i]:
                env_action = adapter.to_env_action(arm, dex, env)
                if render_this:
                    # physics + render (render via hook -> shared renderer thread).
                    obs, r, term, info = env.step(env_action)
                    if self._writer[i] is not None:
                        try:
                            self._writer[i].write(obs["images"]["head"])
                        except Exception:
                            pass
                else:
                    # render-skip: physics + subtask tracking only, no get_obs.
                    # obs stays at its last rendered value (unused by the next
                    # non-query infer; refreshed on the next query-due step).
                    r, term, info = env.step_physics_only(env_action)
                self._n_steps[i] += 1
                self._ever_success[i] = self._ever_success[i] or bool(info.get("success"))
                if self._writer[i] is not None and self._ever_success[i]:
                    # mp4-recorded env whose subtasks have all latched: do NOT
                    # end the episode early -- keep stepping to max_steps so the
                    # saved video covers the full horizon.
                    done = bool(self._n_steps[i] >= ctx["max_steps"])
                else:
                    done = bool(term or info.get("timed_out") or info.get("success")
                                or self._n_steps[i] >= ctx["max_steps"])
            else:
                done = False
            if done:
                if self._writer[i] is not None:
                    try:
                        self._writer[i].close()
                    except Exception:
                        pass
                    self._writer[i] = None
                self._finish_episode(i, info)
                obs = self._start_next_episode(i)
                if obs is None:
                    self.idle[i] = True
                    self.obs_slot[i] = self._dummy_obs()
                else:
                    self.idle[i] = False
                    self.obs_slot[i] = obs
            elif self.idle[i]:
                self.obs_slot[i] = self._dummy_obs()
            elif render_this:
                # Refresh with the freshly-rendered obs so the next policy query
                # sees the CURRENT (post-action) state. Without this, obs_slot stays
                # pinned at the episode's first frame -> every fetch re-plans from
                # the reset pose -> the arm "approaches then retreats" each chunk.
                self.obs_slot[i] = obs
            # else (render-skip, not done, not idle): keep the last rendered obs —
            # a stale obs is unused on non-query steps and is refreshed at the next
            # query-due (render) step.
            self.obs_event[i].set()
        # Stop: close this env IN ITS OWN THREAD (thread-affine cleanup of any
        # per-env state). The shared renderer is NOT closed here (_owns_renderer=False);
        # the manager closes it once in the render thread after all envs join.
        try:
            env.close()
        except Exception:
            pass

    def run(self):
        """Drive the lockstep loop until all episodes are done. Returns episode results."""
        if os.environ.get("BIDEX_VEC_INLINE_RENDER", "0") == "1":
            return self.run_inline()
        self._seed = [0] * self.N
        self._ep = [0] * self.N
        self._n_steps = [0] * self.N
        self._writer = [None] * self.N
        # Latched "all subtasks succeeded at least once" per env. mp4-recorded
        # envs keep rolling to max_steps after success (full-horizon videos);
        # scores are latch-based so post-success physics cannot undo them.
        self._ever_success = [False] * self.N

        # Start the render thread FIRST and wait for it to build the shared
        # renderer, so env-threads' first get_obs (via hook) has a renderer ready.
        self._render_thread = threading.Thread(target=self._render_thread_guarded, daemon=True)
        self._render_thread.start()
        self._render_ready.wait()
        self._raise_thread_error()

        self.threads = [threading.Thread(target=self._env_thread_guarded, args=(i,), daemon=True)
                        for i in range(self.N)]
        for t in self.threads:
            t.start()
        # Collect init obs from all envs.
        for i in range(self.N):
            self.obs_event[i].wait()
            self.obs_event[i].clear()
            self._raise_thread_error()

        rank = self.ctx.get("rank", 0)
        suite = self.ctx.get("suite", "")
        task = self.ctx.get("task", "")
        max_steps = self.ctx.get("max_steps", 0)
        ctrl_step = 0
        hb_every = max(1, int(os.environ.get("BIDEX_VEC_HB_STEPS", "200")))  # heartbeat interval (control steps)
        next_hb = hb_every
        qd = getattr(self.infer_batch_fn, "query_due", None)  # None -> render every step
        # per-heartbeat phase accumulators (main-thread only; reset each heartbeat)
        _hb_inf_ms = 0.0    # time in infer_batch_fn (real query-days ~5s; replay-days ~0)
        _hb_phase_ms = 0.0  # time waiting on env threads to step+render (the 1-card choke)
        _hb_inf_n = 0
        while True:
            obs_list = [self.obs_slot[i] for i in range(self.N)]
            adapter_list = [self.adapters[i] for i in range(self.N)]
            if self._dbg:
                _t_inf0 = time.perf_counter()
            try:
                arms, dexs = self.infer_batch_fn(obs_list, adapter_list)
            except Exception as e:
                print(f"[vec rank {rank}] infer_batch failed ({e}); broadcasting zero actions", flush=True)
                arms = np.zeros((self.N, 14), dtype=np.float32)
                dexs = np.zeros((self.N, 44), dtype=np.float32)
            if self._dbg:
                _hb_inf_ms += (time.perf_counter() - _t_inf0) * 1000.0
                _hb_inf_n += 1
                _t_phase0 = time.perf_counter()
            for i in range(self.N):
                is_real = not self.idle[i]
                # query_due(i) is the post-call chunk state -> True iff the NEXT
                # infer will re-query env i; if so, render this step so the obs
                # fed to that next query is fresh. Recorded envs render every step.
                render_this = is_real and ((qd(i) if qd else True) or self._recording[i])
                self.action_slot[i] = (np.asarray(arms[i], dtype=np.float32),
                                       np.asarray(dexs[i], dtype=np.float32),
                                       is_real, render_this)
                self.action_event[i].set()
            for i in range(self.N):
                self.obs_event[i].wait()
                self.obs_event[i].clear()
                self._raise_thread_error()
            if self._dbg:
                _hb_phase_ms += (time.perf_counter() - _t_phase0) * 1000.0
            ctrl_step += 1
            # heartbeat: log per-env episode/step progress every hb_every steps
            if ctrl_step >= next_hb:
                next_hb = ctrl_step + hb_every
                parts = []
                n_active = 0
                for i in range(self.N):
                    if self.idle[i]:
                        parts.append(f"e{i}:idle")
                    else:
                        n_active += 1
                        parts.append(f"e{i}:s{self._seed[i]}_ep{self._ep[i]} {self._n_steps[i]}/{max_steps}")
                print(f"[vec rank {rank} | {suite}/{task}] step {ctrl_step} | active {n_active}/{self.N} | "
                      + " ".join(parts), flush=True)
                if self._dbg:
                    # Per-window rollout-time accountant: splits each ~hb_every
                    # control-steps window into inference ns the env-thread step+render phase *
                    # render-thread per-frame cost (CPU update_scene vs GPU render)
                    # & env-thread render-wait. qsize == sustained backpressure.
                    rf, rms = self._rt_frames, self._rt_ms
                    rus, rrd = self._rt_us_ms, self._rt_rd_ms
                    wn, wms = self._wait_n, self._wait_ms
                    snap = self._snap if self._snap is not None else \
                        {"rf": rf, "rms": rms, "rus": rus, "rrd": rrd, "wn": wn, "wms": wms}
                    d_rf = max(0, rf - snap["rf"]); d_rms = max(0.0, rms - snap["rms"])
                    d_rus = max(0.0, rus - snap["rus"]); d_rrd = max(0.0, rrd - snap["rrd"])
                    d_wn = max(0, wn - snap["wn"]); d_wms = max(0.0, wms - snap["wms"])
                    self._snap = {"rf": rf, "rms": rms, "rus": rus, "rrd": rrd, "wn": wn, "wms": wms}
                    avg_f = (d_rms / d_rf) if d_rf else 0.0
                    avg_w = (d_wms / d_wn) if d_wn else 0.0
                    try:
                        _qs = self._render_queue.qsize()
                    except Exception:
                        _qs = -1
                    print(f"[vec rank {rank} | {suite}/{task}] dbg hb: inferences={_hb_inf_n} "
                          f"infer={_hb_inf_ms:.0f}ms step+render={_hb_phase_ms:.0f}ms | "
                          f"rendering: frames={d_rf} avgT={avg_f:.1f}ms "
                          f"(update_scene={d_rus:.0f}ms render={d_rrd:.0f}ms) | "
                          f"envs render-wait: n={d_wn} ms={d_wms:.0f} (avg={avg_w:.1f}ms) | "
                          f"qsize={_qs} qdepth_max={self._max_qdepth}", flush=True)
                    _hb_inf_ms = 0.0
                    _hb_phase_ms = 0.0
                    _hb_inf_n = 0
            if all(self.idle):
                break
        for i in range(self.N):
            self.stop[i] = True
            self.action_slot[i] = (None, None, False, False)
            self.action_event[i].set()
        for t in self.threads:
            t.join(timeout=5.0)
        # All env-threads joined -> safe to stop the render thread (it closes the
        # shared renderer, the last GL touch, in the only thread that touched it).
        self._render_queue.put(None)
        self._render_thread.join(timeout=5.0)
        print(f"[vec rank {rank} | {suite}/{task}] max render-queue depth = {self._max_qdepth} "
              f"(over N={self.N} envs; 0 = render thread kept up, ~N = backlogged/stutter)", flush=True)
        # Close any still-open writers (safety).
        for w in getattr(self, "_writer", []):
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        return list(self._results)

    def run_inline(self):
        """Single-thread inline-render rollout (Xid-31-free, no render thread).

        ``BIDEX_VEC_INLINE_RENDER=1`` drives the whole task in one thread. Env
        steps and all renderer objects stay in that thread; there is no render
        thread and no per-env thread. The threaded path's single GL consumer could die on one
        transient EGL fault under load, wedging every env blocked on its render
        event (silent, ~step 200); and N env-threads starving that one GL thread
        under the GIL produced partial vertical-flip frames. Single-thread inline
        has neither failure mode (one thread owns the GL, no shared-consumer to
        die, no GIL starvation) — verified flip-free on L20X. The cost is losing
        cross-env physics parallelism (mj_step no longer overlaps across envs);
        throughput is sustained by the ONE batched infer per step over all N envs.
        """
        import mujoco
        self._seed = [0] * self.N
        self._ep = [0] * self.N
        self._n_steps = [0] * self.N
        self._writer = [None] * self.N
        self._ever_success = [False] * self.N
        self._max_qdepth = 0   # unused (no queue) but reset for the stats line

        # per-camera Renderer cache (inline path): built lazily by the inline hook.
        self._cam_renderers = {}
        self._shared_so = None  # set below
        so = mujoco.MjvOption()
        so.geomgroup[0] = 0  # hide collision proxies
        so.geomgroup[1] = 1
        so.geomgroup[2] = 1
        self._shared_so = so
        self._render_ready.set()  # inline: no separate render thread, but harmless
        # re-install INLINE hooks (render in this thread, no queue) before any get_obs.
        for i, e in enumerate(self.envs):
            e._render_hook = self._make_inline_hook(i)
            e._owns_renderer = False

        rank = self.ctx.get("rank", 0)
        suite = self.ctx.get("suite", "")
        task = self.ctx.get("task", "")
        max_steps = self.ctx.get("max_steps", 0)

        # init episodes (each _start_next_episode → env.reset → get_obs → inline render)
        self.obs_slot = [None] * self.N
        self.idle = [True] * self.N
        for i in range(self.N):
            obs = self._start_next_episode(i)
            if obs is None:
                self.idle[i] = True
                self.obs_slot[i] = self._dummy_obs()
            else:
                self.idle[i] = False
                self.obs_slot[i] = obs

        ctrl_step = 0
        hb_every = max(1, int(os.environ.get("BIDEX_VEC_HB_STEPS", "200")))
        next_hb = hb_every
        qd = getattr(self.infer_batch_fn, "query_due", None)
        while True:
            obs_list = [self.obs_slot[i] for i in range(self.N)]
            adapter_list = [self.adapters[i] for i in range(self.N)]
            try:
                arms, dexs = self.infer_batch_fn(obs_list, adapter_list)
            except Exception as e:
                print(f"[vec rank {rank}] infer_batch failed ({e}); using zero actions", flush=True)
                arms = np.zeros((self.N, 14), dtype=np.float32)
                dexs = np.zeros((self.N, 44), dtype=np.float32)
            for i in range(self.N):
                if self.idle[i]:
                    continue
                env = self.envs[i]
                adapter = self.adapters[i]
                env_action = adapter.to_env_action(
                    np.asarray(arms[i], dtype=np.float32),
                    np.asarray(dexs[i], dtype=np.float32), env)
                render_this = ((qd(i) if qd else True) or self._recording[i])
                if render_this:
                    obs, r, term, info = env.step(env_action)
                    if self._writer[i] is not None:
                        try:
                            self._writer[i].write(obs["images"]["head"])
                        except Exception:
                            pass
                else:
                    r, term, info = env.step_physics_only(env_action)
                self._n_steps[i] += 1
                self._ever_success[i] = self._ever_success[i] or bool(info.get("success"))
                if self._writer[i] is not None and self._ever_success[i]:
                    done = bool(self._n_steps[i] >= max_steps)
                else:
                    done = bool(term or info.get("timed_out") or info.get("success")
                                or self._n_steps[i] >= max_steps)
                if done:
                    if self._writer[i] is not None:
                        try:
                            self._writer[i].close()
                        except Exception:
                            pass
                        self._writer[i] = None
                    self._finish_episode(i, info)
                    obs = self._start_next_episode(i)
                    if obs is None:
                        self.idle[i] = True
                        self.obs_slot[i] = self._dummy_obs()
                    else:
                        self.idle[i] = False
                        self.obs_slot[i] = obs
                elif render_this:
                    # refresh obs so the next policy query sees the post-action state.
                    self.obs_slot[i] = obs
                # else (render-skip, not done, not idle): keep last rendered obs.
            ctrl_step += 1
            if ctrl_step >= next_hb:
                next_hb = ctrl_step + hb_every
                parts = []
                n_active = 0
                for i in range(self.N):
                    if self.idle[i]:
                        parts.append(f"e{i}:idle")
                    else:
                        n_active += 1
                        parts.append(f"e{i}:s{self._seed[i]}_ep{self._ep[i]} {self._n_steps[i]}/{max_steps}")
                print(f"[vec rank {rank} | {suite}/{task}] step {ctrl_step} | active {n_active}/{self.N} | "
                      + " ".join(parts), flush=True)
            if all(self.idle):
                break
        for w in getattr(self, "_writer", []):
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        for r in self._cam_renderers.values():
            try:
                r.close()
            except Exception:
                pass
        print(f"[vec rank {rank} | {suite}/{task}] inline single-thread render "
              f"(N={self.N} envs, GL owned by caller, no render thread)", flush=True)
        return list(self._results)
