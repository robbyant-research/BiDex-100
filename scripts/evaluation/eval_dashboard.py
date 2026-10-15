#!/usr/bin/env python3
"""Beautiful live terminal dashboard for BiDex-100 vectorized 8-card eval.

Runs in the FOREGROUND while the eval (local_eval.sh's combo loop) runs in a
background subshell. Reads only — writes nothing that can perturb the run:

  * vec_rank{r}.log           → per-worker current task + per-env step progress
                                (heartbeat every BIDEX_VEC_HB_STEPS) + EP done +
                                DONE lines  (vectorized server-client path)
  * results/<MID>/*.progress.json  → per-setting done/total task bar
  * results/<MID>/*.json      → finished settings
  * openpi_server.log         → policy-server readiness / warmup / errors
  * serve_policy.py pgrep     → server alive

Auto-discovers the model_id (most-recent results/ dir) and the active combo run
dir (`runs/<MID>/<tag>_*`) by mtime, so it survives relaunches. Exits cleanly once the
eval background PID (BIDEX_EVAL_PID / --eval-pid) is gone and progress stalls,
after a short grace flush so the final results are visible.

Usage (normally driven by local_eval.sh):
    python scripts/evaluation/eval_dashboard.py [--out DIR] [--model-id MID]
           [--eval-pid PID] [--tag pi05_joint_allmixed] [--settings a,b,c]
           [--refresh 1.0] [--max-workers 7]
"""
from __future__ import annotations
import os, sys, re, time, json, signal, subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Why a fixed dict instead of TrueColor hex: terminals that don't report 24-bit
# color still render the 8-color variants fine, and we avoid the rainbow soup
# that made the old watch_eval.sh unreadable. Palette tuned for dark backgrounds.
C_RUN   = "#7aa2f7"   # wings-of-progress blue          (running)
C_DONE  = "#9ece6a"   # green                            (completed)
C_WARN  = "#e0af68"   # amber                            (in-progress / warm)
C_BAD   = "#f7768e"   # red/pink                         (failure / crash)
C_IDLE  = "#6e7681"   # grey                             (idle / queued)
C_ACC   = "#bb9af7"   # purple                           (accents / title)
C_HDR   = "#2ac3de"   # cyan                             (header kpis)
C_DIM   = "#565f89"   # dim grey                         (labels)

# --------------------------------------------------------------------------- #
# data model                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class RankState:
    rank: int
    log: Optional[Path] = None
    offset: int = 0
    task: str = ""                 # suite/task
    max_steps: int = 0
    envs: list = field(default_factory=list)   # [{active,seed,ep,step,max}]
    eps_done: int = 0
    eps_total: int = 0
    successes: int = 0
    fails: int = 0
    last_ep: str = ""              # "✓ s0_ep2 50/100"  /  "✗ ..."  /  ""
    task_started_ts: float = 0.0
    status: str = "queued"         # queued|starting|running|idle|crashed|done
    last_seen: float = 0.0
    done_tasks: list = field(default_factory=list)   # [(task, normalized score, sr)]


@dataclass
class SettingState:
    setting: str
    done: int = 0
    total: int = 0
    finished: bool = False
    score: float = 0.0          # normalized live score (legacy JSON key: mean_rate)
    success_rate: float = 0.0   # live
    n_tasks: int = 0            # aggregation weight; each task counts once
    has_json: bool = False      # a result .json exists (in-progress or done)
    per_task: dict = field(default_factory=dict)  # {task_id: {score, success_rate, n_episodes}}

# --------------------------------------------------------------------------- #
# log parsers                                                                 #
# --------------------------------------------------------------------------- #
RE_HB  = re.compile(r"\[vec rank (\d+)[^\]]*\]\s+step (\d+)\s*\|\s*active (\d+)/(\d+)\s*\|\s*(.*)")
RE_EPI = re.compile(r"e(\d+):(?:s(\d+)_ep(\d+)\s+(\d+)/(\d+)|idle)")
RE_EP  = re.compile(r"\[vec rank (\d+)\s*\|\s*([^/\]]+)/([^\]]+)\]\s+EP done s(\d+)_ep(\d+)\s+success=(\w+)\s+score=(\d+)/(?:\d+|\?)\s+steps=(\d+)/(?:\d+|\?)\s+t=(\d+)s\s+\[done\s+(\d+)/(\d+)\]")
RE_DONE = re.compile(r"\[vec rank (\d+)\]\s+DONE\s+(\S+):\s+mean_rate=([\d.]+)\s+success_rate=([\d.]+)\s+\((\d+)\s+eps,\s+(\d+)s\)")
RE_TASK = re.compile(r"\[vec rank (\d+)\]\s+(\S+)\s*:\s+(\d+)\s+eps over\s+(\d+)\s+envs\s+\(max_steps=(\d+)\)")
RE_CRASH = re.compile(r"Segmentation|Killed|core dumped|Traceback \(most recent", re.I)
RE_INFERFAIL = re.compile(r"infer failed|reconnect", re.I)
RE_SETTING = re.compile(
    r".+_(?:franka|astribot)_(?:joint|camera_ee|robot_ee)_(?:in_domain|out_of_domain)$")

ENVSEP = "  "   # gap between per-env mini-bars in a worker cell

# --------------------------------------------------------------------------- #
# monitor engine                                                              #
# --------------------------------------------------------------------------- #


class EvalMonitor:
    def __init__(self, out: Path, tag: str, settings_expected: list[str],
                 eval_pid: int, refresh: float, max_workers: int, model_id: str = "",
                 log_path: Optional[Path] = None):
        self.out = Path(out)
        self.tag = tag
        self.refresh = refresh
        self.eval_pid = eval_pid
        self.max_workers = max_workers
        self.console = Console()
        # Continuous progress log: each refresh OVERWRITES the file with the latest
        # rendered panel (plain text) — only the newest frame is kept, so the log
        # always mirrors the current dashboard state (open/scroll it anytime).
        self.log_path = Path(log_path) if log_path else None
        self._log_fh = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "w", buffering=1)  # truncate; rewritten each refresh
        # auto-detect results/<MID>: most recently modified leaf under results/
        self.results_root = self.out / "results"
        self.expected = list(settings_expected)
        self.settings: dict[str, SettingState] = {}
        for s in self.expected:
            self.settings[s] = SettingState(s)
        self.ranks: dict[int, RankState] = {}
        self.mid = model_id
        self.run_dir: Optional[Path] = None
        self._last_combo_check = 0.0
        self._prev_envstep_sum = 0
        self._throughput = 0.0          # env-steps/s (EMA)
        self._ep_rate = 0.0
        self._prev_eps_done_sum = 0
        self.start_ts = time.time()
        self.server_state = "booting"
        self.server_log: Optional[Path] = None
        self._crash_ranks: set = set()
        self._infer_fails = 0
        self._eval_done_seen = False
        self._final_hold_until = 0.0

    # ---- discovery -------------------------------------------------------- #
    def _detect_mid(self):
        if self.mid:
            return self.mid
        cands = []
        if self.results_root.is_dir():
            for d in self.results_root.iterdir():
                if d.is_dir():
                    try:
                        cands.append((d.stat().st_mtime, d))
                    except OSError:
                        pass
        cands.sort(reverse=True)
        if cands:
            self.mid = cands[0][1].name

    def _detect_combo_dir(self):
        """Most-recently-active runs/ dir matching the tag (has vec/rank logs)."""
        now = time.time()
        if self.run_dir and (now - self._last_combo_check) < 3.0:
            return
        self._last_combo_check = now
        best = None; best_ts = 0.0
        for d in (self.out / "runs" / (self.mid or self.tag)).glob(f"{self.tag}*"):
            if not d.is_dir():
                continue
            # recursive: vec_rank{r}.log now live under per-domain sub-dirs
            # (in_domain/ out_of_domain/) since launch_vector isolates shards.
            ts = max((f.stat().st_mtime for f in d.rglob("vec_rank*.log")
                      if f.is_file()), default=0.0)
            if ts > best_ts:
                best_ts = ts; best = d
        if best:
            self.run_dir = best
        elif (self.out / "runs").is_dir():
            # fall back to any dir whose name starts with the tag
            cands = sorted((self.out / "runs" / (self.mid or self.tag)).glob(f"{self.tag}*"),
                           key=lambda p: p.name, reverse=True)
            if cands:
                self.run_dir = cands[0]

    def _open_server_log(self):
        root = self.out / "runs" / (self.mid or self.tag)
        candidates = [root / "openpi_server.log"]
        candidates.extend(sorted(root.glob("*_server.0.log")))
        candidates.extend(sorted(root.glob("serve_policy_batched.0.log")))
        for path in candidates:
            if path.is_file():
                self.server_log = path
                return

    # ---- log tailing ------------------------------------------------------ #
    def _tail_log(self, st: RankState):
        if not st.log or not st.log.is_file():
            return
        try:
            sz = st.log.stat().st_size
        except OSError:
            return
        if st.offset > sz:           # file truncated (new run/relaunch) -> reset
            st.offset = 0
        if sz == st.offset:
            return
        with open(st.log, "r", errors="replace") as f:
            f.seek(st.offset)
            new = f.read()
            st.offset = sz
        for line in new.splitlines():
            self._parse_line(st, line)

    def _parse_line(self, st: RankState, line: str):
        if RE_CRASH.search(line):
            st.status = "crashed"
            self._crash_ranks.add(st.rank)
        if RE_INFERFAIL.search(line):
            self._infer_fails += 1
        m = RE_TASK.search(line)
        if m:   # new task starting (vectorized)
            st.task = f"{m.group(2)}/{m.group(3)}" if "/" not in m.group(2) else m.group(2)
            # RE_TASK group2 is "<suite/task>" already; normalize
            st.task = m.group(2)
            st.max_steps = int(m.group(5))
            st.eps_done = 0; st.eps_total = int(m.group(3))
            st.successes = 0; st.fails = 0
            st.envs = []
            st.task_started_ts = time.time()
            st.status = "running"
            st.last_seen = time.time()
            return
        m = RE_HB.search(line)
        if m:
            env_part = m.group(5)
            envs = []
            for em in RE_EPI.finditer(env_part):
                if em.group(2) is None:
                    envs.append({"active": False, "seed": None, "ep": None,
                                 "step": 0, "max": st.max_steps})
                else:
                    envs.append({"active": True, "seed": int(em.group(2)),
                                 "ep": int(em.group(3)), "step": int(em.group(4)),
                                 "max": int(em.group(5)) or st.max_steps})
                    if envs[-1]["max"]:
                        st.max_steps = envs[-1]["max"]
            st.envs = envs
            st.status = "running"
            st.last_seen = time.time()
            return
        m = RE_EP.search(line)
        if m:
            success = m.group(6) == "True"
            st.rank = int(m.group(1))
            if success:
                st.successes += 1
            else:
                st.fails += 1
            st.eps_done = int(m.group(10))
            st.eps_total = int(m.group(11)) or st.eps_total
            # short task name + full name for the cell
            _, tname = os.path.split(f"{m.group(2)}/{m.group(3)}")
            mark = "✓" if success else "✗"
            st.last_ep = f"{mark} s{m.group(4)}_ep{m.group(5)} {m.group(7)}/{m.group(8)}s"
            st.last_seen = time.time()
            return
        m = RE_DONE.search(line)
        if m:
            st.done_tasks.append((m.group(2), float(m.group(3)), float(m.group(4))))
            st.last_seen = time.time()
            return
        if "gpu" in line and st.status == "queued":
            st.status = "starting"
            st.last_seen = time.time()

    # ---- setting progress files ------------------------------------------ #
    def _read_settings(self):
        if not self.mid or not (self.results_root / self.mid).is_dir():
            return
        d = self.results_root / self.mid
        # in progress: *.progress.json (presence ⇒ running; absence ⇒ finished/not-started)
        prog_settings = set()
        for pf in d.glob("*.progress.json"):
            setting = pf.name[:-len(".progress.json")]
            prog_settings.add(setting)
            st = self.settings.setdefault(setting, SettingState(setting))
            try:
                j = json.loads(pf.read_text())
                st.done = int(j.get("done", 0)); st.total = int(j.get("total", 0))
                st.finished = False
            except Exception:
                pass
        # result *.json (exclude .progress.json/.tmp): written live (may be partial)
        # → used both to mark finished (when progress.json absent) and to read the
        # live score / success_rate for display.
        for jf in d.glob("*.json"):
            nm = jf.name
            if nm.endswith(".progress.json") or nm.endswith(".tmp"):
                continue
            setting = nm[:-5]  # strip .json
            # summary.json and any future model-level reports are not eval
            # settings. Treating summary.json as a setting used to add a bogus
            # 11th row with a zero score and biased the dashboard aggregate.
            if not RE_SETTING.fullmatch(setting):
                continue
            st = self.settings.setdefault(setting, SettingState(setting))
            st.has_json = True
            try:
                j = json.loads(jf.read_text())
                # Use per_task mean to compute score/sr (matching web leaderboard _setting_metrics)
                pt = j.get("per_task", {})
                if pt and isinstance(pt, dict):
                    score_vals = [v.get("score", v.get("mean_rate", 0.0))
                                  for v in pt.values() if isinstance(v, dict)]
                    sr_vals = [v.get("success_rate", 0.0) for v in pt.values() if isinstance(v, dict)]
                    st.score = sum(score_vals) / len(score_vals) if score_vals else 0.0
                    st.success_rate = sum(sr_vals) / len(sr_vals) if sr_vals else 0.0
                    st.n_tasks = len(score_vals)
                    st.per_task = {k: {"score": v.get("score", v.get("mean_rate", 0.0)),
                                       "success_rate": v.get("success_rate", 0.0),
                                       "n_episodes": v.get("n_episodes", 0)}
                                   for k, v in pt.items() if isinstance(v, dict)}
                else:
                    st.score = float(j.get("score", j.get("mean_rate", 0.0)) or 0.0)
                    st.success_rate = float(j.get("success_rate", 0.0) or 0.0)
                    st.n_tasks = max(0, int(j.get("n_tasks", 0)))
                if setting not in prog_settings:
                    st.finished = True
                    st.done = st.total = max(1, int(j.get("n_tasks", 1)))
            except Exception:
                pass

    # ---- server / pid ----------------------------------------------------- #
    def _eval_alive(self) -> bool:
        if not self.eval_pid:
            return True   # unknown → assume alive, rely on stall detection
        try:
            os.kill(self.eval_pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _server_status(self):
        if self.server_log and self.server_log.is_file():
            try:
                tail = self.server_log.read_text()[-4000:]
            except Exception:
                tail = ""
            if "server warm" in tail:
                self.server_state = "warm"
            elif "listening on" in tail and "warming" not in tail:
                self.server_state = "listening"
            elif "warming" in tail:
                self.server_state = "warming (JIT)"
            elif re.search(r"DIED early|readiness TIMEOUT", tail):
                self.server_state = "ERROR"
            else:
                self.server_state = "loading"
        if self.server_state in ("loading",):
            # cross-check pgrep when log is silent
            try:
                out = subprocess.check_output(["pgrep", "-f", "serve_policy.py"],
                                              stderr=subprocess.DEVNULL, text=True)
                self.server_state = "loading (pid %s)" % out.split()[0]
            except Exception:
                pass

    # ---- main tick -------------------------------------------------------- #
    def update(self):
        self._detect_mid()
        self._detect_combo_dir()
        if not hasattr(self, "_server_logged"):
            self._open_server_log(); self._server_logged = True
        self._read_settings()
        self._server_status()
        # (re)scan rank logs in the active combo dir
        if self.run_dir and self.run_dir.is_dir():
            for f in sorted(self.run_dir.rglob("vec_rank*.log"), key=lambda p: p.name):
                # rank id from name: vec_rank0.log
                m = re.search(r"(\d+)", f.stem)
                if not m:
                    continue
                rid = int(m.group(1))
                st = self.ranks.setdefault(rid, RankState(rank=rid))
                if st.log != f:
                    st.log = f; st.offset = 0
                self._tail_log(st)
        # throughput EMA
        envstep_sum = sum(
            e["step"] for st in self.ranks.values() for e in st.envs
        ) if self.ranks else 0
        eps_done_sum = sum(st.eps_done for st in self.ranks.values())
        dt = self.refresh or 1.0
        if not getattr(self, "_throughput_seeded", False):
            # cold start: don't let the first reading (Σsteps/1s) read as a spike
            self._prev_envstep_sum = envstep_sum
            self._prev_eps_done_sum = eps_done_sum
            self._throughput_seeded = True
        else:
            inst = max(0.0, (envstep_sum - self._prev_envstep_sum) / dt)
            self._throughput = self._throughput * 0.6 + inst * 0.4 if (self._throughput or inst) else 0.0
            self._ep_rate = self._ep_rate * 0.6 + max(0.0, (eps_done_sum - self._prev_eps_done_sum) / dt) * 0.4 if (self._ep_rate or (eps_done_sum - self._prev_eps_done_sum)) else 0.0
            self._prev_envstep_sum = envstep_sum
            self._prev_eps_done_sum = eps_done_sum
        for st in self.ranks.values():
            if st.status == "running" and time.time() - st.last_seen > 25:
                st.status = "idle"
        # eval-done detection
        if not self._eval_alive():
            if not self._eval_done_seen:
                self._eval_done_seen = True
                self._final_hold_until = time.time() + 6.0  # grace flush
        if self._eval_done_seen and time.time() > self._final_hold_until:
            self._done = True

    @property
    def finished(self) -> bool:
        return getattr(self, "_done", False)

    # ---- rendering -------------------------------------------------------- #
    def _hbar(self, frac: float, color=C_RUN, width: int = 10) -> "Text":
        """A two-tone progress bar as a styled Text (filled in `color`, empty in
        dim grey). Returns a Text with explicit style spans — NOT a markup string
        — so the tags never leak literally (Text("string") does not parse markup)."""
        frac = max(0.0, min(1.0, frac))
        full = int(round(frac * width))
        t = Text(no_wrap=True)
        if full:
            t.append("█" * full, style=color)
        if width - full:
            t.append("░" * (width - full), style=C_IDLE)
        return t

    def render(self, width: Optional[int] = None) -> Panel:
        # `width` is the column budget used to size panels/columns/bars. It
        # defaults to the live console width (a real TTY → the on-screen terminal
        # width). The progress-log writer passes a wider explicit width (see
        # _log_width) so the .dashboard.log file is NOT collapsed to rich's 80-col
        # fallback when the monitor runs without a TTY (k8s/batch) → columns then
        # fit and task names / env bars / step ranges render in full, 1:1 with a
        # wide terminal (no '…' cuts).
        if width is None:
            width = self.console.width
        # ----- grouping (collapse franka/astribot per the user's request) --- #
        def _strip_robot(s):
            return s.replace("_franka_", "_").replace("_astribot_", "_")
        groups = {}  # stripped -> list[SettingState]
        for s in self.expected:
            groups.setdefault(_strip_robot(s), []).append(
                self.settings.get(s) or SettingState(s))
        exp_set = set(self.expected)
        for k, st in self.settings.items():
            if k not in exp_set:
                groups.setdefault(_strip_robot(k), []).append(st)

        def _grp_state(members):
            done = sum(m.done for m in members)
            total = sum(m.total for m in members)
            all_done = bool(members) and all(m.finished for m in members)
            any_run = (any(m.total and not m.finished for m in members)
                       or any(self._setting_running(getattr(m, "setting", "")) for m in members))
            jscores = [m for m in members if m.has_json and m.n_tasks > 0]
            weight = sum(m.n_tasks for m in jscores)
            score = (sum(m.score * m.n_tasks for m in jscores) / weight) if weight else None
            sr = (sum(m.success_rate * m.n_tasks for m in jscores) / weight) if weight else None
            return done, total, all_done, any_run, score, sr

        # overall progress in GROUPS (robot-agnostic), not per-robot jsons.
        exp_total = len(groups) or max(1, len(self.settings))
        done_n = sum(1 for members in groups.values()
                     if _grp_state(members)[2])  # all_done
        overall_frac = done_n / max(1, exp_total)
        # live aggregate score across every group whose .json has flushed so far.
        all_js = [m for members in groups.values() for m in members
                  if m.has_json and m.n_tasks > 0]
        agg_weight = sum(m.n_tasks for m in all_js)
        agg_score = (sum(m.score * m.n_tasks for m in all_js) / agg_weight) if agg_weight else None
        agg_sr = (sum(m.success_rate * m.n_tasks for m in all_js) / agg_weight) if agg_weight else None
        elapsed = time.time() - self.start_ts
        el = self._fmt_dur(elapsed)

        title = Text()
        title.append(" BiDex-100  ", style=f"bold {C_ACC}")
        title.append("vectorized 8-GPU eval  ", style="bold white")
        title.append("│ MID ", style=C_DIM)
        title.append(self.mid or "—", style=f"bold {C_HDR}")
        title.append("  │ tag ", style=C_DIM)
        title.append(self.tag, style=C_HDR)
        title.append("  │ ckpt ", style=C_DIM)
        title.append(Path(os.environ.get("BIDEX_CKPT", "")).name or "—", style=C_HDR)

        # ----- KPI row ---------------------------------------------------- #
        active_envs = sum(1 for st in self.ranks.values() for e in st.envs if e["active"])
        total_envs = sum(len(st.envs) for st in self.ranks.values())
        running_workers = sum(1 for st in self.ranks.values() if st.status == "running")
        eps_done_sum = sum(st.eps_done for st in self.ranks.values())
        eps_total_sum = sum(st.eps_total for st in self.ranks.values() if st.eps_total)
        succ_sum = sum(st.successes for st in self.ranks.values())
        # overall ETA from settings completion
        eta_txt = "—"
        if done_n >= 1 and exp_total > done_n:
            eta = elapsed * (exp_total - done_n) / done_n
            eta_txt = self._fmt_dur(eta)
        elif done_n == exp_total and exp_total:
            eta_txt = "done"
        # server color + symbol
        srv = self.server_state
        if srv in ("warm", "listening"):
            srv_sym, srv_col = "●", C_RUN
        elif "ERROR" in str(srv):
            srv_sym, srv_col = "✖", C_BAD
        elif srv in ("loading", "warming (JIT)") or "loading" in str(srv):
            srv_sym, srv_col = "◐", C_WARN
        elif srv == "booting":
            srv_sym, srv_col = "○", C_IDLE
        else:
            srv_sym, srv_col = "●", C_RUN
        kpi1 = Text(no_wrap=True)
        kpi1.append("⏱ ", C_DIM); kpi1.append(el, "bold white"); kpi1.append("   ")
        kpi1.append(f"{srv_sym} server ", C_DIM); kpi1.append(srv, f"bold {srv_col}"); kpi1.append("   ")
        kpi1.append("workers ", C_DIM); kpi1.append(f"{running_workers}/{len(self.ranks) or self.max_workers}", f"bold {C_ACC}"); kpi1.append("   ")
        kpi1.append("envs ", C_DIM); kpi1.append(f"{active_envs}/{total_envs or '-'}", f"bold {C_ACC}"); kpi1.append("   ")
        kpi1.append("stps/s ", C_DIM); kpi1.append(f"{self._throughput:,.0f}", f"bold {C_HDR}"); kpi1.append("   ")
        kpi1.append("eps ", C_DIM); kpi1.append(f"{eps_done_sum}/{eps_total_sum or '-'}", f"bold {C_HDR}"); kpi1.append("   ")
        kpi1.append("✓ ", C_DIM); kpi1.append(f"{succ_sum}", f"bold {C_DONE}"); kpi1.append("   ")
        # live aggregate score across all flushed settings
        if agg_score is not None:
            kpi1.append("SR ", C_DIM)
            kpi1.append("%.2f%%" % (agg_sr * 100), f"bold {C_DONE if agg_sr else C_DIM}")
            kpi1.append("  score ", C_DIM)
            kpi1.append("%.2f" % (agg_score * 100), f"bold {C_HDR}")
        crash_txt = (",".join(str(r) for r in sorted(self._crash_ranks))) if self._crash_ranks else "none"
        kpi2 = Text(no_wrap=True)
        kpi2.append("ETA ", C_DIM); kpi2.append(eta_txt, "bold white"); kpi2.append("   ")
        kpi2.append("settings ", C_DIM); kpi2.append(f"{done_n}/{exp_total}", f"bold {C_ACC}"); kpi2.append("   ")
        kpi2.append("infer-fail ", C_DIM); kpi2.append(f"{self._infer_fails}", C_BAD if self._infer_fails else C_DIM); kpi2.append("   ")
        kpi2.append("crash ", C_DIM); kpi2.append(crash_txt, C_BAD if self._crash_ranks else C_DIM)

        # ----- overall progress bar -------------------------------------- #
        barw = max(20, width - 30)
        ovline = Text(no_wrap=True)
        ovline.append("overall ", style=C_DIM)
        ovline.append_text(self._hbar(overall_frac, C_ACC, barw))
        ovline.append(" ")
        ovline.append(f"{int(overall_frac*100):3d}%", style="bold white")
        ovline.append(f"  {done_n}/{exp_total}", style=C_DIM)

        # ----- settings / combos table (grouped by robot-stripped name) ---- #
        # groups / _grp_state computed above (overall bar); reused here.
        barw2 = max(16, width - 92)
        sset = Table(expand=True, box=box.SIMPLE_HEAVY, show_header=True,
                     header_style=f"bold {C_HDR}", pad_edge=False)
        sset.add_column("setting (benchmark · mode · domain)", overflow="ellipsis",
                        max_width=max(26, width // 3), no_wrap=True)
        sset.add_column("status", justify="left", width=11, no_wrap=True)
        sset.add_column("tasks", justify="right", width=7, no_wrap=True)
        sset.add_column("SR / score", justify="right", width=18, no_wrap=True)
        sset.add_column("progress", ratio=1)

        def _group_order(item):
            _, members = item
            done, total, all_done, any_run, *_ = _grp_state(members)
            if all_done: return 2
            if any_run or total: return 0
            return 1
        for gk, members in sorted(groups.items(), key=_group_order):
            done, total, all_done, any_run, score_value, sr = _grp_state(members)
            if all_done:
                col = C_DONE; stt = Text("done ✓", f"bold {C_DONE}")
                frac = 1.0; dk, tk = done, total or done
            elif total:
                col = C_RUN; stt = Text("running ●", f"bold {C_RUN}")
                frac = done / total if total else 0.0; dk, tk = done, total
            elif any_run:
                col = C_WARN; stt = Text("preparing", f"bold {C_WARN}")
                frac = 0.0; dk, tk = "—", "—"
            else:
                col = C_IDLE; stt = Text("queued", C_IDLE)
                frac = 0.0; dk, tk = "—", "—"
            bar = self._hbar(frac, col, barw2)
            # live score / success-rate (shown from the first live flush onward)
            if score_value is not None:
                score = Text.assemble(("%.2f%%" % (sr * 100), C_DONE if sr else C_DIM),
                                      ("  score ", C_DIM),
                                      ("%.2f" % (score_value * 100), C_HDR))
            else:
                score = Text("—", C_DIM)
            sset.add_row(self._short_setting(gk), stt, Text(f"{dk}/{tk}", "white"),
                         score, bar)

        # ----- workers table --------------------------------------------- #
        w = Table(expand=True, box=box.SIMPLE_HEAVY, show_header=True,
                  header_style=f"bold {C_HDR}", pad_edge=False)
        w.add_column("rank", width=5, justify="right")
        w.add_column("task", overflow="ellipsis", max_width=max(18, width // 4), no_wrap=True)
        w.add_column("env progress (active/total · leader bar · step range)", ratio=3)
        w.add_column("eps", justify="right", width=8)
        w.add_column("✓/✗", justify="right", width=7)
        w.add_column("last", overflow="ellipsis", max_width=max(16, width // 9), no_wrap=True)
        rank_items = sorted(self.ranks.items())
        if not rank_items:
            for r in range(self.max_workers):
                w.add_row(Text(str(r), C_DIM), Text("queued", C_IDLE),
                          Text("waiting for server warmup…", C_DIM),
                          "—", "—", "—")
        for rid, st in rank_items:
            # env cell
            if not st.envs:
                if st.status == "crashed":
                    envline = Text("■ crashed — see log", f"bold {C_BAD}")
                elif st.status == "idle":
                    envline = Text("idle (no active episode)", C_IDLE)
                elif st.status == "starting":
                    envline = Text("building envs / first infer…", C_WARN)
                elif st.status == "running":
                    envline = Text("running (awaiting heartbeat)…", C_WARN)
                else:
                    envline = Text("—", C_DIM)
            else:
                # compact one-line aggregation (was: one line per env → N=32 made
                # the panel far too tall). Show active/total envs, a single progress
                # bar (leader env frac), and the min–max step range across envs.
                envline = Text(no_wrap=True)
                barw_env = 12
                active = [e for e in st.envs if e["active"]]
                n_act, n_tot = len(active), len(st.envs)
                if not n_act:
                    envline.append(f"0/{n_tot} active", C_IDLE)
                else:
                    fracs = [(e["step"] / e["max"]) if e["max"] else 0.0 for e in active]
                    fmax = max(fracs)
                    done = fmax >= 1.0
                    col = C_DONE if done else C_RUN
                    smin = min(e["step"] for e in active)
                    smax = max(e["step"] for e in active)
                    smaxmx = max(e["max"] for e in active) or 0
                    envline.append(f"{n_act}/{n_tot} ", C_DIM)
                    envline.append_text(self._hbar(fmax, col, barw_env))   # leader env bar
                    envline.append(f" {smin}-{smax}/{smaxmx if smaxmx else '?'}", C_DONE if done else C_HDR)
            # task color
            if st.status == "running":
                tcol = C_ACC
            elif st.status == "crashed":
                tcol = C_BAD
            elif st.status == "idle":
                tcol = C_IDLE
            elif st.status == "done":
                tcol = C_DONE
            else:
                tcol = C_WARN if st.status == "starting" else C_DIM
            w.add_row(
                Text(str(rid), f"bold {tcol}"),
                Text(st.task.split("/")[-1] or "—", tcol),
                envline,
                Text.assemble((f"{st.eps_done}", "bold white"),
                              (f"/{st.eps_total}" if st.eps_total else "/?", C_DIM)),
                Text.assemble((f"{st.successes}", C_DONE), ("/", C_DIM), (f"{st.fails}", C_BAD if st.fails else C_DIM)),
                Text(st.last_ep, C_DIM),
            )

        # ----- footer ----------------------------------------------------- #
        foot = Text()
        if self._eval_done_seen:
            foot.append("● eval process exited — flushing final results…", f"bold {C_DONE}")
        else:
            foot.append("● monitoring — ", C_DIM)
            foot.append("one command shows everything; no second terminal needed.", C_ACC)
        if self._crash_ranks:
            foot.append(f"  ⚠ crashed ranks: {','.join(map(str, sorted(self._crash_ranks)))}", f"bold {C_BAD}")

        body = Group(kpi1, kpi2, Text(""), ovline, Text(""), sset, Text(""), w, Text(""), foot)
        return Panel(body, title=title, border_style=C_ACC, subtitle=foot, padding=(0, 1))

    # ---- helpers ---------------------------------------------------------- #
    def _setting_running(self, setting: str) -> bool:
        # a setting is "running" now if one of our worker tasks belongs to it
        dom = setting.split("_")[-1]
        bench = setting.split("_")[0]
        # crude: active combo dir name encodes bench+robot; accept if combo dir name
        # includes bench AND a rank is running
        if self.run_dir and bench in self.run_dir.name and dom in self.run_dir.name:
            return any(st.status == "running" for st in self.ranks.values())
        return False

    def _setting_json(self, setting: str):
        if not self.mid:
            return None
        p = self.results_root / self.mid / f"{setting}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _short_setting(self, s: str) -> str:
        # input is robot-stripped: "<Bench>_<mode>_<domain>". Parse it cleanly.
        abbr = {"Limited_Demonstrations": "Limited",
                "Instruction_Following": "Instr", "Long_Horizon": "LongHorizon",
                "Generalization": "Gen"}
        bench = s
        mode = dom = ""
        m = re.match(r"(.+)_(joint|camera_ee|robot_ee)_(in_domain|out_of_domain)$", s)
        if m:
            bench, mode, dom = m.group(1), m.group(2), m.group(3)
        for k, v in abbr.items():
            bench = bench.replace(k, v)
        dom_s = "in" if dom == "in_domain" else ("ood" if dom == "out_of_domain" else dom)
        if mode:
            return f"{bench} · {mode} · {dom_s}"
        return bench

    @staticmethod
    def _fmt_dur(s: float) -> str:
        s = int(max(0, s))
        h, s = divmod(s, 3600); m, s = divmod(s, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _log_width(self) -> int:
        """Column width for rendering the progress-log panel.

        The monitor often runs with stdout NOT a TTY (k8s/batch), in which case
        rich's auto-detected console width falls back to 80 cols and every
        width-scaled column (task name, env-progress bar, step range, last-ep)
        collapses to ~max 18-20 and is cut with '…'. Render the log at a wide
        width instead so columns fit and nothing is elided. If the live console
        already reports a real (wide) terminal, use it verbatim so the file
        mirrors the on-screen terminal exactly. Override with
        BIDEX_DASH_WIDTH=<cols> (e.g. your `tput cols`)."""
        w = os.environ.get("BIDEX_DASH_WIDTH", "")
        if w.lstrip("+-").isdigit():
            return max(80, int(w))
        cw = self.console.width or 80
        return cw if cw >= 160 else 90

    def _render_text(self) -> str:
        """Render the current dashboard panel to plain text (no ANSI) for the
        progress log. Sized to a wide width (see _log_width) — NOT the live
        console width — so the .dashboard.log is not trimmed with '…' when the
        monitor runs without a TTY; it reads 1:1 like a wide terminal."""
        from io import StringIO
        w = self._log_width()
        buf = StringIO()
        tc = Console(file=buf, width=w, color_system=None,
                     highlight=False, soft_wrap=True, force_terminal=False)
        tc.print(self.render(width=w))
        return buf.getvalue().rstrip("\n")

    def log(self):
        """OVERWRITE the progress log with the latest rendered panel (plain text)
        + a single timestamp line. Called every refresh so the file always holds
        only the current dashboard state (not an accumulating transcript)."""
        if not self._log_fh:
            return
        ts = time.strftime("%H:%M:%S")
        self._log_fh.seek(0)
        self._log_fh.truncate()
        self._log_fh.write(f"=============================== [{ts}] ===============================\n")
        self._log_fh.write(self._render_text())
        self._log_fh.write("\n")
        self._log_fh.flush()

    def close_log(self):
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

# entry point
# --------------------------------------------------------------------------- #


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    ap.add_argument("--model-id", default=os.environ.get("BIDEX_DASH_MID", ""))
    ap.add_argument("--tag", default=os.environ.get("BIDEX_DASH_TAG", "pi05_joint_allmixed"))
    ap.add_argument("--settings", default=os.environ.get("BIDEX_EXPECTED_SETTINGS", ""),
                    help="comma list of expected setting names (bench_robot_mode_domain)")
    ap.add_argument("--eval-pid", type=int, default=int(os.environ.get("BIDEX_EVAL_PID", "0") or "0"))
    ap.add_argument("--refresh", type=float, default=float(os.environ.get("BIDEX_DASH_REFRESH", "1.0")))
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("BIDEX_DASH_MAX_WORKERS", "7")))
    ap.add_argument("--dash-log", default=os.environ.get("BIDEX_DASH_LOG", ""),
                    help="overwrite this file with the latest dashboard every refresh (only newest frame kept)")
    args = ap.parse_args()
    if not args.out:
        ap.error("--out is required (or set OUT)")
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    out = Path(args.out)
    log_path = Path(args.dash_log) if args.dash_log else (out / "runs" / (args.model_id or args.tag) / f"{args.tag}.dashboard.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mon = EvalMonitor(out, args.tag, settings, args.eval_pid,
                      args.refresh, args.max_workers, args.model_id, log_path=log_path)
    # graceful Ctrl-C: leave the eval running, just stop the dashboard
    def _sig(_s, _f):
        mon._done = True
    try:
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
    except Exception:
        pass
    mon.console.print(f"[bold {C_ACC}]dashboard progress log (latest frame only, overwritten each refresh): {log_path}[/]")
    # Original in-place Live full-screen display (clips to the terminal window;
    # not scrollable). The compact per-rank env row keeps the panel short so it
    # fits the window for the normal N. For the full content, read the log file.
    with Live(mon.render(), console=mon.console, refresh_per_second=4,
              screen=True, transient=False) as live:
        while True:
            time.sleep(args.refresh)
            mon.update()
            live.update(mon.render())
            mon.log()                       # latest frame -> log file (overwritten)
            if mon.finished:
                break
    # final static render (after leaving alt-screen) so the summary stays on-screen
    mon.console.print(mon.render())
    mon.console.print(f"[bold {C_DONE}]● dashboard exited (eval_pid={args.eval_pid} "
                      f"alive={mon._eval_alive()})[/]")
    mon.close_log()
    return 0

if __name__ == "__main__":
    sys.exit(main())
