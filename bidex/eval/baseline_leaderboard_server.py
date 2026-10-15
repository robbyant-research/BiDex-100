#!/usr/bin/env python3
"""BiDex-100 baseline leaderboard — Flask web server (cloudproxy-safe, videos).

robodojo-benchmark.com-style leaderboard:
  * models × benchmarks table with clickable Metric (Success Rate / Score) and
    Domain (In-Domain / Out-of-Domain) toggles — all client-side.
  * model detail page with a clickable In-Domain / Out-of-Domain toggle; within
    the selected domain, tasks are grouped by benchmark (Memory … Articulated),
    each with an embedded rollout video.
Links/video srcs are resolved via a single browser-derived `BASE` (keeps the full
cloudproxy prefix on click/fetch); NO innerHTML destruction (so JS handlers and
rendered tables/videos survive).

Result data: <LB_ROOT>/results/<model_id>/<benchmark>_<robot>_<mode>_<domain>.json
Rollout videos: <LB_ROOT>/mp4s/<model_id>/<benchmark>_<robot>_<mode>_<domain>/<task_dir>/s{seed}_ep{ep:03d}.mp4
Model rows are discovered from <LB_ROOT>/results/<model_id>/ — one row per dir on
disk, NO hardcoded model list and NO default pi05/gr00t/beigh rows. Each row's
name = the model_id dir; its framework + checkpoint are pulled live from the
saved result JSON (nothing hardcoded).

Launch (env has flask):
  export BIDEX_LEADERBOARD_ROOT=outputs/eval/leaderboard
  python bidex/eval/baseline_leaderboard_server.py \
      --lb-root "$BIDEX_LEADERBOARD_ROOT" --port 12006 --base-url /baselines
"""
import os, json, argparse, glob, html
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

try:  # task YAMLs are parsed with PyYAML; degrade gracefully if absent.
    import yaml
except ImportError:
    yaml = None

from flask import Flask, send_file, request


def _first_existing(*cands):
    """First path in cands that exists on disk (None if none do). Used so the
    leaderboard picks the right data root for whichever env it runs in."""
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


# Result JSON + rollout videos live under LB_ROOT, supplied at startup via
# --lb-root (CLI) or DSB_LB_ROOT (env). NO path is hardcoded: the leaderboard
# root can move freely (the user points the server at their own eval-output
# tree) while its on-disk structure — results/<model_id>/, mp4s/<model_id>/ —
# stays the same. Resolved in create_app() before any route runs; left None at
# import so importing the module has no side effect.
LB_ROOT = None
RESULTS_DIR = None
MP4_DIR = None
# Task definitions (language + subtask score weights) come from the in-repo
# suite YAMLs, resolved relative to THIS file so no absolute path is baked in;
# a DSB_SUITES_DIR override lets a custom task tree be used instead.
_SUITES_DEFAULT = Path(__file__).resolve().parent.parent / "tasks_data" / "suites"
SUITES_DIR = _first_existing(os.environ.get("DSB_SUITES_DIR"), _SUITES_DEFAULT) or _SUITES_DEFAULT

_META_CACHE = {}  # mid -> (framework, ckpt_leaf). Only non-empty hits are
                  # cached, so a dir that gains its first result JSON after a
                  # page load still refreshes its framework/ckpt next render.


def _model_meta(mid):
    """(framework, ckpt_leaf) read LIVE from the model's saved result JSON — i.e.
    the `model` + `ckpt` fields the eval writes into results/<mid>/*.json. This
    is the leaderboard's ONLY source of per-model display info: nothing about a
    model's name / framework / checkpoint is hardcoded or defaulted. Returns
    ("", "") while only a *.progress.json exists (no real result JSON yet);
    parses the SMALLEST result JSON to keep it cheap. Cached per mid once a real
    hit is found (framework/ckpt don't change for an existing run)."""
    cached = _META_CACHE.get(mid)
    if cached:
        return cached
    framework, ckpt_leaf = "", ""
    d_root = RESULTS_DIR / mid if RESULTS_DIR is not None else None
    if d_root is not None and d_root.is_dir():
        cands = [p for p in d_root.glob("*.json") if not p.name.endswith(".progress.json")]
        if cands:
            cands.sort(key=lambda p: p.stat().st_size)
            for p in cands:
                try:
                    d = json.load(open(p))
                except Exception:
                    continue
                if d.get("model"):
                    framework = str(d["model"])
                ck = d.get("ckpt")
                if ck:
                    sck = str(ck).rstrip("/")
                    parent = os.path.basename(os.path.dirname(sck))
                    base = os.path.basename(sck)
                    ckpt_leaf = f"{parent}/{base}" if parent else base
                break
    if framework:
        _META_CACHE[mid] = (framework, ckpt_leaf)
    return framework, ckpt_leaf


def discover_models():
    """One row per results/<model_id>/ dir actually on disk — NO hardcoded model
    list and NO default pi05/gr00t/beigh rows, so only models with saved results
    appear (evals writing under any model_id show up live). A row's headline name
    IS its model_id dir; the framework + checkpoint come live from the saved
    result JSON (_model_meta). Junk (_-/.-prefixed) dirs are skipped."""
    rows = []
    if RESULTS_DIR is not None and RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or d.name.startswith("_") or d.name.startswith("."):
                continue
            mid = d.name
            framework, ckpt_leaf = _model_meta(mid)
            rows.append({"id": mid, "name": mid, "org": framework, "tag": ckpt_leaf})
    return rows


def model_ids():
    return [m["id"] for m in discover_models()]


BENCHMARKS = ["Memory", "Long_Horizon", "Generalization", "Instruction_Following",
              "Limited_Demonstrations"]
DOMAINS = ["in_domain", "out_of_domain"]
ROBOTS = {"Memory": ["franka"], "Long_Horizon": ["franka"], "Generalization": ["astribot", "franka"],
          "Instruction_Following": ["franka"], "Limited_Demonstrations": ["franka"]}
MODES = ["joint", "robot_ee", "camera_ee"]


def _score_value(row):
    """Canonical normalized score, with compatibility for pre-rename results."""
    if not isinstance(row, dict):
        return 0.0
    value = row.get("score")
    if value is None:
        value = row.get("mean_rate", 0.0)
    return float(value or 0.0)


def load_result(model_id, benchmark, robot, mode, domain):
    p = RESULTS_DIR / model_id / f"{benchmark}_{robot}_{mode}_{domain}.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _setting_metrics(res):
    pt = (res or {}).get("per_task", {})
    if not pt:
        return None
    return {"score": sum(_score_value(v) for v in pt.values()) / len(pt),
            "sr": sum(v.get("success_rate", 0) for v in pt.values()) / len(pt),
            "n": len(pt)}


def bench_domain_metrics(mid, bench, dom, mode_filter="all"):
    rows = []
    modes = [mode_filter] if mode_filter != "all" else MODES
    for robot in ROBOTS.get(bench, []):
        for mode in modes:
            r = load_result(mid, bench, robot, mode, dom)
            if r:
                m = _setting_metrics(r)
                if m:
                    rows.append(m)
    if not rows:
        return None
    # Per-task micro-average: weight each (robot, mode) setting by its task
    # count so e.g. Generalization's 30-task franka outweighs its 10-task
    # astribot — matching scripts/evaluation/summarize_eval_results.py, which averages
    # every task once. A plain mean over settings would give a 10-task robot
    # (was +2.8 / +3.8 on Generalization). `n` stays the total task-row count,
    # which the frontend uses as the weight when averaging benchmarks.
    n_tot = sum(m["n"] for m in rows)
    return {"score": sum(m["score"] * m["n"] for m in rows) / n_tot,
            "sr": sum(m["sr"] * m["n"] for m in rows) / n_tot,
            "n": n_tot}


def all_table_data():
    # {mid: {bench: {domain: {mode: metrics-or-None}}}}  (mode in "all"+MODES)
    modes = ["all"] + MODES
    return {m["id"]: {b: {d: {mode: bench_domain_metrics(m["id"], b, d, mode) for mode in modes}
                          for d in DOMAINS} for b in BENCHMARKS} for m in discover_models()}


@lru_cache(maxsize=None)
def _load_suite_yaml(benchmark, leaf, ood):
    """One parsed task YAML, cached. A result JSON's per_task key is
    <robot>/<group>/<leaf>; the <leaf> maps to suites/<benchmark>/<leaf>.yaml
    (in-domain) or suites/<benchmark>/<leaf>__ood.yaml (out-of-domain). Returns
    {} on any miss / parse error so callers degrade gracefully (empty language
    and subtask list). This is the canonical source for task language + subtask
    definitions — the leaderboard no longer reads an external annotations dump."""
    p = SUITES_DIR / benchmark / f"{leaf}{'__ood' if ood else ''}.yaml"
    if not p.exists() or yaml is None:
        return {}
    try:
        return yaml.safe_load(open(p)) or {}
    except Exception:
        return {}


def task_language(tname, benchmark, domain):
    """Natural-language instruction for a task, read from its suite YAML."""
    leaf = tname.split("/")[-1]
    return _load_suite_yaml(benchmark, leaf, domain == "out_of_domain").get("language", "") or ""


def task_subtask_defs(tname, benchmark, domain):
    """[(name, max_score)] subtask definitions (names + per-subtask score weights)
    straight from the suite YAML — the canonical subtask structure."""
    leaf = tname.split("/")[-1]
    return [(s.get("name", ""), s.get("score", 0))
            for s in _load_suite_yaml(benchmark, leaf, domain == "out_of_domain").get("subtasks", []) or []]


def find_videos(mid, setting, task_dir):
    d = MP4_DIR / mid / setting / task_dir
    return sorted(glob.glob(str(d / "*.mp4"))) if d.exists() else []


def load_progress(model_id, benchmark, robot, mode, domain):
    """In-progress marker written by run_leaderboard_eval.launch() per setting.
    Returns {done, total} if a *.progress.json exists and no final result yet, else None."""
    p = RESULTS_DIR / model_id / f"{benchmark}_{robot}_{mode}_{domain}.progress.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def collect_tags():
    """Per-(model_id) tag = the checkpoint leaf read live from the saved result
    JSON (results/<mid>/*.json -> `ckpt`), so each row's small tag shows the
    actual checkpoint of that model run — sourced from the eval results, not
    hardcoded. Mirrored across all modes (the checkpoint is per model run, not
    per control mode), with an 'all' alias the leaderboard client uses."""
    tags = {}
    for m in discover_models():
        mid = m["id"]
        leaf = m.get("tag", "")
        tags[mid] = {mode: leaf for mode in MODES}
        tags[mid]["all"] = leaf
    return tags


def all_progress():
    """{mid: {bench: {domain: {mode: {done,total}|None}}}} for in-progress settings.
    A setting counts as in-progress when it has a progress.json AND no REAL result
    yet — where "real" means a completed result with n_tasks>0. A 0-task placeholder
    (a run killed before any task finished) does NOT block the progress bar, so a
    re-run still shows live progress."""
    out = {m["id"]: {b: {d: {mode: None for mode in MODES} for d in DOMAINS} for b in BENCHMARKS}
           for m in discover_models()}
    for m in discover_models():
        mid = m["id"]
        for b in BENCHMARKS:
            for robot in ROBOTS.get(b, []):
                for dom in DOMAINS:
                    for mode in MODES:
                        res = load_result(mid, b, robot, mode, dom)
                        if res and res.get("n_tasks", 0) > 0:
                            continue  # real completed result -> no progress bar
                        pg = load_progress(mid, b, robot, mode, dom)
                        if pg:
                            out[mid][b][dom][mode] = pg
    return out


def detail_tasks_for(mid, mode_filter="all"):
    """Per-domain, per-benchmark grouped task rows for a model's detail page.
    mode_filter: 'all' or a specific mode (joint/robot_ee/camera_ee) to slice by."""
    modes = MODES if mode_filter == "all" else [mode_filter]
    out = {d: {b: [] for b in BENCHMARKS} for d in DOMAINS}
    for bench in BENCHMARKS:
        for robot in ROBOTS.get(bench, []):
            for mode in modes:
                for dom in DOMAINS:
                    res = load_result(mid, bench, robot, mode, dom)
                    setting = f"{bench}_{robot}_{mode}_{dom}"
                    if res:
                        for tname, tres in sorted(res.get("per_task", {}).items()):
                            task_dir = tname.replace("/", "_")
                            vids = [os.path.relpath(v, LB_ROOT) for v in find_videos(mid, setting, task_dir)]
                            # Subtask STRUCTURE (names + score weights) comes from the
                            # suite YAML; the measured completion rate per subtask is
                            # overlaid from the eval result's per_subtask by name.
                            _defs = task_subtask_defs(tname, bench, dom)
                            _measured = {s.get("name", ""): s.get("done_rate", 0)
                                         for s in (tres.get("per_subtask", []) or [])}
                            subs = ([{"name": nm, "done_rate": _measured.get(nm, 0)}
                                     for nm, _sc in _defs]
                                    if _defs else (tres.get("per_subtask", []) or []))
                            per_ep = [{"ep": e.get("ep"), "seed": e.get("seed"),
                                       "n_steps": e.get("n_steps", 0), "time_s": e.get("time_s", 0.0),
                                       "subs": [{"name": s.get("name", ""), "done": bool(s.get("done")),
                                                 "ok": bool(s.get("ok", True)), "must_hold": bool(s.get("must_hold", False)),
                                                 "latched_at_frame": s.get("latched_at_frame")}
                                                for s in e.get("subtasks", [])]}
                                      for e in tres.get("per_episode", [])]
                            out[dom][bench].append({
                                "task": tname.split("/")[-1],
                                "robot": robot, "mode": mode,
                                "lang": task_language(tname, bench, dom),
                                "score": _score_value(tres),
                                "sr": tres.get("success_rate", 0),
                                "eps": tres.get("n_episodes", 0),
                                "subs": "; ".join(f'{s.get("name","")}:{s.get("done_rate",0)*100:.0f}%' for s in subs),
                                "videos": vids,
                                "per_ep": per_ep,
                            })
                    # Surface tasks whose rollout mp4s are already on disk but not
                    # yet in the result JSON (eval in progress with a per-task-flushed
                    # driver, or a partial run) so videos appear as they're written —
                    # not only after the whole setting finishes.
                    _mp4_setting = MP4_DIR / mid / setting
                    if _mp4_setting.exists():
                        _have = {tn.replace("/", "_") for tn in (res.get("per_task", {}) if res else {})}
                        for _td in sorted(d.name for d in _mp4_setting.iterdir() if d.is_dir()):
                            if _td in _have:
                                continue
                            _vids = sorted(glob.glob(str(_mp4_setting / _td / "*.mp4")))
                            if not _vids:
                                continue
                            _tname = _td.replace("_", "/", 2)
                            out[dom][bench].append({
                                "task": _td.split("_", 2)[-1] if _td.count("_") >= 2 else _td,
                                "robot": robot, "mode": mode,
                                "lang": task_language(_tname, bench, dom),
                                "score": None, "sr": None, "eps": len(_vids),
                                "subs": "(in progress)",
                                "videos": [os.path.relpath(v, LB_ROOT) for v in _vids],
                                "per_ep": [], "_partial": True,
                            })
    for d in DOMAINS:
        for b in BENCHMARKS:
            out[d][b].sort(key=lambda t: (t["score"] is None, -(t["score"] or 0), t["task"]))
    return out


def _css():
    return """
:root{--bg:#f4f6fa;--card:#fff;--border:#e3e8ef;--text:#1f2937;--muted:#6b7280;
  --accent:#2563eb;--accent2:#1d4ed8;--good:#16a34a;--warn:#d97706;--bad:#9ca3af;--hover:#eef2ff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh}
.shell{max-width:1500px;margin:0 auto;padding:28px 24px}
header.top{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
header.top h1{font-size:22px;font-weight:700;letter-spacing:-.01em}
header.top .sub{color:var(--muted);font-size:13px;margin-top:2px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden;
  box-shadow:0 1px 2px rgba(16,24,40,.04)}
.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid var(--border);background:#fbfcfe}
.ctrl-group{display:inline-flex;align-items:center;gap:7px}
.seg{display:inline-flex;background:var(--card);border:1px solid var(--border);border-radius:999px;padding:2px}
.seg button{border:0;background:transparent;color:var(--muted);font-size:13px;font-weight:600;
  padding:5px 11px;border-radius:999px;cursor:pointer;transition:.12s}
.seg button.on{background:#eaf0ff;color:var(--accent2)}
.seg a{display:inline-block;border:0;background:transparent;color:var(--muted);font-size:13px;font-weight:600;
  padding:7px 12px;border-radius:6px;cursor:pointer;transition:.12s;text-decoration:none;white-space:nowrap}
.seg a:hover{background:var(--hover)}
.seg a.on{background:var(--card);color:var(--accent2);box-shadow:0 1px 2px rgba(16,24,40,.08)}
.ctrl-label{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.pill-select{min-height:28px;padding:0 28px 0 11px;border:1px solid var(--border);border-radius:999px;
  background:var(--card);color:var(--text);font-size:11px;font-weight:700;letter-spacing:.02em;
  cursor:pointer;appearance:none;background-image:linear-gradient(45deg,transparent 50%,#6b7280 50%),linear-gradient(135deg,#6b7280 50%,transparent 50%);
  background-position:calc(100% - 13px) 11px,calc(100% - 9px) 11px;background-size:4px 4px,4px 4px;background-repeat:no-repeat}
.pill-select:hover,.pill-select:focus{border-color:#9db5ed;outline:none;box-shadow:0 0 0 2px rgba(37,99,235,.1)}
table{width:100%;border-collapse:collapse}
th{background:#f8fafc;color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.05em;padding:11px 12px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}
th.l{text-align:left}
th.is-sorted{color:var(--accent2);background:#eef3ff}
th.is-sorted::after{content:' ↓';font-size:10px}
td{padding:13px 12px;border-bottom:1px solid var(--border);font-size:14px;text-align:right;white-space:nowrap}
td.l{text-align:left}
td.is-sorted{background:rgba(37,99,235,.035)}
tbody tr{cursor:pointer;transition:background .1s}
tbody tr:hover{background:var(--hover)}
.cell-link{display:block;color:inherit;text-decoration:none;margin:-13px -12px;padding:13px 12px}
.rank{color:var(--muted);font-weight:700;font-size:13px;width:42px}
.model{font-weight:700;font-size:14px;color:var(--text)}
.model .org{font-weight:500;font-size:11px;color:var(--muted);display:block}
.model .tag{font-weight:500;font-size:10px;color:var(--muted)}
.val{font-variant-numeric:tabular-nums;font-weight:600}
.val.good{color:var(--good)} .val.mid{color:var(--warn)} .val.none{color:#c7cdd6;font-weight:500}
.overall{font-weight:800;font-size:15px;color:var(--accent2)}
.note{text-align:center;color:var(--muted);font-size:12px;padding:14px}
.back{display:inline-flex;align-items:center;gap:6px;color:var(--accent2);text-decoration:none;font-size:14px;font-weight:600;margin-bottom:14px;cursor:pointer}
.back:hover{text-decoration:underline}
h2.cap{font-size:13px;font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:.05em;margin:16px 14px 6px}
.tr{display:flex;gap:10px;align-items:flex-start;padding:11px 14px;border-bottom:1px solid var(--border)}
.tr:last-child{border-bottom:0}
.tr .info{flex:1;min-width:0}
.tr .tname{font-weight:700;font-size:13px}
.tr .lang{font-size:11px;color:var(--muted)}
.tr .subs{font-size:11px;color:var(--muted);margin-top:2px}
.tr .mets{font-size:12px;font-weight:600;margin-top:3px}
.tr .mets .s{color:var(--accent2)} .tr .mets .r{color:var(--muted);margin-left:8px}
video{width:260px;max-width:260px;display:block;background:#000;border-radius:6px;margin:3px 0}
/* Subtask trigger lights: one chip per subtask, turns green once playback passes
   the frame at which that subtask first latched done (latched_at_frame). Clicking
   an episode re-evaluates the lights against that episode's timeline. */
.sublights{display:flex;flex-direction:column;gap:3px;margin:4px 0 2px;max-width:560px}
.sublights .row{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text)}
.sublights .dot{flex:0 0 auto;width:11px;height:11px;border-radius:50%;background:#cbd2dc;transition:background .12s,box-shadow .12s}
.sublights .dot.on{background:var(--good);box-shadow:0 0 0 2px rgba(22,163,74,.18)}
.sublights .dot.never{background:#e5e7eb}
.sublights .nm{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#4b5563}
.sublights .at{flex:0 0 auto;font-size:10px;color:var(--good);font-variant-numeric:tabular-nums;font-weight:700;min-width:46px;text-align:right}
.sublights .at.pending{color:#c7cdd6;font-weight:500}
/* Timeline strip with a marker at each triggered subtask's frame. */
.tlbar{position:relative;height:10px;width:260px;background:#eef1f6;border-radius:5px;margin:2px 0;overflow:hidden}
.tlbar .mk{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--good);border-radius:2px}
.tlbar .playhead{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--accent2);opacity:.7}
/* Episode links: wrap into a tidy multi-row chip grid (30 eps no longer one row). */
.eplinks{margin-top:5px;max-width:560px}
.eplinks .grp{display:flex;align-items:center;flex-wrap:wrap;gap:3px 4px;margin:2px 0}
.eplinks .glbl{font-size:10px;color:var(--muted);font-weight:700;margin-right:3px;flex:0 0 auto}
.eplinks a{display:inline-block;font-size:11px;color:var(--accent2);cursor:pointer;flex:0 0 auto;
  padding:2px 6px;border-radius:5px;background:#f1f4fa;border:1px solid #e3e8ef;line-height:1.2}
.eplinks a:hover{background:var(--hover)}
.eplinks a.cur{font-weight:700;background:var(--accent);color:#fff;border-color:var(--accent)}
.pbar{width:60px;height:8px;background:#eef1f6;border-radius:4px;display:inline-block;vertical-align:middle;overflow:hidden}
.pbar span{display:block;height:100%;background:var(--accent);border-radius:4px;transition:width .3s}
.cellprog{font-size:10px;color:var(--muted)}
/* Top "currently evaluating" panel: a nice dedicated spot for in-progress bars. */
.progpanel{margin-bottom:16px}
.progpanel .hd{font-size:13px;font-weight:700;color:var(--accent2);margin-bottom:8px;display:flex;align-items:center;gap:8px}
.progpanel .hd .live{width:8px;height:8px;border-radius:50%;background:var(--bad);animation:pulse 1.2s infinite}
@keyframes pulse{0%{opacity:.3}50%{opacity:1}100%{opacity:.3}}
.progpanel .prow{display:flex;align-items:center;gap:10px;padding:6px 10px;border:1px solid var(--border);
  border-radius:8px;margin-bottom:6px;background:#fbfcfe;font-size:12px}
.progpanel .prow .ptag{flex:0 0 auto;font-weight:700;color:var(--text)}
.progpanel .prow .pmeta{flex:0 0 auto;color:var(--muted);font-size:11px}
.progpanel .prow .pbar{flex:1 1 auto;width:auto;height:8px;max-width:340px}
.progpanel .prow .pnum{flex:0 0 auto;font-variant-numeric:tabular-nums;font-weight:600;color:var(--muted)}
.progpanel.hide{display:none}
.pane{display:none} .pane.on{display:block}
.benchsec{display:none} .benchsec.on{display:block}
.benchpill{display:none;margin:10px 14px;padding:8px 12px;border:1px solid var(--accent);border-radius:8px;
  background:#eef2ff;color:var(--accent2);font-size:12px;font-weight:600;cursor:pointer;text-align:center}
.benchpill:hover{background:var(--hover)}
.empty{padding:18px;color:var(--muted);font-size:13px;text-align:center}
"""


def base_script(base_url):
    """Define a global BASE (full cloudproxy-safe prefix) WITHOUT destroying the
    DOM. Links/videos reference it via JS so handlers + rendered tables survive."""
    seg = base_url.strip("/")
    marker = f"/{seg}" if seg else ""
    return f"""
<script>
window.BASE = (function(){{
  var p = window.location.pathname;
  if(!"{seg}"){{ return ""; }}
  var marker = "{marker}";
  var i = p.lastIndexOf(marker);
  if(i < 0){{ return "{marker}"; }}
  return p.slice(0, i + marker.length);
}})();
</script>
"""


def _esc(s):
    return html.escape(str(s)) if s is not None else ""


def render_leaderboard(base_url):
    payload = {"models": discover_models(), "benchmarks": BENCHMARKS, "domains": DOMAINS,
               "data": all_table_data(), "tags": collect_tags(), "progress": all_progress()}
    bench_headers = "".join(
        f'<th data-dimension="{b}">{b.replace("_", " ")}</th>' for b in BENCHMARKS)
    dimension_options = '<option value="overall">Overall</option>' + "".join(
        f'<option value="{b}">{b.replace("_", " ")}</option>' for b in BENCHMARKS)
    # Server-render a navigation-capable skeleton. JavaScript replaces it with
    # live scores on initialization, but proxy timing or blocked inline handlers
    # can no longer make the model rows unclickable.
    initial_rows = []
    for idx, model in enumerate(payload["models"], 1):
        mid = quote(str(model["id"]), safe="")
        href = f'model/{mid}?mode=all&amp;domain=all'
        cells = "".join(
            f'<td><a class="cell-link" href="{href}&amp;bench={bi}">—</a></td>'
            for bi, _benchmark in enumerate(BENCHMARKS)
        )
        initial_rows.append(
            f'<tr><td class="l rank">{idx}</td>'
            f'<td class="l"><a class="cell-link" href="{href}"><span class="model">'
            f'{_esc(model["name"])}<span class="org">{_esc(model.get("org", ""))}</span>'
            f'<span class="tag">{_esc(model.get("tag", ""))}</span></span></a></td>'
            f'<td class="overall none"><a class="cell-link" href="{href}">—</a></td>'
            f'{cells}</tr>'
        )
    initial_rows = "".join(initial_rows)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BiDex-100 Leaderboard</title>
{base_script(base_url)}
<style>{_css()}</style></head><body>
<div class="shell">
<header class="top"><div><h1>BiDex-100 Leaderboard</h1>
  <div class="sub">BiDex-100 baselines · toggle Metric / Domain / Control Mode · click a model for per-task rollout videos</div></div></header>
<div class="card">
  <div class="controls">
    <span class="ctrl-group"><label class="ctrl-label" for="selDimension">Dimension</label>
      <select class="pill-select" id="selDimension">{dimension_options}</select>
    </span>
    <span class="ctrl-group"><span class="ctrl-label">Rank by</span>
      <span class="seg" id="segMetric">
        <button class="on" data-v="sr">Success Rate</button>
        <button data-v="score">Score</button>
      </span>
    </span>
    <span class="ctrl-group"><span class="ctrl-label">Domain</span>
      <span class="seg" id="segDomain">
        <button class="on" data-v="all">All</button>
        <button data-v="in_domain">In-Domain</button>
        <button data-v="out_of_domain">Out-of-Domain</button>
      </span>
    </span>
    <span class="ctrl-group"><span class="ctrl-label">Control Mode</span>
      <span class="seg" id="segMode">
        <button class="on" data-v="all">All</button>
        <button data-v="joint">joint</button>
        <button data-v="robot_ee">robot_ee</button>
        <button data-v="camera_ee">camera_ee</button>
      </span>
    </span>
  </div>
  <div id="progpanel" class="progpanel hide"></div>
  <table><thead><tr>
    <th class="l">Rank</th><th class="l">Model</th><th data-dimension="overall">Overall</th>{bench_headers}
  </tr></thead><tbody id="rows">{initial_rows}</tbody></table>
  <div class="note">Success Rate = mean episode success · Score = mean subtask completion rate · click a row for per-task results + rollout video</div>
</div>
</div>
<script>
const P = {json.dumps(payload)};
let metric = 'sr', dimension = 'overall', domain = 'all', mode = 'all';
function fmt(m){{ return (m==null)? '—' : (m*100).toFixed(2); }}
function cls(v100){{ return (v100==null)?'none':(v100>=50?'good':(v100>=1?'mid':'none')); }}
function cell(mid,b){{
  if(domain === 'all'){{
    // "All" = the per-task average across In-Domain and Out-of-Domain for this
    // (model, benchmark, mode) — each domain weighted by its task count (matches
    // the backend per-task micro-average), NOT a flat 50/50 of the two scores.
    // Averages whatever's available so the live leaderboard still shows a number
    // as results stream in (returns null only when BOTH domains are empty).
    const dd = P.data[mid] && P.data[mid][b];
    if(!dd) return null;
    var id = dd['in_domain'] ? dd['in_domain'][mode] : null;
    var od = dd['out_of_domain'] ? dd['out_of_domain'][mode] : null;
    if(!id && !od) return null;
    var s=0, r=0, nn=0;
    if(id){{ s+=id.score*id.n; r+=id.sr*id.n; nn+=id.n; }}
    if(od){{ s+=od.score*od.n; r+=od.sr*od.n; nn+=od.n; }}
    return {{ score:s/nn, sr:r/nn, n:nn }};
  }}
  const d=P.data[mid]&&P.data[mid][b]&&P.data[mid][b][domain]; return d? d[mode] : null;
}}
function overall(mid){{
  // Overall = per-task micro across the selected domain (and, for 'all', across
  // both domains): each benchmark weighted by its task count, matching
  // scripts/evaluation/summarize_eval_results.py. A flat mean over benchmarks would
  // equal-weight e.g. Memory (20 tasks) and Generalization (40 tasks); weighting
  // by cell.n gives the true task-level average.
  let ws=0, wn=0;
  for(const b of P.benchmarks){{ const m=cell(mid,b); if(m && m.n>0){{ ws+=m[metric]*m.n; wn+=m.n; }} }}
  return wn? ws/wn : null;
}}
function sortValue(mid){{
  if(dimension === 'overall') return overall(mid);
  const m = cell(mid, dimension);
  return m ? m[metric] : null;
}}
const MODELNAME = {json.dumps({m["id"]: m["name"] for m in discover_models()})};
// Build the top "currently evaluating" panel: one row per (model,mode,benchmark,domain)
// that has a progress.json but no final result yet. Final results hide automatically
// (all_progress only lists settings whose terminal JSON is absent).
function renderProg(){{
  var rows=[];
  for(const mid in P.progress){{
    for(const b in P.progress[mid]){{
      for(const d in P.progress[mid][b]){{
        for(const md in P.progress[mid][b][d]){{
          var pg=P.progress[mid][b][d][md]; if(!pg) continue;
          var ttl = pg.total||1; var pct = Math.round(100*(pg.done||0)/ttl);
          rows.push('<div class="prow"><span class="ptag">'+MODELNAME[mid]+' · '+md+'</span>'
            +'<span class="pmeta">'+b.replace(/_/g,' ')+' / '+d.replace(/_/g,'-')+'</span>'
            +'<span class="pbar"><span style="width:'+pct+'%"></span></span>'
            +'<span class="pnum">'+(pg.done||0)+'/'+ttl+'</span></div>');
        }}
      }}
    }}
  }}
  var el=document.getElementById('progpanel');
  if(!rows.length){{ el.className='progpanel hide'; el.innerHTML=''; return; }}
  el.className='progpanel';
  el.innerHTML='<div class="hd"><span class="live"></span>Currently evaluating ('+rows.length+' settings)</div>'+rows.join('');
}}
function render(){{
  renderProg();
  document.querySelectorAll('th[data-dimension]').forEach(function(th){{
    th.classList.toggle('is-sorted', th.dataset.dimension === dimension);
  }});
  const order=[...P.models].sort((a,b)=>{{
    const av=sortValue(a.id), bv=sortValue(b.id);
    return (bv==null?-1:bv) - (av==null?-1:av) || a.name.localeCompare(b.name);
  }});
  document.getElementById('rows').innerHTML = order.map((m,idx)=>{{
    const ov=overall(m.id); const ov100=ov==null?null:ov*100;
    const tag = (P.tags[m.id] && P.tags[m.id][mode]) ? P.tags[m.id][mode] : (m.tag||'');
    let cells=''; P.benchmarks.forEach(function(b,bi){{
      const mm=cell(m.id,b); const v=mm? mm[metric]*100 : null;
      // in-progress setting? show a progress bar instead of '—'
      const pg = (P.progress[m.id]&&P.progress[m.id][b]&&P.progress[m.id][b][domain]&&P.progress[m.id][b][domain][mode])||null;
      let inner;
      if(mm){{ inner = '<span class="val '+cls(v)+'">'+fmt(mm[metric])+'</span>'; }}
      else if(pg){{ const pct = pg.total? Math.round(100*pg.done/pg.total):0; inner = '<span class="cellprog"><span class="pbar"><span style="width:'+pct+'%"></span></span> '+pg.done+'/'+pg.total+'</span>'; }}
      else{{ inner = '<span class="val none">—</span>'; }}
      const cellHref=BASE+'/model/'+encodeURIComponent(m.id)+'?mode='+encodeURIComponent(mode)+'&domain='+encodeURIComponent(domain)+'&bench='+bi;
      cells += '<td class="'+(dimension===b?'is-sorted':'')+'"><a class="cell-link" href="'+cellHref+'">'+inner+'</a></td>';
    }});
    const href=BASE+'/model/'+encodeURIComponent(m.id)+'?mode='+encodeURIComponent(mode)+'&domain='+encodeURIComponent(domain);
    return '<tr>'+
      '<td class="l rank">'+(idx+1)+'</td>'+
      '<td class="l"><a class="cell-link" href="'+href+'"><span class="model">'+m.name+'<span class="org">'+m.org+'</span><span class="tag">'+tag+'</span></span></a></td>'+
      '<td class="overall '+cls(ov100)+(dimension==='overall'?' is-sorted':'')+'"><a class="cell-link" href="'+href+'">'+fmt(ov)+'</a></td>'+cells+'</tr>';
  }}).join('');
}}
function initLeaderboard(){{
  document.getElementById('selDimension').onchange=function(){{ dimension=this.value; render(); }};
  document.querySelectorAll('#segMetric button').forEach(function(b){{
    b.onclick=function(){{ document.querySelectorAll('#segMetric button').forEach(function(x){{x.classList.remove('on');}}); b.classList.add('on'); metric=b.dataset.v; render(); }};
  }});
  document.querySelectorAll('#segDomain button').forEach(function(b){{
    b.onclick=function(){{ document.querySelectorAll('#segDomain button').forEach(function(x){{x.classList.remove('on');}}); b.classList.add('on'); domain=b.dataset.v; render(); }};
  }});
  document.querySelectorAll('#segMode button').forEach(function(b){{
    b.onclick=function(){{ document.querySelectorAll('#segMode button').forEach(function(x){{x.classList.remove('on');}}); b.classList.add('on'); mode=b.dataset.v; render(); }};
  }});
  render();
  // Live polling: refresh scores + the in-progress panel every 5s without
  // losing the current Metric/Domain/Mode toggles (held in JS vars). Eval writes
  // results per-task every few seconds, so this surfaces them in real time.
  setInterval(function(){{
    fetch(BASE+'/api/leaderboard', {{cache:'no-store'}}).then(function(r){{return r.json();}}).then(function(np){{
      P.data=np.data; P.tags=np.tags; P.progress=np.progress; render();
    }}).catch(function(){{}});
  }}, 5000);
}}
if(document.readyState === 'loading'){{
  document.addEventListener('DOMContentLoaded', initLeaderboard, {{once:true}});
}} else {{
  initLeaderboard();
}}
</script>
</body></html>"""


def render_model_detail(mid, base_url, mode_filter="all", domain_filter=None, bench_idx=None):
    _fw, _ck = _model_meta(mid)
    m = next((x for x in discover_models() if x["id"] == mid),
             {"id": mid, "name": mid, "org": _fw, "tag": _ck})
    grouped = detail_tasks_for(mid, mode_filter)
    def render_bench_section(dom, bi, bench, rows):
        if not rows:
            return (f'<h2 class="cap" id="cap_{dom}_{bi}">{bench.replace("_"," ")}</h2>'
                    f'<div class="empty">No {bench.replace("_"," ")} · {dom.replace("_","-")} '
                    f'results yet for this model.</div>')
        items = []
        for ti, t in enumerate(rows):
            vids = t["videos"]
            per_ep = t.get("per_ep", [])
            if vids:
                vid_id = f"v_{dom}_{bi}_{ti}"
                led_id = f"led_{dom}_{bi}_{ti}"
                # Pair each video with ITS OWN episode's subtask LEDs by the (seed,
                # ep) encoded in the mp4 filename — NOT by list index. per_episode is
                # stored in EVAL-FINISH order (ragged, not seed/ep), and only some of
                # its entries have an mp4 on disk; vids[] is a sorted-by-filename
                # glob. Index-pairing vids[i] with per_ep[i] was WRONG: a *failed*
                # episode's video would light all-green from per_ep[0]'s data (often
                # the lone early-finishing success). Emit ONLY the video'd episodes,
                # in vids[] order, each carrying its own truth so the frontend's
                # per[i] (== data-epidx i) matches video i.
                _by_key = {(e.get("seed"), e.get("ep")): e for e in per_ep}
                _vid_eps, aligned = [], []
                for v in vids:
                    stem = os.path.basename(v).rsplit(".", 1)[0]  # s0_ep000
                    pp = stem.split("_")                          # ['s0','ep000']
                    try:
                        sv, ev = int(pp[0][1:]), int(pp[1][2:])
                    except (ValueError, IndexError):
                        sv, ev = None, None
                    _vid_eps.append((sv, ev))
                    aligned.append(_by_key.get((sv, ev), {}))
                groups = {}
                for i, (sv, ev) in enumerate(_vid_eps):
                    groups.setdefault(sv if sv is not None else 0, []).append((i, ev if ev is not None else 0))
                ghtml = []
                for seed in sorted(groups, key=lambda s: (s is None, s)):
                    chips = " ".join(
                        f'<a href="../video/{quote(vids[i], safe="/")}" data-video="{vid_id}" '
                        f'data-rel="{_esc(vids[i])}" data-epidx="{i}" '
                        f'class="{"cur" if i==0 else ""}">e{epp}</a>'
                        for i, epp in groups[seed])
                    ghtml.append(f'<div class="grp"><span class="glbl">s{seed if seed is not None else "?"}</span>{chips}</div>')
                first_video_url = "../video/" + quote(vids[0], safe="/")
                vid = (f'<video id="{vid_id}" data-vid controls playsinline preload="metadata" '
                       f'src="{first_video_url}"></video>'
                       f'<div class="sublights" id="{led_id}" data-per="{_esc(json.dumps(aligned))}"></div>'
                       f'<div class="tlbar" id="tl_{dom}_{bi}_{ti}"></div>'
                       f'<div class="eplinks">{" ".join(ghtml)}</div>')
            else:
                vid = '<span style="color:#c7cdd6;font-size:12px">no video</span>'
            if t.get("_partial"):
                _r_txt, _s_txt = "in progress", "—"
            else:
                _r_txt = f"SR {(t['sr'] or 0) * 100:.2f}%"
                _s_txt = f"Score {(t['score'] or 0) * 100:.2f}"
            items.append(
                f'<div class="tr"><div class="info">'
                f'<div class="tname">{_esc(t["task"])} '
                f'<span style="font-size:10px;color:#5b6472;font-weight:700;background:#e8edf6;'
                f'padding:1px 6px;border-radius:4px;margin-left:4px">{_esc(t["mode"])}</span></div>'
                f'<div class="lang">{_esc(t["lang"])}</div>'
                f'<div class="subs">subs: {t["subs"] or "—"} · eps {t["eps"]}</div>'
                f'<div class="mets"><span class="r">{_r_txt}</span>'
                f'<span class="s">{_s_txt}</span></div></div>'
                f'<div>{vid}</div></div>')
        return f'<h2 class="cap" id="cap_{dom}_{bi}">{bench.replace("_"," ")}</h2>' + "".join(items)

    def render_pane(dom):
        # Render ONLY the selected benchmark (one section per pane) so the page
        # stays small (embedding all 6 benches × 2 domains × per-episode data-per
        # made the page tens of MB and the browser failed to render videos).
        sec = render_bench_section(dom, init_bench_idx, sel_bench, grouped[dom][sel_bench])
        _on = " on" if dom == init_dom else ""  # active domain visible on load
        return f'<div class="pane{_on}" data-d="{dom}">{sec}</div>'

    doms_with_data = [d for d in DOMAINS if any(grouped[d][b] for b in BENCHMARKS)]
    init_dom = domain_filter if domain_filter in doms_with_data else (doms_with_data[0] if doms_with_data else None)
    try:
        init_bench_idx = int(bench_idx) if str(bench_idx).strip() not in ("", "None") else 0
    except Exception:
        init_bench_idx = 0
    if init_bench_idx < 0 or init_bench_idx >= len(BENCHMARKS):
        init_bench_idx = 0
    sel_bench = BENCHMARKS[init_bench_idx]
    dom_buttons = "".join(
        f'<button class="{"on" if d==init_dom else ""}" data-v="{d}">{d.replace("_","-").title()}</button>'
        for d in doms_with_data)
    # Benchmark tabs = navigation links (full page is now just one bench, so a
    # ?bench= reload is fast and not janky). All 6 always visible.
    bench_buttons = "".join(
        f'<a class="{"on" if bi==init_bench_idx else ""}" '
        f'href="?mode={mode_filter}&domain={init_dom or ""}&bench={bi}">{b.replace("_"," ")}</a>'
        for bi, b in enumerate(BENCHMARKS))
    mode_buttons = "".join(
        f'<button class="{"on" if mode_filter==md else ""}" onclick="location.href=\'?mode={md}'
        f'&domain={init_dom or ""}&bench={init_bench_idx}\'">{md}</button>'
        for md in ["all"] + MODES)
    mode_query = mode_filter
    panes = "".join(render_pane(d) for d in doms_with_data)
    _sub_bits = [_esc(b) for b in
                 (m.get('org') or '', (collect_tags().get(mid, {}).get(mode_filter)) or m.get('tag', '') or '')
                 if b]
    _sub_prefix = (' · '.join(_sub_bits) + ' · ') if _sub_bits else ''
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(m['name'])} — BiDex-100</title>
{base_script(base_url)}
<style>{_css()}</style></head><body>
<div class="shell">
<a class="back" id="backBtn" href="../">← Back to Leaderboard</a>
<header class="top"><div><h1>{_esc(m['name'])}</h1>
  <div class="sub">{_sub_prefix}per-task rollout videos · green LED = subtask completed (must_hold checked at final frame)</div></div></header>
<div class="card">
  <div class="controls">
    <span class="ctrl-label">Domain</span>
    <span class="seg" id="segDom">{dom_buttons}</span>
    <span class="ctrl-label" style="margin-left:14px">Benchmark</span>
    <span class="seg" id="segBench">{bench_buttons}</span>
    <span class="ctrl-label" style="margin-left:14px">Control Mode</span>
    <span class="seg" id="segMode">{mode_buttons}</span>
  </div>
  <div class="benchpill" id="livePill">↻ new rollout results — click to refresh</div>
  {panes if panes else '<div class="empty">No evaluation results for this model.</div>'}
</div>
</div>
<script>
const MODE = {json.dumps(mode_query)};
const INIT_DOM = {json.dumps(init_dom)};
const BENCH_IDX = {json.dumps(bench_idx)};
// Build the subtask trigger-light chips + timeline markers for one episode.
// Each chip lights green the moment playback (frame = currentTime/duration *
// n_steps) passes that subtask's latched_at_frame. latched_at_frame == null ->
// never triggered (stays grey). This makes a false/early sim trigger visible:
// a light turning green at an implausible frame tells the viewer the subtask
// declared done at that moment.
function renderLights(ledEl, tlEl, epIdx){{
  var per = JSON.parse(ledEl.getAttribute('data-per')||'[]');
  ledEl.activeEp = epIdx;
  if(!per.length || epIdx>=per.length){{ ledEl.innerHTML = '<span class="lbl" style="color:#c7cdd6">no episode data</span>'; tlEl.innerHTML=''; return; }}
  var ep = per[epIdx]; var n = ep.n_steps||1; var subs = ep.subs||[];
  ledEl._n = n;
  var h = '';
  subs.forEach(function(s,i){{
    var lat = s.latched_at_frame;
    var ok = s.ok !== false;
    var atTxt = (lat!=null && ok) ? ('@'+lat+' ('+(lat/25).toFixed(1)+'s)') : '—';
    h += '<div class="row"><span class="dot '+(lat==null||!ok?'never':'')+'" data-lat="'+(lat==null?'':lat)+'" data-ok="'+(ok?'true':'false')+'" data-i="'+i+'"></span>'
       + '<span class="nm" title="'+(s.name||'').replace(/"/g,'&quot;')+'">'+i+': '+_q(s.name||'')+'</span>'
       + '<span class="at '+(lat!=null&&ok?'':'pending')+'">'+atTxt+'</span></div>';
  }});
  ledEl.innerHTML = h;
  // timeline markers at each triggered subtask's frame (only for ok=true)
  var tl = '';
  subs.forEach(function(s,i){{
    var lat = s.latched_at_frame; var sok = s.ok !== false;
    if(lat==null || !sok) return;
    var pct = Math.max(0, Math.min(100, 100*lat/n));
    tl += '<div class="mk" style="left:'+pct+'%" title="'+_q(s.name||'')+' @'+lat+'"></div>';
  }});
  tl += '<div class="playhead" id="'+tlEl.id+'_ph" style="left:0%"></div>';
  tlEl.innerHTML = tl;
}}
function _q(s){{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').slice(0,40); }}
// Recompute light colours from the video's current playback position.
function syncLights(v, ledEl, tlEl){{
  if(!ledEl || !ledEl._n) return;
  var n = ledEl._n; var dur = v.duration||0; var cur = v.currentTime||0;
  var frame = (dur>0) ? (cur/dur)*n : 0;
  ledEl.querySelectorAll('.dot').forEach(function(d){{
    var lat = d.getAttribute('data-lat');
    var ok = d.getAttribute('data-ok') !== 'false';
    if(lat!=='' && lat!=null && parseFloat(lat)<=frame && ok){{ d.classList.add('on'); }}
    else {{ d.classList.remove('on'); }}
  }});
  var ph = document.getElementById(tlEl.id+'_ph');
  if(ph) ph.style.left = (dur>0 ? Math.min(100,100*cur/dur) : 0) + '%';
}}
function initDetail(){{
  // initial video src + lights for ep0 of each task; bind timeupdate for live lighting.
  document.querySelectorAll('video[data-vid]').forEach(function(v){{
    var links = document.querySelectorAll('a[data-video="'+v.id+'"]');
    if(!v.getAttribute('src') && links.length){{ v.src = BASE + '/video/' + links[0].getAttribute('data-rel'); }}
    var ledId = v.id.replace('v_','led_');
    var tlId  = v.id.replace('v_','tl_');
    var ledEl = document.getElementById(ledId);
    var tlEl  = document.getElementById(tlId);
    if(ledEl && tlEl) renderLights(ledEl, tlEl, 0);
    v.addEventListener('timeupdate', function(){{ syncLights(v, ledEl, tlEl); }});
    v.addEventListener('loadedmetadata', function(){{ syncLights(v, ledEl, tlEl); }});
  }});
  // ep click: swap video + re-render lights for that ep, then re-sync.
  document.querySelectorAll('a[data-video]').forEach(function(a){{
    a.style.cursor='pointer';
    a.onclick = function(e){{
      e.preventDefault();
      var vid = a.getAttribute('data-video');
      var v = document.getElementById(vid);
      v.src = BASE + '/video/' + a.getAttribute('data-rel');
      v.load(); try{{ v.play(); }}catch(err){{}}
      document.querySelectorAll('a[data-video="'+vid+'"]').forEach(function(x){{ x.classList.remove('cur'); }});
      a.classList.add('cur');
      var ledEl = document.getElementById(vid.replace('v_','led_'));
      var tlEl  = document.getElementById(vid.replace('v_','tl_'));
      if(ledEl && tlEl){{ renderLights(ledEl, tlEl, parseInt(a.getAttribute('data-epidx')||'0',10)); syncLights(v, ledEl, tlEl); }}
    }};
  }});
  // ---- Domain toggle (benchmark tabs are plain ?bench= nav links now) ----
  function _setUrl(k, v){{ try{{ var u=new URL(window.location.href); u.searchParams.set(k, v); window.history.replaceState({{}},'',u.toString()); }}catch(e){{}} }}
  document.querySelectorAll('#segDom button').forEach(function(b){{
    b.onclick = function(){{
      document.querySelectorAll('#segDom button').forEach(function(x){{ x.classList.remove('on'); }});
      b.classList.add('on');
      var v = b.dataset.v;
      document.querySelectorAll('.pane').forEach(function(p){{ p.classList.toggle('on', p.dataset.d === v); }});
      _setUrl('domain', v);
      window.scrollTo(0, 0);
    }};
  }});
  // restore scroll only after a "refresh" reload (not on fresh navigations)
  try{{ if(sessionStorage.getItem('detailReload')){{ sessionStorage.removeItem('detailReload'); var sc=parseInt(sessionStorage.getItem('detailScroll')||'0',10); window.scrollTo(0, sc); }} }}catch(e){{}}
}}
if(document.readyState === 'loading'){{
  document.addEventListener('DOMContentLoaded', initDetail, {{once:true}});
}} else {{
  initDetail();
}}
// Live detail: poll every 6s; if new tasks/videos landed AND no video is playing,
// surface a non-intrusive "click to refresh" pill instead of a destructive reload
// (which used to reset the domain and make all videos disappear). Clicking the
// pill saves the scroll position, reloads, and restores it here.
function _anyVideoPlaying(){{
  return Array.prototype.some.call(document.querySelectorAll('video'), function(v){{ return v && !v.paused && v.currentTime>0 && !v.ended; }});
}}
setInterval(function(){{
  if(_anyVideoPlaying()) return;
  fetch(window.location.href, {{cache:'no-store'}}).then(function(r){{return r.text();}}).then(function(html){{
    var nNew=(html.match(/class="tr"/g)||[]).length;
    var nOld=document.querySelectorAll('.tr').length;
    if(nNew>nOld){{ var p=document.getElementById('livePill'); if(p) p.style.display='block'; }}
  }}).catch(function(){{}});
}}, 6000);
function initLivePill(){{
  var pill=document.getElementById('livePill');
  if(pill) pill.onclick = function(){{ try{{ sessionStorage.setItem('detailReload','1'); sessionStorage.setItem('detailScroll', window.scrollY); }}catch(e){{}} window.location.reload(); }};
}}
if(document.readyState === 'loading'){{
  document.addEventListener('DOMContentLoaded', initLivePill, {{once:true}});
}} else {{
  initLivePill();
}}
</script>
</body></html>"""


def _resolve_lb_root(cli_lb_root):
    """LB_ROOT from --lb-root (CLI) or DSB_LB_ROOT (env). NO default path is
    hardcoded: the eval-output root must be supplied because its location may
    move (only its on-disk structure — results/<model_id>/, mp4s/<model_id>/ —
    is assumed stable). Raises SystemExit with a clear message if neither is
    given or the path does not exist."""
    cand = cli_lb_root or os.environ.get("DSB_LB_ROOT")
    if not cand:
        raise SystemExit(
            "baseline_leaderboard_server: --lb-root <path> is required "
            "(or set DSB_LB_ROOT). No eval-output path is hardcoded; pass your "
            "leaderboard root, e.g. --lb-root outputs/eval/leaderboard")
    p = Path(cand).expanduser()
    if not p.exists():
        raise SystemExit(f"baseline_leaderboard_server: --lb-root does not exist: {p}")
    return p.resolve()


def create_app(base_url="/baselines", lb_root=None):
    global LB_ROOT, RESULTS_DIR, MP4_DIR
    LB_ROOT = _resolve_lb_root(lb_root)
    RESULTS_DIR = LB_ROOT / "results"
    MP4_DIR = LB_ROOT / "mp4s"
    app = Flask(__name__)
    prefix = base_url.rstrip("/")

    @app.after_request
    def _no_cache(resp):
        # Eval writes results every few seconds; never let a browser/proxy serve a
        # stale page, JSON, or video-range. This is what keeps scores/videos live.
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    if prefix:
        _orig = app.wsgi_app

        def _strip(environ, start_response):
            path = environ.get("PATH_INFO", "") or "/"
            if path == prefix or path.startswith(prefix + "/"):
                environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + prefix
                environ["PATH_INFO"] = path[len(prefix):] or "/"
            return _orig(environ, start_response)

        app.wsgi_app = _strip

    @app.route("/")
    def index():
        return render_leaderboard(base_url)

    @app.route("/model/<mid>")
    def model_detail(mid):
        return render_model_detail(mid, base_url, request.args.get("mode", "all"),
                                   request.args.get("domain"), request.args.get("bench"))

    @app.route("/api/leaderboard")
    def api_leaderboard():
        from flask import jsonify
        return jsonify({"models": discover_models(), "benchmarks": BENCHMARKS, "domains": DOMAINS,
                        "data": all_table_data(), "tags": collect_tags(),
                        "progress": all_progress()})

    @app.route("/video/<path:relpath>")
    def video(relpath):
        target = (LB_ROOT / relpath).resolve()
        try:
            target.relative_to(LB_ROOT.resolve())
        except ValueError:
            return "out of root", 400
        if not target.exists() or not target.is_file():
            return "not found", 404
        return send_file(str(target), mimetype="video/mp4", as_attachment=False, conditional=True)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=12006)
    ap.add_argument("--base-url", default="/baselines",
                    help="reverse-proxy prefix (empty for direct http://host:port)")
    ap.add_argument("--lb-root", default=None,
                    help="leaderboard root dir (holds results/ and mp4s/). Required "
                         "unless DSB_LB_ROOT is set; NO default path is hardcoded so "
                         "the leaderboard can serve any eval-output tree (its location "
                         "may move; only the results/<model_id>/ + mp4s/<model_id>/ "
                         "structure is assumed stable).")
    args = ap.parse_args()
    app = create_app(args.base_url, args.lb_root)
    print(f"[leaderboard] listening on {args.host}:{args.port}  base_url={args.base_url!r}  lb_root={LB_ROOT}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
