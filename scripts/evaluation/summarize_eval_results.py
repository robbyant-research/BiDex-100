#!/usr/bin/env python3
"""Summarize BiDex-100 eval results into a single JSON per model.

Reads all ``<results_dir>/<model_id>/*.json`` (excluding .progress.json and .tmp),
aggregates per-benchmark / per-domain / per-task success rates and scores, matching
the web leaderboard format, and writes ``<results_dir>/<model_id>/summary.json``.

Usage:
    python scripts/evaluation/summarize_eval_results.py --mid pi05_alljoint_60k
    python scripts/evaluation/summarize_eval_results.py --mid pi05_alljoint_60k --results-dir outputs/eval/leaderboard/results
"""
from __future__ import annotations
import argparse, json, os, re
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("OUT", "")) / "results"

BENCHMARK_ORDER = ["Memory", "Long_Horizon", "Generalization",
                   "Instruction_Following", "Limited_Demonstrations"]



def score_value(row: dict) -> float:
    """Normalized score, accepting legacy result files that used mean_rate."""
    value = row.get("score")
    if value is None:
        value = row.get("mean_rate", 0.0)
    return float(value or 0.0)


def parse_setting_name(name: str):
    """Parse '<Bench>_<robot>_<mode>_<domain>' into (bench, robot, mode, domain)."""
    m = re.match(r"(.+)_(franka|astribot)_(joint|camera_ee|robot_ee)_(in_domain|out_of_domain)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def summarize(mid: str, results_dir: Path):
    mid_dir = results_dir / mid
    if not mid_dir.is_dir():
        print(f"No results dir for model_id={mid}: {mid_dir}")
        return False

    # Collect all setting results
    settings = {}  # setting_name → {bench, robot, mode, domain, per_task, score, sr}
    for jf in sorted(mid_dir.glob("*.json")):
        if jf.name.endswith(".progress.json") or jf.name.endswith(".tmp"):
            continue
        setting_name = jf.name[:-5]  # strip .json
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        parsed = parse_setting_name(setting_name)
        if not parsed:
            continue
        bench, robot, mode, domain = parsed
        pt = data.get("per_task", {})
        # Compute setting-level score/sr like web leaderboard _setting_metrics
        if pt and isinstance(pt, dict):
            score_vals = [score_value(v) for v in pt.values() if isinstance(v, dict)]
            sr_vals = [v.get("success_rate", 0.0) for v in pt.values() if isinstance(v, dict)]
            score = sum(score_vals) / len(score_vals) if score_vals else 0.0
            sr = sum(sr_vals) / len(sr_vals) if sr_vals else 0.0
        else:
            score = score_value(data)
            sr = float(data.get("success_rate", 0.0) or 0.0)

        settings[setting_name] = {
            "bench": bench, "robot": robot, "mode": mode, "domain": domain,
            "success_rate": sr, "score": score,
            "per_task": {k: {"success_rate": v.get("success_rate", 0.0),
                             "score": score_value(v),
                             "n_episodes": v.get("n_episodes", 0)}
                         for k, v in pt.items() if isinstance(v, dict)} if pt else {},
        }

    if not settings:
        print(f"No result JSONs found for {mid}")
        return False

    # Aggregate by benchmark → domain
    benchmarks = {}
    for sname, sv in settings.items():
        bench = sv["bench"]
        domain = sv["domain"]
        benchmarks.setdefault(bench, {domain: {"settings": [], "tasks": {}} for domain in ("in_domain", "out_of_domain")})
        if domain not in benchmarks[bench]:
            benchmarks[bench][domain] = {"settings": [], "tasks": {}}
        benchmarks[bench][domain]["settings"].append(sv)
        # Merge per_task (dedup by task_id; later settings overwrite if same task)
        for tid, tv in sv["per_task"].items():
            benchmarks[bench][domain]["tasks"][tid] = tv

    # Build output structure. Overall is accumulated raw (per-task micro, each
    # (benchmark, domain) weighted by its task count) alongside the per-domain
    # numbers — NOT from a flat task-id dict, which would let out_of_domain
    # overwrite in_domain (both share the same 107 task ids) and silently drop
    # the in-domain rows, yielding an out-of-domain-only "overall".
    _ov_score = _ov_sr = 0.0
    _ov_n = 0
    output = {
        "model_id": mid,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall": {"success_rate": 0.0, "score": 0.0, "n_tasks": 0},
        "benchmarks": {},
    }

    for bench in BENCHMARK_ORDER:
        if bench not in benchmarks:
            continue
        bench_data = {"in_domain": None, "out_of_domain": None}
        for domain in ("in_domain", "out_of_domain"):
            dom_info = benchmarks[bench].get(domain, {})
            tasks = dom_info.get("tasks", {})
            if not tasks:
                continue
            task_scores = [v["score"] for v in tasks.values()]
            task_srs = [v["success_rate"] for v in tasks.values()]
            dom_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
            dom_sr = sum(task_srs) / len(task_srs) if task_srs else 0.0

            # Sort tasks by name for consistent output
            sorted_tasks = dict(sorted(tasks.items(), key=lambda x: x[0]))
            bench_data[domain] = {
                "success_rate": round(dom_sr, 4),
                "score": round(dom_score, 4),
                "n_tasks": len(tasks),
                "tasks": {k: {"success_rate": round(v["success_rate"], 4),
                               "score": round(v["score"], 4),
                               "n_episodes": v.get("n_episodes", 0)}
                          for k, v in sorted_tasks.items()},
            }
            _n = len(task_srs)
            _ov_score += dom_score * _n
            _ov_sr += dom_sr * _n
            _ov_n += _n

        output["benchmarks"][bench] = bench_data

    # Overall: per-task micro across all (benchmark, domain) — each weighted by
    # its task count so in_domain and out_of_domain are BOTH counted (matching the
    # web leaderboard's task-count-weighted Overall). n_tasks is the total number
    # of (task, domain) measurements averaged.
    if _ov_n:
        output["overall"] = {
            "success_rate": round(_ov_sr / _ov_n, 4),
            "score": round(_ov_score / _ov_n, 4),
            "n_tasks": _ov_n,
        }

    # Write summary
    summary_path = mid_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {summary_path}")
    print(f"  Overall: success_rate={output['overall']['success_rate'] * 100:.2f}%  "
          f"score={output['overall']['score'] * 100:.2f}  "
          f"n_tasks={output['overall']['n_tasks']}")
    for bench in BENCHMARK_ORDER:
        if bench not in output["benchmarks"]:
            continue
        bd = output["benchmarks"][bench]
        parts = []
        for dom in ("in_domain", "out_of_domain"):
            di = bd.get(dom)
            if di:
                parts.append(f"{dom[:2]}: SR {di['success_rate'] * 100:.2f}% "
                             f"score {di['score'] * 100:.2f} ({di['n_tasks']}t)")
        if parts:
            print(f"  {bench:24s} {'  |  '.join(parts)}")

    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid", required=True, help="Model ID (results/<mid>/ dir)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="Root results directory")
    args = ap.parse_args()
    if not summarize(args.mid, Path(args.results_dir)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
