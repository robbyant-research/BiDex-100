"""Per-subtask scoring + aggregation for the unified BiDex-100 evaluation.

The env's `info["subtasks"]` (env.py:1110) is a latching per-subtask list
[{name, done, holds_now, must_hold, latched_at_frame, ok}]. The per-subtask
SCORE WEIGHTS come from the task yaml's top-level `subtasks:` block
([{name, score}], synced from task_annotations.json by
scripts/sync_annotations_to_yamls.py) — the yaml is the single source of truth
at eval time. The join is by subtask INDEX (first-appearance order of
goal[*].subtask groups == SubtaskTracker order).

Episode score = sum of score weights for subtasks that latched `done` (the
latching view rewards progress even if a later subtask regresses). "Grab wrong
object" early-termination: the harness stops stepping and keeps the latched
subtask score so far.
"""
from __future__ import annotations


def subtask_weights_from_task(task) -> list[int]:
    """Per-subtask score weights from the loaded Task's yaml `subtasks:` block.
    [] if the yaml has no scored subtasks (caller falls back to equal split)."""
    subs = getattr(task, "subtasks", None) or []
    return [int(round(float(s.get("score", 0)))) for s in subs]


def load_subtask_scores(robot: str, suite: str, task: str) -> list[int]:
    """Per-subtask score weights (by index). The original-dataset
    task_annotations.json fallback was removed (all task yamls carry a scored
    `subtasks:` block, which `subtask_weights` reads first); returns []."""
    return []


def subtask_weights(task, robot: str, suite: str, task_name: str) -> list[int]:
    """Per-subtask score weights from the task yaml's `subtasks:` block (the
    single source of truth). Returns [] if the yaml has no scored subtasks
    (caller falls back to equal split)."""
    return subtask_weights_from_task(task)


def episode_score(info_subtasks: list[dict], weights: list[int]) -> dict:
    """Aggregate one episode's per-subtask results into a score dict.

    `info_subtasks` = env.step info["subtasks"] (latched list, by index).
    `weights`       = load_subtask_scores(...) (by index).
    Returns {score, max_score, subtasks: [{name, done, ok, must_hold, score,
              weight, latched_at_frame, holds_now}], success, latch_tick_hz}.

    `latched_at_frame` is the sim step (30/25 Hz control tick) at which the
    subtask first latched `done` (None if never done). Kept so the leaderboard
    can light a green LED at the trigger moment when a user watches an episode
    mp4 (and to spot subtasks that falsely/early-trigger). `latch_tick_hz` is the
    control frequency used to convert frames -> seconds for the LED bar.

    Scoring uses `done` (latching) by default — milestone subtasks that latch
    once stay credited even if the condition later regresses. But subtasks
    declared `must_hold: true` require the condition to ALSO hold at the final
    frame: their score key is `ok` (= done AND holds_now), not just `done`.
    This prevents a transient inside/on match from claiming credit when the
    object was later moved away.
    """
    n = len(info_subtasks)
    if weights and len(weights) == n:
        w = weights
    elif weights:
        # mismatch length: align by available; fill missing with 0
        w = list(weights) + [0] * (n - len(weights))
    else:
        w = [int(100 / n)] * n if n else []
    sub = []
    score = 0
    all_done = True
    for i, st in enumerate(info_subtasks):
        done = bool(st.get("done"))
        ok = bool(st.get("ok", done))          # fallback: ok = done (backward compat)
        must_hold = bool(st.get("must_hold", False))
        violated = bool(st.get("violated", False))  # latch_violation guard
        wi = w[i] if i < len(w) else 0
        # must_hold subtasks score only when ok (final-frame truth);
        # non-must_hold (milestone) subtasks score on latching (done);
        # latch_violation subtasks (e.g. "not watered" guards) must NOT be
        # credited when violated (ok=False from SubtaskTracker even though
        # done=True) — use ok for those too.
        effective = ok if (must_hold or violated) else done
        sub.append({"name": st.get("name", f"subtask_{i}"), "done": done,
                    "ok": ok, "must_hold": must_hold,
                    "weight": wi, "score": wi if effective else 0,
                    "latched_at_frame": st.get("latched_at_frame"),
                    "holds_now": bool(st.get("holds_now", False))})
        if effective:
            score += wi
        else:
            all_done = False
    return {"score": score, "max_score": sum(w),
            "success": all_done and n > 0, "subtasks": sub,
            "latch_tick_hz": 25}


def aggregate(per_episode: list[dict]) -> dict:
    """Aggregate a list of episode-score dicts (from episode_score) into a
    benchmark-level summary: per-subtask success-rate + mean normalized score."""
    if not per_episode:
        return {"n_episodes": 0, "mean_score": 0.0, "mean_rate": 0.0,
                "per_subtask": []}
    n = len(per_episode)
    mean_score = sum(e["score"] for e in per_episode) / n
    max_score = per_episode[0].get("max_score", 1) or 1
    mean_rate = mean_score / max_score
    # per-subtask success-rate (across episodes, by index) — uses `ok` so
    # must_hold subtasks only count when the condition held at the final frame.
    nsub = max(len(e["subtasks"]) for e in per_episode)
    per_sub = []
    for i in range(nsub):
        ok_cnt = sum(1 for e in per_episode
                     if i < len(e["subtasks"]) and e["subtasks"][i].get("ok", False))
        names = [e["subtasks"][i]["name"] for e in per_episode if i < len(e["subtasks"])]
        per_sub.append({"index": i, "name": names[0] if names else "",
                        "done_rate": ok_cnt / n})
    return {"n_episodes": n, "mean_score": mean_score, "mean_rate": mean_rate,
            "max_score": max_score, "per_subtask": per_sub,
            "per_episode": per_episode}