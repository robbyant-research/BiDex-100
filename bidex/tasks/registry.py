"""Task suite registry — discover and load YAML tasks.

Layout (category-organized, mirrors the final bidex_data dataset)::

    bidex/tasks_data/
        suites/
            Memory/
                pour_ice_between_two_cups.yaml
            Generalization/
            Instruction_Following/
            Limited_Demonstrations/
            Long_Horizon/
        init_states/
            <Category>/
                <task>.npy   # (N, 1+nq+nv)

`load_task("Memory", "pour_ice_between_two_cups")` -> Task. No decorator-based
registration is needed because YAML files are the source of truth — the
filesystem is the registry. The `suite` argument is normally one of the five
benchmark categories. A global by-name fallback keeps task lookup tolerant of
older callers; current task names are unique across the 106 benchmark tasks.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

from bidex.tasks.task_spec import Task


_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SUITES_DIR = os.path.join(_PKG_ROOT, "tasks_data", "suites")


def list_suites(suites_dir: Optional[str] = None) -> List[str]:
    base = suites_dir or SUITES_DIR
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def list_tasks(
    suite: str,
    suites_dir: Optional[str] = None,
    *,
    include_ood: bool = False,
) -> List[str]:
    """List benchmark tasks in ``suite``.

    OOD variant YAMLs are implementation details of the matched evaluation
    protocol and are excluded by default. Pass ``include_ood=True`` when the
    variants themselves need to be inspected.
    """
    base = suites_dir or SUITES_DIR
    sdir = os.path.join(base, suite)
    if not os.path.isdir(sdir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(sdir)
        if f.endswith((".yaml", ".yml"))
        and (include_ood or not os.path.splitext(f)[0].endswith("__ood"))
    )


def resolve_task_path(
    suite: str,
    name: str,
    suites_dir: Optional[str] = None,
) -> str:
    """Resolve `<suite>/<name>` to a yaml path. Tries `<suite>/<name>.yaml`
    first, then falls back to a by-name search across all category directories.
    Raises FileNotFoundError if the task is absent or ambiguous."""
    base = suites_dir or SUITES_DIR
    for ext in (".yaml", ".yml"):
        path = os.path.join(base, suite, name + ext)
        if os.path.isfile(path):
            return path
    matches = []
    for ext in (".yaml", ".yml"):
        matches.extend(glob.glob(os.path.join(base, "*", name + ext)))
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Task name '{name}' is ambiguous across suites dirs: {matches}")
    raise FileNotFoundError(
        f"Task '{suite}/{name}' not found under {base}. "
        f"Available suites: {list_suites(base)}"
    )


def load_task(
    suite: str,
    name: str,
    suites_dir: Optional[str] = None,
) -> Task:
    """Load `<suites_dir>/<suite>/<name>.yaml` (also `.yml`) as a Task."""
    return Task.from_yaml(resolve_task_path(suite, name, suites_dir), suite=suite)
