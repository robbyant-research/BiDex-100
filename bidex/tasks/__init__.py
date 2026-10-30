"""LIBERO-style task / domain-randomization layer for BiDex-100.

Concepts (mirrors LIBERO):

  - Arena            scene XML + robot base offset table (arena.py)
  - Region           named xy[+yaw] sampling rectangle on a target surface
  - ObjectSpec       which catalog object goes where, optional articulation init
  - GoalPredicate    success criterion evaluated against MjData at runtime
  - Task             one YAML file = one task (task_spec.py)
  - Suite / Registry collection of tasks discoverable by name (registry.py)
  - InitStates       per-task library of validated flat (time+qpos+qvel) snapshots

Public API re-exports for `from bidex.tasks import ...`.
"""

from bidex.tasks.task_spec import (
    Region,
    ObjectInit,
    FixtureInit,
    InitPredicate,
    GoalPredicate,
    Task,
)
from bidex.tasks.arena import (
    ARENA_BASE_OFFSETS,
    arena_for_scene,
    base_offset_for,
)
from bidex.tasks.region_sampler import RegionSampler
from bidex.tasks.predicates import (
    evaluate_goal, PREDICATE_REGISTRY, SubtaskTracker,
)
from bidex.tasks.domain import apply_task_initial_state
from bidex.tasks.init_states import (
    load_init_states,
    save_init_states,
    init_states_path,
)
from bidex.tasks.registry import (
    list_suites,
    list_tasks,
    load_task,
    SUITES_DIR,
)

__all__ = [
    "Region",
    "ObjectInit",
    "FixtureInit",
    "InitPredicate",
    "GoalPredicate",
    "Task",
    "ARENA_BASE_OFFSETS",
    "arena_for_scene",
    "base_offset_for",
    "RegionSampler",
    "evaluate_goal",
    "PREDICATE_REGISTRY",
    "SubtaskTracker",
    "apply_task_initial_state",
    "load_init_states",
    "save_init_states",
    "init_states_path",
    "list_suites",
    "list_tasks",
    "load_task",
    "SUITES_DIR",
]
