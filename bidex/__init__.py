"""BiDex-100 — minimal MuJoCo simulation benchmark for imitation-learning eval.

Public API:

    from bidex import BiDexEnv, load_task, list_tasks, list_suites

    task = load_task("Generalization", "put_apple_in_white_bowl")
    env = BiDexEnv(task, control_mode="joint", cameras=["head"])
    obs = env.reset(seed=0)
    obs, reward, terminated, info = env.step(action)

See ``bidex.env.BiDexEnv`` for the full obs/action spec and
``bidex.replay`` for the replay mode.
"""

from bidex.env import BiDexEnv
from bidex.config import BiDexConfig, default_config, load_config
from bidex.replay import replay_episode, record_episode
from bidex.tasks import (
    Task,
    SubtaskTracker,
    evaluate_goal,
    list_suites,
    list_tasks,
    load_task,
)
from bidex.robots.registry import available_robot_types

__all__ = [
    "BiDexEnv",
    "BiDexConfig",
    "default_config",
    "load_config",
    "replay_episode",
    "record_episode",
    "Task",
    "SubtaskTracker",
    "evaluate_goal",
    "list_suites",
    "list_tasks",
    "load_task",
    "available_robot_types",
]

__version__ = "0.1.0"
