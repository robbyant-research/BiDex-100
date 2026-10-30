"""BiDex-100 unified training interface: benchmark + control-mode registry.

Single source of truth shared by all model training frameworks (openpi, Isaac-GR00T,
...) so that a (model, benchmark, mode) choice maps deterministically to the
training data (task list) and the parquet column layout. No data fitting, no
per-model duplication of the benchmark/mode definitions.

Benchmarks (5) × control modes (3). General control space is always 58D =
<ee-or-arm>[14] + dex[44].
    Benchmarks:
      Memory                 : franka, dual
      Long_Horizon           : franka, dual
      Generalization         : astribot+franka, dual
      Instruction_Following  : franka, dual
      Limited_Demonstrations : franka, dual
    Modes (control action space, 58D = <ee-or-arm>[14] + dex[44]):
      joint    : actions.arm_joint_position[14] + dex[44]      (delta for openpi)
      robot_ee : actions.abs_end_effector[14]      + dex[44]    (ABSOLUTE, base frame)
      camera_ee: actions.abs_end_effector_camera[14] + dex[44]  (ABSOLUTE, head-cam frame)

Task discovery is driven by the repo's YAML registry (bidex.tasks.registry,
tasks_data/suites/) — this module no longer reads the on-disk original dataset.
`task_dirs` / `benchmark_robot_tasks` are retained as empty stubs for backwards
compatibility; the registry is the source of truth.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlMode:
    name: str
    action_state_col: str   # observation.states.<X>  (14-D, right-first)
    action_action_col: str  # actions.<X>             (14-D)
    dex_state_col: str = "observation.states.dex_joint_position"
    dex_action_col: str = "actions.dex_joint_position"
    delta_actions: bool = False     # openpi: True for joint only; ee modes absolute
    is_ee: bool = True              # ee-family (absolute pose); joint=False

    @property
    def arm_action_col(self):
        return "actions.arm_joint_position" if self.name == "joint" else None


MODES = {
    "joint": ControlMode("joint", "observation.states.arm_joint_position",
                         "actions.arm_joint_position", delta_actions=True, is_ee=False),
    "robot_ee": ControlMode("robot_ee", "observation.states.abs_end_effector",
                            "actions.abs_end_effector"),
    "camera_ee": ControlMode("camera_ee", "observation.states.abs_end_effector_camera",
                            "actions.abs_end_effector_camera"),
}


@dataclass(frozen=True)
class Benchmark:
    name: str
    category: str                # task_annotations.category
    robots: tuple                # ("franka",) | ("astribot","franka") | ...
    arms: tuple = ("dual",)      # ("dual",) both arms 58D
    placeholder: bool = False    # True if no annotated tasks yet


BENCHMARKS = {
    "Memory": Benchmark("Memory", "Memory", ("franka",)),
    "Long_Horizon": Benchmark("Long_Horizon", "Long_Horizon", ("franka",)),
    "Generalization": Benchmark("Generalization", "Generalization", ("astribot", "franka")),
    "Instruction_Following": Benchmark("Instruction_Following", "Instruction_Following", ("franka",)),
    "Limited_Demonstrations": Benchmark("Limited_Demonstrations", "Limited_Demonstrations", ("franka",)),
}
BENCHMARK_ORDER = ["Memory", "Long_Horizon", "Generalization", "Instruction_Following",
                   "Limited_Demonstrations"]
MODE_ORDER = ["joint", "robot_ee", "camera_ee"]


def task_dirs(benchmark: str, robot: str, split: str = "success", arm: str | None = None):
    """Dataset-backed task dirs are no longer used — the YAML registry
    (bidex.tasks.registry) is the source of truth. Retained as an empty
    stub for backwards-compat callers."""
    return []


def arm_for_task(robot: str, suite: str, task: str) -> str:
    """Active arm for a task. All current benchmarks are dual-arm; return "dual"."""
    return "dual"


def task_dirs_with_arm(benchmark: str, robot: str, split: str = "success", arm: str | None = None):
    """Like task_dirs but returns [(path, arm)]. Dataset-backed discovery removed;
    returns []."""
    return []


def benchmark_arms(benchmark: str) -> tuple:
    """Arms emitted as separate training configs. All benchmarks -> (dual,)
    (one config; the 58D dataset columns carry both arms)."""
    return BENCHMARKS[benchmark].arms


def is_single_arm_benchmark(benchmark: str) -> bool:
    """True if the benchmark uses a single-arm (29D) config. Always False for
    the current 5 benchmarks (all dual-arm 58D)."""
    return "single_arm" in BENCHMARKS[benchmark].arms


def config_name(benchmark: str, robot: str, mode: str, arm: str = "dual") -> str:
    """Canonical openpi/submit config name."""
    rb = "FR" if robot == "franka" else "AS"
    if arm and arm not in ("dual", "single_arm"):
        return f"BiDex-{benchmark}-{rb}-{arm}-{mode}-1"
    return f"BiDex-{benchmark}-{rb}-{mode}-1"


def benchmark_robot_tasks(benchmark: str, split: str = "success"):
    """{robot: [task_dirs]} for a benchmark (Generalization -> both robots)."""
    return {robot: task_dirs(benchmark, robot, split) for robot in BENCHMARKS[benchmark].robots}


def benchmark_robot_arm_tasks(benchmark: str, split: str = "success"):
    """{robot: {arm: [task_dirs]}} for a benchmark. All current benchmarks collapse
    to a single "dual" arm key."""
    bm = BENCHMARKS[benchmark]
    return {robot: {arm: task_dirs(benchmark, robot, split, arm)
                    for arm in benchmark_arms(benchmark)}
            for robot in bm.robots}


def all_train_configs():
    """Enumerate the (benchmark, mode[, arm]) training configs per model.

    Yields dicts: {benchmark, mode, arm, robot_arm_tasks: {robot: [dirs]}, placeholder}.
    All current benchmarks yield a single "dual" arm entry per mode."""
    for bm in BENCHMARK_ORDER:
        rat = benchmark_robot_arm_tasks(bm)
        empty = all(len(p) == 0 for arm_map in rat.values() for p in arm_map.values())
        for mode in MODE_ORDER:
            for arm in benchmark_arms(bm):
                yield {"benchmark": bm, "mode": mode, "arm": arm,
                       "robot_arm_tasks": {r: rat[r][arm] for r in rat},
                       "placeholder": BENCHMARKS[bm].placeholder or empty}


if __name__ == "__main__":
    n_cfgs = sum(len(benchmark_arms(bm)) for bm in BENCHMARK_ORDER) * len(MODES)
    print(f"benchmarks={len(BENCHMARKS)} modes={len(MODES)} -> {n_cfgs} configs/model")
    for c in all_train_configs():
        n = sum(len(v) for v in c["robot_arm_tasks"].values())
        robo = {k: len(v) for k, v in c["robot_arm_tasks"].items()}
        arm = c["arm"] if c["arm"] != "dual" else "dual"
        print(f"  {c['benchmark']:22} {c['mode']:9} arm={arm:5} "
              f"placeholder={c['placeholder']!s:5} tasks={n} {robo}")
