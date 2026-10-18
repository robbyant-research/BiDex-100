<p align="center">
  <img src="https://raw.githubusercontent.com/BiDex-100/BiDex-100/gh-pages/static/images/title.png" alt="BiDex-100" width="760">
</p>

<p align="center">
  <strong>A comprehensive benchmark for bimanual dexterous manipulation across 100+ simulation tasks</strong>
</p>

<p align="center">
  <a href="https://bidex-100.github.io/BiDex-100/">Website</a> ·
  <a href="https://huggingface.co/datasets/Robbyant-Research/BiDex-100/tree/main/datasets">Datasets</a> ·
  <a href="https://huggingface.co/datasets/Robbyant-Research/BiDex-100/tree/main/assets">3D Assets</a>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="MuJoCo" src="https://img.shields.io/badge/Engine-MuJoCo-475569?style=flat-square">
  <img alt="LeRobot v2.1 and v3.0" src="https://img.shields.io/badge/Data-LeRobot_v2.1_%7C_v3.0-ffcc00?style=flat-square">
  <img alt="Code: MIT" src="https://img.shields.io/badge/Code-MIT-blue?style=flat-square">
  <img alt="Data and original assets: CC BY-NC 4.0" src="https://img.shields.io/badge/Data%20%26%20Original%20Assets-CC_BY--NC_4.0-8A2BE2?style=flat-square">
</p>

## Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/BiDex-100/BiDex-100/gh-pages/static/images/figures/overview.png" alt="BiDex-100 benchmark overview" width="92%">
</p>

BiDex-100 evaluates vision-language-action policies on coordinated, bimanual,
dexterous manipulation. It combines declarative task definitions, reproducible
MuJoCo environments, matched in-domain and out-of-domain evaluation, 3D assets,
and 8,080 success-validated demonstrations.

- **106 tasks across five capability axes:** Memory, Long-Horizon,
  Instruction-Following, Limited-Demonstrations, and Generalization.
- **Two bimanual embodiments:** dual Franka FR3 v2 arms and Astribot S1,
  both equipped with dexterous hands.
- **8,080 demonstrations and 6.25M frames:** six synchronized RGB-D streams at
  25 FPS, released as task-level LeRobot v2.1 and v3.0 datasets.
- **58-D unified action space:** 14 arm/end-effector dimensions plus 44
  dexterous-hand joint dimensions.
- **Matched ID/OOD protocol:** deterministic seeds, scene and object shifts,
  predicate-based progress scoring, and a strict score-100 success criterion.

## Installation

```bash
git clone https://github.com/robbyant-research/BiDex-100.git
cd BiDex-100

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[eval]"
```

The Hugging Face dataset repository has three independently downloadable
components:

```text
assets/                    compressed simulation asset packs
datasets/lerobotv21/       LeRobot v2.1 datasets
datasets/lerobotv30/       LeRobot v3.0 datasets
```

Download everything and prepare the simulation assets:

```bash
python -m pip install -U huggingface_hub
hf auth login  # omit for public, unauthenticated downloads

bash scripts/download_release.sh all downloads/BiDex-100
bash scripts/setup_assets_links.sh downloads/BiDex-100
```

Each component can instead be downloaded separately into the same destination:

```bash
bash scripts/download_release.sh assets downloads/BiDex-100
bash scripts/download_release.sh lerobotv21 downloads/BiDex-100
bash scripts/download_release.sh lerobotv30 downloads/BiDex-100
```

`scripts/download_assets.sh` and `scripts/download_datasets.sh` remain as
convenience wrappers. Run `download_datasets.sh lerobotv21`,
`download_datasets.sh lerobotv30`, or `download_datasets.sh all` to fetch one
or both dataset formats. The download destination preserves the repository
layout shown above.

The asset setup script accepts either `downloads/BiDex-100` or
`downloads/BiDex-100/assets`. It extracts each archive in place, creates the
repository links, and recreates the relative aliases used by assembly tasks.
When `assets/SHA256SUMS` is present, archive checksums are verified; otherwise
the script records each local archive digest so unchanged extractions can be
reused safely. It is idempotent, so an interrupted setup can be rerun safely.
Alternatively set
`BIDEX_ASSETS=/path/to/BiDex-100/assets`. Export `HF_TOKEN` instead of using
`hf auth login` in non-interactive environments.

For headless GPU rendering, select EGL before starting Python:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Tasks

Task YAML files under `bidex/tasks_data/suites/<suite>/<task>.yaml` define the
scene, robot, objects, initialization distribution, instruction, success
predicates, subtasks, and OOD counterpart. The five suites contain 10, 10, 20,
26, and 40 tasks respectively (106 total).

This minimal example is complete and directly runnable. It disables cameras so
that the first simulation check does not require a graphics context, resets one
task, holds the reset robot pose, advances physics with `env.step`, and closes
all MuJoCo resources:

```python
from bidex import BiDexEnv, load_task

task = load_task("Memory", "uncover_red_cube")
env = BiDexEnv(task, control_mode="joint", cameras=[])

try:
    obs = env.reset(seed=0)
    action = {
        "arm_joint_position": obs["arm_joint_position"].copy(),
        "dex_joint_position": obs["dex_joint_position"].copy(),
    }

    for _ in range(64):
        obs, reward, terminated, info = env.step(action)
        if terminated:
            break

    print({"reward": reward, "terminated": terminated, "score": info.get("score")})
finally:
    env.close()
```

Use `list_suites()` and `list_tasks(<suite>)` to discover the 106 base tasks.
Pass `include_ood=True` to `list_tasks` only when inspecting the paired OOD
variant definitions. Pass
`cameras=["head", "left_wrist", "right_wrist"]` when RGB observations are
needed.

## Training

BiDex-100 is model-agnostic: this repository provides the benchmark and data
contract, while each VLA implementation owns its trainer. The supported 58-D
representations are:

| Mode | Arm component | Hand component |
| --- | --- | --- |
| `joint` | `actions.arm_joint_position` (14-D) | `actions.dex_joint_position` (44-D) |
| `robot_ee` | `actions.abs_end_effector` (14-D) | `actions.dex_joint_position` (44-D) |
| `camera_ee` | `actions.abs_end_effector_camera` (14-D) | `actions.dex_joint_position` (44-D) |

Use `bidex.tasks.benchmark.BENCHMARKS` and `MODES` as the canonical suite and
modality registry when adding a trainer integration.

### Reported training setup

The currently reported model results use **co-training**: demonstrations from
multiple benchmark tasks are merged into one training corpus and used to train
one shared policy checkpoint. Evaluation then runs that same checkpoint across
all applicable tasks; the reported numbers are not an ensemble of separately
trained per-task policies.

## Evaluation

BiDex-100 exposes one evaluation contract for different VLA policies. A model
adapter receives the same three camera observations, proprioceptive state, and
language instruction; the benchmark owns task discovery, seeded resets,
ID/OOD randomization, rollout execution, scoring, videos, and result schemas.
This keeps model comparisons independent of model-specific launch code.

### ID/OOD protocol summary

The canonical protocol evaluates 25 episodes for each of two ID seeds (`0,1`)
and two OOD seeds (`2,3`). Both domains use deterministic per-episode reset
seeds and the same task instructions, success predicates, rollout horizon, and
scoring rules.

| Setting | In-domain (ID) | Out-of-domain (OOD) |
| --- | --- | --- |
| Task definition | Base task YAML. | The task's committed `__ood.yaml` variant (all 106 tasks are covered). |
| Robot initial state | Deterministically samples a recorded working pose from the task's evaluation H5 and restores robot joints only; objects retain the fresh base-YAML placement. | Uses the task's default robot reset pose; it does not sample an H5 demonstration pose. |
| Objects and layout | Base-YAML regions and object identities. | Task-specific baked region/layout shifts; selected tasks additionally use left/right arm mirroring, collision-safe distractors, or per-episode articulated-object scale changes. |
| Scene appearance | Original floor, wall, table, and lights. | Deterministic floor/wall/table textures and randomized lighting. |
| Cameras | Nominal head and wrist extrinsics. | A collision-audited finite pose bank independently controls head and wrist changes. |

OOD appearance randomization is intentionally bounded but visually strong:

- Lighting cycles through five exposure profiles from `dark` to
  `high_bright` with key/fill/ambient levels of `0.11/0.035/0.014`,
  `0.18/0.060/0.022`, `0.29/0.100/0.040`, `0.40/0.145/0.058`, and
  `0.48/0.180/0.072`. The key light casts shadows and spans the full azimuth
  circle, elevations of 25/38/52/66/76 degrees, neutral/warm/cool tints, and
  distances of 3.0/3.5/4.0/4.4 m. The non-shadowing fill light provides an
  illumination floor to avoid unusably dark observations.
- Head-camera offsets are bounded to 1.8 cm and 6 degrees. Wrist-camera offsets
  are bounded to 1.35 cm and 6 degrees; the mount-clearance axis is restricted
  to outward motion for both Franka and Astribot so the robot connector cannot
  move in front of the lens. Across the canonical 50 OOD episodes per task,
  camera occurrence counts are exactly 20 nominal, 10 head-only, 10
  wrists-only, and 10 head-and-wrists.
- Object/layout changes come from each task's committed OOD YAML. Distractor
  identity is fixed per task and OOD seed while placement is resampled per
  episode; task geometry and reachability constraints keep objects on usable,
  collision-safe tabletop regions.

Randomization uses component-scoped SHA-256 keys, NumPy PCG64 streams, and
hash-validated texture/camera manifests. Consequently, a fixed task, seed, and
episode produces the same plan on different machines without depending on GPU
or renderer output.

Seven reference integrations are included: π0.5, Galaxea G0.5, GR00T N1.7,
LingBot-VLA, XR-1, InternVLA-A1.5, and DM0.5. Additional models can use the same
driver by implementing the observation-to-action adapter used by
`bidex/eval/run_leaderboard_eval.py`.

The launcher has no machine-specific defaults. Every integration supplies:

- `MODEL_PY`: Python interpreter for the model environment.
- `MODEL_REPO`: model source directory.
- `MODEL_CKPT`: checkpoint path.
- `BIDEX_LEADERBOARD_ROOT`: output directory.

`MODEL_VLM_PATH` is additionally required by LingBot-VLA, XR-1, and
InternVLA-A1.5 because those adapters load a separate local VLM backbone.
`OPENPI_DATA_HOME` is an optional native OpenPI cache override; when unset,
OpenPI uses its normal cache.

For example, configure π0.5 without embedding paths in this repository:

```bash
export MODEL_PY=/path/to/openpi/.venv/bin/python
export MODEL_REPO=/path/to/openpi
export MODEL_CKPT=/path/to/pi05-checkpoint
export BIDEX_LEADERBOARD_ROOT=/path/to/evaluation-output

# Optional for an offline, pre-populated OpenPI cache.
export OPENPI_DATA_HOME=/path/to/openpi-cache
```

Run one deterministic end-to-end smoke task before the full benchmark:

```bash
MID_OVERRIDE=pi05_smoke \
BENCHES=Memory \
BIDEX_TASKS=Memory/pour_ice_between_two_cups \
SEEDS=0 ID_SEEDS=0 OOD_SEEDS=2 EPS=1 \
MODEL_GPUS=0 RENDER_GPUS=1 \
  bash scripts/evaluation/run_vec_models_eval.sh pi05 smoke

MID_OVERRIDE=pi05_alljoint \
MODEL_GPUS=0,1,2,3,4,5,6,7 \
RENDER_GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/evaluation/run_vec_models_eval.sh pi05
```

### GPU layouts

`MODEL_GPUS` assigns policy servers and `RENDER_GPUS` assigns vectorized
MuJoCo workers. The benchmark protocol and output schema do not change with
the layout. Every layout uses the same server-client transport; a GPU present
in both lists hosts separate policy-server and render-worker processes.

| Layout | `MODEL_GPUS` | `RENDER_GPUS` | Description |
| --- | --- | --- | --- |
| Single GPU | `0` | `0` | Policy and simulation share one card. |
| Dedicated split | `0` | `1,2,3,4,5,6,7` | One policy card and seven render-only cards. |
| Multi-server split | `0,1` | `2,3,4,5,6,7` | Two policy servers distribute requests from six workers. |
| Partial overlap | `0,1` | `1,2,3,4,5,6,7` | GPU 1 hosts both roles. |
| Full overlap | `0,1,2,3,4,5,6,7` | `0,1,2,3,4,5,6,7` | Every card hosts one server and one worker. |

Use a distinct `MID_OVERRIDE` for every smoke or full run.
`BENCHES`, `BIDEX_TASKS`, `SEEDS`, `ID_SEEDS`, `OOD_SEEDS`, `EPS`,
`ENVS_PER_GPU`, `MAX_RECORD_ENVS`, and `BIDEX_MAX_ENVS_PER_GPU` refine the
protocol without changing its scoring contract. `GPUS` remains a compatibility
alias; explicit model/render lists are recommended.

Workers execute the full action chunk returned by each model by default. Set
`BIDEX_VEC_CHUNK_REUSE` to a positive integer only to cap that native horizon
for an explicit ablation; empty or zero keeps the model's maximum chunk size.

### View results

`MID_OVERRIDE` identifies one evaluated checkpoint/configuration. Results,
videos, and live logs are isolated under that identifier:

```text
results/<MID_OVERRIDE>/<benchmark>_<robot>_<mode>_<domain>.json
mp4s/<MID_OVERRIDE>/<setting>/<task>/s<seed>_ep<episode>.mp4
runs/<MID_OVERRIDE>/<run_name>/{in_domain,out_of_domain}/...
```

Start the leaderboard on all local interfaces:

```bash
python bidex/eval/baseline_leaderboard_server.py \
  --lb-root "$BIDEX_LEADERBOARD_ROOT" \
  --host 0.0.0.0 --port 12006 --base-url /baselines
```

Open `http://<host-ip>:12006/baselines/` from the local network. The page
discovers model rows from `results/<MID_OVERRIDE>/`, refreshes live results,
and serves rollout videos. Leaderboards rank by strict success rate by default;
progress score is the secondary metric and can be selected from the metric toggle.

## Licensing

| Component | License |
| --- | --- |
| Codebase | [MIT License](LICENSE) |
| BiDex-100 datasets | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| BiDex-100 original 3D assets | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| Third-party assets | Their original upstream licenses, retained inside each asset subdirectory |

## Acknowledgements

BiDex-100 builds on
[MuJoCo](https://mujoco.readthedocs.io/),
[LeRobot](https://github.com/huggingface/lerobot),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[RoboCasa](https://github.com/robocasa/robocasa),
[robosuite](https://github.com/ARISE-Initiative/robosuite),
[dexjoco](https://github.com/brave-eai/dexjoco), and
[Articraft](https://github.com/mattzh72/articraft).

## Citation

```bibtex
@misc{bidex100,
  title        = {BiDex-100: A Comprehensive Benchmark for Bimanual Dexterous
                  Manipulation Across 100+ Simulation Tasks},
  year         = {2026},
}
```
