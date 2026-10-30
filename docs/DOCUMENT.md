# BiDex-100 — Setup, Training and Evaluation

This guide covers simulation setup, release downloads, the task API, 58-D action representations, model preparation and co-training, the ID/OOD evaluation protocol, GPU layouts, result inspection, licensing and citation. The full evaluation uses **100 rollouts per task: 50 ID + 50 OOD**.

## Contents

- [Simulation environment](#installation)
- [Create and run a task](#simulation)
- [Training data and action spaces](#training)
- [Model environments and training](#models)
- [π0.5](#pi05)
- [LingBot-VLA-v2](#lingbot)
- [GR00T-N1.7](#gr00t)
- [Additional model adapters](#additional-models)
- [Unified evaluation](#evaluation)
- [Licensing and acknowledgements](#licensing)
- [Citation](#citation)

<a id="installation"></a>


## Simulation environment


```bash
git clone https://github.com/robbyant-research/BiDex-100.git
cd BiDex-100

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[eval]"
```

The release is available on [Hugging Face](https://huggingface.co/datasets/Robbyant-Research/BiDex-100) and [ModelScope](https://modelscope.cn/datasets/Robbyant-Research/BiDex-100), with the same three independently downloadable components:


```text
assets/                    compressed simulation asset packs
datasets/lerobotv21/       LeRobot v2.1 datasets
datasets/lerobotv30/       LeRobot v3.0 datasets
```

Download everything from Hugging Face (the default source) and prepare the simulation assets:


```bash
python -m pip install -U huggingface_hub
hf auth login  # omit for public, unauthenticated downloads

bash scripts/download_release.sh all downloads/BiDex-100
bash scripts/setup_assets_links.sh downloads/BiDex-100
```

To download from ModelScope instead, install its CLI and select the source:


```bash
python -m pip install -U modelscope

bash scripts/download_release.sh --source modelscope all downloads/BiDex-100
bash scripts/setup_assets_links.sh downloads/BiDex-100
```

Public ModelScope downloads do not require login. Each component can instead be downloaded separately into the same destination:


```bash
bash scripts/download_release.sh assets downloads/BiDex-100
bash scripts/download_release.sh lerobotv21 downloads/BiDex-100
bash scripts/download_release.sh lerobotv30 downloads/BiDex-100

# The same component selection works with ModelScope.
bash scripts/download_release.sh --source modelscope assets downloads/BiDex-100
bash scripts/download_release.sh --source modelscope lerobotv21 downloads/BiDex-100
bash scripts/download_release.sh --source modelscope lerobotv30 downloads/BiDex-100
```

`scripts/download_assets.sh` and `scripts/download_datasets.sh` remain as convenience wrappers. Run `download_datasets.sh lerobotv21`, `download_datasets.sh lerobotv30`, or `download_datasets.sh all` to fetch one or both dataset formats. The download destination preserves the repository layout shown above.

Both wrappers accept `--source modelscope` before their positional arguments:


```bash
bash scripts/download_assets.sh --source modelscope downloads/BiDex-100
bash scripts/download_datasets.sh --source modelscope all downloads/BiDex-100
```

Set `BIDEX_DOWNLOAD_SOURCE=modelscope` to select ModelScope by default; `--source` overrides this setting. `BIDEX_HF_REPO` and `BIDEX_MODELSCOPE_REPO` override the repository ID for their respective source (both default to `Robbyant-Research/BiDex-100`). Rerun the same command to reuse downloaded files and resume an interrupted download. After downloading, the script checks for all five asset packs or all 106 task directories per selected dataset format, including task metadata, Parquet data, and videos.

The asset setup script accepts either `downloads/BiDex-100` or `downloads/BiDex-100/assets`. It extracts each archive in place, creates the repository links, and recreates the relative aliases used by assembly tasks. It is idempotent, so an interrupted setup can be rerun safely. Alternatively set `BIDEX_ASSETS=/path/to/BiDex-100/assets`. Export `HF_TOKEN` instead of using `hf auth login` in non-interactive environments.

For headless GPU rendering, select EGL before starting Python:


```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

<a id="simulation"></a>


## Create and run a task

Task YAML files under `bidex/tasks_data/suites/<suite>/<task>.yaml` define the scene, robot, objects, initialization distribution, instruction, success predicates, subtasks, and OOD counterpart. The five suites contain 10, 10, 20, 26, and 40 tasks respectively (106 total).

This minimal example is complete and directly runnable. It disables cameras so that the first simulation check does not require a graphics context, resets one task, holds the reset robot pose, advances physics with `env.step`, and closes all MuJoCo resources:


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

Use `list_suites()` and `list_tasks(<suite>)` to discover the 106 base tasks. Pass `include_ood=True` to `list_tasks` only when inspecting the paired OOD variant definitions. Pass `cameras=["head", "left_wrist", "right_wrist"]` when RGB observations are needed.

<a id="training"></a>


## Training data and action spaces

BiDex-100 is model-agnostic: this repository provides the benchmark and data contract, while each VLA implementation owns its trainer. The supported 58-D representations are:

| Mode | Arm component | Hand component |
| --- | --- | --- |
| `joint` | `actions.arm_joint_position` (14-D) | `actions.dex_joint_position` (44-D) |
| `robot_ee` | `actions.abs_end_effector` (14-D) | `actions.dex_joint_position` (44-D) |
| `camera_ee` | `actions.abs_end_effector_camera` (14-D) | `actions.dex_joint_position` (44-D) |

Use `bidex.tasks.benchmark.BENCHMARKS` and `MODES` as the canonical suite and modality registry when adding a trainer integration.


### Reported training setup

The currently reported model results use **co-training**: demonstrations from multiple benchmark tasks are merged into one training corpus and used to train one shared policy checkpoint. Evaluation then runs that same checkpoint across all applicable tasks; the reported numbers are not an ensemble of separately trained per-task policies.

<a id="models"></a>


## Model environments and training

Install the simulator and each model in separate environments. Run commands below from the indicated repository root. Use the same downloaded task corpus for co-training: one shared policy across tasks and embodiments. The reported joint setup uses the three RGB cameras, 14 arm joints and 44 hand joints, in right-first order. Pretrained model weights are separate downloads from the source code.

| Model | BiDex-100 source | Training runtime | Reported checkpoint |
| --- | --- | --- | --- |
| π0.5 | [openpi-bidex100](https://github.com/WangYong-999/openpi-bidex100) | Python 3.11 · JAX / OpenPI | 60k steps · chunk 50 |
| LingBot-VLA-v2 | [lingbot-vla-v2-bidex100](https://github.com/WangYong-999/lingbot-vla-v2-bidex100) | Python 3.12 · PyTorch / FSDP2 | 60k steps · chunk 50 |
| GR00T-N1.7 | [Isaac-GR00T-bidex100](https://github.com/WangYong-999/Isaac-GR00T-bidex100) | Python 3.12 · PyTorch | 400k steps · chunk 16 |

Each model repository includes a `BIDEX100.md` guide. The commands below use the public BiDex-100 names. A few serialized identifiers retain aliases for checkpoints created before the benchmark was renamed.

<a id="pi05"></a>


## π0.5


### Create the environment


```bash
git clone https://github.com/WangYong-999/openpi-bidex100.git
cd openpi-bidex100
python -m pip install -U uv
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

The repository lockfile selects OpenPI's LeRobot revision and JAX dependencies. Use this environment for both the trainer and policy server. The unified evaluation launcher also uses this interpreter for simulation workers, so install the benchmark dependencies into it before evaluation.


### Prepare data and train

Set an absolute path to the downloaded **LeRobot v2.1** root. It contains the five capability directories, each with task-level `meta/info.json` files.


```bash
export BIDEX_DATA_ROOT=/absolute/path/to/BiDex-100/downloads/BiDex-100/datasets/lerobotv21

uv run scripts/compute_norm_stats.py --config-name BiDex-All-joint-1
uv run scripts/train.py BiDex-All-joint-1 \
  --exp-name bidex_all_joint \
  --batch-size 256 \
  --num-train-steps 60000 \
  --fsdp-devices 8
```

The 58-D projection layers are initialized for BiDex-100 while compatible weights are transferred from `gs://openpi-assets/checkpoints/pi05_base/params`. The joint transform trains on deltas and restores absolute targets for evaluation. For a single GPU, use `--fsdp-devices 1` and choose a batch size that fits memory. The checkpoint path is `checkpoints/BiDex-All-joint-1/bidex_all_joint/60000`.


### Evaluate

Run from the **BiDex-100** repository after preparing the model environment and checkpoint above. Replace the absolute paths, activate the simulation environment, install the benchmark dependencies into `MODEL_PY` as described under [Unified evaluation](#evaluation), and use a fresh `MID_OVERRIDE` for each run. This launches the full benchmark: 50 ID + 50 OOD rollouts per task, with the native task horizon and model action chunk.

The config name above matches newly trained public checkpoints. For an earlier checkpoint using `assets/dexsim_all_joint`, select the retained `DexSim-All-joint-1` config alias and the matching checkpoint path.


```bash
export MODEL_REPO=/absolute/path/to/openpi-bidex100
export MODEL_PY="$MODEL_REPO/.venv/bin/python"
export MODEL_CKPT="$MODEL_REPO/checkpoints/BiDex-All-joint-1/bidex_all_joint/60000"
export BATCHED_CONFIG=BiDex-All-joint-1
export BIDEX_LEADERBOARD_ROOT=/absolute/path/to/evaluation-output
export ID_SEEDS=0,1 OOD_SEEDS=2,3 EPS=25
export MODEL_GPUS=0,1,2,3,4,5,6,7
export RENDER_GPUS=0,1,2,3,4,5,6,7
export MID_OVERRIDE=pi05_alljoint_60k_full_norm20

bash scripts/evaluation/run_vec_models_eval.sh pi05
```

<a id="lingbot"></a>


## LingBot-VLA-v2


### Create the environment


```bash
git clone --recurse-submodules https://github.com/WangYong-999/lingbot-vla-v2-bidex100.git
cd lingbot-vla-v2-bidex100
bash tools/create_train_env.sh --env-name bidex-lingbot
conda activate bidex-lingbot
```

The repository installer creates Python 3.12 and pins the PyTorch 2.8 stack. It also installs FlashAttention and the depth packages. A matching local FlashAttention wheel can be supplied with `--flash-attn-wheel /path/to/wheel.whl`. Download the LingBot-VLA-v2-6B base checkpoint, Qwen3-VL tokenizer/processor, MoGe weights, and the depth/video teacher files described in the upstream README.


### Prepare the 58-D checkpoint and data

BiDex-100 uses 58 action dimensions; the pretrained LingBot checkpoint uses 55. Use the migrated projection-expansion utility before fine-tuning:


```bash
python scripts/bidex_create_58d_ckpt.py \
  --src /absolute/path/to/lingbot-vla-v2-6b \
  --dst /absolute/path/to/lingbot-vla-v2-6b-58d

python scripts/prepare_bidex100.py \
  --data-root /absolute/path/to/BiDex-100/downloads/BiDex-100/datasets/lerobotv30 \
  --model-path /absolute/path/to/lingbot-vla-v2-6b-58d \
  --tokenizer-path /absolute/path/to/Qwen3-VL-4B-Instruct \
  --teacher-root /absolute/path/to/lingbot-vla-v2-6b \
  --moge-path /absolute/path/to/moge-2-vitb-normal/model.pt \
  --output-dir /absolute/path/to/outputs/lingbot-bidex
```

This writes `assets/training_data/bidex_all_joint.txt` and `configs/vla/bidex/local_joint.yaml`. It keeps the joint delta-action convention and checkpoint milestones from the source integration, while supplying your data, tokenizer, teacher and output paths. The released joint normalization statistics are included at `assets/norm_stats/bidex_joint.json`.


### Train


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/vla/bidex/local_joint.yaml
```

The generated recipe trains for 60k steps with the source 58-D architecture and alignment objectives. For fewer GPUs, adjust micro/global batch sizes and gradient accumulation together. Evaluate the saved step directory (or its `hf_ckpt` subdirectory) that contains the Hugging Face weights.


### Evaluate

Run from the **BiDex-100** repository after preparing the model environment and checkpoint above. Replace the absolute paths, activate the simulation environment, install the benchmark dependencies into `MODEL_PY` as described under [Unified evaluation](#evaluation), and use a fresh `MID_OVERRIDE` for each run. This launches the full benchmark: 50 ID + 50 OOD rollouts per task, with the native task horizon and model action chunk.

Set `MODEL_PY` to the Python executable in the `bidex-lingbot` environment created above. The saved step is under `checkpoints/global_step_60000`.


```bash
export MODEL_REPO=/absolute/path/to/lingbot-vla-v2-bidex100
export MODEL_PY=/absolute/path/to/conda/envs/bidex-lingbot/bin/python
export MODEL_VLM_PATH=/absolute/path/to/Qwen3-VL-4B-Instruct
export MODEL_CKPT=/absolute/path/to/outputs/lingbot-bidex/checkpoints/global_step_60000
export BIDEX_LEADERBOARD_ROOT=/absolute/path/to/evaluation-output
export ID_SEEDS=0,1 OOD_SEEDS=2,3 EPS=25
export MODEL_GPUS=0,1,2,3,4,5,6,7
export RENDER_GPUS=0,1,2,3,4,5,6,7
export MID_OVERRIDE=lingbot_alljoint_60k_full_norm20

bash scripts/evaluation/run_vec_models_eval.sh lingbot
```

<a id="gr00t"></a>


## GR00T-N1.7


### Create the environment


```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/WangYong-999/Isaac-GR00T-bidex100.git
cd Isaac-GR00T-bidex100
python -m pip install -U uv
uv sync --python 3.12 --all-extras
```

For an NVIDIA dGPU, use a CUDA-compatible driver and the dependency setup in `scripts/deployment/dgpu/install_deps.sh`. Download the GR00T-N1.7 pretrained model separately. The BiDex-100 integration keeps the upstream installation manifests and training loop.


### Prepare data and train

The preparation utility writes the state/action/camera mapping required by GR00T to each task's `meta/modality.json`. It uses the **LeRobot v2.1** download. The joint embodiment is shared by both robots, giving one normalization profile for the co-trained policy.


```bash
uv run python examples/BiDex-100/prepare_training.py \
  --data-root /absolute/path/to/BiDex-100/downloads/BiDex-100/datasets/lerobotv21

NUM_GPUS=8 MAX_STEPS=400000 GLOBAL_BATCH_SIZE=256 \
  uv run bash examples/BiDex-100/train.sh \
    --data-root /absolute/path/to/BiDex-100/downloads/BiDex-100/datasets/lerobotv21 \
    --base-model-path /absolute/path/to/GR00T-N1.7-3B \
    --output-dir /absolute/path/to/outputs/gr00t-bidex
```

`examples/BiDex-100/dex_sim_joint_config.py` defines the three RGB cameras, 58-D relative arm/hand targets and native 16-step action chunk. The persisted `dex_sim_joint` identifier is retained for compatibility with the benchmark's existing checkpoint loader; the integration is for BiDex-100.


### Evaluate

Run from the **BiDex-100** repository after preparing the model environment and checkpoint above. Replace the absolute paths, activate the simulation environment, install the benchmark dependencies into `MODEL_PY` as described under [Unified evaluation](#evaluation), and use a fresh `MID_OVERRIDE` for each run. This launches the full benchmark: 50 ID + 50 OOD rollouts per task, with the native task horizon and model action chunk.


```bash
export MODEL_REPO=/absolute/path/to/Isaac-GR00T-bidex100
export MODEL_PY="$MODEL_REPO/.venv/bin/python"
export MODEL_CKPT=/absolute/path/to/outputs/gr00t-bidex/checkpoint-400000
export BIDEX_LEADERBOARD_ROOT=/absolute/path/to/evaluation-output
export ID_SEEDS=0,1 OOD_SEEDS=2,3 EPS=25
export MODEL_GPUS=0,1,2,3,4,5,6,7
export RENDER_GPUS=0,1,2,3,4,5,6,7
export MID_OVERRIDE=gr00t_allmixed_400k_full_norm20

bash scripts/evaluation/run_vec_models_eval.sh gr00t
```

<a id="additional-models"></a>


## Additional model adapters

The benchmark also includes evaluation adapters for the four models below. Their source repositories are linked here; the three BiDex-100 forks above are the model integrations released with this guide. Use a BiDex-100-trained checkpoint with the adapter's expected schema, rather than an unrelated upstream checkpoint. Follow each source repository's installation and fine-tuning guide when preparing its model environment.

| Model | Source / environment and training guide | Evaluation selector | Extra path |
| --- | --- | --- | --- |
| Galaxea G0.5 | [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) | `galaxea` | — |
| OpenDM DM0.5 | [OpenDM](https://github.com/dexmal/opendm) | `dm05` | — |
| Xiaomi XR1 | [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) | `xr1` | `MODEL_VLM_PATH` |
| InternVLA-A1.5 | [InternVLA-A1.5](https://github.com/InternRobotics/InternVLA-A-series) | `internvla` | `MODEL_VLM_PATH` |

For each adapter, export `MODEL_PY`, `MODEL_REPO`, `MODEL_CKPT` and `BIDEX_LEADERBOARD_ROOT`, plus `MODEL_VLM_PATH` where listed. Then run, for example, `bash scripts/evaluation/run_vec_models_eval.sh galaxea smoke`. Use a separate output identifier for each checkpoint. The next section describes the full 100-rollout protocol shared by all seven models.

<a id="evaluation"></a>


## Unified evaluation

BiDex-100 exposes one evaluation contract for different VLA policies. A model adapter receives the same three camera observations, proprioceptive state, and language instruction; the benchmark owns task discovery, seeded resets, ID/OOD randomization, rollout execution, scoring, videos, and result schemas. This keeps model comparisons independent of model-specific launch code.


### ID/OOD protocol summary

**Each task has 100 rollouts in total: 50 ID + 50 OOD.**

The canonical protocol evaluates 25 episodes for each of two ID seeds (`0,1`) and two OOD seeds (`2,3`). Both domains use deterministic per-episode reset seeds and the same task instructions, success predicates, rollout horizon, and scoring rules.

| Setting | In-domain (ID) | Out-of-domain (OOD) |
| --- | --- | --- |
| Task definition | Base task YAML. | The task's committed `__ood.yaml` variant (all 106 tasks are covered). |
| Robot initial state | Deterministically samples a recorded working pose from the task's evaluation H5 and restores robot joints only; objects retain the fresh base-YAML placement. | Uses the task's default robot reset pose; it does not sample an H5 demonstration pose. |
| Objects and layout | Base-YAML regions and object identities. | Task-specific baked region/layout shifts; selected tasks additionally use left/right arm mirroring, collision-safe distractors, or per-episode articulated-object scale changes. |
| Scene appearance | Original floor, wall, table, and lights. | Deterministic floor/wall/table textures and randomized lighting. |
| Cameras | Nominal head and wrist extrinsics. | A collision-audited finite pose bank independently controls head and wrist changes. |

OOD appearance randomization is intentionally bounded but visually strong:

- Lighting cycles through five exposure profiles from `dark` to `high_bright` with key/fill/ambient levels of `0.11/0.035/0.014`, `0.18/0.060/0.022`, `0.29/0.100/0.040`, `0.40/0.145/0.058`, and `0.48/0.180/0.072`. The key light casts shadows and spans the full azimuth circle, elevations of 25/38/52/66/76 degrees, neutral/warm/cool tints, and distances of 3.0/3.5/4.0/4.4 m. The non-shadowing fill light provides an illumination floor to avoid unusably dark observations.
- Head-camera offsets are bounded to 1.8 cm and 6 degrees. Wrist-camera offsets are bounded to 1.35 cm and 6 degrees; the mount-clearance axis is restricted to outward motion for both Franka and Astribot so the robot connector cannot move in front of the lens. Across the canonical 50 OOD episodes per task, camera occurrence counts are exactly 20 nominal, 10 head-only, 10 wrists-only, and 10 head-and-wrists.
- Object/layout changes come from each task's committed OOD YAML. Distractor identity is fixed per task and OOD seed while placement is resampled per episode; task geometry and reachability constraints keep objects on usable, collision-safe tabletop regions.

Randomization uses component-scoped SHA-256 keys, NumPy PCG64 streams, and hash-validated texture/camera manifests. Consequently, a fixed task, seed, and episode produces the same plan on different machines without depending on GPU or renderer output.

Seven reference integrations are included: π0.5, Galaxea G0.5, GR00T N1.7, LingBot-VLA, XR-1, InternVLA-A1.5, and DM0.5. Additional models can use the same driver by implementing the observation-to-action adapter used by `bidex/eval/run_leaderboard_eval.py`.

The launcher has no machine-specific defaults. Every integration supplies:

- `MODEL_PY`: Python interpreter for the model environment.
- `MODEL_REPO`: model source directory.
- `MODEL_CKPT`: checkpoint path.
- `BIDEX_LEADERBOARD_ROOT`: output directory.

`MODEL_VLM_PATH` is additionally required by LingBot-VLA, XR-1, and InternVLA-A1.5 because those adapters load a separate local VLM backbone. `OPENPI_DATA_HOME` is an optional native OpenPI cache override; when unset, OpenPI uses its normal cache.


### Full evaluation settings

The three model examples above follow the full evaluation recipe: `ID_SEEDS=0,1`, `OOD_SEEDS=2,3` and `EPS=25`, giving **100 rollouts per task**. All eight GPUs host both policy servers and render workers. Run from a fresh shell with no earlier task filters or debug overrides; the launcher defaults to all tasks with native horizons and action chunks.

Install the simulation dependencies in the active BiDex-100 environment as described in [Simulation environment](#installation). The launcher uses `MODEL_PY` for both policy servers and simulation workers. Install the benchmark requirements into that environment too, before running any of the three full examples:


```bash
"$MODEL_PY" -m pip install -r requirements.txt h5py
```

For an environment created by `uv sync` without pip, use:


```bash
uv pip install --python "$MODEL_PY" -r requirements.txt h5py
```

For offline π0.5 evaluation, point OpenPI at a pre-populated cache containing the tokenizer and base-model assets. Leave this unset to use OpenPI's normal cache and download behavior:


```bash
export OPENPI_DATA_HOME=/absolute/path/to/openpi-cache
```


### GPU layouts

`MODEL_GPUS` assigns policy servers and `RENDER_GPUS` assigns vectorized MuJoCo workers. The benchmark protocol and output schema do not change with the layout. Every layout uses the same server-client transport; a GPU present in both lists hosts separate policy-server and render-worker processes.

| Layout | `MODEL_GPUS` | `RENDER_GPUS` | Description |
| --- | --- | --- | --- |
| Single GPU | `0` | `0` | Policy and simulation share one card. |
| Dedicated split | `0` | `1,2,3,4,5,6,7` | One policy card and seven render-only cards. |
| Multi-server split | `0,1` | `2,3,4,5,6,7` | Two policy servers distribute requests from six workers. |
| Partial overlap | `0,1` | `1,2,3,4,5,6,7` | GPU 1 hosts both roles. |
| Full overlap | `0,1,2,3,4,5,6,7` | `0,1,2,3,4,5,6,7` | Every card hosts one server and one worker. |

Use a distinct `MID_OVERRIDE` for every smoke or full run. `BENCHES`, `BIDEX_TASKS`, `SEEDS`, `ID_SEEDS`, `OOD_SEEDS`, `EPS`, `ENVS_PER_GPU`, `MAX_RECORD_ENVS`, and `BIDEX_MAX_ENVS_PER_GPU` refine the protocol without changing its scoring contract. `GPUS` remains a compatibility alias; explicit model/render lists are recommended.

Workers execute the full action chunk returned by each model by default. Set `BIDEX_VEC_CHUNK_REUSE` to a positive integer only to cap that native horizon for an explicit ablation; empty or zero keeps the model's maximum chunk size.


### View results

A full joint evaluation produces 12 domain-result JSON files and `summary.json`. The files cover six suite/robot settings × two domains: Generalization evaluates both robots; the other four suites use Franka. Each file aggregates task results. Inspect the summary and logs for episode totals and failures.

`MID_OVERRIDE` identifies one evaluated checkpoint/configuration. Results, videos, and live logs are isolated under that identifier:


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

Open `http://<host-ip>:12006/baselines/` from the local network. The page discovers model rows from `results/<MID_OVERRIDE>/`, refreshes live results, and serves rollout videos. Leaderboards rank by strict success rate by default; progress score is the secondary metric and can be selected from the metric toggle.

<a id="licensing"></a>


## Licensing and acknowledgements

| Component | License |
| --- | --- |
| Codebase | [MIT License](https://github.com/robbyant-research/BiDex-100/blob/master/LICENSE) |
| BiDex-100 datasets | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| BiDex-100 original 3D assets | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| Third-party assets | Their original upstream licenses, retained inside each asset subdirectory |


### Acknowledgements

BiDex-100 builds on [MuJoCo](https://mujoco.readthedocs.io/), [LeRobot](https://github.com/huggingface/lerobot), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [RoboCasa](https://github.com/robocasa/robocasa), [robosuite](https://github.com/ARISE-Initiative/robosuite), [dexjoco](https://github.com/brave-eai/dexjoco), and [Articraft](https://github.com/mattzh72/articraft).

<a id="citation"></a>


## Citation


```bibtex
@article{wang2026bidex,
  title={BiDex-100: A Comprehensive Benchmark for Bimanual Dexterous Manipulation Across 100+ Simulation Tasks},
  author={Wang, Yong and Zheng, Kecheng and Zhu, Xing and Xu, Chao and Shen, Yujun and Cai, Shengze},
  journal={arXiv preprint arXiv:2610.25459},
  year={2026}
}
```
