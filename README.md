<p align="center">
  <img src="https://raw.githubusercontent.com/BiDex-100/BiDex-100/gh-pages/static/images/title.png" alt="BiDex-100" width="760">
</p>

<p align="center">
  <strong>A comprehensive benchmark for bimanual dexterous manipulation across 100+ simulation tasks</strong>
</p>

<p align="center">
  <a href="https://bidex-100.github.io/BiDex-100/"><img alt="Project Website" src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/datasets/Robbyant-Research/BiDex-100"><img alt="Dataset on Hugging Face" src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Dataset&message=HuggingFace&color=yellow"></a>
  <a href="https://modelscope.cn/datasets/Robbyant-Research/BiDex-100"><img alt="Dataset on ModelScope" src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Dataset&message=ModelScope&color=purple"></a>
  <a href="LICENSE"><img alt="Code license: MIT" src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="Data and original assets license: CC BY-NC 4.0" src="https://img.shields.io/badge/Data%20%26%20Original%20Assets-CC_BY--NC_4.0-8A2BE2"></a>
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

# Download simulation assets and link them into the repository.
python -m pip install -U huggingface_hub
bash scripts/download_release.sh assets downloads/BiDex-100
bash scripts/setup_assets_links.sh downloads/BiDex-100

# Headless GPU rendering.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

For dataset downloads, ModelScope alternatives, asset layout and setup details,
see [Simulation environment](https://bidex-100.github.io/BiDex-100/document.html#installation).
Data and assets: [Hugging Face](https://huggingface.co/datasets/Robbyant-Research/BiDex-100) ·
[ModelScope](https://modelscope.cn/datasets/Robbyant-Research/BiDex-100).

## Tasks

See [Create and run a task](https://bidex-100.github.io/BiDex-100/document.html#simulation)
for the simulation API and [Task gallery](https://bidex-100.github.io/BiDex-100/#get-started)
for all 106 tasks.

## Training

See [Training data and action spaces](https://bidex-100.github.io/BiDex-100/document.html#training)
and the model environment, data preparation and training guides:
[π0.5](https://bidex-100.github.io/BiDex-100/document.html#pi05) ·
[LingBot-VLA-v2](https://bidex-100.github.io/BiDex-100/document.html#lingbot) ·
[GR00T-N1.7](https://bidex-100.github.io/BiDex-100/document.html#gr00t).

## Evaluation

The full protocol runs **100 rollouts per task**: 25 episodes for each of two ID
seeds (`0,1`) and two OOD seeds (`2,3`). The example below uses π0.5; prepare its
[model environment and checkpoint](https://bidex-100.github.io/BiDex-100/document.html#pi05)
first, then run from this repository with the BiDex-100 environment activated.
Replace the absolute paths and use a fresh run identifier.

```bash
export MODEL_REPO=/absolute/path/to/openpi-bidex100
export MODEL_PY="$MODEL_REPO/.venv/bin/python"
export MODEL_CKPT="$MODEL_REPO/checkpoints/BiDex-All-joint-1/bidex_all_joint/60000"
export BATCHED_CONFIG=BiDex-All-joint-1
# Include simulation dependencies in the model environment used by workers.
uv pip install --python "$MODEL_PY" -r requirements.txt h5py
export BIDEX_LEADERBOARD_ROOT=/absolute/path/to/evaluation-output
export ID_SEEDS=0,1 OOD_SEEDS=2,3 EPS=25
export MODEL_GPUS=0,1,2,3,4,5,6,7
export RENDER_GPUS=0,1,2,3,4,5,6,7
export MID_OVERRIDE=pi05_alljoint_60k_full_norm20

bash scripts/evaluation/run_vec_models_eval.sh pi05
```

See [Unified evaluation](https://bidex-100.github.io/BiDex-100/document.html#evaluation)
for the ID/OOD protocol, GPU layouts, offline cache, result files and local results
server. Full model-specific examples are available for
[π0.5](https://bidex-100.github.io/BiDex-100/document.html#pi05),
[LingBot-VLA-v2](https://bidex-100.github.io/BiDex-100/document.html#lingbot) and
[GR00T-N1.7](https://bidex-100.github.io/BiDex-100/document.html#gr00t).
Reported results: [Leaderboard and detailed results](https://bidex-100.github.io/BiDex-100/leaderboard.html).

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
