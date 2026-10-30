<p align="center">
  <img src="https://raw.githubusercontent.com/robbyant-research/BiDex-100/gh-pages/static/images/title.png" alt="BiDex-100" width="760">
</p>

<p align="center">
  <strong>A comprehensive benchmark for bimanual dexterous manipulation across 100+ simulation tasks</strong>
</p>

<p align="center">
  <a href="https://robbyant-research.github.io/BiDex-100/"><img alt="Project Website" src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://arxiv.org/abs/2610.25459"><img alt="arXiv: 2610.25459" src="https://img.shields.io/badge/arXiv-2610.25459-B31B1B?logo=arxiv&logoColor=white"></a>
  <a href="https://github.com/robbyant-research/BiDex-100"><img alt="GitHub Code" src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/Robbyant-Research/BiDex-100"><img alt="Dataset on Hugging Face" src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Dataset&message=HuggingFace&color=yellow"></a>
  <a href="https://modelscope.cn/datasets/Robbyant-Research/BiDex-100"><img alt="Dataset on ModelScope" src="https://img.shields.io/static/v1?label=Dataset&amp;message=ModelScope&amp;color=purple&amp;logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMCAyMCI%2BPHBhdGggZD0iTTAsOS4zMzU3NTY3Nzg3MTcwNDFMMi4yOTA0MjcyLDkuMzM1NzU2Nzc4NzE3MDQxTDIuMjkwNDI3MiwxMS42NjI1MDQ2Nzg3MTcwNDJMMCwxMS42NjI1MDQ2Nzg3MTcwNDJMMCw5LjMzNTc1Njc3ODcxNzA0MVpNOC44NTM4ODE4LDExLjY2MjUwNDY3ODcxNzA0MkwxMS4xNDUxNzksMTEuNjYyNTA0Njc4NzE3MDQyTDExLjE0NTE3OSwxMy45ODc0ODYzNzg3MTcwNEw4Ljg1Mzg4MTgsMTMuOTg3NDg2Mzc4NzE3MDRMOC44NTM4ODE4LDExLjY2MDczOTY3ODcxNzA0MUw4Ljg1Mzg4MTgsMTEuNjYyNTA0Njc4NzE3MDQyWk0xNS43MjYwMzMsMTMuOTg3NDg2Mzc4NzE3MDRMMTMuNDM1NjA2LDEzLjk4NzQ4NjM3ODcxNzA0TDEzLjQzNTYwNiwxNS45OTk5OTk5Nzg3MTcwNEwxNy43MDg2MzMsMTUuOTk5OTk5OTc4NzE3MDRMMTcuNzA4NjMzLDExLjY2MDczOTY3ODcxNzA0MUwxNS43MjYwMzEsMTEuNjYwNzM5Njc4NzE3MDQxTDE1LjcyNjAzMywxMy45ODc0ODYzNzg3MTcwNFoiIGZpbGw9IiM2MjRBRkYiLz48cGF0aCBkPSJNMTEuMTQ1MTgsOS4zMzY2NjM1MjA3MDAwNzNMMTMuNDM1NjA3LDkuMzM2NjYzNTIwNzAwMDczTDEzLjQzNTYwNywxMS42NjI1Mjc4MjA3MDAwNzRMMTEuMTQ1MTgsMTEuNjYyNTI3ODIwNzAwMDc0TDExLjE0NTE4LDkuMzM2NjYzNTIwNzAwMDczWk0wLDcuMDEyNTY0NDIwNzAwMDczTDIuMjkwNDI3LDcuMDEyNTY0NDIwNzAwMDczTDIuMjkwNDI3LDkuMzM3NTQ2MTIwNzAwMDc0TDAsOS4zMzc1NDYxMjA3MDAwNzRMMCw3LjAxMjU2NDQyMDcwMDA3M0wwLDcuMDEzNDQ2ODg2MjEwMDczTDAsNy4wMTI1NjQ0MjA3MDAwNzNaIiBmaWxsPSIjMzZDRkQxIi8%2BPHBhdGggZD0iTTE3LjcwODcwMzk5NDc1MDk3Nyw5LjMzNTc1Njc3ODcxNzA0MUwyMC4wMDAwMDAxOTQ3NTA5NzgsOS4zMzU3NTY3Nzg3MTcwNDFMMjAuMDAwMDAwMTk0NzUwOTc4LDExLjY2MjUwNDQ3ODcxNzA0MUwxNy43MDk1NzMzNDQ3MzA5NzcsMTEuNjYyNTA0NDc4NzE3MDQxTDE3LjcwOTU3MzM0NDczMDk3Nyw5LjMzNTc1NzcyMTUyMjEwMkwxNy43MDg3MDM5OTQ3NTA5NzcsOS4zMzU3NTY3Nzg3MTcwNDFaIiBmaWxsPSIjNjI0QUZGIi8%2BPHBhdGggZD0iTTE3LjcwODcwMzk5NDc1MDk3Nyw3LjAxMjU2NDQyMDcwMDA3M0wyMC4wMDAwMDAxOTQ3NTA5NzgsNy4wMTI1NjQ0MjA3MDAwNzNMMjAuMDAwMDAwMTk0NzUwOTc4LDkuMzM3NTQ2MTIwNzAwMDc0TDE3LjcwOTU3MzM0NDczMDk3Nyw5LjMzNzU0NjEyMDcwMDA3NEwxNy43MDk1NzMzNDQ3MzA5NzcsNy4wMTI1NjQ0MjA3MDAwNzNMMTcuNzA5NTczMzQ0NzMwOTc3LDcuMDEzNDQ2ODg2MjEwMDczTDE3LjcwODcwMzk5NDc1MDk3Nyw3LjAxMjU2NDQyMDcwMDA3M1oiIGZpbGw9IiMzNkNGRDEiLz48cGF0aCBkPSJNMTMuNDM1NDgzOTMyNDk1MTE3LDVMMTMuNDM1NDgzOTMyNDk1MTE3LDcuMDEyNTEzMkwxNS43MjU5MTA5MzI0OTUxMTgsNy4wMTI1MTMyTDE1LjcyNTkxMDkzMjQ5NTExOCw5LjMzNzQ5NDlMMTcuNzA4NTEzNzMyNDk1MTE3LDkuMzM3NDk0OUwxNy43MDg1MTM3MzI0OTUxMTcsNUwxMy40MzU0ODM5MzI0OTUxMTcsNVoiIGZpbGw9IiM2MjRBRkYiLz48cGF0aCBkPSJNNi41NjI0Nzc1ODg2NTM1NjQ1LDkuMzM1NzU2Nzc4NzE3MDQxTDguODUzNzczNzg4NjUzNTY0LDkuMzM1NzU2Nzc4NzE3MDQxTDguODUzNzczNzg4NjUzNTY0LDExLjY2MjUwNDQ3ODcxNzA0MUw2LjU2MjQ3NzU4ODY1MzU2NDUsMTEuNjYyNTA0NDc4NzE3MDQxTDYuNTYyNDc3NTg4NjUzNTY0NSw5LjMzNTc1Njc3ODcxNzA0MVoiIGZpbGw9IiMzNkNGRDEiLz48cGF0aCBkPSJNNC4yNzM3NjI3ODM0OTYwOTQsNy4wMTI1MTMyTDYuNTY0MTg5ODgzNDk2MDk0LDcuMDEyNTEzMkw2LjU2NDE4OTg4MzQ5NjA5NCw1TDIuMjkxMTYwNTgzNDk2MDkzOCw1TDIuMjkxMTYwNTgzNDk2MDkzOCw5LjMzNzQ5NDlMNC4yNzM3NjI3ODM0OTYwOTQsOS4zMzc0OTQ5TDQuMjczNzYyNzgzNDk2MDk0LDcuMDEyNTEzMlpNNC4yNzM3NjI3ODM0OTYwOTQsMTEuNjYyNDc2NUwyLjI5MTE2MTA0Nzg5MjQ0MzYsMTEuNjYyNDc2NUwyLjI5MTE2MTA0Nzg5MjQ0MzYsMTUuOTk5MDg4TDYuNTY0MTg5ODgzNDk2MDk0LDE1Ljk5OTA4OEw2LjU2NDE4OTg4MzQ5NjA5NCwxMy45ODY1NzQyTDQuMjczNzYyNzgzNDk2MDk0LDEzLjk4NjU3NDJMNC4yNzM3NjI3ODM0OTYwOTQsMTEuNjYyNDc2NVoiIGZpbGw9IiM2MjRBRkYiLz48L3N2Zz4%3D"></a>
</p>

## Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/robbyant-research/BiDex-100/gh-pages/static/images/figures/overview.png" alt="BiDex-100 benchmark overview" width="92%">
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
see [Simulation environment](docs/DOCUMENT.md#installation).
Data and assets: [Hugging Face](https://huggingface.co/datasets/Robbyant-Research/BiDex-100) ·
[ModelScope](https://modelscope.cn/datasets/Robbyant-Research/BiDex-100).

## Tasks

See [Create and run a task](docs/DOCUMENT.md#simulation)
for the simulation API and [Task gallery](https://robbyant-research.github.io/BiDex-100/#get-started)
for all 106 tasks.

## Training

See [Training data and action spaces](docs/DOCUMENT.md#training)
and the model environment, data preparation and training guides:
[π0.5](docs/DOCUMENT.md#pi05) ·
[LingBot-VLA-v2](docs/DOCUMENT.md#lingbot) ·
[GR00T-N1.7](docs/DOCUMENT.md#gr00t).

## Evaluation

The full protocol runs **100 rollouts per task**: 25 episodes for each of two ID
seeds (`0,1`) and two OOD seeds (`2,3`). The example below uses π0.5; prepare its
[model environment and checkpoint](docs/DOCUMENT.md#pi05)
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

See [Unified evaluation](docs/DOCUMENT.md#evaluation)
for the ID/OOD protocol, GPU layouts, offline cache, result files and local results
server. Full model-specific examples are available for
[π0.5](docs/DOCUMENT.md#pi05),
[LingBot-VLA-v2](docs/DOCUMENT.md#lingbot) and
[GR00T-N1.7](docs/DOCUMENT.md#gr00t).
Reported results: [Leaderboard and detailed results](https://robbyant-research.github.io/BiDex-100/leaderboard.html).

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
@article{wang2026bidex,
  title={BiDex-100: A Comprehensive Benchmark for Bimanual Dexterous Manipulation Across 100+ Simulation Tasks},
  author={Wang, Yong and Zheng, Kecheng and Zhu, Xing and Xu, Chao and Shen, Yujun and Cai, Shengze},
  journal={arXiv preprint arXiv:2610.25459},
  year={2026}
}
```
