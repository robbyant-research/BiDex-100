import numpy as np
import pytest

from bidex.eval import run_leaderboard_eval as evaluation
from bidex.eval import scoring, vector_rollout
from scripts.evaluation import summarize_eval_results
from scripts.evaluation.policy_protocol import packb, unpackb


def test_full_chunk_is_the_default_and_override_is_validated(monkeypatch):
    monkeypatch.delenv("BIDEX_VEC_CHUNK_REUSE", raising=False)
    assert evaluation._configured_chunk_reuse() == evaluation.FULL_CHUNK_REUSE

    monkeypatch.setenv("BIDEX_VEC_CHUNK_REUSE", "7")
    assert evaluation._configured_chunk_reuse() == 7

    monkeypatch.setenv("BIDEX_VEC_CHUNK_REUSE", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        evaluation._configured_chunk_reuse()


def test_chunk_reuse_clamps_to_native_horizon_and_resets_episode():
    calls = []

    def forward(observations, _adapters):
        calls.append(list(observations))
        return [
            (
                np.full((3, 14), len(calls), dtype=np.float32),
                np.full((3, 44), len(calls), dtype=np.float32),
            )
            for _ in observations
        ]

    infer = evaluation._ChunkReusedVecInfer(
        forward, evaluation.FULL_CHUNK_REUSE, label="test"
    )
    observations = [{"env": 0}, {"env": 1}]
    adapters = [object(), object()]

    for _ in range(3):
        arms, dexs = infer(observations, adapters)
        assert arms.shape == (2, 14)
        assert dexs.shape == (2, 44)
    assert infer.H == 3
    assert len(calls) == 1

    infer(observations, adapters)
    assert len(calls) == 2

    infer.reset_env(1)
    infer(observations, adapters)
    assert len(calls) == 3
    assert calls[-1] == [{"env": 1}]


def test_policy_protocol_round_trip_preserves_arrays():
    source = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        "state": np.linspace(-1, 1, 7, dtype=np.float32),
        "language": "move the cup",
    }
    restored = unpackb(packb(source))

    assert restored["language"] == source["language"]
    assert restored["image"].dtype == source["image"].dtype
    assert restored["state"].dtype == source["state"].dtype
    assert np.array_equal(restored["image"], source["image"])
    assert np.array_equal(restored["state"], source["state"])


def test_public_cli_exposes_only_server_client_evaluation():
    parser = evaluation.build_arg_parser()
    args = parser.parse_args(["--model", "openpi", "--ckpt", "/tmp/model"])
    help_text = parser.format_help()

    assert args.render_gpus == "1"
    assert "--vectorized" not in help_text
    assert "--gpus" not in help_text
    assert "--chunk-reuse" not in help_text
    assert "--modality-config-path" not in help_text


def test_vector_rollout_resolves_episode_score_from_scoring_module():
    _restore, episode_score, _writer = vector_rollout._helpers()

    assert episode_score is scoring.episode_score


def test_vector_worker_failure_is_propagated():
    class FailedWorker:
        def wait(self):
            return 17

    with pytest.raises(RuntimeError, match=r"rank 3 exited with code 17"):
        evaluation._wait_for_vector_workers([(3, FailedWorker())])


@pytest.mark.parametrize(
    ("domain", "seeds"),
    [("in_domain", [0, 1]), ("out_of_domain", [2, 3])],
)
@pytest.mark.parametrize(
    "render_gpus",
    ["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"],
)
def test_vector_workers_receive_canonical_seeds_independent_of_rank(
    monkeypatch, tmp_path, domain, seeds, render_gpus
):
    commands = []

    class SuccessfulWorker:
        def wait(self):
            return 0

    def capture_popen(cmd, **_kwargs):
        commands.append(cmd)
        return SuccessfulWorker()

    monkeypatch.setattr(
        evaluation,
        "enumerate_task_ids",
        lambda *_args: [f"Memory/task_{rank}" for rank in range(8)],
    )
    monkeypatch.setattr(evaluation, "_server_port_list", lambda _model: ["8765"])
    monkeypatch.setattr(evaluation.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(evaluation.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(evaluation, "_merge_shards", lambda *_args: {})

    evaluation.launch_vector(
        model="openpi",
        ckpt="/tmp/checkpoint",
        benchmark="Memory",
        control_mode="joint",
        robot="franka",
        domain=domain,
        seeds=seeds,
        eps_per_seed=1,
        render_gpus=render_gpus,
        out_dir=tmp_path / "runs",
        envs_per_gpu=1,
        max_record_envs=1,
        record_video=False,
        model_id="seed-contract-test",
        mp4_root=str(tmp_path / "mp4s"),
        results_dir=str(tmp_path / "results"),
        arm="dual",
    )

    expected = ",".join(map(str, seeds))
    worker_seeds = [cmd[cmd.index("--seeds") + 1] for cmd in commands]
    assert worker_seeds == [expected] * len(render_gpus.split(","))


def test_empty_result_summary_is_a_failure(tmp_path):
    assert summarize_eval_results.summarize("missing-model", tmp_path) is False
