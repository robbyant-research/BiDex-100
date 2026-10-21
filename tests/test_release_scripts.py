from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_local_eval_initializes_eval_pid_before_cleanup_trap() -> None:
    script = (REPO_ROOT / "scripts/evaluation/local_eval.sh").read_text(
        encoding="utf-8"
    )

    assert script.index('EVAL_PID=""') < script.index("_cleanup_all()")


@pytest.fixture
def download_cli(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_log = tmp_path / "download-args"
    mock_hf = bin_dir / "hf"
    mock_hf.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${2:-} == --help ]]; then\n"
        "  if [[ ${MODELSCOPE_LEGACY_CLI:-0} == 1 ]]; then echo --local_dir; else echo --local-dir; fi\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$@\" > \"$HF_ARGS_LOG\"\n"
        "argv=(\"$@\")\n"
        "destination=''\n"
        "for ((i=0; i<${#argv[@]}; i++)); do\n"
        "  if [[ ${argv[$i]} == --local-dir || ${argv[$i]} == --local_dir ]]; then destination=${argv[$((i+1))]}; fi\n"
        "done\n"
        "for arg in \"${argv[@]}\"; do\n"
        "  case $arg in\n"
        "    'assets/**')\n"
        "      for pack in articraft_objects assembly_objects astribot_descriptions libero robocasa_assets; do\n"
        "        mkdir -p \"$destination/assets/$pack\"\n"
        "        printf archive > \"$destination/assets/$pack/$pack.tar.gz\"\n"
        "      done\n"
        "      ;;\n"
        "    'datasets/lerobotv21/**'|'datasets/lerobotv30/**')\n"
        "      format=${arg#datasets/}; format=${format%/**}\n"
        "      mkdir -p \"$destination/datasets/$format/task/meta\" \"$destination/datasets/$format/task/data/chunk-000\" \"$destination/datasets/$format/task/videos/chunk-000/camera\"\n"
        "      printf '{}\\n' > \"$destination/datasets/$format/task/meta/info.json\"\n"
        "      printf parquet > \"$destination/datasets/$format/task/data/chunk-000/file-000.parquet\"\n"
        "      printf video > \"$destination/datasets/$format/task/videos/chunk-000/camera/file-000.mp4\"\n"
        "      ;;\n"
        "  esac\n"
        "done\n",
        encoding="utf-8",
    )
    mock_hf.chmod(0o755)
    shutil.copy2(mock_hf, bin_dir / "modelscope")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HF_ARGS_LOG": str(args_log),
        "BIDEX_EXPECTED_TASKS": "1",
        "BIDEX_DOWNLOAD_SOURCE": "huggingface",
    }
    env.pop("BIDEX_HF_REPO", None)
    env.pop("BIDEX_MODELSCOPE_REPO", None)
    return env, args_log


@pytest.mark.parametrize("source", ["huggingface", "modelscope", "modelscope-legacy"])
def test_download_release_component_filters(
    tmp_path: Path, download_cli: tuple[dict[str, str], Path], source: str,
) -> None:
    env, args_log = download_cli
    legacy = source == "modelscope-legacy"
    if legacy:
        source = "modelscope"
        env = {**env, "MODELSCOPE_LEGACY_CLI": "1"}
    expected = {
        "assets": ["assets/**"],
        "lerobotv21": ["datasets/lerobotv21/**"],
        "lerobotv30": ["datasets/lerobotv30/**"],
        "datasets": [
            "datasets/lerobotv21/**",
            "datasets/lerobotv30/**",
        ],
        "all": [
            "assets/**",
            "datasets/lerobotv21/**",
            "datasets/lerobotv30/**",
        ],
    }
    for component, patterns in expected.items():
        destination = tmp_path / component
        _run(
            "bash",
            "scripts/download_release.sh",
            "--source",
            source,
            component,
            str(destination),
            env=env,
        )
        argv = args_log.read_text(encoding="utf-8").splitlines()
        assert argv[:4] == ["download", "Robbyant-Research/BiDex-100", "--repo-type", "dataset"]
        include_index = argv.index("--include")
        local_dir_index = argv.index("--local_dir" if legacy else "--local-dir")
        assert argv[include_index + 1 : local_dir_index] == patterns
        assert argv[local_dir_index + 1] == str(destination)


@pytest.mark.parametrize(
    ("script", "args", "patterns"),
    [
        ("download_assets.sh", [], ["assets/**"]),
        ("download_datasets.sh", ["all"], ["datasets/lerobotv21/**", "datasets/lerobotv30/**"]),
        ("download_datasets.sh", ["v21"], ["datasets/lerobotv21/**"]),
        ("download_datasets.sh", ["3.0"], ["datasets/lerobotv30/**"]),
        ("download_datasets.sh", [], ["datasets/lerobotv21/**", "datasets/lerobotv30/**"]),
    ],
)
def test_modelscope_wrappers_forward_source_and_preserve_layout(
    tmp_path: Path, download_cli: tuple[dict[str, str], Path],
    script: str, args: list[str], patterns: list[str],
) -> None:
    env, args_log = download_cli
    destination = tmp_path / "download with spaces"
    _run("bash", f"scripts/{script}", "--source", "modelscope", *args, str(destination), env=env)
    argv = args_log.read_text().splitlines()
    assert argv[:4] == ["download", "Robbyant-Research/BiDex-100", "--repo-type", "dataset"]
    assert argv[argv.index("--include") + 1:argv.index("--local-dir")] == patterns
    assert argv[-1] == str(destination)


def test_download_source_environment_and_explicit_override(
    tmp_path: Path, download_cli: tuple[dict[str, str], Path],
) -> None:
    env, args_log = download_cli
    env = {**env, "BIDEX_DOWNLOAD_SOURCE": "modelscope", "BIDEX_MODELSCOPE_REPO": "example/mirror"}
    destination = str(tmp_path / "release")
    _run("bash", "scripts/download_release.sh", "assets", destination, env=env)
    assert args_log.read_text().splitlines()[:4] == ["download", "example/mirror", "--repo-type", "dataset"]
    _run("bash", "scripts/download_release.sh", "--source", "huggingface", "assets", destination, env=env)
    assert args_log.read_text().splitlines()[:3] == ["download", "Robbyant-Research/BiDex-100", "--repo-type"]


@pytest.mark.parametrize("script", ["download_release.sh", "download_assets.sh", "download_datasets.sh"])
@pytest.mark.parametrize("source_args", [["--source"], ["--source", "invalid"]])
def test_download_rejects_invalid_source_arguments(
    tmp_path: Path, download_cli: tuple[dict[str, str], Path], script: str, source_args: list[str],
) -> None:
    env, args_log = download_cli
    args = ["assets", str(tmp_path / "release")] if script == "download_release.sh" else [str(tmp_path / "release")]
    if source_args == ["--source"]:
        args = []
    result = subprocess.run(["bash", f"scripts/{script}", *source_args, *args], cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "source" in result.stderr
    assert not args_log.exists()


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_download_release_rejects_empty_successful_download(tmp_path: Path, source: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mock_hf = bin_dir / "hf"
    mock_hf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    mock_hf.chmod(0o755)
    shutil.copy2(mock_hf, bin_dir / "modelscope")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "BIDEX_DOWNLOAD_SOURCE": source}

    result = subprocess.run(
        [
            "bash",
            "scripts/download_release.sh",
            "lerobotv30",
            str(tmp_path / "empty"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "download is incomplete" in result.stderr
    assert "Download complete." not in result.stdout


@pytest.mark.parametrize("source", ["huggingface", "modelscope"])
def test_download_release_rejects_metadata_without_payload(tmp_path: Path, source: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mock_hf = bin_dir / "hf"
    mock_hf.write_text(
        "#!/usr/bin/env bash\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ $1 == --local-dir || $1 == --local_dir ]]; then\n"
        "    mkdir -p \"$2/datasets/lerobotv21/task/meta\"\n"
        "    printf '{}\\n' > \"$2/datasets/lerobotv21/task/meta/info.json\"\n"
        "    exit 0\n"
        "  fi\n"
        "  shift\n"
        "done\n",
        encoding="utf-8",
    )
    mock_hf.chmod(0o755)
    shutil.copy2(mock_hf, bin_dir / "modelscope")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "BIDEX_EXPECTED_TASKS": "1",
        "BIDEX_DOWNLOAD_SOURCE": source,
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/download_release.sh",
            "lerobotv21",
            str(tmp_path / "metadata-only"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "no parquet data" in result.stderr


def _write_pack(asset_root: Path, pack: str, members: dict[str, bytes]) -> str:
    pack_dir = asset_root / pack
    pack_dir.mkdir(parents=True)
    archive = pack_dir / f"{pack}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(archive.read_bytes()).hexdigest()


@pytest.mark.parametrize("with_manifest", [False, True])
def test_setup_assets_extracts_and_links_idempotently(
    tmp_path: Path, with_manifest: bool,
) -> None:
    fake_repo = tmp_path / "repo"
    scripts_dir = fake_repo / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts/setup_assets_links.sh", scripts_dir)
    (fake_repo / "bidex/robocasa").mkdir(parents=True)
    (fake_repo / "bidex/robots/assembly/astribot_s1/shared").mkdir(parents=True)

    download_root = tmp_path / "download"
    asset_root = download_root / "assets"
    checksums = {
        "articraft_objects": _write_pack(
            asset_root,
            "articraft_objects",
            {"object_a/object_a.xml": b"<mujoco/>\n"},
        ),
        "assembly_objects": _write_pack(
            asset_root,
            "assembly_objects",
            {"assembly_cube/assembly_cube.xml": b"<mujoco/>\n"},
        ),
        "astribot_descriptions": _write_pack(
            asset_root,
            "astribot_descriptions",
            {"urdf/astribot_s1_urdf/meshes/dummy.stl": b"solid dummy\nendsolid\n"},
        ),
        "libero": _write_pack(
            asset_root,
            "libero",
            {
                "objects/dexjoco_objects/threading_rod/threading_rod.xml":
                    b"<mujoco/>\n"
            },
        ),
        "robocasa_assets": _write_pack(
            asset_root,
            "robocasa_assets",
            {
                "objects/lightwheel/tiered_basket/TieredBasket007/model.xml":
                    b"<mujoco/>\n",
                "fixtures/dummy/model.xml": b"<mujoco/>\n",
            },
        ),
    }
    if with_manifest:
        manifest = "".join(
            f"{digest}  {pack}/{pack}.tar.gz\n"
            for pack, digest in checksums.items()
        )
        (asset_root / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    first = subprocess.run(
        ["bash", str(scripts_dir / "setup_assets_links.sh"), str(download_root)],
        cwd=fake_repo,
        check=True,
        text=True,
        capture_output=True,
    )
    expected_message = (
        "Checksum manifest found"
        if with_manifest
        else "Checksum manifest not present"
    )
    assert expected_message in first.stdout
    assert first.stdout.count("  EXTRACT ") == 5
    assert (asset_root / "assembly_objects/cube_a/cube_a.xml").is_symlink()
    assert (
        asset_root
        / "libero/objects/dexjoco_objects/threading_rod_left/threading_rod_left.xml"
    ).is_symlink()
    assert (fake_repo / "bidex/libero").resolve() == (asset_root / "libero").resolve()
    assert (fake_repo / "bidex/robocasa/assets").resolve() == (
        asset_root / "robocasa_assets"
    ).resolve()

    second = subprocess.run(
        ["bash", str(scripts_dir / "setup_assets_links.sh"), str(asset_root)],
        cwd=fake_repo,
        check=True,
        text=True,
        capture_output=True,
    )
    assert second.stdout.count("  READY ") == 5
