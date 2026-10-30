import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "bidex/eval/baseline_leaderboard_server.py"
_SPEC = importlib.util.spec_from_file_location("baseline_leaderboard_server", _MODULE_PATH)
leaderboard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(leaderboard)


def _stub_leaderboard(monkeypatch):
    models = [{"id": "model-a", "name": "Model A", "org": "Org", "tag": "ckpt"}]
    monkeypatch.setattr(leaderboard, "discover_models", lambda: models)
    monkeypatch.setattr(leaderboard, "all_table_data", lambda: {"model-a": {}})
    monkeypatch.setattr(leaderboard, "collect_tags", lambda: {"model-a": {"all": "ckpt"}})
    monkeypatch.setattr(leaderboard, "all_progress", lambda: {})


def test_leaderboard_has_working_no_javascript_navigation(monkeypatch):
    _stub_leaderboard(monkeypatch)

    page = leaderboard.render_leaderboard("/baselines")

    assert "BiDex-100 Baseline Leaderboard" not in page
    assert "\x11" not in page
    assert "content:' ↓'" in page
    assert 'href="model/model-a?mode=all&amp;domain=all"' in page
    assert "document.readyState === 'loading'" in page


def test_detail_page_renders_initial_video_without_javascript(monkeypatch):
    _stub_leaderboard(monkeypatch)
    grouped = {
        domain: {benchmark: [] for benchmark in leaderboard.BENCHMARKS}
        for domain in leaderboard.DOMAINS
    }
    grouped["in_domain"]["Memory"] = [{
        "task": "pick_object",
        "robot": "franka",
        "mode": "joint",
        "lang": "Pick the object.",
        "score": 1.0,
        "sr": 1.0,
        "eps": 1,
        "subs": "done:100%",
        "videos": ["mp4s/model-a/Memory_franka_joint_in_domain/task/s1_ep000.mp4"],
        "per_ep": [],
    }]
    monkeypatch.setattr(leaderboard, "detail_tasks_for", lambda *_args: grouped)

    page = leaderboard.render_model_detail("model-a", "/baselines")

    assert 'href="../"' in page
    assert 'preload="metadata"' in page
    assert 'src="../video/mp4s/model-a/Memory_franka_joint_in_domain/task/s1_ep000.mp4"' in page
    assert "document.readyState === 'loading'" in page
