from pathlib import Path

from bidex.tasks.registry import list_tasks


def test_list_tasks_excludes_ood_variants_by_default(tmp_path: Path) -> None:
    suite_dir = tmp_path / "Memory"
    suite_dir.mkdir()
    (suite_dir / "task_a.yaml").touch()
    (suite_dir / "task_a__ood.yaml").touch()
    (suite_dir / "task_b.yml").touch()

    assert list_tasks("Memory", str(tmp_path)) == ["task_a", "task_b"]
    assert list_tasks("Memory", str(tmp_path), include_ood=True) == [
        "task_a",
        "task_a__ood",
        "task_b",
    ]
