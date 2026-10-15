from __future__ import annotations

from pathlib import Path

import pytest

from bidex import load_task
from bidex.eval import run_leaderboard_eval as evaluation


SUITE = "Instruction_Following"

COMBO_TIMEOUTS = {
    "put_jam_and_lemon_from_tray_onto_shelf": (1325, 53.00),
    "take_jam_and_lemon_from_shelf_into_tray": (1414, 56.56),
    "take_jam_into_tray_and_put_milk_on_shelf": (1312, 52.48),
    "take_milk_and_orange_from_shelf_into_tray": (1115, 44.60),
    "take_teapot_into_tray_and_put_orange_on_shelf": (1286, 51.44),
}

SWITCH_TIMEOUTS = {
    "put_lemon_on_shelf_then_take_jam_into_tray": (782, 1456, 58.24),
    "put_orange_on_shelf_then_take_back_into_tray": (750, 1373, 54.92),
    "take_jam_into_tray_then_put_back_on_shelf": (674, 1217, 48.68),
    "take_milk_and_lemon_from_shelf_into_tray": (492, 1232, 49.28),
    "take_teapot_into_tray_then_put_back_on_shelf": (536, 1281, 51.24),
}

PAIRED_TIMEOUTS = {
    (
        "hammer_one_nail_each_hand",
        "hammer_two_nails_each_hand",
    ): (1241, 49.64),
    (
        "open_bottle_and_drop_one_sugar_cube_in_bottle",
        "open_bottle_and_drop_two_sugar_cubes_in_bottle",
    ): (1289, 51.56),
    (
        "pack_two_mangoes_into_two_baskets",
        "pack_four_mangoes_into_baskets",
    ): (919, 36.76),
    (
        "thread_bagel_and_donut_onto_side_rods",
        "thread_bagel_and_donut_onto_middle_and_side_rods",
    ): (630, 25.20),
    (
        "water_one_plant_each_hand",
        "water_two_plants_each_hand",
    ): (508, 20.32),
}


def _base_and_ood(task_name: str):
    return (
        load_task(SUITE, task_name),
        load_task(SUITE, f"{task_name}__ood"),
    )


@pytest.mark.parametrize("task_name,expected", COMBO_TIMEOUTS.items())
def test_combo_tasks_use_sum_of_atomic_budgets(task_name, expected):
    expected_steps, expected_seconds = expected

    for task in _base_and_ood(task_name):
        assert task.max_steps == expected_steps
        assert task.max_seconds == pytest.approx(expected_seconds)
        assert task.max_seconds == pytest.approx(task.max_steps / 25)

    assert evaluation.max_steps_for_task(SUITE, task_name) == expected_steps


@pytest.mark.parametrize("task_name,expected", SWITCH_TIMEOUTS.items())
def test_switch_tasks_use_absolute_atomic_phase_budget(task_name, expected):
    switch_step, expected_steps, expected_seconds = expected

    for task in _base_and_ood(task_name):
        assert task.max_steps == expected_steps
        assert task.max_seconds == pytest.approx(expected_seconds)
        assert task.max_seconds == pytest.approx(task.max_steps / 25)
        assert task.language_schedule[0]["until_step"] == switch_step
        assert all("until_fraction" not in segment for segment in task.language_schedule)

    assert evaluation.max_steps_for_task(SUITE, task_name) == expected_steps


@pytest.mark.parametrize("task_pair,expected", PAIRED_TIMEOUTS.items())
def test_instruction_following_pairs_share_larger_p90_budget(task_pair, expected):
    expected_steps, expected_seconds = expected
    computed_pair_max = max(
        evaluation.compute_max_steps_p90(SUITE, task_name)
        for task_name in task_pair
    )
    assert computed_pair_max == expected_steps

    for task_name in task_pair:
        for task in _base_and_ood(task_name):
            assert task.max_steps == expected_steps
            assert task.max_seconds == pytest.approx(expected_seconds)
            assert task.max_seconds == pytest.approx(task.max_steps / 25)
        assert evaluation.max_steps_for_task(SUITE, task_name) == expected_steps


def test_full_launcher_does_not_set_debug_max_steps_cap():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/evaluation/local_eval.sh",
        "scripts/evaluation/run_vec_models_eval.sh",
    ):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "BIDEX_MAX_STEPS_CAP=" not in source
