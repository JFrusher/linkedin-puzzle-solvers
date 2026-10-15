"""Real LinkedIn screenshots: each parser must read its board exactly, and each board must solve."""

from dataclasses import replace

import pytest

from bench import real
from bench.benchmark_runner import same_partition
from core.difficulty import rate
from main import solve_image
from solvers import get_game
from solvers.patches.solver import is_valid_solution as patches_valid
from solvers.queens.solver import is_valid_solution as queens_valid
from solvers.sudoku.solver import is_valid_solution as sudoku_valid
from solvers.tango.solver import is_valid_solution as tango_valid
from solvers.zip.solver import is_valid_solution as zip_valid


def test_queens() -> None:
    parsed = get_game("queens").parser().parse(real.image("queens"))
    assert same_partition(parsed.regions, real.QUEENS.regions)
    assert queens_valid(real.QUEENS, get_game("queens").solver().solve(parsed))


def test_tango() -> None:
    parsed = get_game("tango").parser().parse(real.image("tango"))
    assert dict(parsed.givens) == dict(real.TANGO.givens)
    assert set(parsed.constraints) == set(real.TANGO.constraints)
    assert tango_valid(real.TANGO, get_game("tango").solver().solve(parsed))


def test_zip() -> None:
    parsed = get_game("zip").parser().parse(real.image("zip"))
    assert parsed == real.ZIP
    assert zip_valid(real.ZIP, get_game("zip").solver().solve(parsed))


def test_mini_sudoku() -> None:
    parsed = get_game("mini-sudoku").parser().parse(real.image("mini-sudoku"))
    assert dict(parsed.givens) == dict(real.SUDOKU.givens)
    assert sudoku_valid(real.SUDOKU, get_game("mini-sudoku").solver().solve(parsed))


def test_patches() -> None:
    parsed = get_game("patches").parser().parse(real.image("patches"))
    assert parsed == real.PATCHES
    assert patches_valid(real.PATCHES, get_game("patches").solver().solve(parsed))


def test_wend() -> None:
    parser = get_game("wend").parser()
    parsed = replace(parser.parse(real.image("wend")), word_lengths=parser.slot_lengths(real.image("wend-slots")))
    assert parsed == real.WEND
    assert sorted(p.word for p in get_game("wend").solver().solve(parsed)) == sorted(real.WEND_ANSWER)


def test_cli_path_takes_wend_slots_from_their_own_screenshot() -> None:
    outcome = solve_image("wend", real.image("wend"), real.image("wend-slots"))
    assert sorted(p.word for p in outcome["solution"]) == sorted(real.WEND_ANSWER)
    with pytest.raises(ValueError, match="only applies to Wend"):
        solve_image("zip", real.image("zip"), real.image("wend-slots"))


@pytest.mark.parametrize("name", ["queens", "tango", "zip", "mini-sudoku", "patches", "wend"])
def test_overlay_draws_on_the_screenshot(name: str) -> None:
    game = get_game(name)
    picture = real.image(name)
    board = game.parser().parse(picture)
    overlay = game.parser().generate_overlay(picture, board, game.solver().solve(board))
    assert overlay.shape[:2] == picture.shape[:2]


@pytest.mark.parametrize("name", ["queens", "zip", "tango", "mini-sudoku", "patches", "wend"])
def test_game_is_detected_from_the_screenshot_alone(name: str) -> None:
    assert solve_image(None, real.image(name))["game"] == name


@pytest.mark.parametrize(
    "name, board, difficulty",
    [
        ("queens", real.QUEENS, "Medium: Confinement ×1, Shared Attack ×1"),
        ("tango", real.TANGO, "Medium: Sign Balance ×3"),
        ("mini-sudoku", real.SUDOKU, "Easy: Naked Single ×24"),
        ("patches", real.PATCHES, "Hard: Try and Fail ×2"),
        ("wend", real.WEND, "Easy: Only Word ×3, Nook ×2"),
    ],
)
def test_real_boards_get_a_difficulty(name: str, board: object, difficulty: str) -> None:
    assert rate(get_game(name).solver().run(board).explanation.steps) == difficulty


def test_wend_reads_small_slots_from_the_same_screenshot() -> None:
    parsed = get_game("wend").parser().parse(real.image("wend-2"))
    assert parsed == real.WEND_2
    assert sorted(p.word for p in get_game("wend").solver().solve(parsed)) == sorted(real.WEND_2_ANSWER)


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("queens", queens_valid),
        ("tango", tango_valid),
        ("zip", zip_valid),
        ("mini-sudoku", sudoku_valid),
        ("patches", patches_valid),
        ("wend", None),
    ],
)
def test_detection_names_the_game_of_each_second_screenshot(name: str, valid) -> None:  # type: ignore[no-untyped-def]
    """Patches once read as an all-but-empty Mini Sudoku, which any grid solves."""
    outcome = solve_image(None, real.image(f"{name}-2"))
    assert outcome["game"] == name
    if valid is not None:
        assert valid(outcome["board"], outcome["solution"])
    else:
        assert outcome["board"] == real.WEND_2
        assert sorted(p.word for p in outcome["solution"]) == sorted(real.WEND_2_ANSWER)
