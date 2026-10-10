"""Real LinkedIn screenshots: each parser must read its board exactly, and each board must solve."""

from dataclasses import replace

import pytest

from bench import real
from bench.benchmark_runner import same_partition
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
