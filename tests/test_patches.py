"""Patches: rectangle candidates, DLX exact cover, Only Fit / Only Cover explanations and icon vision."""

from itertools import islice

import pytest

from bench.synthetic import generate_patches, render_patches
from core.solver_base import UnsolvableError
from solvers.patches.solver import Clue, PatchesBoard, PatchesSolver, candidates, is_valid_solution
from solvers.patches.vision import PatchesVisionParser


@pytest.mark.parametrize("n", (5, 6, 7))
def test_solver_tiles_the_grid(n: int) -> None:
    board, truth = generate_patches(n, n)
    result = PatchesSolver().run(board)
    assert is_valid_solution(board, result.solution)
    assert {p.cells for p in result.solution} == {p.cells for p in truth}


@pytest.mark.parametrize("seed", range(3))
def test_generator_makes_one_answer_boards(seed: int) -> None:
    board, truth = generate_patches(6, seed + 100)
    assert list(islice(PatchesSolver().solutions(board), 2)) == [truth]


def test_candidates_respect_shape_and_other_clues() -> None:
    board = PatchesBoard(2, (Clue((0, 0), 2, "wide"), Clue((1, 1), 2, "any")))
    assert [(p.top, p.left, p.height, p.width) for p in candidates(board, board.clues[0])] == [(0, 0, 1, 2)]


@pytest.mark.parametrize("seed", range(3))
def test_explanation_places_every_patch_of_the_answer(seed: int) -> None:
    board, truth = generate_patches(6, seed)
    steps = PatchesSolver().run(board).explanation.steps
    assert len(steps) == len(board.clues)
    assert {s.cell for s in steps} == {clue.cell for clue in board.clues}
    assert {s.message.split(" because")[0] for s in steps} == {
        f"clue {p.clue.area} at {p.clue.cell} takes the {p.height}x{p.width} patch at ({p.top}, {p.left})"
        for p in truth
    }


def test_impossible_board_raises() -> None:
    with pytest.raises(UnsolvableError):
        PatchesSolver().solve(PatchesBoard(2, (Clue((0, 0), 2, "tall"), Clue((0, 1), 2, "wide"))))


def test_board_validation() -> None:
    with pytest.raises(ValueError):
        PatchesBoard(2, (Clue((0, 0), 3, "any"),))
    with pytest.raises(ValueError):
        Clue((0, 0), 2, "round")


@pytest.mark.parametrize(("n", "seed"), [(5, 2), (5, 3), (6, 6), (7, 1)])
def test_vision_reads_areas_and_shapes(n: int, seed: int) -> None:
    board, _ = generate_patches(n, seed)
    assert PatchesVisionParser().parse(render_patches(board, seed)) == board
