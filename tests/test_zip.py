"""Zip: A* Hamiltonian path search, wall/checkpoint vision."""

from itertools import islice

import pytest

from bench.synthetic import generate_zip, render_zip
from core.solver_base import UnsolvableError
from solvers.zip.solver import ZipBoard, ZipSolver, is_valid_solution
from solvers.zip.vision import ZipVisionParser


@pytest.mark.parametrize("n,seed", [(4, 0), (5, 1), (6, 2), (7, 3)])
def test_solver_finds_valid_path(n: int, seed: int) -> None:
    board, _ = generate_zip(n, seed)
    result = ZipSolver().run(board)
    assert is_valid_solution(board, result.solution)
    assert result.explanation.steps[0].rule == "Start"


def test_walls_block_moves() -> None:
    board = ZipBoard(2, ((0, 0), (1, 0)), frozenset({frozenset({(0, 0), (1, 0)})}))
    assert ZipSolver().solve(board) == [(0, 0), (0, 1), (1, 1), (1, 0)]


def test_unsolvable_board_raises() -> None:
    # A 9-cell path starts and ends on the same checkerboard colour; (0, 1) is the other colour.
    board = ZipBoard(3, ((0, 0), (0, 1)))
    with pytest.raises(UnsolvableError):
        ZipSolver().solve(board)


def test_board_validation() -> None:
    with pytest.raises(ValueError):
        ZipBoard(3, ((0, 0),))
    with pytest.raises(ValueError):
        ZipBoard(3, ((0, 0), (3, 3)))


@pytest.mark.parametrize("n,seed", [(5, 0), (6, 1), (7, 2)])
def test_vision_reads_checkpoints_and_walls(n: int, seed: int) -> None:
    board, _ = generate_zip(n, seed)
    assert ZipVisionParser().parse(render_zip(board, seed)) == board


@pytest.mark.parametrize("n", [5, 6])
def test_generator_makes_one_answer_boards(n: int) -> None:
    board, truth = generate_zip(n, seed=n + 100)
    assert list(islice(ZipSolver().solutions(board), 2)) == [truth]
