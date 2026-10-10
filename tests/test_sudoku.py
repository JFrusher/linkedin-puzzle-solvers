"""Mini Sudoku: SAT encoding, single-based explanations and per-cell OCR vision."""

from itertools import islice

import pytest

from bench.synthetic import generate_sudoku, render_sudoku
from core.solver_base import UnsolvableError
from solvers.sudoku.solver import SudokuBoard, SudokuSolver, is_valid_solution
from solvers.sudoku.vision import SudokuVisionParser


@pytest.mark.parametrize("seed", range(5))
def test_solver_satisfies_every_rule(seed: int) -> None:
    board, truth = generate_sudoku(seed)
    result = SudokuSolver().run(board)
    assert result.solution == truth and is_valid_solution(board, result.solution)


@pytest.mark.parametrize("seed", range(3))
def test_generator_makes_one_answer_boards(seed: int) -> None:
    board, truth = generate_sudoku(seed + 100)
    assert list(islice(SudokuSolver().solutions(board), 2)) == [truth]


@pytest.mark.parametrize("seed", range(3))
def test_explanation_fills_every_open_cell_with_the_answer(seed: int) -> None:
    board, truth = generate_sudoku(seed)
    steps = SudokuSolver().run(board).explanation.steps
    assert len(steps) == 36 - len(board.givens)
    assert all(f"must be {truth[s.cell[0]][s.cell[1]]} " in s.message for s in steps)


def test_contradiction_raises() -> None:
    with pytest.raises(UnsolvableError):
        SudokuSolver().solve(SudokuBoard({(0, 0): 1, (0, 1): 1}))


def test_board_validation() -> None:
    with pytest.raises(ValueError):
        SudokuBoard({(0, 0): 7})
    with pytest.raises(ValueError):
        SudokuBoard({(6, 0): 1})


@pytest.mark.parametrize("seed", range(4))
def test_vision_reads_givens(seed: int) -> None:
    board, _ = generate_sudoku(seed)
    assert dict(SudokuVisionParser().parse(render_sudoku(board, seed)).givens) == dict(board.givens)
