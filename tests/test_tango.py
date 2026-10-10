"""Tango: PySAT CNF encoding, rule explanations and template-matching vision."""

import pytest

from bench.synthetic import generate_tango, render_tango
from core.solver_base import UnsolvableError
from solvers.tango.solver import Constraint, Relation, Symbol, TangoBoard, TangoSolver, is_valid_solution
from solvers.tango.vision import TangoVisionParser


@pytest.mark.parametrize("seed", range(5))
def test_solver_satisfies_every_rule(seed: int) -> None:
    board, _ = generate_tango(6, seed)
    result = TangoSolver().run(board)
    assert is_valid_solution(board, result.solution)
    assert len(result.explanation) == 36


def test_relations_are_enforced() -> None:
    board = TangoBoard(4, {(0, 0): Symbol.SUN}, (Constraint((0, 0), (0, 1), Relation.OPPOSITE),))
    grid = TangoSolver().solve(board)
    assert grid[0][1] is Symbol.MOON and is_valid_solution(board, grid)


def test_contradiction_raises() -> None:
    board = TangoBoard(4, {(0, 0): Symbol.SUN, (0, 1): Symbol.SUN}, (Constraint((0, 0), (0, 1), Relation.OPPOSITE),))
    with pytest.raises(UnsolvableError):
        TangoSolver().solve(board)


def test_board_validation() -> None:
    with pytest.raises(ValueError):
        TangoBoard(5)
    with pytest.raises(ValueError):
        TangoBoard(4, constraints=(Constraint((0, 0), (2, 0), Relation.EQUAL),))


@pytest.mark.parametrize("seed", range(6))
def test_vision_reads_symbols_and_signs(seed: int) -> None:
    board, _ = generate_tango(6, seed)
    parsed = TangoVisionParser().parse(render_tango(board, seed))
    assert dict(parsed.givens) == dict(board.givens)
    assert set(parsed.constraints) == set(board.constraints)
