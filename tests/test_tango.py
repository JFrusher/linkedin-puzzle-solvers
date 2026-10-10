"""Tango: PySAT CNF encoding, rule explanations and template-matching vision."""

from itertools import islice

import pytest

from bench.synthetic import generate_tango, render_tango
from core.solver_base import UnsolvableError
from solvers.tango import rules
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


@pytest.mark.parametrize("seed", range(3))
def test_generator_makes_one_answer_boards(seed: int) -> None:
    board, truth = generate_tango(6, seed + 100)
    assert list(islice(TangoSolver().solutions(board), 2)) == [truth]


def test_opposite_pair_counts_towards_balance() -> None:
    # Row 0: two suns plus an unknown × pair at (0,2)-(0,3) already make three suns, so (0,4), (0,5) are moons.
    board = TangoBoard(
        6,
        {(0, 0): Symbol.SUN, (0, 1): Symbol.MOON, (0, 5): Symbol.SUN},
        (Constraint((0, 2), (0, 3), Relation.OPPOSITE),),
    )
    deduction = rules.sign_balance(board, dict(board.givens))
    assert deduction is not None and deduction[:3] == ((0, 4), Symbol.MOON, "Sign Balance")


@pytest.mark.parametrize("seed", range(10))
def test_rules_never_contradict_the_answer(seed: int) -> None:
    board, truth = generate_tango(6, seed + 300)
    known = dict(board.givens)
    while (deduction := rules.next_deduction(board, known)) is not None:
        cell, symbol, _, _ = deduction
        assert truth[cell[0]][cell[1]] is symbol
        known[cell] = symbol
    assert len(known) == 36
