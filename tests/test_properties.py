"""Property-based tests: Hypothesis draws the inputs, an independent oracle or a symmetry checks the answer."""

from itertools import combinations, permutations

from hypothesis import given, settings
from hypothesis import strategies as st

from bench.synthetic import generate_patches, generate_sudoku
from core.dlx import DancingLinks
from solvers.patches.solver import Clue, PatchesBoard, PatchesSolver
from solvers.patches.solver import is_valid_solution as patches_valid
from solvers.queens.solver import QueensBoard, QueensSolver
from solvers.queens.solver import is_valid_solution as queens_valid
from solvers.sudoku.solver import SudokuBoard, SudokuSolver

SLOW = settings(max_examples=40, deadline=None)


@st.composite
def exact_cover(draw: st.DrawFn) -> tuple[int, int, list[list[int]]]:
    primary = draw(st.integers(0, 5))
    secondary = draw(st.integers(0, 3))
    columns = st.lists(st.integers(0, primary + secondary - 1), min_size=1, unique=True)
    rows = draw(st.lists(columns, max_size=8)) if primary + secondary else []
    return primary, secondary, rows


@given(exact_cover())
def test_dlx_finds_exactly_the_brute_force_covers(problem: tuple[int, int, list[list[int]]]) -> None:
    primary, secondary, rows = problem
    links = DancingLinks(primary, secondary)
    for row_id, columns in enumerate(rows):
        links.add_row(row_id, columns)
    found = sorted(sorted(cover) for cover in links.search())

    def is_cover(chosen: tuple[int, ...]) -> bool:
        hits = [column for row_id in chosen for column in rows[row_id]]
        # Algorithm X stops once every primary column is covered, so it never picks a secondary-only row.
        picks_primary = all(min(rows[row_id]) < primary for row_id in chosen)
        return picks_primary and len(hits) == len(set(hits)) and set(range(primary)) <= set(hits)

    expected = [list(chosen) for k in range(len(rows) + 1) for chosen in combinations(range(len(rows)), k)]
    assert found == sorted(chosen for chosen in expected if is_cover(tuple(chosen)))


@st.composite
def queens_boards(draw: st.DrawFn) -> QueensBoard:
    """Any colouring of an N x N grid that uses every region id; regions need not be connected."""
    n = draw(st.integers(4, 6))
    cells = draw(st.lists(st.integers(0, n - 1), min_size=n * n, max_size=n * n))
    cells[:n] = draw(st.permutations(range(n)))  # every id appears at least once
    order = draw(st.permutations(range(n * n)))
    regions = [0] * (n * n)
    for index, region in zip(order, cells):
        regions[index] = region
    return QueensBoard(tuple(tuple(regions[r * n : (r + 1) * n]) for r in range(n)))


@SLOW
@given(queens_boards())
def test_queens_finds_every_placement_and_no_other(board: QueensBoard) -> None:
    n = board.size
    expected = sorted(
        sorted(enumerate(cols)) for cols in permutations(range(n)) if queens_valid(board, list(enumerate(cols)))
    )
    assert sorted(QueensSolver().solutions(board)) == expected


@SLOW
@given(queens_boards())
def test_queens_answers_turn_with_the_board(board: QueensBoard) -> None:
    transposed = QueensBoard(tuple(zip(*board.regions)))
    turned = sorted(sorted((c, r) for r, c in queens) for queens in QueensSolver().solutions(board))
    assert sorted(QueensSolver().solutions(transposed)) == turned


@SLOW
@given(st.integers(0, 10_000), st.permutations(range(1, 7)))
def test_sudoku_answer_follows_a_digit_relabelling(seed: int, relabel: list[int]) -> None:
    board, truth = generate_sudoku(seed)
    swap = dict(zip(range(1, 7), relabel))
    relabelled = SudokuBoard({cell: swap[digit] for cell, digit in board.givens.items()})
    assert list(SudokuSolver().solutions(relabelled)) == [[[swap[d] for d in row] for row in truth]]


@SLOW
@given(st.integers(5, 7), st.integers(0, 10_000))
def test_patches_answer_turns_with_the_board(n: int, seed: int) -> None:
    board, truth = generate_patches(n, seed)
    turn = {"tall": "wide", "wide": "tall", "square": "square", "any": "any"}
    transposed = PatchesBoard(n, tuple(Clue(clue.cell[::-1], clue.area, turn[clue.shape]) for clue in board.clues))
    (answer,) = PatchesSolver().solutions(transposed)
    assert patches_valid(transposed, answer)
    assert {frozenset((c, r) for r, c in p.cells) for p in answer} == {p.cells for p in truth}
