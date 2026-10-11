"""The Rust Dancing Links twin must behave exactly like core/dlx.py: same covers, same order, same node count."""

import random

import pytest

import solvers.patches.solver as patches_solver
import solvers.queens.solver as queens_solver
from bench import real, synthetic
from core.dlx import DancingLinks

dlx_rs = pytest.importorskip("dlx_rs", reason="build it with: pip install ./rust/dlx")


def build(engine: type, primary: int, secondary: int, rows: list[list[int]]):  # type: ignore[no-untyped-def]
    links = engine(primary, secondary)
    for row_id, columns in enumerate(rows):
        links.add_row(row_id, columns)
    return links


def random_matrix(seed: int) -> tuple[int, int, list[list[int]]]:
    rng = random.Random(seed)
    primary, secondary = rng.randint(1, 8), rng.randint(0, 3)
    rows = [
        sorted(rng.sample(range(primary + secondary), rng.randint(1, min(3, primary + secondary))))
        for _ in range(rng.randint(1, 18))
    ]
    return primary, secondary, rows


@pytest.mark.parametrize("seed", range(300))
def test_every_cover_matches_in_order_with_the_same_node_count(seed: int) -> None:
    primary, secondary, rows = random_matrix(seed)
    python, rust = build(DancingLinks, primary, secondary, rows), build(dlx_rs.DancingLinks, primary, secondary, rows)
    assert list(rust.search()) == list(python.search())
    assert rust.nodes_expanded == python.nodes_expanded


def test_empty_matrix_has_one_empty_cover() -> None:
    assert list(dlx_rs.DancingLinks(0, 0).search()) == [[]]


def test_a_new_search_restarts_after_a_partial_one() -> None:
    primary, secondary, rows = 3, 0, [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]
    links = build(dlx_rs.DancingLinks, primary, secondary, rows)
    first = next(links.search())
    assert list(links.search())[0] == first and len(list(links.search())) == 4


def test_bad_column_is_rejected() -> None:
    with pytest.raises(IndexError, match="out of range"):
        dlx_rs.DancingLinks(2, 0).add_row(0, [2])


@pytest.mark.parametrize("n", [6, 8, 10])
def test_queens_solver_gives_the_same_answers_on_either_engine(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    board, _ = synthetic.generate_queens(n, n)
    expected = list(queens_solver.QueensSolver().solutions(board))
    monkeypatch.setattr(queens_solver, "DancingLinks", dlx_rs.DancingLinks)
    assert list(queens_solver.QueensSolver().solutions(board)) == expected


def test_patches_solver_gives_the_same_answer_on_the_real_board(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = list(patches_solver.PatchesSolver().solutions(real.PATCHES))
    monkeypatch.setattr(patches_solver, "DancingLinks", dlx_rs.DancingLinks)
    assert list(patches_solver.PatchesSolver().solutions(real.PATCHES)) == expected
