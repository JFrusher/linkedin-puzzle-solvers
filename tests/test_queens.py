"""Queens: DLX exact cover, explanations and K-Means region vision."""

from itertools import islice

import numpy as np
import pytest

from bench.benchmark_runner import same_partition
from bench.synthetic import generate_queens, regions_contiguous, render_queens
from core.dlx import DancingLinks
from core.solver_base import UnsolvableError
from solvers.queens import rules
from solvers.queens.solver import QueensBoard, QueensSolver, is_valid_solution
from solvers.queens.vision import QueensVisionParser


def test_dancing_links_finds_every_exact_cover() -> None:
    # Knuth's example matrix: the unique exact cover is rows {0, 3, 4}.
    links = DancingLinks(primary=7, secondary=0)
    for row_id, columns in enumerate([[2, 4, 5], [0, 3, 6], [1, 2, 5], [0, 3], [1, 6], [3, 4, 6]]):
        links.add_row(row_id, columns)
    assert [sorted(cover) for cover in links.search()] == [[0, 3, 4]]


@pytest.mark.parametrize("n", range(5, 13))
def test_solver_satisfies_every_rule(n: int) -> None:
    board, _ = generate_queens(n, seed=n)
    assert regions_contiguous(board)
    result = QueensSolver().run(board)
    assert is_valid_solution(board, result.solution)
    placements = [s for s in result.explanation.steps if s.rule in ("Single Choice", "Search")]
    assert sorted(s.cell for s in placements) == result.solution
    assert result.profile.latency_us > 0 and result.profile.nodes_expanded >= n


def test_unsolvable_board_raises() -> None:
    # Regions 0 and 1 are both confined to row 0, so two crowns would share a row.
    regions = ((0, 0, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (3, 3, 3, 3))
    with pytest.raises(UnsolvableError):
        QueensSolver().solve(QueensBoard(regions))


def test_board_rejects_bad_region_ids() -> None:
    with pytest.raises(ValueError):
        QueensBoard(((0, 0), (0, 5)))


@pytest.mark.parametrize("n,seed", [(6, 1), (8, 2), (10, 3), (12, 4)])
def test_vision_recovers_regions(n: int, seed: int) -> None:
    board, _ = generate_queens(n, seed)
    image = render_queens(board, seed)
    parsed = QueensVisionParser().parse(image)
    assert same_partition(parsed.regions, board.regions)


def test_overlay_matches_image_and_draws_crowns() -> None:
    board, _ = generate_queens(8, seed=5)
    image = render_queens(board, seed=5)
    parser = QueensVisionParser()
    layer = parser.overlay_layer(image, board, QueensSolver().solve(board))
    assert layer.shape == (*image.shape[:2], 4)
    assert np.count_nonzero(layer[..., 3]) > 0
    assert parser.generate_overlay(image, board, QueensSolver().solve(board)).shape == image.shape


@pytest.mark.parametrize("n", [6, 8, 10])
def test_generator_makes_one_answer_boards(n: int) -> None:
    board, truth = generate_queens(n, seed=n + 100)
    assert regions_contiguous(board)
    assert list(islice(QueensSolver().solutions(board), 2)) == [truth]


def test_domino_confined_to_a_row_clears_that_row() -> None:
    # Region 0 is the 2x1 domino (0,0)-(0,1); whichever cell holds its queen, (0,2)..(0,4) are out.
    regions = ((0, 0, 1, 1, 1), (2, 2, 1, 3, 3), (2, 2, 4, 3, 3), (2, 4, 4, 4, 3), (2, 4, 4, 4, 4))
    state = rules.QueensState.start(QueensBoard(regions))
    reasons = []
    while (deduction := rules.shared_attack(state)) is not None:
        reasons.append(deduction.reason)
        state.apply(deduction)
    assert not {(0, 2), (0, 3), (0, 4)} & state.candidates
    assert any(reason.startswith("every slot left in Region 0") for reason in reasons)


@pytest.mark.parametrize("n", [7, 9, 11])
def test_rules_never_contradict_the_answer(n: int) -> None:
    for seed in range(5):
        board, truth = generate_queens(n, seed)
        state = rules.QueensState.start(board)
        while (deduction := rules.next_deduction(state)) is not None and not state.solved():
            assert deduction.place is None or deduction.place in truth
            assert not deduction.eliminate & set(truth)
            state.apply(deduction)
