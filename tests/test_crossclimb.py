"""Crossclimb: Hamming graph ladder search, SBERT clue answering and OCR vision."""

import pytest

from bench.benchmark_runner import word_recall
from bench.synthetic import CROSSCLIMB_ANSWER, CROSSCLIMB_SAMPLE, render_crossclimb
from solvers.crossclimb.solver import CrossclimbBoard, CrossclimbSolver, hamming, is_valid_ladder
from solvers.crossclimb.vision import CrossclimbVisionParser


def test_hamming() -> None:
    assert hamming("COLD", "CORD") == 1
    with pytest.raises(ValueError):
        hamming("COLD", "CORDS")


def test_ladder_search_prefers_higher_similarity() -> None:
    solver = CrossclimbSolver()
    candidates = [[("COLD", 0.9), ("BOLD", 0.2)], [("CORD", 0.8)], [("CARD", 0.7), ("WORD", 0.6)]]
    ladder = solver._best_ladder(candidates)
    assert [word for _, word, _ in ladder] == ["CARD", "CORD", "COLD"]


def test_board_validation() -> None:
    with pytest.raises(ValueError):
        CrossclimbBoard(("one clue",), 4)


def test_solver_answers_sample_ladder() -> None:
    result = CrossclimbSolver().run(CROSSCLIMB_SAMPLE)
    assert is_valid_ladder(CROSSCLIMB_SAMPLE, result.solution)
    assert {rung.word for rung in result.solution} == set(CROSSCLIMB_ANSWER)


def test_vision_reads_rows_and_clues() -> None:
    parsed = CrossclimbVisionParser().parse(render_crossclimb(CROSSCLIMB_SAMPLE))
    assert parsed.word_length == CROSSCLIMB_SAMPLE.word_length
    assert all(word_recall(p, t) >= 0.8 for p, t in zip(parsed.clues, CROSSCLIMB_SAMPLE.clues, strict=True))
