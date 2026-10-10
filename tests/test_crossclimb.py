"""Crossclimb: Hamming graph ladder search, SBERT clue answering and OCR vision."""

import numpy as np
import pytest

import solvers.crossclimb.solver as solver_module
from bench.benchmark_runner import word_recall
from bench.synthetic import CROSSCLIMB_ANSWER, CROSSCLIMB_SAMPLE, render_crossclimb
from solvers.crossclimb.solver import (
    CrossclimbBoard,
    CrossclimbSolver,
    end_candidates,
    hamming,
    is_valid_ladder,
    pair_text,
)
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
        CrossclimbBoard(("one clue",), "end clue", 4)
    with pytest.raises(ValueError):
        CrossclimbBoard(("one", "two"), " ", 4)


def test_end_candidates_are_one_letter_from_each_end() -> None:
    tops, bottoms = end_candidates(["REAM", "ROAM", "WORM"], ("TEAM", "ROAM", "WORK", "BEAM", "WARM", "COLD"))
    assert tops == ["TEAM", "BEAM"] and bottoms == ["WORK", "WARM"]


def test_pair_text_uses_wordnet_compounds() -> None:
    assert pair_text("TEAM", "WORK").startswith("teamwork: ")
    assert pair_text("WORK", "TEAM") == "work team"


def test_ends_pick_the_pair_matching_the_shared_clue(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stand-in encoder that only knows whether a text mentions "cooperative": enough to test the selection.
    def fake_embed(texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] if "cooperative" in t else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(solver_module, "embed", fake_embed)
    words = ("TEAM", "BEAM", "REAM", "ROAM", "WORM", "WORK", "WARM")
    top, bottom, _ = CrossclimbSolver()._best_ends(["REAM", "ROAM", "WORM"], "cooperative work", words)
    assert (top, bottom) == ("TEAM", "WORK")


def test_solver_answers_sample_ladder() -> None:
    result = CrossclimbSolver().run(CROSSCLIMB_SAMPLE)
    assert is_valid_ladder(CROSSCLIMB_SAMPLE, result.solution)
    assert tuple(rung.word for rung in result.solution) in (CROSSCLIMB_ANSWER, CROSSCLIMB_ANSWER[::-1])


def test_vision_reads_rows_and_clues() -> None:
    parsed = CrossclimbVisionParser().parse(render_crossclimb(CROSSCLIMB_SAMPLE))
    assert parsed.word_length == CROSSCLIMB_SAMPLE.word_length
    assert word_recall(parsed.end_clue, CROSSCLIMB_SAMPLE.end_clue) >= 0.8
    assert all(word_recall(p, t) >= 0.8 for p, t in zip(parsed.clues, CROSSCLIMB_SAMPLE.clues, strict=True))
