"""Typed boards: Crossclimb and Wend can be written in instead of read from a screenshot."""

from pathlib import Path

import pytest

from bench import real, synthetic
from main import solve_text
from solvers.crossclimb.solver import board_from_text as crossclimb_from_text
from solvers.wend.solver import board_from_text as wend_from_text

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_wend_example_is_the_real_board() -> None:
    assert wend_from_text((EXAMPLES / "wend.txt").read_text()) == real.WEND


def test_crossclimb_example_is_the_sample_board() -> None:
    assert crossclimb_from_text((EXAMPLES / "crossclimb.txt").read_text()) == synthetic.CROSSCLIMB_SAMPLE


def test_wend_lengths_are_optional_and_case_is_ignored() -> None:
    board = wend_from_text("cat\ndog\npig\n")
    assert board.letters[0] == ("C", "A", "T") and board.word_lengths == ()


def test_typed_wend_solves_to_the_real_answer() -> None:
    outcome = solve_text("wend", (EXAMPLES / "wend.txt").read_text())
    assert sorted(p.word for p in outcome["solution"]) == sorted(real.WEND_ANSWER)


@pytest.mark.parametrize(
    ("parse", "text"),
    [
        (wend_from_text, ""),
        (wend_from_text, "AB\nC\n"),  # not square
        (wend_from_text, "A1\nBC\n"),  # not a letter
        (crossclimb_from_text, "four\nend clue\nclue one\nclue two\n"),  # no word length
        (crossclimb_from_text, "4\nend clue\nonly one middle clue\n"),
    ],
)
def test_bad_typed_boards_fail_fast(parse, text: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        parse(text)


def test_untypeable_game_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be typed in"):
        solve_text("zip", "anything")
