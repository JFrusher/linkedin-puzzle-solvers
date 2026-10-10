"""Pinpoint: moving centroid, WordNet hypernym candidates and the clue stream."""

import numpy as np
import pytest
from rich.console import Console

from bench.synthetic import PINPOINT_SAMPLES
from core.nlp import embed
from solvers.pinpoint import semantic_engine, wordplay
from solvers.pinpoint.cli_stream import run_stream
from solvers.pinpoint.semantic_engine import PinpointSolver, SemanticEngine


def test_centroid_is_normalised_mean_of_clue_embeddings() -> None:
    engine = SemanticEngine()
    for clue in ("pawn", "rook"):
        engine.add_clue(clue)
    mean = embed(["pawn", "rook"]).mean(axis=0)
    assert np.allclose(engine.centroid, mean / np.linalg.norm(mean), atol=1e-5)


def test_empty_clue_rejected() -> None:
    with pytest.raises(ValueError):
        SemanticEngine().add_clue("   ")


@pytest.mark.parametrize("clues,expected", PINPOINT_SAMPLES)
def test_expected_category_in_top_three(clues: tuple[str, ...], expected: str) -> None:
    ranking = PinpointSolver().run(list(clues)).solution
    assert expected in [guess.label for guess in ranking[:3]]
    assert sum(guess.confidence for guess in ranking) == pytest.approx(100.0)


def test_stream_reads_until_blank_line() -> None:
    answers = iter(["pawn", "rook", "knight", ""])
    ranking = run_stream(SemanticEngine(), read_clue=lambda _: next(answers), console=Console(quiet=True))
    assert ranking and ranking[0].coverage == 1.0


BREAKER = ["Code", "Circuit", "Ice", "Deal", "Tie"]  # LinkedIn Pinpoint #891, 8 Oct 2026
WHALES = ["Blue", "Right", "Bowhead", "Humpback", "Orca"]  # LinkedIn Pinpoint #892, 9 Oct 2026
TREES = ["oak", "maple", "birch", "willow", "elm"]


def test_join_word_found_for_breaker() -> None:
    hits = sum('words before "breaker"' in wordplay.joins(clue) for clue in BREAKER)
    assert hits == 3  # circuit breaker, icebreaker, tiebreaker; WordNet lacks code/deal breaker


def test_subtype_compounds_are_not_word_play() -> None:
    assert 'words after "red"' not in wordplay.joins("oak")  # a red oak is just an oak


def test_class_word_found_through_compounds() -> None:
    assert any(s.name() == "blue_whale.n.01" for s in wordplay.compound_hypernyms("Blue", depth=3))


def test_many_senses() -> None:
    assert wordplay.senses(["Score", "Tip", "Location", "Purpose", "Dot"])['meanings of "point"'] >= 3


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic stand-in for SBERT so ranking logic runs without the model."""
    rows = [np.random.default_rng(abs(hash(t)) % 2**32).normal(size=8) for t in texts]
    return np.array([row / np.linalg.norm(row) for row in rows])


@pytest.mark.parametrize("clues,leader", [(BREAKER, 'words before "breaker"'), (TREES, None), (WHALES, None)])
def test_word_play_leads_only_when_meaning_explains_less(
    monkeypatch: pytest.MonkeyPatch, clues: list[str], leader: str | None
) -> None:
    monkeypatch.setattr(semantic_engine, "embed", _fake_embed)
    engine = SemanticEngine()
    for clue in clues:
        ranking = engine.add_clue(clue)
    top = ranking[0].label
    if leader is None:
        assert not top.startswith(("words ", "meanings "))
    else:
        assert top == leader
