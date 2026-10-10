"""Pinpoint: moving centroid, WordNet hypernym candidates and the clue stream."""

import numpy as np
import pytest
from rich.console import Console

from bench.synthetic import PINPOINT_SAMPLES
from core.nlp import embed
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
