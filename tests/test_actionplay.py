"""Pinpoint action candidates need a clue relation or a whole-word fallback."""

import pytest

from solvers.pinpoint import actionplay


def test_unrelated_gloss_verbs_are_not_actions() -> None:
    # The noun gloss contains "images" and "objects", both of which have verb senses.
    assert actionplay.get_conceptnet_actions("telescope") == set()


def test_derivational_actions_are_preserved_without_inflected_gloss_words() -> None:
    assert actionplay.get_conceptnet_actions("observation") == {"notice", "observe", "watch"}


@pytest.mark.parametrize("clue", ["deadline", "headline", "hotdog", "streetlight", "broadband"])
def test_fallback_does_not_match_substrings(clue: str) -> None:
    assert not {"walk", "cross"} & actionplay.get_conceptnet_actions(clue)


@pytest.mark.parametrize(
    "clue",
    [
        "dog",
        "street",
        "line",
        "plank",
        "tightrope",
        "bridge",
        "road",
        "rope",
        "The LINE!",
        "a street",
        "tightrope-walker",
    ],
)
def test_fallback_matches_whole_words(clue: str) -> None:
    assert {"walk", "cross"} <= actionplay.get_conceptnet_actions(clue)


def test_action_scores_use_validated_candidates() -> None:
    assert actionplay.score_actions(["deadline", "headline", "telescope"]) == {}
    assert actionplay.score_actions(["dog", "tightrope", "road"]) == {
        "things you can walk": 3,
        "things you can cross": 3,
    }
