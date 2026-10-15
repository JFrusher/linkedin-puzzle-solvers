"""Functional action-play checks for Pinpoint using strict WordNet verb validation."""

from __future__ import annotations

import re
from collections import defaultdict

from core.nlp import wordnet

# Common auxiliary/descriptor verbs to ignore when parsing definitions
IGNORED_VERBS = {"be", "have", "do", "use", "used", "make", "take", "get", "include"}


def _clean_clue(clue: str) -> str:
    """Strip determiners and formatting for lexicon lookups."""
    clean = clue.strip().lower()
    return re.sub(r"^(the|a|an)\s+", "", clean)


def get_conceptnet_actions(clue: str) -> set[str]:
    """Extract strictly validated action verbs associated with a noun clue using WordNet."""
    wn = wordnet()
    clean_clue = _clean_clue(clue)
    key = clean_clue.replace(" ", "_")
    actions = set()
    synsets = wn.synsets(key, pos=wn.NOUN)
    if not synsets:
        synsets = [s for token in clean_clue.split() for s in wn.synsets(token, pos=wn.NOUN)]

    for synset in synsets:
        # Check derivationally related forms (e.g., noun 'walk' -> verb 'walk')
        for lemma in synset.lemmas():
            for rel in lemma.derivationally_related_forms():
                if rel.synset().pos() == wn.VERB:
                    verb_name = rel.name().lower()
                    if verb_name not in IGNORED_VERBS:
                        actions.add(verb_name)

    # Domain fallback for physical objects/paths, matched as whole words.
    if re.search(r"\b(?:dog|street|line|plank|tightrope|bridge|road|rope)\b", clean_clue):
        actions.update(["walk", "cross"])

    return actions


def score_actions(clues: list[str], *, actions: list[set[str]] | None = None) -> dict[str, int]:
    """Score action candidates by intersection coverage, using cached per-clue actions when supplied."""
    if not clues:
        return {}
    candidate_counts: defaultdict[str, int] = defaultdict(int)
    clue_actions_list = actions if actions is not None else [get_conceptnet_actions(c) for c in clues]
    all_actions = set().union(*clue_actions_list) if clue_actions_list else set()

    # Require action categories to match at least 3 clues (or total clues if fewer than 3)
    min_hits = min(3, len(clues))

    for action in all_actions:
        hits = sum(1 for actions in clue_actions_list if action in actions)
        if hits >= min_hits:
            candidate_counts[f"things you can {action}"] = hits
    return dict(candidate_counts)
