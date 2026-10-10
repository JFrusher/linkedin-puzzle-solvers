"""Pinpoint: guess the shared category of a clue stream.

Meaning: for K clues the context vector is ``C = (1/K) * sum(Embed(clue_i))``. Candidate categories are
the WordNet hypernyms of every clue's noun senses, plus the parents of ``clue_X`` compounds that are a kind
of X (blue → blue whale → whale). Each is scored by ``cosine(C, Embed(category)) * coverage``, where
coverage is the fraction of clues that have the candidate as a hypernym.

Word play (:mod:`solvers.pinpoint.wordplay`): a join word ("words before *breaker*") or a word whose
meanings the clues spread across ("meanings of *point*"). Such a label ranks first only when it covers at
least three clues and more clues than the best specific meaning category; otherwise meaning explains the
clues as well (oak, birch and elm are trees even though red oak and red birch exist).

Scores become 0-100% confidences with a softmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.nlp import embed, wordnet
from core.solver_base import BaseSolver
from solvers.pinpoint import wordplay

MAX_HYPERNYM_DEPTH = 6
SPECIFIC_DEPTH = 7
"""Hypernyms at least this deep in WordNet (tree, whale, chessman) count as a specific meaning."""
MIN_WORDPLAY_CLUES = 3
SOFTMAX_TEMPERATURE = 0.05
TOP_N = 10


@dataclass(frozen=True)
class CategoryGuess:
    label: str
    confidence: float
    """Percentage in 0-100."""
    coverage: float
    cosine: float


class SemanticEngine:
    """Stateful engine: feed clues one at a time and read the evolving ranking."""

    def __init__(self) -> None:
        self.clues: list[str] = []
        self._clue_vectors: list[np.ndarray] = []
        self._hypernyms: list[dict[str, int]] = []
        """Per clue: hypernym label -> its WordNet depth."""
        self._joins: list[set[str]] = []
        self._label_cache: dict[str, np.ndarray] = {}
        self.candidate_count = 0
        """Hypernym categories scored by the latest :meth:`ranking` call."""

    def reset(self) -> None:
        self.clues.clear()
        self._clue_vectors.clear()
        self._hypernyms.clear()
        self._joins.clear()

    @property
    def centroid(self) -> np.ndarray:
        if not self._clue_vectors:
            raise ValueError("no clues have been added yet")
        mean = np.mean(self._clue_vectors, axis=0)
        return mean / np.linalg.norm(mean)

    def add_clue(self, clue: str) -> list[CategoryGuess]:
        """Add a clue, update the centroid and return the ranked categories."""
        clue = clue.strip()
        if not clue:
            raise ValueError("clue must not be empty")
        self.clues.append(clue)
        self._clue_vectors.append(embed([clue])[0])
        self._hypernyms.append(self._hypernym_labels(clue))
        self._joins.append(wordplay.joins(clue))
        return self.ranking()

    @staticmethod
    def _hypernym_labels(clue: str) -> dict[str, int]:
        wn = wordnet()
        synsets = wn.synsets(clue.lower().replace(" ", "_"), pos=wn.NOUN)
        if not synsets:
            synsets = [s for token in clue.lower().split() for s in wn.synsets(token, pos=wn.NOUN)]
        synsets += wordplay.compound_hypernyms(clue, depth=3)
        labels: dict[str, int] = {}
        for synset in synsets:
            for parent in synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms(), depth=MAX_HYPERNYM_DEPTH):
                label = parent.lemma_names()[0].replace("_", " ")
                labels[label] = max(labels.get(label, 0), parent.min_depth())
        return labels

    def ranking(self) -> list[CategoryGuess]:
        clue_count = len(self.clues)
        meaning = sorted(set().union(*self._hypernyms))
        meaning_hits = [sum(label in h for h in self._hypernyms) for label in meaning]
        specific = max(
            (hits for label, hits in zip(meaning, meaning_hits, strict=True) if self._depth(label) >= SPECIFIC_DEPTH),
            default=0,
        )
        play_hits: dict[str, int] = {}
        for labels in self._joins:
            for label in labels:
                play_hits[label] = play_hits.get(label, 0) + 1
        play_hits.update(wordplay.senses(self.clues))
        play = sorted(label for label, hits in play_hits.items() if hits >= MIN_WORDPLAY_CLUES and hits > specific)
        candidates = meaning + play
        self.candidate_count = len(candidates)
        if not candidates:
            return []
        uncached = [label for label in candidates if label not in self._label_cache]
        if uncached:
            self._label_cache.update(zip(uncached, embed(uncached), strict=True))
        cosines = np.stack([self._label_cache[label] for label in candidates]) @ self.centroid
        coverage = np.array(meaning_hits + [play_hits[label] for label in play]) / clue_count
        # Word play that beats every specific meaning outranks meaning outright.
        scores = np.concatenate([cosines[: len(meaning)] * coverage[: len(meaning)], 1 + coverage[len(meaning) :]])
        top = np.argsort(-scores)[:TOP_N]
        weights = np.exp((scores[top] - scores[top].max()) / SOFTMAX_TEMPERATURE)
        confidences = 100 * weights / weights.sum()
        return [
            CategoryGuess(candidates[i], float(conf), float(coverage[i]), float(cosines[i]))
            for i, conf in zip(top, confidences, strict=True)
        ]

    def _depth(self, label: str) -> int:
        return max(h[label] for h in self._hypernyms if label in h)


class PinpointSolver(BaseSolver[list[str], list[CategoryGuess]]):
    """Batch wrapper: feed all clues and return the final ranking."""

    name = "pinpoint"

    def __init__(self) -> None:
        super().__init__()
        self.engine = SemanticEngine()

    def solve(self, board_state: list[str]) -> list[CategoryGuess]:
        if not board_state:
            raise ValueError("Pinpoint needs at least one clue")
        self.engine.reset()
        ranking: list[CategoryGuess] = []
        for clue in board_state:
            ranking = self.engine.add_clue(clue)
            self.nodes_expanded += self.engine.candidate_count
            if ranking:
                top = ranking[0]
                self.explainer.add(
                    "Clue", f"'{clue}' -> leading category '{top.label}' at {top.confidence:.0f}% confidence"
                )
            else:
                self.explainer.add("Clue", f"'{clue}' has no WordNet noun sense; centroid updated only")
        return ranking
