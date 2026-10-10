"""Pinpoint: guess the shared category of a clue stream with a moving embedding centroid.

For K clues the context vector is ``C = (1/K) * sum(Embed(clue_i))``. Candidate
categories are the WordNet hypernyms of every clue's noun senses. Each
candidate is scored by ``cosine(C, Embed(category)) * coverage``, where
coverage is the fraction of clues that have the candidate as a hypernym, and
scores are turned into 0-100% confidences with a softmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.nlp import embed, wordnet
from core.solver_base import BaseSolver

MAX_HYPERNYM_DEPTH = 6
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
        self._hypernyms: list[set[str]] = []
        self._label_cache: dict[str, np.ndarray] = {}
        self.candidate_count = 0
        """Hypernym categories scored by the latest :meth:`ranking` call."""

    def reset(self) -> None:
        self.clues.clear()
        self._clue_vectors.clear()
        self._hypernyms.clear()

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
        return self.ranking()

    @staticmethod
    def _hypernym_labels(clue: str) -> set[str]:
        wn = wordnet()
        synsets = wn.synsets(clue.lower().replace(" ", "_"), pos=wn.NOUN)
        if not synsets:
            synsets = [s for token in clue.lower().split() for s in wn.synsets(token, pos=wn.NOUN)]
        labels: set[str] = set()
        for synset in synsets:
            frontier = [synset]
            for _ in range(MAX_HYPERNYM_DEPTH):
                frontier = [h for s in frontier for h in s.hypernyms() + s.instance_hypernyms()]
                labels.update(h.lemma_names()[0].replace("_", " ") for h in frontier)
        return labels

    def ranking(self) -> list[CategoryGuess]:
        candidates = sorted(set().union(*self._hypernyms))
        self.candidate_count = len(candidates)
        if not candidates:
            return []
        uncached = [label for label in candidates if label not in self._label_cache]
        if uncached:
            self._label_cache.update(zip(uncached, embed(uncached)))
        vectors = np.stack([self._label_cache[label] for label in candidates])
        cosines = vectors @ self.centroid
        coverage = np.array([sum(label in h for h in self._hypernyms) for label in candidates]) / len(self.clues)
        scores = cosines * coverage
        top = np.argsort(-scores)[:TOP_N]
        weights = np.exp((scores[top] - scores[top].max()) / SOFTMAX_TEMPERATURE)
        confidences = 100 * weights / weights.sum()
        return [
            CategoryGuess(candidates[i], float(conf), float(coverage[i]), float(cosines[i]))
            for i, conf in zip(top, confidences)
        ]


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
