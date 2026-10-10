"""Crossclimb: answer trivia clues with embeddings, then order the answers into a word ladder.

1. Every WordNet lemma of the target length is described by its glosses; the
   clue and the glosses are embedded with Sentence-BERT and the top-K most
   similar words become that clue's candidates.
2. Candidates form a graph whose edges join words at Hamming distance 1.
3. A branch-and-bound DFS finds the ladder that uses every clue exactly once
   and maximises the summed clue/answer similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core.nlp import dictionary_words, embed, wordnet
from core.solver_base import BaseSolver, UnsolvableError

TOP_K = 25


@dataclass(frozen=True)
class CrossclimbBoard:
    clues: tuple[str, ...]
    word_length: int

    def __post_init__(self) -> None:
        if len(self.clues) < 2:
            raise ValueError("Crossclimb needs at least two clues")
        if self.word_length < 2:
            raise ValueError(f"word length must be at least 2, got {self.word_length}")


@dataclass(frozen=True)
class Rung:
    clue: str
    word: str
    similarity: float


def hamming(a: str, b: str) -> int:
    """Number of positions at which two equal-length words differ."""
    if len(a) != len(b):
        raise ValueError(f"Hamming distance needs equal lengths: {a!r} vs {b!r}")
    return sum(x != y for x, y in zip(a, b))


@lru_cache(maxsize=8)
def vocabulary(length: int) -> tuple[tuple[str, ...], np.ndarray]:
    """Words of ``length`` letters and the embeddings of their WordNet glosses."""
    wn = wordnet()
    words = tuple(sorted(word for word in dictionary_words(length) if len(word) == length))
    glosses = [f"{word.lower()}: " + "; ".join(s.definition() for s in wn.synsets(word.lower())[:3]) for word in words]
    return words, embed(glosses)


class CrossclimbSolver(BaseSolver[CrossclimbBoard, list[Rung]]):
    name = "crossclimb"

    def solve(self, board_state: CrossclimbBoard) -> list[Rung]:
        words, gloss_vectors = vocabulary(board_state.word_length)
        clue_vectors = embed(list(board_state.clues))
        candidates: list[list[tuple[str, float]]] = []
        for clue, vector in zip(board_state.clues, clue_vectors):
            scores = gloss_vectors @ vector
            top = np.argsort(-scores)[:TOP_K]
            candidates.append([(words[i], float(scores[i])) for i in top])
            best_word, best_score = candidates[-1][0]
            self.explainer.add("Clue", f"'{clue}' -> best guess {best_word} (cosine {best_score:.2f})")
        ladder = self._best_ladder(candidates)
        if ladder is None:
            raise UnsolvableError("no combination of clue candidates forms a one-letter-change ladder")
        rungs = [Rung(board_state.clues[i], word, score) for i, word, score in ladder]
        for upper, lower in zip(rungs, rungs[1:]):
            position = next(k for k, (x, y) in enumerate(zip(upper.word, lower.word)) if x != y)
            self.explainer.add("Ladder", f"{upper.word} -> {lower.word} changes letter {position + 1}")
        return rungs

    def _best_ladder(self, candidates: list[list[tuple[str, float]]]) -> list[tuple[int, str, float]] | None:
        clue_count = len(candidates)
        best_remaining = [max(score for _, score in options) for options in candidates]
        best: list[tuple[int, str, float]] | None = None
        best_total = -np.inf

        def dfs(path: list[tuple[int, str, float]], used: int, total: float) -> None:
            nonlocal best, best_total
            self.nodes_expanded += 1
            if len(path) == clue_count:
                if total > best_total:
                    best, best_total = list(path), total
                return
            bound = total + sum(best_remaining[i] for i in range(clue_count) if not used >> i & 1)
            if bound <= best_total:
                return
            last_word = path[-1][1]
            taken = {word for _, word, _ in path}
            for i in range(clue_count):
                if used >> i & 1:
                    continue
                for word, score in candidates[i]:
                    if word not in taken and hamming(word, last_word) == 1:
                        path.append((i, word, score))
                        dfs(path, used | 1 << i, total + score)
                        path.pop()

        for i, options in enumerate(candidates):
            for word, score in options:
                dfs([(i, word, score)], 1 << i, score)
        if best is None:
            return None
        # A ladder reads the same both ways; present it starting from the alphabetically first end.
        return best if best[0][1] <= best[-1][1] else best[::-1]


def is_valid_ladder(board: CrossclimbBoard, rungs: list[Rung]) -> bool:
    """Every clue used once, every word the right length, adjacent words one letter apart."""
    if sorted(r.clue for r in rungs) != sorted(board.clues):
        return False
    if any(len(r.word) != board.word_length for r in rungs):
        return False
    return all(hamming(a.word, b.word) == 1 for a, b in zip(rungs, rungs[1:]))
