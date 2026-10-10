"""Crossclimb: answer trivia clues with embeddings, order them into a word ladder, then add the ends.

The board is a 7-word ladder: five middle words with their own clues, plus a top and a bottom word that
unlock last and share one clue (they form a phrase or pairing, e.g. TEAM + WORK).

1. Every WordNet lemma of the target length is described by its glosses; the clue and the glosses are
   embedded with Sentence-BERT and the top-K most similar words become that clue's candidates.
2. Candidates form a graph whose edges join words at Hamming distance 1.
3. A branch-and-bound DFS finds the middle ladder that uses every clue exactly once and maximises the
   summed clue/answer similarity.
4. The top word is one letter from one end of that ladder and the bottom word one letter from the other.
   Each pair is described in both reading orders ("team work", or a WordNet compound such as "teamwork"
   with its gloss) and the pair whose description best matches the shared clue wins.
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
    """Clues of the middle words, in any order."""
    end_clue: str
    """The one clue shared by the top and bottom words."""
    word_length: int

    def __post_init__(self) -> None:
        if len(self.clues) < 2:
            raise ValueError("Crossclimb needs at least two middle clues")
        if not self.end_clue.strip():
            raise ValueError("Crossclimb needs the shared clue for the top and bottom words")
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
        middle = [Rung(board_state.clues[i], word, score) for i, word, score in ladder]
        top, bottom, score = self._best_ends([rung.word for rung in middle], board_state.end_clue, words)
        self.explainer.add(
            "Ends",
            f"'{board_state.end_clue}' -> {top} + {bottom} (cosine {score:.2f}); "
            f"{top} is one letter from {middle[0].word} and {bottom} one letter from {middle[-1].word}",
        )
        rungs = [Rung(board_state.end_clue, top, score), *middle, Rung(board_state.end_clue, bottom, score)]
        for upper, lower in zip(rungs, rungs[1:]):
            position = next(k for k, (x, y) in enumerate(zip(upper.word, lower.word)) if x != y)
            self.explainer.add("Ladder", f"{upper.word} -> {lower.word} changes letter {position + 1}")
        return rungs

    def _best_ends(self, ladder: list[str], clue: str, words: tuple[str, ...]) -> tuple[str, str, float]:
        """The top/bottom pair, one letter from each end of ``ladder``, that best matches ``clue``."""
        tops, bottoms = end_candidates(ladder, words)
        pairs = [(top, bottom) for top in tops for bottom in bottoms if top != bottom]
        if not pairs:
            raise UnsolvableError(f"no words sit one letter from both ends of {'-'.join(ladder)}")
        texts = sorted({pair_text(a, b) for top, bottom in pairs for a, b in ((top, bottom), (bottom, top))})
        similarity = dict(zip(texts, (embed(texts) @ embed([clue])[0]).tolist(), strict=True))
        self.nodes_expanded += len(pairs)
        top, bottom = max(
            pairs, key=lambda p: max(similarity[pair_text(p[0], p[1])], similarity[pair_text(p[1], p[0])])
        )
        return top, bottom, max(similarity[pair_text(top, bottom)], similarity[pair_text(bottom, top)])

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


def end_candidates(ladder: list[str], words: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Words one letter from the first and from the last word of ``ladder``, excluding ladder words."""
    used = set(ladder)
    tops = [w for w in words if w not in used and hamming(w, ladder[0]) == 1]
    bottoms = [w for w in words if w not in used and hamming(w, ladder[-1]) == 1]
    return tops, bottoms


def pair_text(first: str, second: str) -> str:
    """Describe a word pair read in this order: a WordNet compound with its gloss, else the phrase."""
    wn = wordnet()
    for name in (f"{first}{second}", f"{first}_{second}", f"{first}-{second}"):
        synsets = wn.synsets(name.lower())
        if synsets:
            return f"{name.lower().replace('_', ' ')}: {synsets[0].definition()}"
    return f"{first} {second}".lower()


def is_valid_ladder(board: CrossclimbBoard, rungs: list[Rung]) -> bool:
    """Ends carry the shared clue, every middle clue used once, right lengths, neighbours one letter apart."""
    if len(rungs) != len(board.clues) + 2 or rungs[0].clue != board.end_clue or rungs[-1].clue != board.end_clue:
        return False
    if sorted(r.clue for r in rungs[1:-1]) != sorted(board.clues):
        return False
    if any(len(r.word) != board.word_length for r in rungs):
        return False
    return all(hamming(a.word, b.word) == 1 for a, b in zip(rungs, rungs[1:]))


def board_from_text(text: str) -> CrossclimbBoard:
    """A typed board: the word length, then the clue shared by the top and bottom words, then one middle clue
    per line, in any order."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4 or not lines[0].isdigit():
        raise ValueError("type the word length, then the shared top/bottom clue, then the middle clues, one per line")
    return CrossclimbBoard(clues=tuple(lines[2:]), end_clue=lines[1], word_length=int(lines[0]))
