"""Wend: partition every open tile into dictionary words with Trie-guided DFS.

1. Trie-guided DFS enumerates every word path in the grid (orthogonal steps,
   no tile reused); prefixes absent from the Trie are cut immediately.
2. An exact-tiling search covers each open tile exactly once with those
   placements, always branching on the tile with the fewest options.
3. Flood-fill island pruning rejects any partial tiling that leaves a
   connected component whose size no combination of remaining word lengths fills.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from core.nlp import dictionary_words
from core.solver_base import BaseSolver, UnsolvableError

Cell = tuple[int, int]
MIN_WORD_LENGTH = 3


@dataclass(frozen=True)
class WendBoard:
    """Letters per tile (``None`` marks a wall) and the target word lengths (empty = any)."""

    letters: tuple[tuple[str | None, ...], ...]
    word_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if any(len(row) != len(self.letters) for row in self.letters):
            raise ValueError("Wend board must be square")
        for row in self.letters:
            for letter in row:
                if letter is not None and not (len(letter) == 1 and letter.isalpha() and letter.isupper()):
                    raise ValueError(f"Wend tiles must be single upper-case letters or None, got {letter!r}")
        if self.word_lengths and sum(self.word_lengths) != len(self.open_cells):
            raise ValueError(
                f"word lengths sum to {sum(self.word_lengths)} but the board has {len(self.open_cells)} open tiles"
            )

    @property
    def size(self) -> int:
        return len(self.letters)

    @property
    def open_cells(self) -> list[Cell]:
        return [(r, c) for r, row in enumerate(self.letters) for c, letter in enumerate(row) if letter is not None]


@dataclass(frozen=True)
class WordPlacement:
    word: str
    cells: tuple[Cell, ...]


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_word = False


class Trie:
    """Prefix tree over upper-case words."""

    def __init__(self, words: Iterable[str]) -> None:
        self.root = TrieNode()
        for word in words:
            node = self.root
            for letter in word:
                node = node.children.setdefault(letter, TrieNode())
            node.is_word = True


class WendSolver(BaseSolver[WendBoard, list[WordPlacement]]):
    name = "wend"

    def __init__(self, words: Iterable[str] | None = None) -> None:
        super().__init__()
        self.trie = Trie(dictionary_words(MIN_WORD_LENGTH) if words is None else words)

    def solve(self, board_state: WendBoard) -> list[WordPlacement]:
        placements = self._enumerate(board_state)
        by_cell: dict[Cell, list[WordPlacement]] = {}
        for placement in placements:
            for cell in placement.cells:
                by_cell.setdefault(cell, []).append(placement)
        open_cells = set(board_state.open_cells)
        missing = open_cells - by_cell.keys()
        if missing:
            raise UnsolvableError(f"tiles {sorted(missing)} belong to no dictionary word")
        lengths = Counter(board_state.word_lengths)
        chosen = self._tile(board_state, open_cells, by_cell, lengths, [])
        if chosen is None:
            raise UnsolvableError("no partition of the board into dictionary words exists")
        for placement in chosen:
            self.explainer.add(
                "Word", f"{placement.word} covers {' -> '.join(map(str, placement.cells))}", placement.cells[0]
            )
        return chosen

    def _enumerate(self, board: WendBoard) -> list[WordPlacement]:
        allowed = set(board.word_lengths)
        found: dict[tuple[Cell, ...], WordPlacement] = {}

        def dfs(cell: Cell, node: TrieNode, path: list[Cell]) -> None:
            self.nodes_expanded += 1
            if node.is_word and len(path) >= MIN_WORD_LENGTH and (not allowed or len(path) in allowed):
                word = "".join(board.letters[r][c] or "" for r, c in path)
                found.setdefault(tuple(path), WordPlacement(word, tuple(path)))
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < board.size and 0 <= nc < board.size) or (nr, nc) in path:
                    continue
                letter = board.letters[nr][nc]
                child = node.children.get(letter) if letter is not None else None
                if child is not None:
                    path.append((nr, nc))
                    dfs((nr, nc), child, path)
                    path.pop()

        for r, c in board.open_cells:
            node = self.trie.root.children.get(board.letters[r][c] or "")
            if node is not None:
                dfs((r, c), node, [(r, c)])
        return list(found.values())

    def _tile(
        self,
        board: WendBoard,
        uncovered: set[Cell],
        by_cell: dict[Cell, list[WordPlacement]],
        lengths: Counter[int],
        chosen: list[WordPlacement],
    ) -> list[WordPlacement] | None:
        if not uncovered:
            return list(chosen)

        def usable(placement: WordPlacement) -> bool:
            return all(cell in uncovered for cell in placement.cells) and (
                not lengths or lengths[len(placement.cells)] > 0
            )

        options = {cell: [p for p in by_cell[cell] if usable(p)] for cell in uncovered}
        cell = min(options, key=lambda key: (len(options[key]), key))
        for placement in options[cell]:
            self.nodes_expanded += 1
            remaining = uncovered - set(placement.cells)
            if lengths:
                lengths[len(placement.cells)] -= 1
            if self._islands_fillable(remaining, lengths):
                chosen.append(placement)
                result = self._tile(board, remaining, by_cell, lengths, chosen)
                if result is not None:
                    return result
                chosen.pop()
            if lengths:
                lengths[len(placement.cells)] += 1
        return None

    @staticmethod
    def _islands_fillable(uncovered: set[Cell], lengths: Counter[int]) -> bool:
        """Flood-fill the uncovered tiles; every island must be fillable by remaining words."""
        if lengths:
            sums = 1  # bit k set <=> some sub-multiset of remaining lengths sums to k
            for length, amount in lengths.items():
                for _ in range(amount):
                    sums |= sums << length
        seen: set[Cell] = set()
        for start in uncovered:
            if start in seen:
                continue
            seen.add(start)
            stack, size = [start], 0
            while stack:
                r, c = stack.pop()
                size += 1
                for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nb in uncovered and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if size < MIN_WORD_LENGTH or (lengths and not (sums >> size) & 1):
                return False
        return True


def is_valid_solution(board: WendBoard, words: list[WordPlacement], dictionary: frozenset[str]) -> bool:
    """Check that ``words`` are dictionary paths that tile every open cell exactly once."""
    covered = [cell for placement in words for cell in placement.cells]
    if sorted(covered) != sorted(board.open_cells):
        return False
    if board.word_lengths and sorted(len(p.cells) for p in words) != sorted(board.word_lengths):
        return False
    for placement in words:
        if placement.word not in dictionary:
            return False
        if "".join(board.letters[r][c] or "" for r, c in placement.cells) != placement.word:
            return False
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1 for a, b in zip(placement.cells, placement.cells[1:])):
            return False
    return True
