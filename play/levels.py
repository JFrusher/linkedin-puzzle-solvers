"""Per-game generator settings for Easy / Medium / Hard, and the JSON puzzle the browser plays.

Settings were tuned by sampling ratings: at a single size some levels never appear (Queens 7x7 is never Hard,
Patches never Easy), so sizes change with level and the pool falls back to the closest rating it finds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bench import synthetic
from core.difficulty import LEVELS as RATINGS
from core.difficulty import rate
from solvers import get_game
from solvers.wend.solver import WendSolver

PLAY_GAMES = ("queens", "zip", "tango", "mini-sudoku", "patches", "wend")
LEVEL_NAMES = ("easy", "medium", "hard")
BANDS = {"easy": ("Trivial", "Easy"), "medium": ("Medium",), "hard": ("Hard", "Expert")}
"""Which ``core.difficulty`` ratings count as each level."""


@dataclass(frozen=True)
class LevelSpec:
    make: Callable[[int], Any]
    """seed -> board, from the ``bench.synthetic`` generators."""
    label: str
    """Shown for unrated games (Zip), whose level comes from size and clue count alone."""
    bands: tuple[str, ...] | None = None
    """Ratings accepted at this level when they differ from ``BANDS``: Wend has no Medium rule to rate by."""


LEVELS: dict[str, dict[str, LevelSpec]] = {
    "queens": {
        "easy": LevelSpec(lambda seed: synthetic.generate_queens(6, seed)[0], "6×6"),
        "medium": LevelSpec(lambda seed: synthetic.generate_queens(7, seed)[0], "7×7"),
        "hard": LevelSpec(lambda seed: synthetic.generate_queens(9, seed)[0], "9×9"),
    },
    "zip": {
        "easy": LevelSpec(lambda seed: synthetic.generate_zip(5, seed, 8, 0)[0], "5×5"),
        "medium": LevelSpec(lambda seed: synthetic.generate_zip(6, seed, 6, 2)[0], "6×6"),
        "hard": LevelSpec(lambda seed: synthetic.generate_zip(7, seed, 4, 4)[0], "7×7"),
    },
    "tango": {
        "easy": LevelSpec(lambda seed: synthetic.generate_tango(6, seed, 8, 8)[0], "6×6"),
        "medium": LevelSpec(lambda seed: synthetic.generate_tango(6, seed, 4, 4)[0], "6×6"),
        "hard": LevelSpec(lambda seed: synthetic.generate_tango(6, seed, 0, 2)[0], "6×6"),
    },
    "mini-sudoku": {level: LevelSpec(lambda seed: synthetic.generate_sudoku(seed)[0], "6×6") for level in LEVEL_NAMES},
    "patches": {
        "easy": LevelSpec(lambda seed: synthetic.generate_patches(4, seed)[0], "4×4"),
        "medium": LevelSpec(lambda seed: synthetic.generate_patches(6, seed)[0], "6×6"),
        "hard": LevelSpec(lambda seed: synthetic.generate_patches(7, seed)[0], "7×7"),
    },
    # LinkedIn-like: everyday words, a few long ones, straight wall bars. Wend's rules rate Easy (Only Word, Nook)
    # or Hard (Try and Fail) and nothing between, so Medium is the bigger board at any rating up to Medium.
    "wend": {
        "easy": LevelSpec(lambda seed: synthetic.generate_wend(5, seed, 0, (3, 7), True, 2), "5×5"),
        "medium": LevelSpec(
            lambda seed: synthetic.generate_wend(6, seed, 0, (4, 8), True, 2), "6×6", ("Trivial", "Easy", "Medium")
        ),
        "hard": LevelSpec(lambda seed: synthetic.generate_wend(7, seed, 0, (5, 9), True, 3), "7×7"),
    },
}


def level_of(rating: str) -> str:
    """The play level a ``core.difficulty.rate`` string falls in."""
    band = rating.split(":")[0]
    return next(level for level, bands in BANDS.items() if band in bands)


def distance(rating: str, level: str, bands: tuple[str, ...] | None = None) -> int:
    """How many rating bands ``rating`` sits from ``level`` (or from ``bands``, when given); 0 inside it."""
    index = RATINGS.index(rating.split(":")[0])
    return min(abs(index - RATINGS.index(band)) for band in bands or BANDS[level])


def _cells(cells: Any) -> list[list[int]]:
    return [[r, c] for r, c in cells]


def _serialise(game: str, board: Any, solution: Any) -> tuple[dict[str, Any], Any]:
    """Plain JSON shapes for the browser: grids as nested lists, cells as ``[r, c]``."""
    if game == "queens":
        return {"size": len(board.regions), "regions": [list(row) for row in board.regions]}, _cells(solution)
    if game == "zip":
        walls = [_cells(sorted(wall)) for wall in board.walls]
        return {"size": board.size, "checkpoints": _cells(board.checkpoints), "walls": walls}, _cells(solution)
    if game == "tango":
        symbols: list[list[str | None]] = [[None] * board.size for _ in range(board.size)]
        for (r, c), symbol in board.givens.items():
            symbols[r][c] = symbol.value
        constraints = [{"a": list(k.a), "b": list(k.b), "relation": k.relation.value} for k in board.constraints]
        grid = [[symbol.value for symbol in row] for row in solution]
        return {"size": board.size, "givens": symbols, "constraints": constraints}, grid
    if game == "mini-sudoku":
        givens = [[0] * 6 for _ in range(6)]
        for (r, c), digit in board.givens.items():
            givens[r][c] = digit
        return {"size": 6, "givens": givens}, [list(row) for row in solution]
    if game == "patches":
        clues = [{"cell": list(clue.cell), "area": clue.area, "shape": clue.shape} for clue in board.clues]
        patches = [
            {"clue": board.clues.index(p.clue), "top": p.top, "left": p.left, "height": p.height, "width": p.width}
            for p in solution
        ]
        return {"size": board.size, "clues": clues}, patches
    if game == "wend":
        letters = [list(row) for row in board.letters]
        words = [{"word": p.word, "cells": _cells(p.cells)} for p in solution]
        return {"size": board.size, "letters": letters, "word_lengths": list(board.word_lengths)}, words
    raise ValueError(f"{game} is not playable")


def build_puzzle(game: str, level: str, board: Any) -> dict[str, Any]:
    """Solve and explain ``board``, then package everything the browser needs to play it."""
    solver = synthetic._wend_solver() if game == "wend" else get_game(game).solver()  # shares Wend's Trie
    result = solver.run(board, track_memory=False)
    board_json, solution_json = _serialise(game, board, result.solution)
    steps = [
        {"index": s.index, "rule": s.rule, "message": s.message, "cell": list(s.cell) if s.cell else None}
        for s in result.explanation.steps
    ]
    puzzle = {
        "game": game,
        "level": level,
        "rating": rate(result.explanation.steps) if get_game(game).rated else LEVELS[game][level].label,
        "board": board_json,
        "solution": solution_json,
        "steps": steps,
    }
    if game == "wend":
        # Every letter path on this board that spells a word of a slot length: the browser's dictionary.
        assert isinstance(solver, WendSolver)
        puzzle["placements"] = [_cells(p.cells) for p in solver._enumerate(board)]
    return puzzle
