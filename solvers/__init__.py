"""Game registry: one entry per LinkedIn puzzle, mapping names to solver and parser factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.solver_base import BaseSolver
from core.vision_base import BaseVisionParser
from solvers.crossclimb.solver import CrossclimbSolver
from solvers.crossclimb.solver import board_from_text as crossclimb_from_text
from solvers.crossclimb.vision import CrossclimbVisionParser
from solvers.patches.solver import PatchesSolver
from solvers.patches.vision import PatchesVisionParser
from solvers.pinpoint.semantic_engine import PinpointSolver
from solvers.queens.solver import QueensSolver
from solvers.queens.vision import QueensVisionParser
from solvers.sudoku.solver import SudokuSolver
from solvers.sudoku.vision import SudokuVisionParser
from solvers.tango.solver import TangoSolver
from solvers.tango.vision import TangoVisionParser
from solvers.wend.solver import WendSolver
from solvers.wend.solver import board_from_text as wend_from_text
from solvers.wend.vision import WendVisionParser
from solvers.zip.solver import ZipSolver
from solvers.zip.vision import ZipVisionParser


@dataclass(frozen=True)
class Game:
    name: str
    title: str
    solver: Callable[[], BaseSolver[Any, Any]]
    parser: Callable[[], BaseVisionParser[Any, Any]] | None
    """``None`` for games played from a text stream rather than a screenshot."""
    text: Callable[[str], Any] | None = None
    """Builds the board from typed text, for games whose screenshot can be skipped."""


GAMES: dict[str, Game] = {
    game.name: game
    for game in (
        Game("queens", "Queens (Crowns) — DLX exact cover", QueensSolver, QueensVisionParser),
        Game("zip", "Zip (Pathfinder) — A* search", ZipSolver, ZipVisionParser),
        Game("tango", "Tango (Sun & Moon) — SAT", TangoSolver, TangoVisionParser),
        Game("pinpoint", "Pinpoint (Categories) — SBERT centroid", PinpointSolver, None),
        Game(
            "crossclimb",
            "Crossclimb (Word Ladder) — embeddings + Hamming graph",
            CrossclimbSolver,
            CrossclimbVisionParser,
            crossclimb_from_text,
        ),
        Game("wend", "Wend (Word Partitioning) — Trie DFS", WendSolver, WendVisionParser, wend_from_text),
        Game("mini-sudoku", "Mini Sudoku (6x6) — SAT", SudokuSolver, SudokuVisionParser),
        Game("patches", "Patches (Rectangles) — DLX exact cover", PatchesSolver, PatchesVisionParser),
    )
}


def get_game(name: str) -> Game:
    """Look up a game by name, failing with the list of valid names."""
    if name not in GAMES:
        raise KeyError(f"unknown game {name!r}; choose from {', '.join(GAMES)}")
    return GAMES[name]
