"""Real LinkedIn screenshots (``tests/fixtures/real/``) with hand-checked boards, for calibration and tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from solvers.patches.solver import Clue, PatchesBoard
from solvers.queens.solver import QueensBoard
from solvers.sudoku.solver import SudokuBoard
from solvers.tango.solver import Constraint, Relation, Symbol, TangoBoard
from solvers.wend.solver import WendBoard
from solvers.zip.solver import ZipBoard

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "real"


def image(name: str) -> np.ndarray:
    picture = cv2.imread(str(FIXTURES / f"{name}.png"))
    if picture is None:
        raise FileNotFoundError(FIXTURES / f"{name}.png")
    return picture


def _regions(rows: str) -> tuple[tuple[int, ...], ...]:
    """One letter per cell, one letter per region."""
    letters = sorted(set(rows.split()[0] + "".join(rows.split())))
    return tuple(tuple(letters.index(ch) for ch in row) for row in rows.split())


QUEENS = QueensBoard(
    _regions(
        """
        OOOPYYYYY
        OOPPPBYYY
        OOGPBBBYY
        OGGGSBYYY
        ORGSSSYYY
        RRRKSYYYT
        YRKKKYYTT
        YYYKYYTTT
        YYYYYTTTT
        """
    )
)
"""O orange, P purple, Y yellow, B blue, G green, S grey, R red, K pink, T tan."""

S, M = Symbol.SUN, Symbol.MOON
TANGO = TangoBoard(
    6,
    {(0, 0): M, (0, 1): M, (0, 2): S, (1, 0): M, (1, 2): M, (2, 0): S, (2, 1): M, (2, 2): M},
    (
        Constraint((1, 4), (1, 5), Relation.EQUAL),
        Constraint((2, 5), (3, 5), Relation.EQUAL),
        Constraint((3, 1), (4, 1), Relation.OPPOSITE),
        Constraint((4, 5), (5, 5), Relation.OPPOSITE),
        Constraint((5, 1), (5, 2), Relation.OPPOSITE),
        Constraint((5, 3), (5, 4), Relation.OPPOSITE),
    ),
)


def _walls(*pairs: tuple[tuple[int, int], tuple[int, int]]) -> frozenset[frozenset[tuple[int, int]]]:
    return frozenset(frozenset(pair) for pair in pairs)


ZIP = ZipBoard(
    7,
    ((1, 5), (6, 4), (3, 2), (3, 4), (5, 3), (5, 1), (0, 2), (1, 3)),
    _walls(
        ((1, 2), (1, 3)),
        ((1, 2), (2, 2)),
        ((1, 3), (2, 3)),
        ((2, 4), (2, 5)),
        ((3, 4), (3, 5)),
        ((2, 4), (3, 4)),
        ((3, 1), (3, 2)),
        ((4, 1), (4, 2)),
        ((3, 2), (4, 2)),
        ((4, 3), (5, 3)),
        ((4, 4), (5, 4)),
        ((5, 3), (5, 4)),
    ),
)

SUDOKU = SudokuBoard(
    {
        (0, 0): 1, (0, 1): 2,
        (1, 0): 3, (1, 1): 4, (1, 5): 1,
        (2, 5): 2,
        (3, 0): 4,
        (4, 0): 5, (4, 4): 4, (4, 5): 3,
        (5, 4): 1, (5, 5): 5,
    }
)  # fmt: skip

PATCHES = PatchesBoard(
    8,
    (
        Clue((1, 2), 6, "any"),
        Clue((1, 6), None, "tall"),
        Clue((2, 1), 6, "any"),
        Clue((2, 3), 9, "any"),
        Clue((2, 7), None, "tall"),
        Clue((3, 6), None, "wide"),
        Clue((4, 1), None, "square"),
        Clue((5, 0), None, "wide"),
        Clue((5, 4), 9, "any"),
        Clue((5, 6), 4, "any"),
        Clue((6, 1), None, "square"),
        Clue((6, 5), 12, "any"),
    ),
)
"""Numbers in '+' badges are areas of any shape; plain coloured icons are shapes of any area."""

WEND = WendBoard(
    letters=tuple(
        tuple(None if ch == "#" else ch for ch in row)
        for row in (
            "##U#####",
            "#GPRECT#",
            "#REIDSIR",
            "#ADROUG#",
            "#FTEETH#",
            "LEOVOLN#",
            "#DEDAOW#",
            "#####D##",
        )
    ),
    word_lengths=(6, 7, 8, 9, 10),
)
"""Word lengths come from the separate ``wend-slots`` screenshot."""
WEND_ANSWER = ("DIRECT", "UPGRADE", "LEFTOVER", "RIGHTEOUS", "DOWNLOADED")
