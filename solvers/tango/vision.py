"""Tango vision: symbols told apart by colour, edge signs by whether ink fills the centre of the sign."""

from __future__ import annotations

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, inner_ink
from solvers.tango.solver import Cell, Constraint, Grid, Relation, Symbol, TangoBoard

CELL_MARGIN = 0.15
EMPTY_STD = 12.0
"""Cells whose grey-level standard deviation is below this are blank."""
SIGN_SIZE = 0.24
"""Rendered edge-sign size as a fraction of a cell."""
PROBE_SIZE = 0.4
"""Edge probe size as a fraction of a cell; larger than a sign so a slightly misplaced sign stays inside."""
SIGN_MIN_INK = 0.04
"""Fraction of dark pixels the probe's centre needs before it can hold a sign."""
SIGN_DARK = 160
"""Grey level below which a pixel counts as sign ink; LinkedIn's brown signs are no darker than 117."""
MAX_OTHER_HUES = 0.1
"""Share of coloured pixels allowed outside sun orange and moon blue; Tango boards measure 0, Patches 0.41+."""
SATURATED = 80
"""HSV saturation above which a pixel belongs to a coloured symbol."""


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)


def _symbol(cell: np.ndarray) -> Symbol:
    """Suns are orange and moons blue, in LinkedIn's icons and in ours, so compare blue with red."""
    coloured = cv2.cvtColor(cell[..., :3], cv2.COLOR_BGR2HSV)[..., 1] > SATURATED
    if not coloured.any():
        raise ValueError("a Tango cell holds marks but no coloured sun or moon")
    blue, _, red = (float(cell[..., i][coloured].mean()) for i in range(3))
    return Symbol.MOON if blue > red else Symbol.SUN


def _relation(ink: np.ndarray) -> Relation:
    """An 'x' has ink where its strokes cross, at the centre of its box; an '=' has the gap between its bars."""
    ys, xs = np.nonzero(ink)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    return Relation.OPPOSITE if ink[cy - 1 : cy + 2, cx - 1 : cx + 2].mean() > 0.5 else Relation.EQUAL


class TangoVisionParser(BaseVisionParser[TangoBoard, Grid]):
    def parse(self, image: np.ndarray) -> TangoBoard:
        grid = self.rectify_grid(image)
        hsv = cv2.cvtColor(grid.image, cv2.COLOR_BGR2HSV)
        hues = hsv[..., 0][hsv[..., 1] > 60]
        other = float(((hues > 30) & (hues < 95) | (hues > 125)).mean()) if hues.size else 0.0  # OpenCV hue: 0-180
        if other > MAX_OTHER_HUES:
            raise ValueError(f"not a Tango board: {other:.0%} of its colour is neither sun orange nor moon blue")
        n = self.detect_grid_size(grid.image)
        if n % 2:
            raise ValueError(f"detected an odd Tango grid size {n}")
        givens: dict[Cell, Symbol] = {}
        for r, row in enumerate(self.extract_cells(grid.image, n, margin=CELL_MARGIN)):
            for c, cell in enumerate(row):
                gray = _gray(cell)
                if gray.std() >= EMPTY_STD:
                    givens[(r, c)] = _symbol(cell)
        return TangoBoard(size=n, givens=givens, constraints=self._constraints(grid.image, n))

    @staticmethod
    def _constraints(grid_image: np.ndarray, n: int) -> tuple[Constraint, ...]:
        step = GRID_SIDE / n
        half = int(step * PROBE_SIZE / 2)
        core = int(step * SIGN_SIZE / 4)
        gray = _gray(grid_image)
        found = []
        for r in range(n):
            for c in range(n):
                for dr, dc in ((0, 1), (1, 0)):
                    if r + dr >= n or c + dc >= n:
                        continue
                    x = int((c + 0.5 + dc * 0.5) * step)
                    y = int((r + 0.5 + dr * 0.5) * step)
                    probe = gray[y - half : y + half, x - half : x + half]
                    centre = gray[y - core : y + core, x - core : x + core]
                    if (centre < SIGN_DARK).mean() < SIGN_MIN_INK:
                        continue
                    ink = inner_ink(probe < SIGN_DARK)
                    if not ink.any():
                        continue  # only lines running through the probe: no sign here
                    found.append(Constraint((r, c), (r + dr, c + dc), _relation(ink)))
        return tuple(found)

    def overlay_layer(self, image: np.ndarray, board: TangoBoard, solution: Grid) -> np.ndarray:
        n = board.size
        canvas = visualizer.blank_canvas(GRID_SIDE)
        size = int(GRID_SIDE / n)
        for r in range(n):
            for c in range(n):
                if (r, c) in board.givens:
                    continue
                center = visualizer.cell_center(r, c, n, GRID_SIDE)
                draw = visualizer.draw_sun if solution[r][c] is Symbol.SUN else visualizer.draw_moon
                draw(canvas, center, size)
        return self.warp_to_image(image, canvas)
