"""Patches vision: area digits read with EasyOCR, shape icons classified by the aspect of their ink.

Layout assumption: a clue cell shows its area in the upper half and a rectangle icon in the lower half
(square, tall or wide); no icon means any shape.
"""

from __future__ import annotations

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, inner_ink, recognize_glyphs
from solvers.patches.solver import Clue, Patch, PatchesBoard

INK_FRACTION = 0.005
"""A half-cell with less than this fraction of dark pixels is blank (empty halves measure 0 once residue is removed)."""
GLYPH_PAD = 6
"""White border around the tight digit crop."""
SQUARE_TOLERANCE = 0.2
"""Icons whose width/height ratio is within this of 1 are squares."""


class PatchesVisionParser(BaseVisionParser[PatchesBoard, list[Patch]]):
    def parse(self, image: np.ndarray) -> PatchesBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        clues = []
        for r, row in enumerate(self.extract_cells(grid.image, n, margin=0.06)):
            for c, cell in enumerate(row):
                ink = inner_ink(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) < 100)
                half = ink.shape[0] // 2
                if ink[:half].mean() < INK_FRACTION:
                    continue
                digits, _ = recognize_glyphs(self._glyph_crop(cell[:half], ink[:half]), allowlist="0123456789")
                if not digits.isdigit():
                    raise ValueError(f"clue cell {(r, c)} has no readable area, read {digits!r}")
                clues.append(Clue((r, c), int(digits), self._shape(ink[half:])))
        return PatchesBoard(n, tuple(clues))

    @staticmethod
    def _glyph_crop(image: np.ndarray, ink: np.ndarray) -> np.ndarray:
        """Crop tight around the digits: a loose crop makes the recognizer invent extra glyphs."""
        ys, xs = np.nonzero(ink)
        crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        return cv2.copyMakeBorder(crop, *[GLYPH_PAD] * 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    @staticmethod
    def _shape(ink: np.ndarray) -> str:
        if ink.mean() < INK_FRACTION:
            return "any"
        ys, xs = np.nonzero(ink)
        ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        if abs(ratio - 1) <= SQUARE_TOLERANCE:
            return "square"
        return "wide" if ratio > 1 else "tall"

    def overlay_layer(self, image: np.ndarray, board: PatchesBoard, solution: list[Patch]) -> np.ndarray:
        n = board.size
        canvas = visualizer.blank_canvas(GRID_SIDE)
        for index, patch in enumerate(solution):
            color = visualizer.PALETTE[index % len(visualizer.PALETTE)]
            (x0, y0), _ = visualizer.cell_rect(patch.top, patch.left, n, GRID_SIDE)
            _, (x1, y1) = visualizer.cell_rect(patch.top + patch.height - 1, patch.left + patch.width - 1, n, GRID_SIDE)
            cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), color, -1)
            cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), visualizer.INK, 3)
        return self.warp_to_image(image, canvas)
