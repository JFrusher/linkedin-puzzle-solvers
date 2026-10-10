"""Patches vision: each clue is a coloured badge; its outline gives the shape and white digits inside give the area.

LinkedIn draws a shape-only clue as a solid square, tall or wide rectangle, and an area-only clue as a
'+'-shaped badge holding the number. A badge whose outline is not rectangular means any shape.
"""

from __future__ import annotations

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, recognize_glyphs, tight_crop
from solvers.patches.solver import Clue, Patch, PatchesBoard

SATURATED = 50
"""HSV saturation above which a pixel belongs to a clue badge; LinkedIn's slate icon averages 78."""
MIN_BADGE = 0.1
"""Badges cover at least this fraction of a cell; smaller coloured specks are ignored."""
RECT_FILL = 0.92
"""A badge filling this much of its bounding box is a plain rectangle (measured 0.98+); '+' badges fill 0.79-0.85."""
MIN_DIGIT_PIXELS = 0.01
"""A badge holds a number when its white holes cover this fraction of a cell."""
SQUARE_TOLERANCE = 0.2
"""Rectangles whose width/height ratio is within this of 1 are squares."""


class PatchesVisionParser(BaseVisionParser[PatchesBoard, list[Patch]]):
    def parse(self, image: np.ndarray) -> PatchesBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        step = GRID_SIDE / n
        coloured = cv2.cvtColor(grid.image, cv2.COLOR_BGR2HSV)[..., 1] > SATURATED
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(coloured.astype(np.uint8))
        clues = []
        for i in range(1, count):
            x, y, w, h, pixels = stats[i]
            if pixels < MIN_BADGE * step * step:
                continue
            badge = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
            outline, _ = cv2.findContours(badge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            filled = cv2.drawContours(np.zeros_like(badge), outline, -1, 1, -1).astype(bool)
            digits = filled & ~badge.astype(bool)
            cell = (int(centroids[i][1] // step), int(centroids[i][0] // step))
            area = self._read_area(digits, cell) if digits.sum() >= MIN_DIGIT_PIXELS * step * step else None
            shape = self._shape(w, h) if filled.mean() >= RECT_FILL else "any"
            clues.append(Clue(cell, area, shape))
        return PatchesBoard(n, tuple(sorted(clues, key=lambda clue: clue.cell)))

    @staticmethod
    def _read_area(digits: np.ndarray, cell: tuple[int, int]) -> int:
        """White digits on the badge, redrawn black on white for the recognizer."""
        black_on_white = cv2.cvtColor(np.where(digits, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        text, _ = recognize_glyphs(tight_crop(black_on_white, digits), allowlist="0123456789")
        if not text.isdigit():
            raise ValueError(f"clue badge at {cell} has no readable area, read {text!r}")
        return int(text)

    @staticmethod
    def _shape(width: int, height: int) -> str:
        ratio = width / height
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
