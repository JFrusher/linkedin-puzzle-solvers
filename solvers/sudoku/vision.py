"""Mini Sudoku vision: digits read per cell with EasyOCR; empty cells hold almost no ink."""

from __future__ import annotations

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, recognize_glyphs
from solvers.sudoku.solver import SIZE, Cell, Grid, SudokuBoard

INK_FRACTION = 0.005
"""Cells with less than this fraction of dark pixels are empty: empty cells measure 0, a real "1" 0.016."""
SATURATED = 50
"""HSV saturation above which a pixel is coloured; Mini Sudoku is greyscale, so colour marks a Patches badge."""
MAX_COLOUR_FRACTION = 0.1
"""A cell more coloured than this is not a Mini Sudoku cell (Patches badges cover 0.1+ of a cell)."""


class SudokuVisionParser(BaseVisionParser[SudokuBoard, Grid]):
    def parse(self, image: np.ndarray) -> SudokuBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        if n != SIZE:
            raise ValueError(f"expected a {SIZE}x{SIZE} Mini Sudoku grid, detected {n}x{n}")
        givens: dict[Cell, int] = {}
        for r, row in enumerate(self.extract_cells(grid.image, n)):
            for c, cell in enumerate(row):
                if (cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)[..., 1] > SATURATED).mean() > MAX_COLOUR_FRACTION:
                    raise ValueError(f"cell {(r, c)} holds a coloured badge, not a Mini Sudoku digit")
                if (cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) < 100).mean() < INK_FRACTION:
                    continue
                digit, _ = recognize_glyphs(cell, allowlist="123456")
                if len(digit) != 1:
                    raise ValueError(f"cell {(r, c)} should hold one digit, read {digit!r}")
                givens[(r, c)] = int(digit)
        return SudokuBoard(givens)

    def overlay_layer(self, image: np.ndarray, board: SudokuBoard, solution: Grid) -> np.ndarray:
        canvas = visualizer.blank_canvas(GRID_SIDE)
        for r in range(SIZE):
            for c in range(SIZE):
                if (r, c) not in board.givens:
                    center = visualizer.cell_center(r, c, SIZE, GRID_SIDE)
                    visualizer.draw_label(canvas, str(solution[r][c]), center, 2.2, visualizer.PATH)
        return self.warp_to_image(image, canvas)
