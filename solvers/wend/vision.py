"""Wend vision: per-tile EasyOCR letters, dark-tile walls and word-length slot rows."""

from __future__ import annotations

import string

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, recognize_glyphs
from solvers.wend.solver import WendBoard, WordPlacement

WALL_GRAY = 90
"""Tiles whose mean grey level is below this are walls."""


class WendVisionParser(BaseVisionParser[WendBoard, list[WordPlacement]]):
    def parse(self, image: np.ndarray) -> WendBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        letters: list[tuple[str | None, ...]] = []
        for r, row in enumerate(self.extract_cells(grid.image, n, margin=0.1)):
            parsed: list[str | None] = []
            for c, cell in enumerate(row):
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                if gray.mean() < WALL_GRAY:
                    parsed.append(None)
                    continue
                letter, _ = recognize_glyphs(cell, allowlist=string.ascii_uppercase)
                if len(letter) != 1:
                    raise ValueError(f"tile {(r, c)} should hold one letter, read {letter!r}")
                parsed.append(letter)
            letters.append(tuple(parsed))
        return WendBoard(letters=tuple(letters), word_lengths=self._slot_lengths(image, grid.corners))

    def _slot_lengths(self, image: np.ndarray, grid_corners: np.ndarray) -> tuple[int, ...]:
        """Count the empty letter boxes per row outside the grid; one row per target word."""
        contours, _ = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grid_area = cv2.contourArea(grid_corners)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center = (float(x + w / 2), float(y + h / 2))
            if cv2.pointPolygonTest(grid_corners, center, False) >= 0:
                continue
            if 0.8 < w / h < 1.25 and 1e-4 * grid_area < w * h < 0.02 * grid_area:
                boxes.append((x, y, w, h))
        rows: list[list[tuple[int, int, int, int]]] = []
        for box in sorted(boxes, key=lambda b: b[1]):
            if rows and abs(rows[-1][0][1] - box[1]) < box[3] / 2:
                rows[-1].append(box)
            else:
                rows.append([box])
        return tuple(sorted(len(row) for row in rows))

    def overlay_layer(self, image: np.ndarray, board: WendBoard, solution: list[WordPlacement]) -> np.ndarray:
        n = board.size
        canvas = visualizer.blank_canvas(GRID_SIDE)
        for index, placement in enumerate(solution):
            color = visualizer.PALETTE[index % len(visualizer.PALETTE)]
            for r, c in placement.cells:
                visualizer.fill_cell(canvas, r, c, n, color)
            points = [visualizer.cell_center(r, c, n, GRID_SIDE) for r, c in placement.cells]
            visualizer.draw_path(canvas, points, thickness=int(GRID_SIDE / n * 0.12), color=color[:3] + (230,))
        return self.warp_to_image(image, canvas)
