"""Wend vision: per-tile EasyOCR letters, dark-tile walls and word-length slot rows."""

from __future__ import annotations

import string

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, inner_ink, recognize_glyphs, tight_crop
from solvers.wend.solver import WendBoard, WordPlacement

WALL_GRAY = 215
"""Tiles whose median grey level is below this are walls: tiles measure 250+, walls 178 (LinkedIn) or 45 (ours)."""


LETTER_DARK = 128
"""Grey level below which a pixel belongs to a letter."""


class WendVisionParser(BaseVisionParser[WendBoard, list[WordPlacement]]):
    def parse(self, image: np.ndarray) -> WendBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        letters: list[tuple[str | None, ...]] = []
        for r, row in enumerate(self.extract_cells(grid.image, n, margin=0.1)):
            parsed: list[str | None] = []
            for c, cell in enumerate(row):
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                if np.median(gray) < WALL_GRAY:
                    parsed.append(None)
                    continue
                ink = inner_ink(gray < LETTER_DARK)
                if not ink.any():
                    raise ValueError(f"tile {(r, c)} is open but holds no letter")
                letter, _ = recognize_glyphs(tight_crop(cell, ink), allowlist=string.ascii_uppercase)
                if len(letter) != 1:
                    raise ValueError(f"tile {(r, c)} should hold one letter, read {letter!r}")
                parsed.append(letter)
            letters.append(tuple(parsed))
        return WendBoard(letters=tuple(letters), word_lengths=self.slot_lengths(image, grid.corners))

    def slot_lengths(self, image: np.ndarray, grid_corners: np.ndarray | None = None) -> tuple[int, ...]:
        """Count the empty letter boxes per row, one row per target word.

        The slots may sit beside the grid in the same screenshot (boxes inside ``grid_corners`` are skipped) or
        come as a screenshot of their own. Boxes are kept when near-square and close to the median box size.
        """
        contours, _ = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center = (float(x + w / 2), float(y + h / 2))
            if grid_corners is not None and cv2.pointPolygonTest(grid_corners, center, False) >= 0:
                continue
            if 0.8 < w / h < 1.25:
                boxes.append((x, y, w, h))
        if not boxes:
            return ()
        typical = float(np.median([w * h for *_, w, h in boxes]))
        boxes = [box for box in boxes if 0.5 * typical < box[2] * box[3] < 2 * typical]
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
