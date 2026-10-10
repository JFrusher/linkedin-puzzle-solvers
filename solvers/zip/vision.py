"""Zip vision: Hough circles + EasyOCR for numbered checkpoints, line-thickness probes for walls."""

from __future__ import annotations

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser, recognize_glyphs
from solvers.zip.solver import Cell, ZipBoard

DARK = 100
"""Grey level below which a pixel counts as ink (walls, checkpoint discs)."""
WALL_FRACTION = 0.075
"""A boundary whose dark band is thicker than this fraction of a cell is a wall.

Measured: walls are at least 0.097 (synthetic) and 0.194 (real); a checkpoint disc reaching into the band
reaches 0.051.
"""


class ZipVisionParser(BaseVisionParser[ZipBoard, list[Cell]]):
    def parse(self, image: np.ndarray) -> ZipBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        return ZipBoard(size=n, checkpoints=self._checkpoints(grid.image, n), walls=self._walls(grid.image, n))

    def _checkpoints(self, grid_image: np.ndarray, n: int) -> tuple[Cell, ...]:
        step = GRID_SIDE / n
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=step * 0.8,
            param1=120,
            param2=30,
            minRadius=int(step * 0.2),
            maxRadius=int(step * 0.45),
        )
        if circles is None:
            raise ValueError("no numbered checkpoints found")
        numbered: dict[int, Cell] = {}
        for x, y, radius in circles[0]:
            cell = (int(y // step), int(x // step))
            inner = int(radius * 0.75)
            crop = gray[max(0, int(y) - inner) : int(y) + inner, max(0, int(x) - inner) : int(x) + inner]
            if crop.size == 0 or np.mean(crop) > 128:
                continue  # hollow ring or a stray arc, not a filled checkpoint disc
            digits, _ = recognize_glyphs(cv2.bitwise_not(crop), allowlist="0123456789")
            if not digits:
                raise ValueError(f"checkpoint disc at cell {cell} has no readable number")
            number = int(digits)
            if number in numbered:
                raise ValueError(f"checkpoint {number} detected twice ({numbered[number]} and {cell})")
            numbered[number] = cell
        expected = list(range(1, len(numbered) + 1))
        if sorted(numbered) != expected:
            raise ValueError(f"checkpoint numbers {sorted(numbered)} are not a 1..K sequence")
        return tuple(numbered[k] for k in expected)

    @staticmethod
    def _walls(grid_image: np.ndarray, n: int) -> frozenset[frozenset[Cell]]:
        step = GRID_SIDE / n
        dark = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY) < DARK
        band = int(step * 0.15)
        walls: set[frozenset[Cell]] = set()
        for r in range(n):
            for c in range(n):
                y0, y1 = int((r + 0.3) * step), int((r + 0.7) * step)
                x0, x1 = int((c + 0.3) * step), int((c + 0.7) * step)
                if c + 1 < n:
                    x = int((c + 1) * step)
                    thickness = dark[y0:y1, x - band : x + band].sum(axis=1).mean()
                    if thickness > step * WALL_FRACTION:
                        walls.add(frozenset({(r, c), (r, c + 1)}))
                if r + 1 < n:
                    y = int((r + 1) * step)
                    thickness = dark[y - band : y + band, x0:x1].sum(axis=0).mean()
                    if thickness > step * WALL_FRACTION:
                        walls.add(frozenset({(r, c), (r + 1, c)}))
        return frozenset(walls)

    def overlay_layer(self, image: np.ndarray, board: ZipBoard, solution: list[Cell]) -> np.ndarray:
        canvas = visualizer.blank_canvas(GRID_SIDE)
        points = [visualizer.cell_center(r, c, board.size, GRID_SIDE) for r, c in solution]
        visualizer.draw_path(canvas, points, thickness=int(GRID_SIDE / board.size * 0.3))
        return self.warp_to_image(image, canvas)
