"""Queens vision: LAB colour sampling + K-Means clustering into N regions."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser
from solvers.queens.solver import Cell, QueensBoard


class QueensVisionParser(BaseVisionParser[QueensBoard, list[Cell]]):
    """Detect the N x N grid and group cells into N colour regions."""

    def parse(self, image: np.ndarray) -> QueensBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        cells = self.extract_cells(grid.image, n, margin=0.3)
        colors = np.array(
            [np.median(cv2.cvtColor(cell, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0) for row in cells for cell in row],
            dtype=np.float32,
        )
        labels = KMeans(n_clusters=n, n_init=10, random_state=0).fit_predict(colors)
        # Renumber regions in reading order so ids are stable across runs.
        order: dict[int, int] = {}
        for label in labels:
            order.setdefault(int(label), len(order))
        regions = tuple(tuple(order[int(labels[r * n + c])] for c in range(n)) for r in range(n))
        return QueensBoard(regions=regions)

    def overlay_layer(self, image: np.ndarray, board: QueensBoard, solution: list[Cell]) -> np.ndarray:
        n = board.size
        canvas = visualizer.blank_canvas(GRID_SIDE)
        for r, c in solution:
            visualizer.draw_crown(canvas, visualizer.cell_center(r, c, n, GRID_SIDE), int(GRID_SIDE / n * 0.6))
        return self.warp_to_image(image, canvas)
