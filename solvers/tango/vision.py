"""Tango vision: template matching of cell symbols and edge signs against rendered glyphs."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from core import visualizer
from core.vision_base import GRID_SIDE, BaseVisionParser
from solvers.tango.solver import Cell, Constraint, Grid, Relation, Symbol, TangoBoard

CELL_MARGIN = 0.15
EMPTY_STD = 12.0
"""Cells whose grey-level standard deviation is below this are blank."""
SIGN_SIZE = 0.24
"""Rendered edge-sign size as a fraction of a cell."""
PROBE_SIZE = 0.4
"""Edge probe size as a fraction of a cell; larger than a sign so the template can slide."""
SIGN_MIN_INK = 0.04
"""Fraction of dark pixels the probe's centre needs before it can hold a sign."""


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)


@lru_cache(maxsize=16)
def symbol_templates(n: int) -> dict[Symbol, np.ndarray]:
    """Grey cell crops of a rendered sun and moon at the rectified scale for an N x N grid."""
    step = int(GRID_SIDE / n)
    pad = int(step * CELL_MARGIN)
    templates = {}
    for symbol, draw in ((Symbol.SUN, visualizer.draw_sun), (Symbol.MOON, visualizer.draw_moon)):
        tile = np.full((step, step, 3), 255, dtype=np.uint8)
        draw(tile, (step // 2, step // 2), step)
        templates[symbol] = _gray(tile[pad : step - pad, pad : step - pad])
    return templates


@lru_cache(maxsize=16)
def sign_templates(n: int) -> dict[Relation, np.ndarray]:
    """Grey crops of rendered '=' and 'x' signs at the rectified scale."""
    size = int(GRID_SIDE / n * SIGN_SIZE)
    templates = {}
    for relation in Relation:
        tile = np.full((size + 4, size + 4, 3), 255, dtype=np.uint8)
        visualizer.draw_relation(tile, (size // 2 + 2, size // 2 + 2), size, relation.value)
        templates[relation] = _gray(tile)
    return templates


def best_match(patch: np.ndarray, templates: dict) -> object:
    """Return the template key with the best normalised cross-correlation anywhere inside ``patch``.

    A patch at least as large as the templates is searched by sliding, which
    tolerates small misalignments; a smaller patch is resized to the template.
    """
    scores = {}
    for key, template in templates.items():
        if patch.shape[0] < template.shape[0] or patch.shape[1] < template.shape[1]:
            patch = cv2.resize(patch, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
        scores[key] = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED).max())
    return max(scores, key=scores.__getitem__)


class TangoVisionParser(BaseVisionParser[TangoBoard, Grid]):
    def parse(self, image: np.ndarray) -> TangoBoard:
        grid = self.rectify_grid(image)
        n = self.detect_grid_size(grid.image)
        if n % 2:
            raise ValueError(f"detected an odd Tango grid size {n}")
        templates = symbol_templates(n)
        givens: dict[Cell, Symbol] = {}
        for r, row in enumerate(self.extract_cells(grid.image, n, margin=CELL_MARGIN)):
            for c, cell in enumerate(row):
                gray = _gray(cell)
                if gray.std() >= EMPTY_STD:
                    givens[(r, c)] = best_match(gray, templates)  # type: ignore[assignment]
        return TangoBoard(size=n, givens=givens, constraints=self._constraints(grid.image, n))

    @staticmethod
    def _constraints(grid_image: np.ndarray, n: int) -> tuple[Constraint, ...]:
        step = GRID_SIDE / n
        half = int(step * PROBE_SIZE / 2)
        core = int(step * SIGN_SIZE / 4)
        gray = _gray(grid_image)
        templates = sign_templates(n)
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
                    if (centre < 100).mean() < SIGN_MIN_INK:
                        continue
                    relation = best_match(probe, templates)
                    found.append(Constraint((r, c), (r + dr, c + dc), relation))  # type: ignore[arg-type]
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
