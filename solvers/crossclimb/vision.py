"""Crossclimb vision: letter-box rows give the word length, EasyOCR reads each row's clue.

Layout assumption: each rung is one row of empty letter boxes with its clue text printed to the left of
the boxes. The first and last rows are the top and bottom words, which share one clue.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core import visualizer
from core.vision_base import BaseVisionParser, read_sentence
from solvers.crossclimb.solver import CrossclimbBoard, Rung

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class RungRow:
    boxes: tuple[Box, ...]
    clue: str


class CrossclimbVisionParser(BaseVisionParser[CrossclimbBoard, list[Rung]]):
    def parse(self, image: np.ndarray) -> CrossclimbBoard:
        rows = self.detect_rows(image)
        if len(rows) < 4:
            raise ValueError(f"expected a top row, at least two middle rows and a bottom row, found {len(rows)} rows")
        return CrossclimbBoard(
            clues=tuple(row.clue for row in rows[1:-1]), end_clue=rows[0].clue, word_length=len(rows[0].boxes)
        )

    def detect_rows(self, image: np.ndarray) -> list[RungRow]:
        """Find the letter-box rows top to bottom and read the clue line left of each."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows = []
        for boxes in self._box_rows(image):
            top = min(y for _, y, _, _ in boxes)
            bottom = max(y + h for _, y, _, h in boxes)
            crop = gray[top:bottom, : boxes[0][0] - boxes[0][2] // 4]
            if (crop < 128).mean() < 0.002:
                raise ValueError(f"no clue text found left of the letter boxes at y={top}")
            rows.append(RungRow(boxes=tuple(boxes), clue=read_sentence(crop)))
        return rows

    def _box_rows(self, image: np.ndarray) -> list[list[Box]]:
        contours, _ = cv2.findContours(self.threshold(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_side = min(image.shape[:2]) * 0.02
        squares = [
            (x, y, w, h) for x, y, w, h in map(cv2.boundingRect, contours) if 0.85 < w / h < 1.18 and w > min_side
        ]
        if not squares:
            raise ValueError("no letter boxes found")
        median_side = float(np.median([w for _, _, w, _ in squares]))
        squares = [s for s in squares if abs(s[2] - median_side) < 0.2 * median_side]
        rows: list[list[Box]] = []
        for box in sorted(squares, key=lambda b: b[1]):
            if rows and abs(rows[-1][0][1] - box[1]) < median_side / 2:
                rows[-1].append(box)
            else:
                rows.append([box])
        rows = [sorted(row) for row in rows]
        lengths = {len(row) for row in rows}
        if len(rows) < 2 or len(lengths) != 1:
            raise ValueError(f"expected >= 2 rows of equal box counts, got row lengths {[len(r) for r in rows]}")
        return rows

    def overlay_layer(self, image: np.ndarray, board: CrossclimbBoard, solution: list[Rung]) -> np.ndarray:
        """Write each answer into its row's boxes and tag the row with its ladder position."""
        rows = self.detect_rows(image)
        layer = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        # Middle rows are matched by clue; the top and bottom rows by position, since they share a clue.
        middle = {rung.clue: (index + 1, rung.word) for index, rung in enumerate(solution[1:-1], start=1)}
        placements = (
            [(1, solution[0].word)] + [middle[row.clue] for row in rows[1:-1]] + [(len(solution), solution[-1].word)]
        )
        for row, (rank, word) in zip(rows, placements, strict=True):
            for (x, y, w, h), letter in zip(row.boxes, word):
                visualizer.draw_label(layer, letter, (x + w // 2, y + h // 2), h / 40, visualizer.PATH)
            x, y, w, h = row.boxes[0]
            visualizer.draw_label(layer, f"#{rank}", (x - w // 2 - 4, y + h // 2), h / 60, visualizer.GOLD)
        return layer
