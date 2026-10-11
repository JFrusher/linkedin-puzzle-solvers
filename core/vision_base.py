"""Abstract computer-vision pipeline shared by every game parser."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import cv2
import numpy as np

from core import visualizer

if TYPE_CHECKING:
    import easyocr

BoardT = TypeVar("BoardT")
SolutionT = TypeVar("SolutionT")

GRID_SIDE = 720
"""Side length (px) of the rectified, deskewed grid image."""


class GridNotFoundError(ValueError):
    """Raised when no puzzle grid can be located in an image."""


@dataclass(frozen=True)
class RectifiedGrid:
    """A deskewed square grid and the homography that produced it."""

    image: np.ndarray
    """Square BGR image of side :data:`GRID_SIDE`."""
    matrix: np.ndarray
    """3x3 perspective transform from original-image to grid coordinates."""
    corners: np.ndarray
    """4x2 float32 grid corners in the original image (tl, tr, br, bl)."""


@lru_cache(maxsize=1)
def ocr_reader() -> easyocr.Reader:
    """Return the process-wide EasyOCR reader (model load takes seconds, so it is cached)."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def recognize_glyphs(image: np.ndarray, allowlist: str) -> tuple[str, float]:
    """Recognise a crop that holds one short token (a letter or a number).

    EasyOCR's CRAFT detector often misses lone glyphs, so detection is skipped
    and the whole crop is treated as a single text box.
    """
    ((_, text, conf),) = ocr_reader().recognize(image, allowlist=allowlist)
    return text, float(conf)


def read_sentence(image: np.ndarray) -> str:
    """Read one line of running text: detect word boxes, then join them left to right.

    ``mag_ratio=2`` upscales before detection, which recovers more short words;
    a lone leading article can still be missed.
    """
    results = ocr_reader().readtext(image, mag_ratio=2)
    return " ".join(text for box, text, _ in sorted(results, key=lambda result: result[0][0][0]))


def inner_ink(ink: np.ndarray) -> np.ndarray:
    """Ink mask minus every blob that touches the crop edge: those are grid lines running through it."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    h, w = ink.shape
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    right, bottom = left + stats[:, cv2.CC_STAT_WIDTH], top + stats[:, cv2.CC_STAT_HEIGHT]
    inner = [i for i in range(1, count) if left[i] > 0 and top[i] > 0 and right[i] < w and bottom[i] < h]
    return np.isin(labels, inner)


GLYPH_PAD = 6
"""White border around a tight glyph crop."""


def tight_crop(image: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Crop to the ink plus a white border: a loose crop makes the recognizer invent extra glyphs."""
    ys, xs = np.nonzero(ink)
    crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.copyMakeBorder(
        crop, GLYPH_PAD, GLYPH_PAD, GLYPH_PAD, GLYPH_PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = points.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
        dtype=np.float32,
    )


class BaseVisionParser(ABC, Generic[BoardT, SolutionT]):
    """Locate, deskew and slice a puzzle grid, then delegate game-specific parsing."""

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        """Read an image from disk as BGR."""
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"image not found: {file}")
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {file}")
        return image

    @staticmethod
    def threshold(image: np.ndarray) -> np.ndarray:
        """Inverse adaptive threshold: grid lines and glyphs become white on black."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        block = max(11, (min(gray.shape) // 40) | 1)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4)

    def find_grid_corners(self, image: np.ndarray) -> np.ndarray:
        """Return the ordered corners of the largest quadrilateral contour."""
        binary = self.threshold(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.05 * image.shape[0] * image.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < min_area:
                break
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_corners(approx)
        raise GridNotFoundError("no quadrilateral puzzle grid covering at least 5% of the image was found")

    def rectify_grid(self, image: np.ndarray) -> RectifiedGrid:
        """Isolate, deskew and flatten the puzzle square via a perspective transform."""
        corners = self.find_grid_corners(image)
        target = np.array(
            [[0, 0], [GRID_SIDE - 1, 0], [GRID_SIDE - 1, GRID_SIDE - 1], [0, GRID_SIDE - 1]], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(corners, target)
        warped = cv2.warpPerspective(image, matrix, (GRID_SIDE, GRID_SIDE))
        return RectifiedGrid(image=warped, matrix=matrix, corners=corners)

    @staticmethod
    def detect_grid_size(grid_image: np.ndarray, candidates: range = range(4, 13)) -> int:
        """Infer N for an N x N grid from the periodicity of its line edges.

        The column/row edge-energy profiles are correlated (Pearson) with an
        ideal comb of N-1 equally spaced lines; missing lines and unexplained
        lines both lower the correlation, so the true N scores highest.
        """
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        profile_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
        profile_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1)
        # Ignore the outer border, which every candidate shares.
        trim = gray.shape[0] // 60
        profiles = np.concatenate([profile_x[trim:-trim], profile_y[trim:-trim]])
        positions = np.arange(trim, gray.shape[0] - trim, dtype=np.float32)
        best_n, best_score = 0, -np.inf
        for n in candidates:
            lines = np.arange(1, n) * gray.shape[0] / n
            comb = np.exp(-0.5 * ((positions[:, None] - lines[None, :]) / 2.0) ** 2).sum(axis=1)
            score = float(np.corrcoef(profiles, np.concatenate([comb, comb]))[0, 1])
            if score > best_score:
                best_n, best_score = n, score
        return best_n

    @staticmethod
    def extract_cells(grid_image: np.ndarray, n: int, margin: float = 0.15) -> list[list[np.ndarray]]:
        """Slice the grid into N x N cell crops, trimming ``margin`` of each side to drop grid lines."""
        if n <= 0:
            raise ValueError(f"grid size must be positive, got {n}")
        step = grid_image.shape[0] / n
        pad = int(step * margin)
        return [
            [
                grid_image[
                    int(r * step) + pad : int((r + 1) * step) - pad, int(c * step) + pad : int((c + 1) * step) - pad
                ]
                for c in range(n)
            ]
            for r in range(n)
        ]

    @abstractmethod
    def parse(self, image: np.ndarray) -> BoardT:
        """Convert a screenshot into the game's board model."""

    @abstractmethod
    def overlay_layer(self, image: np.ndarray, board: BoardT, solution: SolutionT) -> np.ndarray:
        """Return a BGRA layer, aligned with ``image``, that contains only the solution drawing."""

    def warp_to_image(self, image: np.ndarray, canvas: np.ndarray) -> np.ndarray:
        """Map a BGRA drawing made in rectified-grid space back onto ``image``'s perspective."""
        grid = self.rectify_grid(image)
        height, width = image.shape[:2]
        return cv2.warpPerspective(
            canvas, np.linalg.inv(grid.matrix), (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
        )

    def generate_overlay(self, image: np.ndarray, board: BoardT, solution: SolutionT) -> np.ndarray:
        """Composite the solution drawing over the original screenshot."""
        return visualizer.composite(image, self.overlay_layer(image, board, solution))

    def debug_views(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Intermediate pipeline images: raw, contours, HSV channels and the cell graph."""
        binary = self.threshold(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_view = image.copy()
        cv2.drawContours(contour_view, contours, -1, (0, 200, 0), 1)
        grid = self.rectify_grid(image)
        cv2.polylines(contour_view, [grid.corners.astype(np.int32)], True, (0, 0, 255), 3)
        hsv = cv2.cvtColor(grid.image, cv2.COLOR_BGR2HSV)
        hsv_view = cv2.cvtColor(np.hstack([hsv[..., 0], hsv[..., 1], hsv[..., 2]]), cv2.COLOR_GRAY2BGR)
        n = self.detect_grid_size(grid.image)
        graph_view = grid.image.copy()
        visualizer.draw_cell_graph(graph_view, n)
        return {"Raw": image, "Contours": contour_view, "HSV Masks": hsv_view, "Graph Nodes": graph_view}
