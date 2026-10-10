"""Drawing primitives for crisp, high-contrast solution overlays.

All ``draw_*`` functions draw on BGRA canvases (alpha 255 where drawn) unless
noted, so the result can be warped back onto a screenshot and composited.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Color = tuple[int, int, int, int]

INK: Color = (20, 20, 20, 255)
WHITE: Color = (255, 255, 255, 255)
GOLD: Color = (0, 190, 255, 255)
SUN: Color = (0, 165, 255, 255)
MOON: Color = (220, 120, 40, 255)
PATH: Color = (230, 80, 160, 235)
PALETTE: tuple[Color, ...] = (
    (80, 80, 230, 140),
    (80, 200, 80, 140),
    (230, 140, 60, 140),
    (60, 200, 230, 140),
    (200, 80, 200, 140),
    (200, 200, 60, 140),
    (120, 120, 120, 140),
    (40, 140, 240, 140),
)


def blank_canvas(side: int) -> np.ndarray:
    """Fully transparent square BGRA canvas."""
    return np.zeros((side, side, 4), dtype=np.uint8)


def cell_center(row: int, col: int, n: int, side: int) -> tuple[int, int]:
    """Pixel (x, y) centre of cell (row, col) in an N x N grid of ``side`` px."""
    step = side / n
    return int((col + 0.5) * step), int((row + 0.5) * step)


def cell_rect(row: int, col: int, n: int, side: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Top-left and bottom-right pixel corners of cell (row, col)."""
    step = side / n
    return (int(col * step), int(row * step)), (int((col + 1) * step), int((row + 1) * step))


def draw_crown(canvas: np.ndarray, center: tuple[int, int], size: int, color: Color = GOLD) -> None:
    """Draw a filled five-point crown with an ink outline."""
    x, y = center
    h = size // 2
    points = np.array(
        [
            (x - h, y + h // 2),
            (x - h, y - h // 2),
            (x - h // 2, y),
            (x, y - h),
            (x + h // 2, y),
            (x + h, y - h // 2),
            (x + h, y + h // 2),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [points], color, cv2.LINE_AA)
    cv2.polylines(canvas, [points], True, INK, max(2, size // 20), cv2.LINE_AA)


def draw_sun(canvas: np.ndarray, center: tuple[int, int], size: int) -> None:
    """Draw a sun: filled disc with an ink ring."""
    radius = size // 3
    cv2.circle(canvas, center, radius, SUN, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, INK, max(2, size // 25), cv2.LINE_AA)


def draw_moon(canvas: np.ndarray, center: tuple[int, int], size: int) -> None:
    """Draw a crescent moon by carving a transparent disc out of a filled one."""
    radius = size // 3
    disc = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(disc, center, radius, 255, -1, cv2.LINE_AA)
    cv2.circle(disc, (center[0] + radius // 2, center[1] - radius // 3), radius, 0, -1, cv2.LINE_AA)
    canvas[disc > 0] = MOON[: canvas.shape[2]]


def draw_relation(canvas: np.ndarray, center: tuple[int, int], size: int, relation: str) -> None:
    """Draw a Tango edge sign ('=' or 'x') on a small white tile centred on ``center``."""
    if relation not in ("=", "x"):
        raise ValueError(f"unknown relation {relation!r}")
    x, y = center
    h = size // 2
    q = size // 4
    thickness = max(2, size // 8)
    cv2.rectangle(canvas, (x - h, y - h), (x + h, y + h), WHITE, -1)
    if relation == "=":
        gap = q * 3 // 4  # LinkedIn's '=' bars sit apart, with a clear gap at the centre
        cv2.line(canvas, (x - q, y - gap), (x + q, y - gap), INK, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x - q, y + gap), (x + q, y + gap), INK, thickness, cv2.LINE_AA)
    else:
        cv2.line(canvas, (x - q, y - q), (x + q, y + q), INK, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x - q, y + q), (x + q, y - q), INK, thickness, cv2.LINE_AA)


def draw_path(canvas: np.ndarray, points: Sequence[tuple[int, int]], thickness: int, color: Color = PATH) -> None:
    """Draw a ribbon through ``points`` with rounded joints and a start marker."""
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(canvas, point, thickness // 2, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, points[0], thickness, color, -1, cv2.LINE_AA)


def draw_label(canvas: np.ndarray, text: str, center: tuple[int, int], scale: float, color: Color = INK) -> None:
    """Draw ``text`` centred on ``center``."""
    thickness = max(1, int(scale * 2))
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    origin = (center[0] - width // 2, center[1] + height // 2)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fill_cell(canvas: np.ndarray, row: int, col: int, n: int, color: Color, inset: int = 4) -> None:
    """Fill one cell (minus ``inset`` px on each side) with ``color``."""
    (x0, y0), (x1, y1) = cell_rect(row, col, n, canvas.shape[0])
    cv2.rectangle(canvas, (x0 + inset, y0 + inset), (x1 - inset, y1 - inset), color, -1)


def draw_cell_graph(image: np.ndarray, n: int) -> None:
    """Draw cell-centre nodes and 4-neighbour edges on a BGR image (debug view)."""
    side = image.shape[0]
    for r in range(n):
        for c in range(n):
            here = cell_center(r, c, n, side)
            if c + 1 < n:
                cv2.line(image, here, cell_center(r, c + 1, n, side), (0, 160, 0), 1, cv2.LINE_AA)
            if r + 1 < n:
                cv2.line(image, here, cell_center(r + 1, c, n, side), (0, 160, 0), 1, cv2.LINE_AA)
            cv2.circle(image, here, max(3, side // (n * 10)), (0, 0, 220), -1, cv2.LINE_AA)


def composite(image: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Alpha-blend a BGRA ``layer`` over a BGR ``image`` of the same size."""
    if image.shape[:2] != layer.shape[:2]:
        raise ValueError(f"layer {layer.shape[:2]} does not match image {image.shape[:2]}")
    alpha = layer[..., 3:4].astype(np.float32) / 255.0
    blended = image.astype(np.float32) * (1 - alpha) + layer[..., :3].astype(np.float32) * alpha
    return blended.astype(np.uint8)
