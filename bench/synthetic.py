"""Seeded synthetic puzzle generators and screenshot-like renderers.

Real LinkedIn screenshots are not bundled, so tests, benchmarks and the web
playground use boards generated here with known ground truth. Renders are
skewed with a random perspective warp to exercise grid rectification.
"""

from __future__ import annotations

import random
from collections import deque

import cv2
import numpy as np

from core import visualizer
from solvers.crossclimb.solver import CrossclimbBoard
from solvers.queens.solver import Cell, QueensBoard
from solvers.tango.solver import Constraint, Grid, Relation, Symbol, TangoBoard
from solvers.wend.solver import WendBoard
from solvers.zip.solver import ZipBoard

CELL = 80
MARGIN = 60
LINE_GRAY = (190, 190, 190)
INK = (20, 20, 20)
REGION_COLORS = (
    (122, 171, 247),
    (247, 196, 163),
    (154, 225, 166),
    (237, 158, 222),
    (128, 222, 247),
    (216, 216, 216),
    (167, 167, 255),
    (255, 213, 140),
    (176, 132, 191),
    (140, 240, 210),
    (100, 150, 200),
    (200, 120, 120),
)

# --------------------------------------------------------------------------- generators


def generate_queens(n: int, seed: int) -> tuple[QueensBoard, list[Cell]]:
    """Random non-touching crown placement, then regions grown from each crown by random BFS."""
    rng = random.Random(seed)
    columns: list[int] = []

    def place(row: int) -> bool:
        if row == n:
            return True
        options = [c for c in range(n) if c not in columns and (not columns or abs(columns[-1] - c) > 1)]
        rng.shuffle(options)
        for c in options:
            columns.append(c)
            if place(row + 1):
                return True
            columns.pop()
        return False

    if n < 4 or not place(0):
        raise ValueError(f"no Queens layout exists for N={n}")
    regions = [[-1] * n for _ in range(n)]
    frontier: list[Cell] = []
    for region, (r, c) in enumerate(enumerate(columns)):
        regions[r][c] = region
        frontier.append((r, c))
    while frontier:
        r, c = frontier.pop(rng.randrange(len(frontier)))
        options = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        free = [(nr, nc) for nr, nc in options if 0 <= nr < n and 0 <= nc < n and regions[nr][nc] < 0]
        if free:
            nr, nc = rng.choice(free)
            regions[nr][nc] = regions[r][c]
            frontier += [(r, c), (nr, nc)]
    return QueensBoard(tuple(tuple(row) for row in regions)), list(enumerate(columns))


def generate_zip(n: int, seed: int, checkpoints: int = 8, walls: int = 6) -> tuple[ZipBoard, list[Cell]]:
    """Randomise a serpentine Hamiltonian path with backbite moves, then add checkpoints and walls."""
    rng = random.Random(seed)
    path: list[Cell] = [(r, c if r % 2 == 0 else n - 1 - c) for r in range(n) for c in range(n)]
    for _ in range(n * n * 20):
        if rng.random() < 0.5:
            path.reverse()
        r, c = path[-1]
        options = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        neighbour = rng.choice([p for p in options if 0 <= p[0] < n and 0 <= p[1] < n])
        k = path.index(neighbour)
        if k != len(path) - 2:
            path[k + 1 :] = reversed(path[k + 1 :])
    stride = (len(path) - 1) / (checkpoints - 1)
    marks = tuple(path[round(i * stride)] for i in range(checkpoints))
    consecutive = {frozenset(pair) for pair in zip(path, path[1:])}
    candidates = [
        frozenset({(r, c), (r + dr, c + dc)})
        for r in range(n)
        for c in range(n)
        for dr, dc in ((0, 1), (1, 0))
        if r + dr < n and c + dc < n and frozenset({(r, c), (r + dr, c + dc)}) not in consecutive
    ]
    return ZipBoard(n, marks, frozenset(rng.sample(candidates, min(walls, len(candidates))))), path


def generate_tango(n: int, seed: int, givens: int = 8, signs: int = 6) -> tuple[TangoBoard, Grid]:
    """Random valid filled grid by backtracking, then reveal some cells and edge relations."""
    rng = random.Random(seed)
    grid: list[list[Symbol | None]] = [[None] * n for _ in range(n)]

    def ok(r: int, c: int) -> bool:
        row = grid[r]
        col = [grid[i][c] for i in range(n)]
        for line in (row, col):
            if line.count(Symbol.SUN) > n // 2 or line.count(Symbol.MOON) > n // 2:
                return False
            if any(line[k] is not None and line[k] == line[k + 1] == line[k + 2] for k in range(n - 2)):
                return False
        return True

    def fill(index: int) -> bool:
        if index == n * n:
            return True
        r, c = divmod(index, n)
        for symbol in rng.sample(list(Symbol), 2):
            grid[r][c] = symbol
            if ok(r, c) and fill(index + 1):
                return True
        grid[r][c] = None
        return False

    fill(0)
    solution: Grid = [[cell for cell in row if cell is not None] for row in grid]
    cells = [(r, c) for r in range(n) for c in range(n)]
    shown = {cell: solution[cell[0]][cell[1]] for cell in rng.sample(cells, givens)}
    edges = [((r, c), (r + dr, c + dc)) for r, c in cells for dr, dc in ((0, 1), (1, 0)) if r + dr < n and c + dc < n]
    constraints = tuple(
        Constraint(a, b, Relation.EQUAL if solution[a[0]][a[1]] == solution[b[0]][b[1]] else Relation.OPPOSITE)
        for a, b in rng.sample(edges, signs)
    )
    return TangoBoard(n, shown, constraints), solution


WEND_SAMPLE = WendBoard(
    letters=(
        tuple("APPLE"),
        tuple("GRAPE"),
        ("L", "I", "M", "E", None),
        tuple("MELON"),
        tuple("PEACH"),
    ),
    word_lengths=(4, 5, 5, 5, 5),
)
"""Rows spell APPLE / GRAPE / LIME + wall / MELON / PEACH; the solver may find any valid tiling."""

CROSSCLIMB_SAMPLE = CrossclimbBoard(
    clues=(
        "having a low temperature",
        "a line made of twisted fibres or threads",
        "a rectangular piece of stiff paper used to send messages",
        "a division of a hospital for a particular group of patients",
        "having or producing a comfortable degree of heat",
    ),
    word_length=4,
)
CROSSCLIMB_ANSWER = ("COLD", "CORD", "CARD", "WARD", "WARM")

PINPOINT_SAMPLES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pawn", "rook", "knight", "bishop", "queen"), "chessman"),
    (("oak", "maple", "birch", "willow", "elm"), "tree"),
    (("violin", "cello", "viola", "harp", "guitar"), "stringed instrument"),
)

# --------------------------------------------------------------------------- renderers


def _canvas(n: int, extra_bottom: int = 0) -> np.ndarray:
    side = n * CELL + 2 * MARGIN
    return np.full((side + extra_bottom, side, 3), 250, dtype=np.uint8)


def _cell_box(r: int, c: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (MARGIN + c * CELL, MARGIN + r * CELL), (MARGIN + (c + 1) * CELL, MARGIN + (r + 1) * CELL)


def _grid_lines(image: np.ndarray, n: int) -> None:
    for i in range(1, n):
        cv2.line(image, (MARGIN + i * CELL, MARGIN), (MARGIN + i * CELL, MARGIN + n * CELL), LINE_GRAY, 2)
        cv2.line(image, (MARGIN, MARGIN + i * CELL), (MARGIN + n * CELL, MARGIN + i * CELL), LINE_GRAY, 2)


def _border(image: np.ndarray, n: int) -> None:
    cv2.rectangle(image, (MARGIN, MARGIN), (MARGIN + n * CELL, MARGIN + n * CELL), INK, 4)


def _edge_segment(a: Cell, b: Cell) -> tuple[tuple[int, int], tuple[int, int]]:
    (r1, c1), (r2, c2) = sorted((a, b))
    if r1 == r2:
        x = MARGIN + c2 * CELL
        return (x, MARGIN + r1 * CELL), (x, MARGIN + (r1 + 1) * CELL)
    y = MARGIN + r2 * CELL
    return (MARGIN + c1 * CELL, y), (MARGIN + (c1 + 1) * CELL, y)


def skew(image: np.ndarray, seed: int, strength: float = 0.04) -> np.ndarray:
    """Apply a random mild perspective warp, padding with the background colour."""
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + rng.uniform(-strength, strength, size=(4, 2)).astype(np.float32) * np.array([w, h], dtype=np.float32)
    dst -= dst.min(axis=0)
    size = tuple(int(v) for v in np.ceil(dst.max(axis=0)))
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, size, borderValue=(250, 250, 250))


def render_queens(board: QueensBoard, seed: int = 0) -> np.ndarray:
    n = board.size
    image = _canvas(n)
    for r in range(n):
        for c in range(n):
            top_left, bottom_right = _cell_box(r, c)
            cv2.rectangle(image, top_left, bottom_right, REGION_COLORS[board.regions[r][c]], -1)
    _grid_lines(image, n)
    for r in range(n):
        for c in range(n):
            if c + 1 < n and board.regions[r][c] != board.regions[r][c + 1]:
                cv2.line(image, *_edge_segment((r, c), (r, c + 1)), INK, 4)
            if r + 1 < n and board.regions[r][c] != board.regions[r + 1][c]:
                cv2.line(image, *_edge_segment((r, c), (r + 1, c)), INK, 4)
    _border(image, n)
    return skew(image, seed)


def render_zip(board: ZipBoard, seed: int = 0) -> np.ndarray:
    n = board.size
    image = _canvas(n)
    cv2.rectangle(image, (MARGIN, MARGIN), (MARGIN + n * CELL, MARGIN + n * CELL), (255, 255, 255), -1)
    _grid_lines(image, n)
    for wall in board.walls:
        cv2.line(image, *_edge_segment(*wall), INK, 8)
    for number, (r, c) in enumerate(board.checkpoints, start=1):
        center = (MARGIN + c * CELL + CELL // 2, MARGIN + r * CELL + CELL // 2)
        cv2.circle(image, center, int(CELL * 0.32), INK, -1, cv2.LINE_AA)
        visualizer.draw_label(image, str(number), center, 0.9, (255, 255, 255, 255))
    _border(image, n)
    return skew(image, seed)


def render_tango(board: TangoBoard, seed: int = 0) -> np.ndarray:
    n = board.size
    image = _canvas(n)
    cv2.rectangle(image, (MARGIN, MARGIN), (MARGIN + n * CELL, MARGIN + n * CELL), (255, 255, 255), -1)
    _grid_lines(image, n)
    for (r, c), symbol in board.givens.items():
        center = (MARGIN + c * CELL + CELL // 2, MARGIN + r * CELL + CELL // 2)
        (visualizer.draw_sun if symbol is Symbol.SUN else visualizer.draw_moon)(image, center, CELL)
    for constraint in board.constraints:
        (y0, x0), (y1, x1) = constraint.a, constraint.b
        center = (MARGIN + int((x0 + x1 + 1) * CELL / 2), MARGIN + int((y0 + y1 + 1) * CELL / 2))
        visualizer.draw_relation(image, center, int(CELL * 0.24), constraint.relation.value)
    _border(image, n)
    return skew(image, seed)


def render_wend(board: WendBoard, seed: int = 0) -> np.ndarray:
    n = board.size
    slot = 34
    image = _canvas(n, extra_bottom=len(board.word_lengths) * (slot + 14) + 20)
    cv2.rectangle(image, (MARGIN, MARGIN), (MARGIN + n * CELL, MARGIN + n * CELL), (255, 255, 255), -1)
    for r in range(n):
        for c in range(n):
            letter = board.letters[r][c]
            top_left, bottom_right = _cell_box(r, c)
            if letter is None:
                cv2.rectangle(image, top_left, bottom_right, (45, 45, 45), -1)
            else:
                center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
                visualizer.draw_label(image, letter, center, 1.6, (20, 20, 20, 255))
    _grid_lines(image, n)
    _border(image, n)
    y = MARGIN + n * CELL + 30
    for length in board.word_lengths:
        for k in range(length):
            x = MARGIN + k * (slot + 8)
            cv2.rectangle(image, (x, y), (x + slot, y + slot), INK, 2)
        y += slot + 14
    return skew(image, seed, strength=0.02)


def render_crossclimb(board: CrossclimbBoard) -> np.ndarray:
    box, gap, row_height, text_width = 56, 8, 84, 1000
    width = text_width + board.word_length * (box + gap) + 2 * 40
    image = np.full((len(board.clues) * row_height + 80, width, 3), 250, dtype=np.uint8)
    for index, clue in enumerate(board.clues):
        y = 40 + index * row_height
        cv2.putText(image, clue, (40, y + box // 2 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, INK, 2, cv2.LINE_AA)
        for k in range(board.word_length):
            x = 40 + text_width + k * (box + gap)
            cv2.rectangle(image, (x, y), (x + box, y + box), INK, 2)
    return image


def regions_contiguous(board: QueensBoard) -> bool:
    """True when every region is contiguous (sanity check for generated boards)."""
    n = board.size
    for region in range(n):
        cells = [(r, c) for r in range(n) for c in range(n) if board.regions[r][c] == region]
        seen = {cells[0]}
        queue = deque([cells[0]])
        while queue:
            r, c = queue.popleft()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(seen) != len(cells):
            return False
    return True
