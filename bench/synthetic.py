"""Seeded synthetic puzzle generators and screenshot-like renderers.

Real LinkedIn screenshots are not bundled, so tests, benchmarks and the web
playground use boards generated here with known ground truth. Renders are
skewed with a random perspective warp to exercise grid rectification.
"""

from __future__ import annotations

import random
from collections import deque
from functools import lru_cache
from itertools import islice

import cv2
import numpy as np

from core import visualizer
from core.nlp import dictionary_words
from solvers.crossclimb.solver import CrossclimbBoard
from solvers.queens.solver import Cell, QueensBoard, QueensSolver
from solvers.tango.solver import Constraint, Grid, Relation, Symbol, TangoBoard, TangoSolver
from solvers.wend.solver import MIN_WORD_LENGTH, WendBoard, WendSolver
from solvers.zip.solver import ZipBoard, ZipSolver

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
    """Random non-touching crown placement, regions grown from each crown, then reshaped until unique.

    Real boards have exactly one answer. Region edges are flipped one cell at a time (never a true crown
    cell, never splitting a region) and a flip is kept when the answer count does not rise. Cells that
    hold crowns in other answers are tried first, since moving them breaks those answers.
    """
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
    truth = list(enumerate(columns))
    return _make_unique(regions, truth, rng), truth


def _make_unique(regions: list[list[int]], truth: list[Cell], rng: random.Random, cap: int = 20) -> QueensBoard:
    n = len(regions)
    crowns = set(truth)

    def answers() -> list[list[Cell]]:
        return list(islice(QueensSolver().solutions(QueensBoard(tuple(tuple(row) for row in regions))), cap))

    def flips(cells: list[Cell]) -> list[tuple[list[Cell], int]]:
        """Moves that hand a cell, plus any piece of its region that would break off, to a neighbour region."""
        moves = []
        for r, c in cells:
            own = regions[r][c]
            rest = {(i, j) for i in range(n) for j in range(n) if regions[i][j] == own and (i, j) != (r, c)}
            keep = _component(rest, truth[own])  # the piece holding this region's true crown stays
            moved = [(r, c), *sorted(rest - keep)]
            foreign = {
                regions[nr][nc]
                for i, j in moved
                for nr, nc in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                if 0 <= nr < n and 0 <= nc < n and regions[nr][nc] != own
            }
            moves += [(moved, region) for region in foreign]
        return moves

    found = answers()
    for _ in range(n * n * 20):
        if len(found) == 1:
            return QueensBoard(tuple(tuple(row) for row in regions))
        targets = {cell for answer in found for cell in answer} - crowns
        anywhere = [(r, c) for r in range(n) for c in range(n) if (r, c) not in crowns]
        # Mostly break other answers directly; sometimes reshape anywhere so stuck targets become movable.
        moves = (rng.random() < 0.7 and flips(sorted(targets))) or flips(anywhere)
        moved, region = rng.choice(moves)
        previous = regions[moved[0][0]][moved[0][1]]
        for r, c in moved:
            regions[r][c] = region
        candidate = answers()
        if len(candidate) <= len(found):
            found = candidate
        else:
            for r, c in moved:
                regions[r][c] = previous
    raise ValueError("could not reshape regions into a one-answer board; try another seed")


def _component(cells: set[Cell], start: Cell) -> set[Cell]:
    """The 4-connected part of ``cells`` that contains ``start``."""
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def hamiltonian_path(n: int, rng: random.Random) -> list[Cell]:
    """A random path through every cell: a serpentine randomised by backbite moves."""
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
    return path


def sparse_zip(n: int, seed: int, checkpoints: int = 4) -> ZipBoard:
    """Few checkpoints and no walls, like many real boards; may have several answers (stress case)."""
    path = hamiltonian_path(n, random.Random(seed))
    return ZipBoard(n, tuple(path[round(i * (len(path) - 1) / (checkpoints - 1))] for i in range(checkpoints)))


def generate_zip(n: int, seed: int, checkpoints: int = 6, walls: int = 2) -> tuple[ZipBoard, list[Cell]]:
    """Random Hamiltonian path, a few checkpoints and walls, then more clues until it is the only answer.

    While a second path exists, either wall off an edge it uses that the true path does not (always breaks
    it) or promote a cell it reaches at a different step into a checkpoint.
    """
    rng = random.Random(seed)
    path = hamiltonian_path(n, rng)
    stride = (len(path) - 1) / (checkpoints - 1)
    marks = {path[round(i * stride)] for i in range(checkpoints)}
    used = {frozenset(pair) for pair in zip(path, path[1:])}
    candidates = [
        frozenset({(r, c), (r + dr, c + dc)})
        for r in range(n)
        for c in range(n)
        for dr, dc in ((0, 1), (1, 0))
        if r + dr < n and c + dc < n and frozenset({(r, c), (r + dr, c + dc)}) not in used
    ]
    blocked = set(rng.sample(candidates, min(walls, len(candidates))))
    while True:
        board = ZipBoard(n, tuple(sorted(marks, key=path.index)), frozenset(blocked))
        other = next((p for p in islice(ZipSolver().solutions(board), 2) if p != path), None)
        if other is None:
            return board, path
        shifted = [cell for cell in path if other.index(cell) != path.index(cell) and cell not in marks]
        if shifted and rng.random() < 0.5:
            marks.add(rng.choice(shifted))
        else:
            blocked.add(rng.choice(sorted({frozenset(pair) for pair in zip(other, other[1:])} - used, key=sorted)))


def generate_tango(n: int, seed: int, givens: int = 4, signs: int = 4) -> tuple[TangoBoard, Grid]:
    """Random valid filled grid, a few revealed cells and signs, then more clues until the answer is unique.

    While a second answer exists, pick a cell where it differs from the truth and either reveal that cell
    or add the sign on an edge whose two cells it gets wrong in different ways.
    """
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

    def sign(a: Cell, b: Cell) -> Constraint:
        same = solution[a[0]][a[1]] == solution[b[0]][b[1]]
        return Constraint(a, b, Relation.EQUAL if same else Relation.OPPOSITE)

    constraints = [sign(a, b) for a, b in rng.sample(edges, signs)]
    while True:
        board = TangoBoard(n, dict(shown), tuple(constraints))
        other = next((g for g in islice(TangoSolver().solutions(board), 2) if g != solution), None)
        if other is None:
            return board, solution
        wrong = {(r, c) for r, c in cells if other[r][c] != solution[r][c]}
        # An edge with exactly one wrong end has the opposite relation in the other answer, so its sign breaks it.
        breaking = [(a, b) for a, b in edges if (a in wrong) != (b in wrong)]
        if breaking and rng.random() < 0.5:
            constraints.append(sign(*rng.choice(breaking)))
        else:
            r, c = rng.choice(sorted(wrong))
            shown[(r, c)] = solution[r][c]


def generate_wend(n: int, seed: int, walls: int = 1) -> WendBoard:
    """Tile the open cells with random snakes of 3–6 cells, write a random dictionary word along each, and
    retry until the board has exactly one answer, like the real game.
    """
    rng = random.Random(seed)
    cells = [(r, c) for r in range(n) for c in range(n)]
    by_length: dict[int, list[str]] = {}
    for word in sorted(dictionary_words(MIN_WORD_LENGTH)):
        by_length.setdefault(len(word), []).append(word)
    for _ in range(1000):
        wall = set(rng.sample(cells, walls))
        free = set(cells) - wall
        paths: list[list[Cell]] = []

        while free:
            # Start from the tightest cell so no tile gets stranded.
            path = [min(free, key=lambda cell: (_open_sides(cell, free), rng.random()))]
            free.discard(path[0])
            target = rng.randint(3, 6)
            while len(path) < target:
                r, c = path[-1]
                options = [
                    (r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (r + dr, c + dc) in free
                ]
                if not options:
                    break
                path.append(rng.choice(options))
                free.discard(path[-1])
            if len(path) < MIN_WORD_LENGTH:
                break
            paths.append(path)
        else:
            letters: list[list[str | None]] = [[None] * n for _ in range(n)]
            for path in paths:
                for (r, c), letter in zip(path, rng.choice(by_length[len(path)]), strict=True):
                    letters[r][c] = letter
            board = WendBoard(tuple(tuple(row) for row in letters), tuple(sorted(len(path) for path in paths)))
            if len(list(islice(_wend_solver().solutions(board), 2))) == 1:
                return board
    raise ValueError(f"could not make a one-answer {n}x{n} board with {walls} wall(s)")


@lru_cache(maxsize=1)
def _wend_solver() -> WendSolver:
    return WendSolver()  # building the WordNet Trie takes seconds, so share one


def _open_sides(cell: Cell, free: set[Cell]) -> int:
    return sum((cell[0] + dr, cell[1] + dc) in free for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


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
        "a mass of small bubbles formed on a liquid",
        "a quantity of paper, 480 or 500 sheets",
        "a small elongated soft-bodied animal such as an earthworm",
        "move about aimlessly without any destination",
        "a category of things distinguished by a common characteristic",
    ),
    end_clue="cooperative work done by a group acting together",
    word_length=4,
)
CROSSCLIMB_ANSWER = ("TEAM", "REAM", "ROAM", "FOAM", "FORM", "WORM", "WORK")
"""LinkedIn Crossclimb, 5 October 2026: the middle clues are shuffled, as in the game."""

PINPOINT_SAMPLES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pawn", "rook", "knight", "bishop", "queen"), "chessman"),
    (("oak", "maple", "birch", "willow", "elm"), "tree"),
    (("violin", "cello", "viola", "harp", "guitar"), "stringed instrument"),
    # Real LinkedIn puzzles: #892 (9 Oct 2026, "Types of whales") and #891 (8 Oct 2026).
    (("Blue", "Right", "Bowhead", "Humpback", "Orca"), "whale"),
    (("Code", "Circuit", "Ice", "Deal", "Tie"), 'words before "breaker"'),
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
    rows = (board.end_clue, *board.clues, board.end_clue)  # top and bottom rows share the end clue
    image = np.full((len(rows) * row_height + 80, width, 3), 250, dtype=np.uint8)
    for index, clue in enumerate(rows):
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
