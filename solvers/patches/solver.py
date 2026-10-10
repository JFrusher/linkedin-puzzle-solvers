"""Patches: split the grid into rectangles, one clue each, as exact cover on Dancing Links.

Each clue gives a patch's area, its shape (``square``, ``tall`` = more rows than columns, ``wide`` = more
columns than rows, or ``any``), or both. LinkedIn shows a number for an area of any shape and a plain shape
icon for a shape of any area. A candidate is a rectangle that fits the clue, contains its clue cell and no
other clue. Exact cover then picks one candidate per clue so that every cell is covered once:
primary items are the N^2 cells plus one item per clue.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from core.dlx import DancingLinks
from core.solver_base import BaseSolver, UnsolvableError

Cell = tuple[int, int]
SHAPES = ("square", "tall", "wide", "any")


@dataclass(frozen=True)
class Clue:
    cell: Cell
    area: int | None
    """``None`` when the clue shows only a shape."""
    shape: str

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")
        if self.area is not None and self.area < 1:
            raise ValueError(f"area must be positive, got {self.area}")

    @property
    def label(self) -> str:
        """The clue as the board shows it: '6', 'tall' or '6 tall'."""
        return " ".join(part for part in (str(self.area or ""), "" if self.shape == "any" else self.shape) if part)


@dataclass(frozen=True)
class PatchesBoard:
    size: int
    clues: tuple[Clue, ...]

    def __post_init__(self) -> None:
        if len({clue.cell for clue in self.clues}) != len(self.clues):
            raise ValueError("two clues share a cell")
        known = [clue.area for clue in self.clues if clue.area is not None]
        if sum(known) > self.size**2 or (len(known) == len(self.clues) and sum(known) != self.size**2):
            raise ValueError(f"clue areas sum to {sum(known)}, which cannot tile {self.size**2} cells")


@dataclass(frozen=True)
class Patch:
    clue: Clue
    top: int
    left: int
    height: int
    width: int

    @property
    def cells(self) -> frozenset[Cell]:
        return frozenset(
            (r, c) for r in range(self.top, self.top + self.height) for c in range(self.left, self.left + self.width)
        )


def fits_shape(height: int, width: int, shape: str) -> bool:
    return {
        "square": height == width,
        "tall": height > width,
        "wide": width > height,
        "any": True,
    }[shape]


def candidates(board: PatchesBoard, clue: Clue) -> list[Patch]:
    """Every rectangle that could be this clue's patch."""
    others = {other.cell for other in board.clues if other != clue}
    r0, c0 = clue.cell
    found = []
    for height, width in product(range(1, board.size + 1), repeat=2):
        if (clue.area is not None and height * width != clue.area) or not fits_shape(height, width, clue.shape):
            continue
        for top in range(max(0, r0 - height + 1), min(r0, board.size - height) + 1):
            for left in range(max(0, c0 - width + 1), min(c0, board.size - width) + 1):
                patch = Patch(clue, top, left, height, width)
                if not patch.cells & others:
                    found.append(patch)
    return found


class PatchesSolver(BaseSolver[PatchesBoard, list[Patch]]):
    name = "patches"

    def solve(self, board_state: PatchesBoard) -> list[Patch]:
        patches = next(self.solutions(board_state), None)
        if patches is None:
            raise UnsolvableError("no set of rectangles covers the grid with one clue each")
        return patches

    def solutions(self, board_state: PatchesBoard) -> Iterator[list[Patch]]:
        """Yield every tiling; take two to test uniqueness."""
        n = board_state.size
        options = [patch for clue in board_state.clues for patch in candidates(board_state, clue)]
        clue_index = {clue: i for i, clue in enumerate(board_state.clues)}
        links = DancingLinks(primary=n * n + len(board_state.clues), secondary=0)
        for row_id, patch in enumerate(options):
            links.add_row(row_id, [r * n + c for r, c in patch.cells] + [n * n + clue_index[patch.clue]])
        for cover in links.search():
            self.nodes_expanded = links.nodes_expanded
            yield sorted((options[row_id] for row_id in cover), key=lambda p: (p.top, p.left))
        self.nodes_expanded = links.nodes_expanded

    def explain(self, board_state: PatchesBoard, solution: list[Patch]) -> None:
        """Replay the answer with human rules; ``Search`` only where none applies.

        * Only Fit: a clue with one rectangle left takes it.
        * Only Cover: a cell that only one rectangle can still cover takes that rectangle.
        * Try and Fail: a rectangle that, followed by Only Fit and Only Cover steps, leaves a clue with no
          rectangle or a cell that nothing can cover is ruled out.
        """
        options = {clue: candidates(board_state, clue) for clue in board_state.clues}
        ruled_out: set[Patch] = set()
        uncovered = {(r, c) for r in range(board_state.size) for c in range(board_state.size)}
        open_clues = list(board_state.clues)
        remaining = list(solution)

        def fitting(cover: set[Cell], clues: list[Clue]) -> dict[Clue, list[Patch]]:
            return {clue: [p for p in options[clue] if p not in ruled_out and p.cells <= cover] for clue in clues}

        def forced(cover: set[Cell], fits: dict[Clue, list[Patch]]) -> tuple[Patch, str, str] | None:
            for opts in fits.values():
                if len(opts) == 1:
                    return opts[0], "Only Fit", "it is the only rectangle left that fits the clue"
            for cell in sorted(cover):
                covering = [p for opts in fits.values() for p in opts if cell in p.cells]
                if len(covering) == 1:
                    return covering[0], "Only Cover", f"no other rectangle can still cover {cell}"
            return None

        def dead_end(patch: Patch) -> bool:
            cover, clues = uncovered - patch.cells, [clue for clue in open_clues if clue != patch.clue]
            while clues:
                fits = fitting(cover, clues)
                if any(not opts for opts in fits.values()):
                    return True
                if any(all(cell not in p.cells for opts in fits.values() for p in opts) for cell in cover):
                    return True
                step = forced(cover, fits)
                if step is None:
                    return False
                cover -= step[0].cells
                clues.remove(step[0].clue)
            return bool(cover)

        while open_clues:
            fits = fitting(uncovered, open_clues)
            step = forced(uncovered, fits)
            if step is None:
                failing = next((p for opts in fits.values() for p in opts if dead_end(p)), None)
                if failing is not None:
                    ruled_out.add(failing)
                    self.explainer.add(
                        "Try and Fail",
                        f"clue {failing.clue.label} at {failing.clue.cell} cannot take the "
                        f"{failing.height}x{failing.width} patch at ({failing.top}, {failing.left}): a few steps "
                        "later a clue has no rectangle or a cell cannot be covered",
                        failing.clue.cell,
                    )
                    continue
                step = remaining[0], "Search", "no rule applies, so the search picks it"
            patch, rule, reason = step
            remaining = [p for p in remaining if p.clue != patch.clue]
            open_clues.remove(patch.clue)
            uncovered -= patch.cells
            self.explainer.add(
                rule,
                f"clue {patch.clue.label} at {patch.clue.cell} takes the {patch.height}x{patch.width} patch at "
                f"({patch.top}, {patch.left}) because {reason}",
                patch.clue.cell,
            )


def is_valid_solution(board: PatchesBoard, patches: list[Patch]) -> bool:
    covered = [cell for patch in patches for cell in patch.cells]
    all_cells = {(r, c) for r in range(board.size) for c in range(board.size)}
    if len(covered) != len(all_cells) or set(covered) != all_cells:
        return False
    if sorted(p.clue.cell for p in patches) != sorted(c.cell for c in board.clues):
        return False
    clue_cells = {clue.cell for clue in board.clues}
    return all(
        (p.clue.area is None or p.height * p.width == p.clue.area)
        and fits_shape(p.height, p.width, p.clue.shape)
        and p.cells & clue_cells == {p.clue.cell}
        for p in patches
    )
