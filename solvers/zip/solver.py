"""Zip as a Hamiltonian path through ordered checkpoints, solved with A* search.

State: (head cell, visited bitmask, next checkpoint index). Every move costs 1
and a complete path has exactly N^2 - 1 moves, so the admissible heuristic
``h = max(unvisited cells, Manhattan chain through the remaining checkpoints)``
makes ``f = g + h`` exact for every feasible state. Ties are broken by depth
(deepest first, i.e. backtracking order) and then by Manhattan distance to the
next checkpoint. Infeasible states are pruned before they enter the queue:

* stepping on a checkpoint out of order;
* unvisited cells no longer connected to the head (flood fill);
* an unvisited cell other than the final checkpoint left with fewer than two
  usable neighbours (degree-constrained subgraph check).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

from core.solver_base import BaseSolver, UnsolvableError

Cell = tuple[int, int]


@dataclass(frozen=True)
class ZipBoard:
    """An N x N grid with numbered checkpoints and walls between adjacent cells."""

    size: int
    checkpoints: tuple[Cell, ...]
    """``checkpoints[i]`` holds the cell labelled ``i + 1``."""
    walls: frozenset[frozenset[Cell]] = frozenset()

    def __post_init__(self) -> None:
        if len(self.checkpoints) < 2:
            raise ValueError("Zip needs at least two checkpoints")
        if len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError("Zip checkpoints must be distinct cells")
        for r, c in self.checkpoints:
            if not (0 <= r < self.size and 0 <= c < self.size):
                raise ValueError(f"checkpoint {(r, c)} lies outside the {self.size}x{self.size} grid")

    def neighbours(self, cell: Cell) -> list[Cell]:
        """Orthogonal neighbours of ``cell`` that are not separated by a wall."""
        r, c = cell
        return [
            (nr, nc)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= nr < self.size and 0 <= nc < self.size and frozenset({cell, (nr, nc)}) not in self.walls
        ]


@dataclass(frozen=True)
class _State:
    head: int
    visited: int
    next_checkpoint: int
    g: int
    parent: _State | None


class ZipSolver(BaseSolver[ZipBoard, list[Cell]]):
    """Find the path that visits every cell once and the checkpoints in order."""

    name = "zip"

    def solve(self, board_state: ZipBoard) -> list[Cell]:
        n = board_state.size
        total = n * n
        adjacency = [[r2 * n + c2 for r2, c2 in board_state.neighbours(divmod(index, n))] for index in range(total)]
        checkpoints = [r * n + c for r, c in board_state.checkpoints]
        checkpoint_order = {cell: i for i, cell in enumerate(checkpoints)}
        end = checkpoints[-1]
        full = (1 << total) - 1

        def manhattan(a: int, b: int) -> int:
            return abs(a // n - b // n) + abs(a % n - b % n)

        def heuristic(state: _State) -> tuple[int, int]:
            remaining = total - state.visited.bit_count()
            chain, here = 0, state.head
            for target in checkpoints[state.next_checkpoint :]:
                chain += manhattan(here, target)
                here = target
            next_distance = (
                manhattan(state.head, checkpoints[state.next_checkpoint])
                if state.next_checkpoint < len(checkpoints)
                else 0
            )
            return max(remaining, chain), next_distance

        def feasible(head: int, visited: int) -> bool:
            unvisited = full & ~visited
            if not unvisited:
                return True
            # Degree check: a path's interior cells need two usable neighbours.
            cell_bits = unvisited
            while cell_bits:
                low = cell_bits & -cell_bits
                cell = low.bit_length() - 1
                cell_bits ^= low
                degree = sum(1 for nb in adjacency[cell] if nb == head or (unvisited >> nb) & 1)
                if degree < (1 if cell == end else 2):
                    return False
            # Flood fill from the head across unvisited cells must reach all of them.
            seen, stack = 0, [head]
            while stack:
                cell = stack.pop()
                for nb in adjacency[cell]:
                    bit = 1 << nb
                    if unvisited & bit and not seen & bit:
                        seen |= bit
                        stack.append(nb)
            return seen == unvisited

        start = _State(head=checkpoints[0], visited=1 << checkpoints[0], next_checkpoint=1, g=0, parent=None)
        tie = count()
        h, d = heuristic(start)
        frontier: list[tuple[int, int, int, int, _State]] = [(h, 0, d, next(tie), start)]
        while frontier:
            *_, state = heapq.heappop(frontier)
            self.nodes_expanded += 1
            if state.visited == full:
                return self._reconstruct(state, n, board_state)
            if state.head == end:
                continue
            for nb in adjacency[state.head]:
                bit = 1 << nb
                if state.visited & bit:
                    continue
                order = checkpoint_order.get(nb)
                if order is not None and order != state.next_checkpoint:
                    continue
                visited = state.visited | bit
                if not feasible(nb, visited):
                    continue
                child = _State(
                    head=nb,
                    visited=visited,
                    next_checkpoint=state.next_checkpoint + (order is not None),
                    g=state.g + 1,
                    parent=state,
                )
                h, d = heuristic(child)
                heapq.heappush(frontier, (child.g + h, -child.g, d, next(tie), child))
        raise UnsolvableError("no path visits every cell once with the checkpoints in order")

    def _reconstruct(self, state: _State, n: int, board: ZipBoard) -> list[Cell]:
        indices: list[int] = []
        node: _State | None = state
        while node is not None:
            indices.append(node.head)
            node = node.parent
        path = [divmod(index, n) for index in reversed(indices)]
        labels = {cell: i + 1 for i, cell in enumerate(board.checkpoints)}
        self.explainer.add("Start", f"Begin on checkpoint 1 at {path[0]}", path[0])
        moves_since = 0
        for cell in path[1:]:
            moves_since += 1
            if cell in labels:
                self.explainer.add(
                    "Checkpoint",
                    f"Reach checkpoint {labels[cell]} at {cell} after {moves_since} move(s), "
                    "keeping every unvisited cell connected and of degree >= 2",
                    cell,
                )
                moves_since = 0
        return path


def is_valid_solution(board: ZipBoard, path: list[Cell]) -> bool:
    """Check that ``path`` is a wall-respecting Hamiltonian path through the checkpoints in order."""
    n = board.size
    if len(path) != n * n or len(set(path)) != n * n:
        return False
    if any(b not in board.neighbours(a) for a, b in zip(path, path[1:])):
        return False
    positions = [path.index(cell) for cell in board.checkpoints]
    return positions == sorted(positions) and positions[0] == 0 and positions[-1] == n * n - 1
