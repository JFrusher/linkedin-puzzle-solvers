"""Zip as a Hamiltonian path through ordered checkpoints, solved with A* search.

State: (head cell, visited bitmask, next checkpoint index). Every move costs 1
and a complete path has exactly N^2 - 1 moves, so the admissible heuristic
``h = max(unvisited cells, Manhattan chain through the remaining checkpoints)``
makes ``f = g + h`` exact for every feasible state. Ties are broken by depth
(deepest first, i.e. backtracking order) and then by Manhattan distance to the
next checkpoint. Infeasible states are pruned before they enter the queue:

* stepping on a checkpoint out of order;
* unvisited cells no longer connected among themselves, or not touching the head;
* a bottleneck the path cannot pass: the end is a cut cell, a cut cell splits off two parts, or the
  part split off away from the end does not touch the head (Tarjan cut cells, DFS rooted at the end);
* an unvisited cell other than the final checkpoint left with fewer than two
  usable neighbours (degree-constrained subgraph check);
* the next checkpoint no longer reachable without stepping on a later one.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
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
        path = next(self.solutions(board_state), None)
        if path is None:
            raise UnsolvableError("no path visits every cell once with the checkpoints in order")
        return path

    def solutions(self, board_state: ZipBoard) -> Iterator[list[Cell]]:
        """Yield every valid path in A* order; take two to test uniqueness."""
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

        # later[k]: checkpoints after the k-th, which the path may not touch while heading for the k-th.
        later = [sum(1 << cp for cp in checkpoints[k + 1 :]) for k in range(len(checkpoints))]

        def bottlenecks_ok(unvisited: int, starts: list[int]) -> bool:
            """Tarjan DFS from the end over the unvisited cells.

            Fails when the cells are disconnected, when the end is a cut cell (the path would have to leave
            it), or when a cut cell splits off a part that the path cannot enter first: the part must be
            the one away from the end, and the head must touch it.
            """
            order = {end: 0}
            low = {end: 0}
            size = {end: 1}
            root_children = 0
            split: set[int] = set()
            stack = [(end, -1, iter(adjacency[end]))]
            while stack:
                cell, parent, neighbours = stack[-1]
                advanced = False
                for nb in neighbours:
                    if not (unvisited >> nb) & 1 or nb == parent:
                        continue
                    if nb in order:
                        low[cell] = min(low[cell], order[nb])
                        continue
                    order[nb] = low[nb] = len(order)
                    size[nb] = 1
                    stack.append((nb, cell, iter(adjacency[nb])))
                    advanced = True
                    break
                if advanced:
                    continue
                stack.pop()
                if parent < 0:
                    continue
                low[parent] = min(low[parent], low[cell])
                size[parent] += size[cell]
                if parent == end:
                    root_children += 1
                elif low[cell] >= order[parent]:
                    # ``parent`` is a cut cell and ``cell``'s subtree lies on the side away from the end.
                    if parent in split:
                        return False  # a second split-off part: the path cannot cover three parts
                    split.add(parent)
                    first, last = order[cell], order[cell] + size[cell]
                    if not any(nb in order and first <= order[nb] < last for nb in starts):
                        return False
            return len(order) == unvisited.bit_count() and root_children <= 1

        def feasible(head: int, visited: int, next_checkpoint: int) -> bool:
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
            # After this move the path never returns to the head, so the rest of the path runs from a cell
            # touching the head, through every unvisited cell, to the end.
            starts = [nb for nb in adjacency[head] if (unvisited >> nb) & 1]
            if not starts or not bottlenecks_ok(unvisited, starts):
                return False
            # Checkpoint order: the next checkpoint must be reachable without stepping on a later one.
            if next_checkpoint == len(checkpoints):
                return True
            target = 1 << checkpoints[next_checkpoint]
            allowed = unvisited & ~later[next_checkpoint]
            seen, stack = 0, [head]
            while stack:
                cell = stack.pop()
                for nb in adjacency[cell]:
                    bit = 1 << nb
                    if allowed & bit and not seen & bit:
                        if bit == target:
                            return True
                        seen |= bit
                        stack.append(nb)
            return False

        # Repeated-state skipping: what can follow a state depends only on (head, visited, next checkpoint), so
        # each key is expanded once. An answer found through one arrival is also an answer through every other
        # arrival at the same key, so those answers are rebuilt by splicing instead of searched again.
        expanded: set[tuple[int, int, int]] = set()
        skipped: dict[tuple[int, int, int], list[_State]] = {}
        suffixes: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}
        # Keyed by id() but holding the state, so a finished state's id cannot be reused by a new one.
        spliced: dict[tuple[int, tuple[int, ...]], _State] = {}
        found: set[tuple[int, ...]] = set()

        def key(state: _State) -> tuple[int, int, int]:
            return state.head, state.visited, state.next_checkpoint

        def record(state: _State, tail: tuple[int, ...]) -> Iterator[list[Cell]]:
            """``state`` reaches the end through ``tail``; yield every answer that implies, once each."""
            if (id(state), tail) in spliced:
                return
            spliced[id(state), tail] = state
            pending: list[tuple[_State, tuple[int, ...]]] = []
            node: _State | None = state
            while node is not None:
                suffixes.setdefault(key(node), []).append(tail)
                pending += [(other, tail) for other in skipped.get(key(node), [])]
                tail = (node.head, *tail)
                node = node.parent
            if tail not in found:
                found.add(tail)
                yield [divmod(index, n) for index in tail]
            for other, other_tail in pending:
                yield from record(other, other_tail)

        start = _State(head=checkpoints[0], visited=1 << checkpoints[0], next_checkpoint=1, g=0, parent=None)
        tie = count()
        h, d = heuristic(start)
        frontier: list[tuple[int, int, int, int, int, _State]] = [(h, 0, 0, d, next(tie), start)]
        while frontier:
            *_, state = heapq.heappop(frontier)
            state_key = key(state)
            if state_key in expanded:
                skipped.setdefault(state_key, []).append(state)
                for tail in list(suffixes.get(state_key, [])):
                    yield from record(state, tail)
                continue
            expanded.add(state_key)
            self.nodes_expanded += 1
            if state.visited == full:
                yield from record(state, ())
                continue
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
                if not feasible(nb, visited, state.next_checkpoint + (order is not None)):
                    continue
                child = _State(
                    head=nb,
                    visited=visited,
                    next_checkpoint=state.next_checkpoint + (order is not None),
                    g=state.g + 1,
                    parent=state,
                )
                h, d = heuristic(child)
                # Warnsdorff: among equal-depth siblings, try the cell with the fewest onward moves first.
                onward = sum(1 for nxt in adjacency[nb] if not (visited >> nxt) & 1)
                heapq.heappush(frontier, (child.g + h, -child.g, onward, d, next(tie), child))

    def explain(self, board_state: ZipBoard, solution: list[Cell]) -> None:
        path = solution
        labels = {cell: i + 1 for i, cell in enumerate(board_state.checkpoints)}
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


def is_valid_solution(board: ZipBoard, path: list[Cell]) -> bool:
    """Check that ``path`` is a wall-respecting Hamiltonian path through the checkpoints in order."""
    n = board.size
    if len(path) != n * n or len(set(path)) != n * n:
        return False
    if any(b not in board.neighbours(a) for a, b in zip(path, path[1:])):
        return False
    positions = [path.index(cell) for cell in board.checkpoints]
    return positions == sorted(positions) and positions[0] == 0 and positions[-1] == n * n - 1
