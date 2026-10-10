"""Knuth's Algorithm X on Dancing Links, shared by every exact-cover game (Queens, Patches)."""

from __future__ import annotations

from collections.abc import Iterator


class Node:
    """A Dancing Links node; column headers additionally track ``size``."""

    __slots__ = ("left", "right", "up", "down", "column", "row_id", "size", "name")

    def __init__(self, column: Node | None = None, row_id: int = -1, name: str = "") -> None:
        self.left: Node = self
        self.right: Node = self
        self.up: Node = self
        self.down: Node = self
        self.column: Node = column if column is not None else self
        self.row_id = row_id
        self.size = 0
        self.name = name


class DancingLinks:
    """Toroidal doubly-linked sparse matrix for Algorithm X."""

    def __init__(self, primary: int, secondary: int) -> None:
        """Columns ``0..primary-1`` must be covered exactly once, the next ``secondary`` at most once."""
        self.root = Node(name="root")
        self.columns: list[Node] = []
        for index in range(primary):
            column = Node(name=str(index))
            column.left, column.right = self.root.left, self.root
            self.root.left.right = column
            self.root.left = column
            self.columns.append(column)
        # Secondary headers stay out of the root list, so search never requires them.
        self.columns.extend(Node(name=str(primary + index)) for index in range(secondary))
        self.nodes_expanded = 0

    def add_row(self, row_id: int, column_ids: list[int]) -> None:
        """Append a matrix row with 1s in ``column_ids``."""
        first: Node | None = None
        for column_id in column_ids:
            column = self.columns[column_id]
            node = Node(column=column, row_id=row_id)
            node.up, node.down = column.up, column
            column.up.down = node
            column.up = node
            column.size += 1
            if first is None:
                first = node
            else:
                node.left, node.right = first.left, first
                first.left.right = node
                first.left = node

    @staticmethod
    def cover(column: Node) -> None:
        column.right.left = column.left
        column.left.right = column.right
        row = column.down
        while row is not column:
            node = row.right
            while node is not row:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row = row.down

    @staticmethod
    def uncover(column: Node) -> None:
        row = column.up
        while row is not column:
            node = row.left
            while node is not row:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        column.right.left = column
        column.left.right = column

    def search(self, partial: list[int] | None = None) -> Iterator[list[int]]:
        """Yield every exact cover as a list of row ids (Knuth's Algorithm X)."""
        partial = [] if partial is None else partial
        if self.root.right is self.root:
            yield list(partial)
            return
        # Minimum-remaining-values heuristic: branch on the tightest column.
        column = self.root.right
        best = column
        while column is not self.root:
            if column.size < best.size:
                best = column
            column = column.right
        if best.size == 0:
            return
        self.cover(best)
        row = best.down
        while row is not best:
            self.nodes_expanded += 1
            partial.append(row.row_id)
            node = row.right
            while node is not row:
                self.cover(node.column)
                node = node.right
            yield from self.search(partial)
            node = row.left
            while node is not row:
                self.uncover(node.column)
                node = node.left
            partial.pop()
            row = row.down
        self.uncover(best)
