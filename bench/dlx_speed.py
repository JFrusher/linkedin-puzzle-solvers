"""Python vs Rust Dancing Links on the same exact-cover rows.

    pip install ./rust/dlx && python -m bench.dlx_speed

Each case builds the matrix and runs the search to exhaustion (every cover, which is also how a solver proves
a board has one answer). Both engines expand exactly the same nodes, so the ratio is pure engine speed.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import dlx_rs

from bench import real, synthetic
from core.dlx import DancingLinks

Rows = tuple[int, int, list[list[int]]]


def queens_rows(regions: tuple[tuple[int, ...], ...]) -> Rows:
    """The LinkedIn Queens matrix, built as solvers/queens/solver.py builds it."""
    n = len(regions)
    rows = []
    for r in range(n):
        for c in range(n):
            windows = [
                3 * n + wr * (n - 1) + wc
                for wr in (r - 1, r)
                for wc in (c - 1, c)
                if 0 <= wr < n - 1 and 0 <= wc < n - 1
            ]
            rows.append([r, n + c, 2 * n + regions[r][c], *windows])
    return 3 * n, (n - 1) ** 2, rows


def patches_rows(board) -> Rows:  # type: ignore[no-untyped-def]
    from solvers.patches.solver import candidates

    n = board.size
    index = {clue: i for i, clue in enumerate(board.clues)}
    rows = [
        [r * n + c for r, c in patch.cells] + [n * n + index[clue]]
        for clue in board.clues
        for patch in candidates(board, clue)
    ]
    return n * n + len(board.clues), 0, rows


def classic_queens_rows(n: int) -> Rows:
    """Classic N-queens: rows and columns exactly once, both diagonals at most once. Many answers, deep search."""
    rows = [[r, n + c, 2 * n + r + c, 2 * n + (2 * n - 1) + (r - c + n - 1)] for r in range(n) for c in range(n)]
    return 2 * n, 2 * (2 * n - 1), rows


def run(engine: type, case: Rows) -> tuple[int, int]:
    primary, secondary, rows = case
    links = engine(primary, secondary)
    for row_id, columns in enumerate(rows):
        links.add_row(row_id, columns)
    covers = sum(1 for _ in links.search())
    return covers, links.nodes_expanded


def timed(engine: type, case: Rows, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(engine, case)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def cases() -> list[tuple[str, Callable[[], Rows]]]:
    return [
        *[
            (f"LinkedIn Queens {n}×{n}", lambda n=n: queens_rows(synthetic.generate_queens(n, n)[0].regions))
            for n in (8, 10, 12)
        ],
        ("LinkedIn Queens, real screenshot", lambda: queens_rows(real.QUEENS.regions)),
        ("Patches, real screenshot", lambda: patches_rows(real.PATCHES)),
        ("Patches 7×7", lambda: patches_rows(synthetic.generate_patches(7, 7)[0])),
        *[(f"Classic {n}-queens, all answers", lambda n=n: classic_queens_rows(n)) for n in (8, 10, 12)],
    ]


def main() -> None:
    print("| Case | Covers | Nodes | Python | Rust | Speed-up |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, make in cases():
        case = make()
        covers, nodes = run(DancingLinks, case)
        if run(dlx_rs.DancingLinks, case) != (covers, nodes):
            raise AssertionError(f"{name}: the engines disagree")
        repeats = 3 if nodes > 100_000 else 15
        python, rust = timed(DancingLinks, case, repeats), timed(dlx_rs.DancingLinks, case, repeats)
        timings = f"{python * 1e3:,.2f} ms | {rust * 1e3:,.3f} ms | {python / rust:,.0f}×"
        print(f"| {name} | {covers:,} | {nodes:,} | {timings} |")


if __name__ == "__main__":
    main()
