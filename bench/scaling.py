"""How solve latency and search effort grow with board size: ``python -m bench.scaling``.

LinkedIn ships fixed sizes (Queens 7-10, Zip 6-7, Tango 6, Patches 5-7), so larger boards come from the seeded
generators in ``bench.synthetic``. Writes ``docs/scaling.svg`` and prints the numbers as a Markdown table.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench import synthetic  # noqa: E402
from core.solver_base import BaseSolver  # noqa: E402
from solvers.patches.solver import PatchesSolver  # noqa: E402
from solvers.queens.solver import QueensSolver  # noqa: E402
from solvers.tango.solver import TangoSolver  # noqa: E402
from solvers.zip.solver import ZipSolver  # noqa: E402

SEEDS = range(5)
CHART = Path(__file__).resolve().parent.parent / "docs" / "scaling.svg"

# Sizes stop where a point takes minutes: generating five Queens 14x14 boards takes 8 s and Patches 13x13 18 s,
# and one Zip 12x12 board (sparse_zip seed 3) costs A* 3.2 million nodes and 10 minutes.
GAMES: dict[str, tuple[type[BaseSolver[Any, Any]], Callable[[int, int], Any], range]] = {
    "Queens (DLX)": (QueensSolver, lambda n, seed: synthetic.generate_queens(n, seed)[0], range(5, 14)),
    "Patches (DLX)": (PatchesSolver, lambda n, seed: synthetic.generate_patches(n, seed)[0], range(5, 13)),
    "Zip (A*)": (ZipSolver, synthetic.sparse_zip, range(5, 12)),
    "Tango (SAT)": (TangoSolver, lambda n, seed: synthetic.generate_tango(n, seed)[0], range(6, 13, 2)),
}


Row = tuple[int, float, float, float]
"""(N, median latency in ms, worst latency in ms, median nodes expanded) over ``SEEDS``."""


def measure() -> dict[str, list[Row]]:
    table: dict[str, list[Row]] = {}
    for game, (solver, generate, sizes) in GAMES.items():
        table[game] = []
        for n in sizes:
            profiles = [solver().run(generate(n, seed), track_memory=False).profile for seed in SEEDS]
            latencies = [p.latency_us / 1000 for p in profiles]
            nodes = statistics.median(p.nodes_expanded for p in profiles)
            table[game].append((n, statistics.median(latencies), max(latencies), nodes))
    return table


def draw(table: dict[str, list[Row]]) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for game, rows in table.items():
        sizes = [n for n, *_ in rows]
        (line,) = left.plot(sizes, [row[1] for row in rows], marker="o", label=game)
        left.plot(sizes, [row[2] for row in rows], linestyle=":", color=line.get_color())  # worst seed
        right.plot(sizes, [row[3] for row in rows], marker="o", label=game)
    labels = {left: "solve latency (ms): median, dotted = worst seed", right: "median nodes expanded (SAT: decisions)"}
    for axis, label in labels.items():
        axis.set_yscale("log")
        axis.set_xlabel("board size N (N x N)")
        axis.set_ylabel(label)
        axis.grid(True, which="major", alpha=0.3)
    left.legend(frameon=False)
    figure.suptitle(f"Scaling past LinkedIn's sizes (median of {len(SEEDS)} seeded boards per point)")
    figure.tight_layout()
    figure.savefig(CHART, metadata={"Date": None})
    plt.close(figure)


def main() -> None:
    table = measure()
    draw(table)
    print("| Game | N | Median ms | Worst ms | Median nodes |")
    print("|---|---:|---:|---:|---:|")
    for game, rows in table.items():
        for n, median, worst, nodes in rows:
            print(f"| {game} | {n} | {median:,.1f} | {worst:,.1f} | {nodes:,.0f} |")
    print(f"\nwrote {CHART}")


if __name__ == "__main__":
    main()
