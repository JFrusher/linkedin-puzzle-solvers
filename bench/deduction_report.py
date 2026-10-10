"""Measure how far human-style rules carry each solver, over many one-answer boards.

    python -m bench.deduction_report

Queens and Tango: share of explanation steps that fall back to search (``Search`` / ``SAT``).
Zip: nodes expanded and solve latency, whose spread shows where pruning is weak.
"""

from __future__ import annotations

import statistics

from bench import synthetic
from solvers.queens.solver import QueensSolver
from solvers.tango.solver import TangoSolver
from solvers.zip.solver import ZipSolver

SEEDS = range(20)


def guess_share(solver_cls: type, boards: list, guess_rule: str) -> str:
    steps = [step for board in boards for step in solver_cls().run(board, track_memory=False).explanation.steps]
    counted = [step for step in steps if step.rule != "Given"]
    return f"{sum(step.rule == guess_rule for step in counted) / len(counted):.0%}"


def main() -> None:
    print("| Game | Size | Metric | Value |")
    print("|---|---|---|---|")
    for n in (7, 8, 9, 10):
        boards = [synthetic.generate_queens(n, seed)[0] for seed in SEEDS]
        print(f"| Queens | {n}x{n} | `Search` steps | {guess_share(QueensSolver, boards, 'Search')} |")
    boards = [synthetic.generate_tango(6, seed)[0] for seed in SEEDS]
    print(f"| Tango | 6x6 | `SAT` steps | {guess_share(TangoSolver, boards, 'SAT')} |")
    for n in (5, 6, 7):
        runs = [ZipSolver().run(synthetic.generate_zip(n, seed)[0], track_memory=False).profile for seed in range(10)]
        nodes = [run.nodes_expanded for run in runs]
        micros = [run.latency_us for run in runs]
        spread = (
            f"{statistics.median(nodes):,.0f} / {max(nodes):,} · {statistics.median(micros):,.0f} / {max(micros):,.0f}"
        )
        print(f"| Zip | {n}x{n} | nodes median / max · µs median / max | {spread} |")


if __name__ == "__main__":
    main()
