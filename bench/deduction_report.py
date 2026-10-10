"""Measure how far human-style rules carry each solver, over many one-answer boards.

    python -m bench.deduction_report

Queens, Tango, Wend, Mini Sudoku, Patches: share of explanation steps that fall back to ``Search`` / ``SAT``.
Zip: solve latency on sparse boards (few checkpoints, no walls), where weak pruning shows as a long tail.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from typing import Any

from bench import synthetic
from solvers.patches.solver import PatchesSolver
from solvers.queens.solver import QueensSolver
from solvers.sudoku.solver import SudokuSolver
from solvers.tango.solver import TangoSolver
from solvers.wend.solver import WendSolver
from solvers.zip.solver import ZipSolver

SEEDS = range(20)


def guess_share(solver_cls: type, boards: list[Any], guess_rule: str, fills: Callable[[Any], int]) -> str:
    """Share of filled cells decided by ``guess_rule``, plus how often each rule fired."""
    steps = [step for board in boards for step in solver_cls().run(board, track_memory=False).explanation.steps]
    fired = Counter(step.rule for step in steps if step.rule != "Given")
    share = fired[guess_rule] / sum(fills(board) for board in boards)
    return f"{share:.0%} ({', '.join(f'{rule} {count}' for rule, count in fired.most_common())})"


def main() -> None:
    boards: list[Any]
    print("| Game | Size | Metric | Value |")
    print("|---|---|---|---|")
    for n in (7, 8, 9, 10):
        boards = [synthetic.generate_queens(n, seed)[0] for seed in SEEDS]
        print(
            f"| Queens | {n}x{n} | `Search` steps | {guess_share(QueensSolver, boards, 'Search', lambda b: b.size)} |"
        )
    for n in (5, 6):
        boards = [synthetic.generate_wend(n, seed) for seed in SEEDS]
        share = guess_share(WendSolver, boards, "Search", lambda b: len(b.word_lengths))
        print(f"| Wend | {n}x{n} | `Search` words | {share} |")
    boards = [synthetic.generate_tango(6, seed)[0] for seed in SEEDS]
    share = guess_share(TangoSolver, boards, "SAT", lambda b: b.size**2 - len(b.givens))
    print(f"| Tango | 6x6 | `SAT` steps | {share} |")
    boards = [synthetic.generate_sudoku(seed)[0] for seed in SEEDS]
    share = guess_share(SudokuSolver, boards, "SAT", lambda b: 36 - len(b.givens))
    print(f"| Mini Sudoku | 6x6 | `SAT` steps | {share} |")
    for n in (5, 6, 7):
        boards = [synthetic.generate_patches(n, seed)[0] for seed in SEEDS]
        share = guess_share(PatchesSolver, boards, "Search", lambda b: len(b.clues))
        print(f"| Patches | {n}x{n} | `Search` patches | {share} |")
    for n in (6, 7, 8):
        runs = [ZipSolver().run(synthetic.sparse_zip(n, seed), track_memory=False).profile for seed in range(30)]
        micros = sorted(run.latency_us for run in runs)
        spread = f"{statistics.median(micros) / 1000:,.1f} / {micros[26] / 1000:,.1f} / {micros[-1] / 1000:,.1f}"
        print(f"| Zip sparse (4 checkpoints) | {n}x{n} | ms median / p90 / max | {spread} |")


if __name__ == "__main__":
    main()
