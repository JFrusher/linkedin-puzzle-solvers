"""Latency, memory and accuracy benchmarks over seeded synthetic boards."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from bench import real, synthetic
from core.nlp import word_forms
from solvers import GAMES, get_game
from solvers.crossclimb.solver import is_valid_ladder
from solvers.patches.solver import is_valid_solution as patches_valid
from solvers.queens.solver import is_valid_solution as queens_valid
from solvers.sudoku.solver import is_valid_solution as sudoku_valid
from solvers.tango.solver import is_valid_solution as tango_valid
from solvers.wend.solver import MIN_WORD_LENGTH
from solvers.wend.solver import is_valid_solution as wend_valid
from solvers.wend.vision import WendVisionParser
from solvers.zip.solver import is_valid_solution as zip_valid


@dataclass(frozen=True)
class Case:
    name: str
    board: Any
    image: np.ndarray | None
    """Rendered screenshot, or ``None`` when the game has no vision stage."""
    parsed_ok: Callable[[Any], bool]
    solved_ok: Callable[[Any], bool]


@dataclass(frozen=True)
class BenchRow:
    game: str
    case: str
    parsed_ok: bool | None
    solved_ok: bool
    median_latency_us: float
    peak_memory_kb: float
    nodes_expanded: int


def same_partition(a: tuple[tuple[int, ...], ...], b: tuple[tuple[int, ...], ...]) -> bool:
    """True when two region maps group cells identically, whatever the region ids."""
    pairs = {(x, y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b)}
    return len(a) == len(b) and len(pairs) == len({x for x, _ in pairs}) == len({y for _, y in pairs})


def word_recall(parsed: str, truth: str) -> float:
    truth_words = truth.lower().split()
    parsed_words = set(parsed.lower().split())
    return sum(word in parsed_words for word in truth_words) / len(truth_words)


def cases(game: str) -> Iterator[Case]:
    """Seeded benchmark cases for ``game``."""
    if game == "queens":
        for n in (6, 8, 10, 12):
            board, _ = synthetic.generate_queens(n, seed=n)
            yield Case(
                f"{n}x{n}",
                board,
                synthetic.render_queens(board, seed=n),
                lambda parsed, b=board: same_partition(parsed.regions, b.regions),
                lambda solution, b=board: queens_valid(b, solution),
            )
    elif game == "zip":
        for n in (5, 6, 7):
            board, _ = synthetic.generate_zip(n, seed=n)
            yield Case(
                f"{n}x{n}",
                board,
                synthetic.render_zip(board, seed=n),
                lambda parsed, b=board: parsed == b,
                lambda solution, b=board: zip_valid(b, solution),
            )
    elif game == "tango":
        for seed in (1, 2, 3):
            board, _ = synthetic.generate_tango(6, seed=seed)
            yield Case(
                f"6x6 seed {seed}",
                board,
                synthetic.render_tango(board, seed=seed),
                lambda parsed, b=board: (
                    dict(parsed.givens) == dict(b.givens) and set(parsed.constraints) == set(b.constraints)
                ),
                lambda solution, b=board: tango_valid(b, solution),
            )
    elif game == "wend":
        board = synthetic.WEND_SAMPLE
        yield Case(
            "5x5 sample",
            board,
            synthetic.render_wend(board),
            lambda parsed: parsed == board,
            lambda solution: wend_valid(board, solution, word_forms(MIN_WORD_LENGTH)),
        )
    elif game == "crossclimb":
        board = synthetic.CROSSCLIMB_SAMPLE
        yield Case(
            "4-letter ladder",
            board,
            synthetic.render_crossclimb(board),
            lambda parsed: (
                parsed.word_length == board.word_length
                and word_recall(parsed.end_clue, board.end_clue) >= 0.8
                and all(word_recall(p, t) >= 0.8 for p, t in zip(parsed.clues, board.clues))
            ),
            lambda rungs: (
                is_valid_ladder(board, rungs)
                and tuple(r.word for r in rungs) in (synthetic.CROSSCLIMB_ANSWER, synthetic.CROSSCLIMB_ANSWER[::-1])
            ),
        )
    elif game == "mini-sudoku":
        for seed in (1, 2, 3):
            board, _ = synthetic.generate_sudoku(seed)
            yield Case(
                f"6x6 seed {seed}",
                board,
                synthetic.render_sudoku(board, seed=seed),
                lambda parsed, b=board: dict(parsed.givens) == dict(b.givens),
                lambda solution, b=board: sudoku_valid(b, solution),
            )
    elif game == "patches":
        for n in (5, 6, 7):
            board, _ = synthetic.generate_patches(n, seed=n)
            yield Case(
                f"{n}x{n}",
                board,
                synthetic.render_patches(board, seed=n),
                lambda parsed, b=board: parsed == b,
                lambda solution, b=board: patches_valid(b, solution),
            )
    elif game == "pinpoint":
        for clues, expected in synthetic.PINPOINT_SAMPLES:
            yield Case(
                expected,
                list(clues),
                None,
                lambda _: True,
                lambda ranking, e=expected: e in [guess.label for guess in ranking[:3]],
            )
    else:
        raise KeyError(f"no benchmark cases for {game!r}")
    if game in REAL_CHECKS:
        board, parsed_ok, solved_ok = REAL_CHECKS[game]
        yield Case("real screenshot", board, real.image(game), parsed_ok, solved_ok)


REAL_CHECKS: dict[str, tuple[Any, Callable[[Any], bool], Callable[[Any], bool]]] = {
    "queens": (
        real.QUEENS,
        lambda parsed: same_partition(parsed.regions, real.QUEENS.regions),
        lambda solution: queens_valid(real.QUEENS, solution),
    ),
    "tango": (
        real.TANGO,
        lambda parsed: (
            dict(parsed.givens) == dict(real.TANGO.givens) and set(parsed.constraints) == set(real.TANGO.constraints)
        ),
        lambda solution: tango_valid(real.TANGO, solution),
    ),
    "zip": (real.ZIP, lambda parsed: parsed == real.ZIP, lambda solution: zip_valid(real.ZIP, solution)),
    "mini-sudoku": (
        real.SUDOKU,
        lambda parsed: dict(parsed.givens) == dict(real.SUDOKU.givens),
        lambda solution: sudoku_valid(real.SUDOKU, solution),
    ),
    "patches": (
        real.PATCHES,
        lambda parsed: parsed == real.PATCHES,
        lambda solution: patches_valid(real.PATCHES, solution),
    ),
    # The word lengths are in a separate screenshot, read here so the whole board is compared.
    "wend": (
        real.WEND,
        lambda parsed: (
            replace(parsed, word_lengths=WendVisionParser().slot_lengths(real.image("wend-slots"))) == real.WEND
        ),
        lambda solution: sorted(p.word for p in solution) == sorted(real.WEND_ANSWER),
    ),
}
"""Real LinkedIn screenshots: the board the solver gets, the parse check and the answer check."""


def benchmark_game(name: str, repeats: int) -> list[BenchRow]:
    """Parse each case's render once, then time ``repeats`` solves without memory tracing plus one with."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    game = get_game(name)
    solver = game.solver()
    rows = []
    for case in cases(name):
        parsed_ok: bool | None = None
        if case.image is not None and game.parser is not None:
            parsed_ok = case.parsed_ok(game.parser().parse(case.image))
        solver.run(case.board, track_memory=False)  # warm-up: model loads and caches stay out of timings
        latencies = [solver.run(case.board, track_memory=False).profile.latency_us for _ in range(repeats)]
        traced = solver.run(case.board, track_memory=True)
        rows.append(
            BenchRow(
                game=name,
                case=case.name,
                parsed_ok=parsed_ok,
                solved_ok=case.solved_ok(traced.solution),
                median_latency_us=statistics.median(latencies),
                peak_memory_kb=traced.profile.peak_memory_kb,
                nodes_expanded=traced.profile.nodes_expanded,
            )
        )
    return rows


def run(games: list[str], repeats: int) -> list[BenchRow]:
    return [row for game in games for row in benchmark_game(game, repeats)]


def _mark(value: bool | None) -> str:
    return "—" if value is None else ("✅" if value else "❌")


def to_markdown(rows: list[BenchRow]) -> str:
    lines = [
        "| Game | Case | Vision | Solved | Median latency (µs) | Peak memory (KiB) | Nodes |",
        "|---|---|:---:|:---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.game} | {row.case} | {_mark(row.parsed_ok)} | {_mark(row.solved_ok)} | "
            f"{row.median_latency_us:,.0f} | {row.peak_memory_kb:,.1f} | {row.nodes_expanded:,} |"
        )
    return "\n".join(lines)


def to_json(rows: list[BenchRow]) -> str:
    return json.dumps([asdict(row) for row in rows], indent=2)


def print_table(rows: list[BenchRow], console: Console | None = None) -> None:
    table = Table(title="LinkedIn Puzzle Solvers — benchmark")
    for column, justify in (
        ("Game", "left"),
        ("Case", "left"),
        ("Vision", "center"),
        ("Solved", "center"),
        ("Median µs", "right"),
        ("Peak KiB", "right"),
        ("Nodes", "right"),
    ):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]
    for row in rows:
        table.add_row(
            row.game,
            row.case,
            _mark(row.parsed_ok),
            _mark(row.solved_ok),
            f"{row.median_latency_us:,.0f}",
            f"{row.peak_memory_kb:,.1f}",
            f"{row.nodes_expanded:,}",
        )
    (console or Console()).print(table)


ALL_GAMES = list(GAMES)
