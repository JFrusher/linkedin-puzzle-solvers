"""Abstract solver base class with microsecond latency and memory profiling."""

from __future__ import annotations

import threading
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from core.explainer import Explainer

_TRACING = threading.RLock()
"""tracemalloc is process-wide, so memory-profiled solves on different threads (Streamlit reruns) take turns."""

BoardT = TypeVar("BoardT")
SolutionT = TypeVar("SolutionT")


class UnsolvableError(RuntimeError):
    """Raised when a solver proves that a board has no valid solution."""


@dataclass
class ProfileRecord:
    """Measurements captured by :meth:`BaseSolver.profile`."""

    latency_us: float = 0.0
    peak_memory_kb: float = 0.0
    nodes_expanded: int = 0


@dataclass
class SolveResult(Generic[SolutionT]):
    """A solution together with its profiling record and deduction log."""

    solution: SolutionT
    profile: ProfileRecord
    explanation: Explainer = field(default_factory=Explainer)


class BaseSolver(ABC, Generic[BoardT, SolutionT]):
    """Uniform interface for every game solver.

    Subclasses implement :meth:`solve` and increment ``self.nodes_expanded`` for
    every search-tree node they visit; :meth:`run` wraps the call in the
    profiler and returns a :class:`SolveResult`.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.nodes_expanded: int = 0
        self.explainer: Explainer = Explainer()

    @abstractmethod
    def solve(self, board_state: BoardT) -> SolutionT:
        """Solve ``board_state`` or raise :class:`UnsolvableError`."""

    def run(self, board_state: BoardT, track_memory: bool = True) -> SolveResult[SolutionT]:
        """Solve ``board_state`` under the profiler.

        ``track_memory`` enables ``tracemalloc``, which slows Python allocation
        several-fold, so latency benchmarks disable it.
        """
        self.nodes_expanded = 0
        self.explainer = Explainer()
        with self.profile(track_memory) as record:
            solution = self.solve(board_state)
        record.nodes_expanded = self.nodes_expanded
        self.explain(board_state, solution)
        return SolveResult(solution=solution, profile=record, explanation=self.explainer)

    def explain(self, board_state: BoardT, solution: SolutionT) -> None:
        """Replay ``solution`` as human deduction steps, outside the profiled region.

        Solvers that log their reasoning while searching keep this default.
        """

    @staticmethod
    @contextmanager
    def profile(track_memory: bool = True) -> Iterator[ProfileRecord]:
        """Measure wall-clock latency (µs) and, optionally, peak traced memory (KiB)."""
        record = ProfileRecord()
        if track_memory:
            _TRACING.acquire()
            try:
                if tracemalloc.is_tracing():
                    raise RuntimeError("tracemalloc is already tracing; nested memory profiling is not supported")
                tracemalloc.start()
            except BaseException:
                _TRACING.release()
                raise
        start = time.perf_counter_ns()
        try:
            yield record
        finally:
            record.latency_us = (time.perf_counter_ns() - start) / 1_000
            if track_memory:
                try:
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                finally:
                    _TRACING.release()
                record.peak_memory_kb = peak / 1024
