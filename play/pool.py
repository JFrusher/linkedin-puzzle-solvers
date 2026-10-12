"""Boards kept ready per (game, level) by a background thread, so New is instant."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from play.levels import LEVEL_NAMES, LEVELS, build_puzzle, distance, level_of
from solvers import get_game

Key = tuple[str, str]


class Pool:
    def __init__(
        self,
        target: int = 3,
        tries: int = 200,
        budget_s: float = 60.0,
        levels: dict[str, dict[str, Any]] | None = None,
        build: Callable[[str, str, Any], dict[str, Any]] = build_puzzle,
    ) -> None:
        self.target, self.tries, self.budget_s = target, tries, budget_s
        self.levels = LEVELS if levels is None else levels
        self.build = build
        self.store: dict[Key, deque[dict[str, Any]]] = {
            (game, level): deque() for game in self.levels for level in self.levels[game]
        }
        # ponytail: one generation at a time; generators share solver state (Wend's Trie solver), and
        # generation is CPU-bound under the GIL anyway.
        self.generating = threading.Lock()
        self.wake = threading.Event()
        self.seeds = random.SystemRandom()

    def take(self, game: str, level: str) -> dict[str, Any]:
        """A ready puzzle, or one generated now when that level's store is empty."""
        if (game, level) not in self.store:
            raise KeyError(f"unknown game or level: {game!r}, {level!r}")
        try:
            puzzle = self.store[(game, level)].popleft()
        except IndexError:
            puzzle = self.make(game, level)
        self.wake.set()
        return puzzle

    def make(self, game: str, level: str) -> dict[str, Any]:
        """Generate until a board rates at ``level``; boards that land elsewhere top up that level's store.

        After ``tries`` boards or ``budget_s`` seconds, the closest-rated board is returned instead.
        """
        rated = get_game(game).rated
        closest: tuple[int, dict[str, Any]] | None = None
        deadline = time.monotonic() + self.budget_s
        with self.generating:
            for _ in range(self.tries):
                try:
                    board = self.levels[game][level].make(self.seeds.randrange(2**31))
                except ValueError:  # a generator gave up on this seed
                    continue
                puzzle = self.build(game, level, board)
                if not rated:
                    return puzzle
                gap = distance(puzzle["rating"], level)
                if gap == 0:
                    return puzzle
                landed = (game, level_of(puzzle["rating"]))
                if landed in self.store and len(self.store[landed]) < self.target:
                    self.store[landed].append({**puzzle, "level": landed[1]})
                elif closest is None or gap < closest[0]:
                    closest = (gap, puzzle)
                if time.monotonic() > deadline:
                    break
        return closest[1] if closest is not None else self._nearest_ready(game, level)

    def _nearest_ready(self, game: str, level: str) -> dict[str, Any]:
        """Every board landed in another level's store: hand over the nearest level's."""
        here = LEVEL_NAMES.index(level)
        for other in sorted(LEVEL_NAMES, key=lambda name: abs(LEVEL_NAMES.index(name) - here)):
            if self.store.get((game, other)):
                return self.store[(game, other)].popleft()
        raise RuntimeError(f"could not generate a {level} {game} board")

    def fill_forever(self) -> None:
        """Top every store up to ``target``, then sleep until a ``take`` empties one."""
        while True:
            short = [key for key, ready in self.store.items() if len(ready) < self.target]
            if not short:
                self.wake.wait()
                self.wake.clear()
                continue
            for game, level in short:
                try:
                    puzzle = self.make(game, level)
                except RuntimeError:
                    continue
                if len(self.store[(game, level)]) < self.target:
                    self.store[(game, level)].append(puzzle)

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.fill_forever, name="play-pool", daemon=True)
        thread.start()
        return thread
