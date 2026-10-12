"""Play app: level settings, the puzzle pool and the local server (plus the JS rule tests when node is installed)."""

import http.client
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import pytest

from play.levels import LEVELS, LevelSpec, build_puzzle, distance, level_of
from play.pool import Pool
from play.server import make_server
from solvers import get_game

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("game", "level"),
    [(game, level) for game, levels in LEVELS.items() for level in levels if game != "wend"] + [("wend", "easy")],
)
def test_every_level_makes_a_board_with_one_answer(game: str, level: str) -> None:
    board = LEVELS[game][level].make(3)
    assert len(list(islice(get_game(game).solver().solutions(board), 2))) == 1


def test_puzzle_carries_board_answer_steps_and_rating() -> None:
    puzzle = build_puzzle("queens", "easy", LEVELS["queens"]["easy"].make(1))
    assert set(puzzle) == {"game", "level", "rating", "board", "solution", "steps"}
    assert len(puzzle["solution"]) == puzzle["board"]["size"]
    assert puzzle["rating"].split(":")[0] in {"Trivial", "Easy", "Medium", "Hard", "Expert"}
    assert build_puzzle("zip", "hard", LEVELS["zip"]["hard"].make(1))["rating"] == "7×7"


def test_ratings_map_to_levels() -> None:
    assert [level_of(f"{band}: x") for band in ("Trivial", "Easy", "Medium", "Hard", "Expert")] == [
        "easy",
        "easy",
        "medium",
        "hard",
        "hard",
    ]
    assert distance("Medium: x", "medium") == 0
    assert distance("Easy: x", "hard") == 2


def stub_pool(bands: list[str], **kwargs: Any) -> Pool:
    """A pool whose 'generator' yields boards rated, in turn, as ``bands`` (queens is rated, so ratings count)."""
    ratings = iter(bands)
    spec = LevelSpec(lambda seed: next(ratings), "stub")
    levels = {"queens": {"easy": spec, "medium": spec, "hard": spec}}
    return Pool(
        levels=levels, build=lambda game, level, band: {"game": game, "level": level, "rating": f"{band}: x"}, **kwargs
    )


def test_take_returns_a_board_at_the_level_and_files_the_misses() -> None:
    pool = stub_pool(["Easy", "Medium", "Hard"])
    assert pool.take("queens", "hard")["rating"] == "Hard: x"
    assert [p["rating"] for p in pool.store[("queens", "easy")]] == ["Easy: x"]
    assert pool.take("queens", "medium")["rating"] == "Medium: x"


def test_unreachable_level_falls_back_to_the_closest_rating() -> None:
    pool = stub_pool(["Easy", "Medium"] * 10, tries=20, target=0)
    assert pool.take("queens", "hard")["rating"] == "Medium: x"


def test_fallback_hands_over_a_board_already_filed_elsewhere() -> None:
    pool = stub_pool(["Easy"], tries=1)
    assert pool.take("queens", "hard")["rating"] == "Easy: x"
    assert not pool.store[("queens", "easy")], "the board was handed over, not copied"


def test_generator_that_always_fails_is_reported() -> None:
    def broken(seed: int) -> None:
        raise ValueError("no board")

    spec = LevelSpec(broken, "stub")
    pool = Pool(levels={"queens": {"easy": spec, "medium": spec, "hard": spec}}, tries=3)
    with pytest.raises(RuntimeError, match="could not generate"):
        pool.take("queens", "easy")


@pytest.fixture
def server() -> Iterator[tuple[str, int]]:
    httpd = make_server("127.0.0.1", 0, stub_pool(["Easy"] * 50))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[:2]  # type: ignore[misc]
    httpd.shutdown()
    httpd.server_close()


def get(address: tuple[str, int], path: str) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection(*address, timeout=10)
    connection.request("GET", path)
    response = connection.getresponse()
    return response.status, response.getheader("Content-Type", ""), response.read()


def test_server_serves_puzzles_and_rejects_bad_requests(server: tuple[str, int]) -> None:
    status, kind, body = get(server, "/api/puzzle?game=queens&level=easy")
    assert (status, kind) == (200, "application/json")
    assert json.loads(body)["rating"] == "Easy: x"
    assert get(server, "/api/puzzle?game=pinpoint&level=easy")[0] == 400
    assert get(server, "/api/puzzle?game=queens&level=extreme")[0] == 400


def test_server_serves_the_page_and_nothing_outside_it(server: tuple[str, int]) -> None:
    status, kind, body = get(server, "/")
    assert status == 200 and kind.startswith("text/html") and b"Puzzle Play" in body
    assert get(server, "/games/queens.js")[1].startswith("text/javascript")
    assert get(server, "/../levels.py")[0] == 404
    assert get(server, "/%2e%2e/levels.py")[0] == 404
    assert get(server, "/missing.js")[0] == 404


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_game_rules_in_javascript() -> None:
    result = subprocess.run(
        ["node", "--test", "tests/play_js/games.test.mjs"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
