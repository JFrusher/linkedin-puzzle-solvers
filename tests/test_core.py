"""Core framework: profiler, explainer, grid rectification and the CLI/REST surface."""

import json

import numpy as np
import pytest

from bench.synthetic import generate_queens, render_queens
from core.explainer import Explainer
from core.solver_base import BaseSolver
from core.vision_base import BaseVisionParser, GridNotFoundError
from main import build_parser, to_jsonable
from solvers.queens.vision import QueensVisionParser
from solvers.tango.solver import Symbol


def test_profile_measures_latency_and_memory() -> None:
    with BaseSolver.profile(track_memory=True) as record:
        _ = [0] * 100_000
    assert record.latency_us > 0 and record.peak_memory_kb > 100


def test_explainer_serialises() -> None:
    explainer = Explainer()
    explainer.add("Single Choice", "Cell (3, 4) must be a Queen | forced", (3, 4))
    assert json.loads(explainer.to_json())[0]["cell"] == [3, 4]
    assert "\\|" in explainer.to_markdown()
    assert str(explainer.steps[0]).startswith("Step 1 [Single Choice]")


def test_rectify_and_slice() -> None:
    board, _ = generate_queens(7, seed=1)
    parser = QueensVisionParser()
    grid = parser.rectify_grid(render_queens(board, seed=1))
    assert parser.detect_grid_size(grid.image) == 7
    cells = parser.extract_cells(grid.image, 7)
    assert len(cells) == 7 and all(len(row) == 7 for row in cells)


def test_blank_image_has_no_grid() -> None:
    with pytest.raises(GridNotFoundError):
        QueensVisionParser().rectify_grid(np.full((300, 300, 3), 255, dtype=np.uint8))


def test_load_image_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        BaseVisionParser.load_image("does/not/exist.png")


def test_to_jsonable_handles_models() -> None:
    assert to_jsonable({(0, 1): Symbol.SUN}) == [{"key": [0, 1], "value": "sun"}]


def test_cli_parses_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["solve", "--game", "queens", "--image", "x.png"]).game == "queens"
    assert parser.parse_args(["benchmark", "--game", "all"]).repeats == 25
    with pytest.raises(SystemExit):
        parser.parse_args(["solve", "--game", "pinpoint", "--image", "x.png"])
