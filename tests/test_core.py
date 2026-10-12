"""Core framework: profiler, explainer, grid rectification and the CLI/REST surface."""

import http.client
import json
import pathlib
import socket
import threading
from collections.abc import Iterator
from http.server import HTTPServer

import numpy as np
import pytest

import main
from bench.synthetic import generate_queens, render_queens
from core.difficulty import rate
from core.explainer import DeductionStep, Explainer
from core.solver_base import BaseSolver
from core.vision_base import BaseVisionParser, GridNotFoundError
from main import SolveHandler, build_parser, to_jsonable
from solvers.queens.vision import QueensVisionParser
from solvers.tango.solver import Symbol


def test_profile_measures_latency_and_memory() -> None:
    with BaseSolver.profile(track_memory=True) as record:
        _ = [0] * 100_000
    assert record.latency_us > 0 and record.peak_memory_kb > 100


def test_profile_serialises_threads_but_rejects_nesting() -> None:
    errors: list[BaseException] = []

    def profiled() -> None:
        try:
            with BaseSolver.profile(track_memory=True):
                _ = [0] * 100_000
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=profiled) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    with BaseSolver.profile(track_memory=True), pytest.raises(RuntimeError, match="nested"):
        with BaseSolver.profile(track_memory=True):
            pass


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


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), SolveHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(address: str, body: bytes, headers: dict[str, str]) -> tuple[int, dict]:
    connection = http.client.HTTPConnection(address, timeout=10)
    connection.putrequest("POST", "/pinpoint")
    for name, value in headers.items():
        connection.putheader(name, value)
    connection.endheaders()
    connection.send(body)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def test_server_rejects_non_json_content_type(server: str) -> None:
    status, payload = _post(server, b'{"clues": []}', {"Content-Type": "text/plain", "Content-Length": "13"})
    assert status == 415 and "application/json" in payload["error"]


def test_server_answers_bad_content_length(server: str) -> None:
    status, payload = _post(server, b"", {"Content-Type": "application/json", "Content-Length": "abc"})
    assert status == 400 and payload["error"].startswith("ValueError")


def test_server_rejects_non_object_body(server: str) -> None:
    status, _ = _post(server, b"[]", {"Content-Type": "application/json", "Content-Length": "2"})
    assert status == 400


def test_stalled_client_does_not_block_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SolveHandler, "timeout", 0.5)
    httpd = HTTPServer(("127.0.0.1", 0), SolveHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    stalled = socket.create_connection(("127.0.0.1", port))  # connects, never sends a request
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/health")
        assert connection.getresponse().status == 200
    finally:
        stalled.close()
        httpd.shutdown()
        httpd.server_close()


def test_paste_solves_the_clipboard_and_copies_the_answer_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from PIL import Image

    board, _ = generate_queens(6, 1)
    screenshot = Image.fromarray(render_queens(board, 1)[..., ::-1].copy())
    monkeypatch.setattr("PIL.ImageGrab.grabclipboard", lambda: screenshot)
    copied: list[pathlib.Path] = []
    monkeypatch.setattr(main, "copy_image_to_clipboard", copied.append)
    output = tmp_path / "solved.png"
    assert main.main(["solve", "--paste", "--output", str(output), "--json"]) == 0
    assert copied == [output] and output.exists()


def test_empty_clipboard_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("PIL.ImageGrab.grabclipboard", lambda: None)
    with pytest.raises(ValueError, match="Win\\+Shift\\+S"):
        main.read_clipboard_image()


def test_clipboard_copy_uses_powershell_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr(main.subprocess, "run", lambda command, check: calls.append(command))
    main.copy_image_to_clipboard(pathlib.Path("solved.png"))
    assert calls[0][:4] == ["powershell", "-NoProfile", "-STA", "-Command"] and "SetImage" in calls[0][4]
    monkeypatch.setattr(main.sys, "platform", "linux")
    with pytest.raises(OSError, match="Windows only"):
        main.copy_image_to_clipboard(pathlib.Path("solved.png"))


def test_typed_boards_still_need_a_game() -> None:
    assert main.main(["solve", "--text", "examples/wend.txt"]) == 2


def test_difficulty_is_the_hardest_rule_and_how_often_it_fired() -> None:
    steps = [DeductionStep(i, rule, "") for i, rule in enumerate(["Single Choice", "Shared Attack", "Shared Attack"])]
    assert rate(steps) == "Medium: Shared Attack ×2"


def test_difficulty_refuses_a_rule_it_has_no_tier_for() -> None:
    with pytest.raises(KeyError, match="Checkpoint"):
        rate([DeductionStep(1, "Checkpoint", "")])
