"""Unified CLI: solve screenshots, stream Pinpoint clues, run benchmarks or serve the REST backend.

python main.py solve --game queens --image board.png
python main.py stream --game pinpoint
python main.py benchmark --game all
python main.py serve --port 8765
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import subprocess
import sys
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from core import visualizer
from core.difficulty import rate
from core.solver_base import UnsolvableError
from core.vision_base import BaseVisionParser, GridNotFoundError
from solvers import DETECTION_ORDER, GAMES, get_game

console = Console()
MAX_REQUEST_BYTES = 20 * 1024 * 1024


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, tuples, frozensets and dicts with tuple keys into JSON types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return [{"key": to_jsonable(k), "value": to_jsonable(v)} for k, v in value.items()]
    if isinstance(value, (list, tuple, frozenset, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def solve_image(game_name: str | None, image: np.ndarray, slots: np.ndarray | None = None) -> dict[str, Any]:
    """Parse, solve and render one screenshot; returns a JSON-ready payload plus the overlay layer.

    ``game_name=None`` detects the game: each one in ``DETECTION_ORDER`` is tried until one reads and solves it.
    ``slots`` is a separate screenshot of Wend's word slots, for when they are not in ``image``.
    """
    if game_name is None:
        for name in DETECTION_ORDER:
            try:
                return solve_image(name, image, slots if name == "wend" else None)
            except (GridNotFoundError, UnsolvableError, ValueError):
                continue
        raise ValueError(f"no game ({', '.join(DETECTION_ORDER)}) could read and solve this screenshot; pass --game")
    game = get_game(game_name)
    if game.parser is None:
        raise ValueError(f"{game_name} has no vision pipeline; use `python main.py stream --game {game_name}`")
    if slots is not None and game_name != "wend":
        raise ValueError("a slots screenshot only applies to Wend")
    parser: BaseVisionParser[Any, Any] = game.parser()
    board = parser.parse(image)
    if slots is not None:
        board = dataclasses.replace(board, word_lengths=parser.slot_lengths(slots))  # type: ignore[attr-defined]
    result = game.solver().run(board)
    return {
        "game": game_name,
        "board": board,
        "solution": result.solution,
        "profile": result.profile,
        "steps": result.explanation.steps,
        "difficulty": rate(result.explanation.steps) if game.rated else None,
        "layer": parser.overlay_layer(image, board, result.solution),
        "markdown": result.explanation.to_markdown(),
    }


def solve_text(game_name: str, text: str) -> dict[str, Any]:
    """Build the board from typed text and solve it; same payload as ``solve_image`` without an overlay."""
    game = get_game(game_name)
    if game.text is None:
        raise ValueError(f"{game_name} cannot be typed in; typed boards work for {', '.join(TYPED_GAMES)}")
    board = game.text(text)
    result = game.solver().run(board)
    return {
        "game": game_name,
        "board": board,
        "solution": result.solution,
        "profile": result.profile,
        "steps": result.explanation.steps,
        "difficulty": rate(result.explanation.steps) if game.rated else None,
        "markdown": result.explanation.to_markdown(),
    }


def read_clipboard_image() -> np.ndarray:
    """The image on the clipboard (Win+Shift+S puts screenshots there), as BGR."""
    from PIL import ImageGrab

    grabbed = ImageGrab.grabclipboard()
    if isinstance(grabbed, list):  # copied files rather than pixels: use the first image file
        grabbed = BaseVisionParser.load_image(grabbed[0]) if grabbed else None
        if grabbed is not None:
            return grabbed
    if grabbed is None:
        raise ValueError("the clipboard holds no image: take a screenshot (Win+Shift+S) first")
    return cv2.cvtColor(np.asarray(grabbed.convert("RGB")), cv2.COLOR_RGB2BGR)


def copy_image_to_clipboard(path: Path) -> None:
    """Put a PNG on the Windows clipboard, so the solved board can be pasted anywhere."""
    if sys.platform != "win32":
        raise OSError("copying images to the clipboard is supported on Windows only")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms, System.Drawing; "
        f"[System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{path.resolve()}'))"
    )
    subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script], check=True)


def encode_png(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_image(data: str) -> np.ndarray:
    raw = base64.b64decode(data.split(",", 1)[-1], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("request image is not a decodable PNG/JPEG")
    return image


TYPED_GAMES = [name for name, game in GAMES.items() if game.text]

# --------------------------------------------------------------------------- commands


def cmd_solve(args: argparse.Namespace) -> int:
    if args.text and args.slots:
        raise ValueError("--slots goes with --image; type the word lengths on the board's last line instead")
    if args.text and not args.game:
        raise ValueError(f"typed boards need --game ({', '.join(TYPED_GAMES)})")
    if args.text:
        return _print_outcome(
            args,
            solve_text(
                args.game, sys.stdin.read() if args.text == "-" else Path(args.text).read_text(encoding="utf-8")
            ),
        )
    image = read_clipboard_image() if args.paste else BaseVisionParser.load_image(args.image)
    slots = BaseVisionParser.load_image(args.slots) if args.slots else None
    outcome = solve_image(args.game, image, slots)
    if args.output:
        output = Path(args.output)
    else:
        output = Path("solved.png") if args.paste else Path(args.image).with_name(f"{Path(args.image).stem}_solved.png")
    cv2.imwrite(str(output), visualizer.composite(image, outcome["layer"]))
    outcome["overlay"] = str(output)
    if args.paste:
        try:
            copy_image_to_clipboard(output)
            outcome["overlay"] += " (also on the clipboard: Ctrl+V it anywhere)"
        except OSError as error:  # the PNG is saved either way
            outcome["overlay"] += f" (not copied to the clipboard: {error})"
    return _print_outcome(args, outcome)


def _print_outcome(args: argparse.Namespace, outcome: dict[str, Any]) -> int:
    if args.json:
        payload = {key: to_jsonable(value) for key, value in outcome.items() if key not in ("layer", "markdown")}
        print(json.dumps(payload, indent=2))
        return 0
    profile = outcome["profile"]
    table = Table(title=f"{get_game(outcome['game']).title}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Solve latency", f"{profile.latency_us:,.1f} µs")
    table.add_row("Peak memory", f"{profile.peak_memory_kb:,.1f} KiB")
    table.add_row("Nodes expanded", f"{profile.nodes_expanded:,}")
    if "overlay" in outcome:
        table.add_row("Overlay", outcome["overlay"])
    if outcome["difficulty"] is not None:
        table.add_row("Difficulty", outcome["difficulty"])
    if outcome["game"] in TYPED_GAMES:
        table.add_row("Answer", ", ".join(item.word for item in outcome["solution"]))
    console.print(table)
    console.print(outcome["markdown"])
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    if args.game != "pinpoint":
        raise ValueError("only pinpoint is played as a clue stream")
    from solvers.pinpoint.cli_stream import run_stream
    from solvers.pinpoint.semantic_engine import SemanticEngine

    run_stream(SemanticEngine(), console=console)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from bench import benchmark_runner

    games = benchmark_runner.ALL_GAMES if args.game == "all" else [args.game]
    rows = benchmark_runner.run(games, repeats=args.repeats)
    benchmark_runner.print_table(rows, console)
    if args.json_out:
        Path(args.json_out).write_text(benchmark_runner.to_json(rows))
    if args.markdown_out:
        Path(args.markdown_out).write_text(benchmark_runner.to_markdown(rows) + "\n")
    return 0 if all(row.solved_ok and row.parsed_ok is not False for row in rows) else 1


REQUEST_TIMEOUT_S = 30


class SolveHandler(BaseHTTPRequestHandler):
    """REST backend for the browser extension.

    ``POST /solve``    ``{"game": str, "image": base64}`` -> solution, steps and a BGRA overlay PNG
    ``POST /pinpoint`` ``{"clues": [str, ...]}``           -> ranked categories
    ``GET  /health``                                        -> ``{"status": "ok"}``
    """

    timeout = REQUEST_TIMEOUT_S  # socket deadline: a stalled client cannot hold the one-request-at-a-time server

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path == "/health":
            self._send(HTTPStatus.OK, {"status": "ok", "games": list(GAMES)})
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        # Only JSON: a browser page cannot send it cross-site without a CORS preflight, which is never granted.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError(f"Content-Length must be between 0 and {MAX_REQUEST_BYTES} bytes")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            if self.path == "/solve":
                outcome = solve_image(str(request["game"]), decode_image(str(request["image"])))
                payload = {
                    key: to_jsonable(value) for key, value in outcome.items() if key not in ("layer", "markdown")
                }
                payload["overlay"] = encode_png(outcome["layer"])
                self._send(HTTPStatus.OK, payload)
            elif self.path == "/pinpoint":
                clues = request["clues"]
                if not isinstance(clues, list) or not all(isinstance(clue, str) for clue in clues):
                    raise ValueError("clues must be a list of strings")
                result = get_game("pinpoint").solver().run(clues)
                self._send(
                    HTTPStatus.OK, {"ranking": to_jsonable(result.solution), "profile": to_jsonable(result.profile)}
                )
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": f"no route {self.path}"})
        except (KeyError, ValueError, GridNotFoundError, UnsolvableError, json.JSONDecodeError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": f"{type(error).__name__}: {error}"})
        except Exception as error:  # the client still gets an answer it can show
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(error).__name__}: {error}"})
            raise


def cmd_serve(args: argparse.Namespace) -> int:
    # One request at a time: solves profile memory with tracemalloc, which cannot run twice at once.
    server = HTTPServer((args.host, args.port), SolveHandler)
    console.print(f"Serving on http://{args.host}:{args.port} (POST /solve, POST /pinpoint, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Shutting down")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="LinkedIn Puzzle Solvers Engine")
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="solve a screenshot and write an overlay image")
    solve.add_argument(
        "--game", choices=[name for name, game in GAMES.items() if game.parser], help="leave out to detect it"
    )
    source = solve.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="path to the screenshot")
    source.add_argument("--paste", action="store_true", help="read the screenshot from the clipboard (Win+Shift+S)")
    source.add_argument("--text", help=f"typed board ({', '.join(TYPED_GAMES)}): a text file, or - for stdin")
    solve.add_argument("--slots", help="Wend only: screenshot of the word slots, when they are not in --image")
    solve.add_argument("--output", help="overlay path (default: <image>_solved.png)")
    solve.add_argument("--json", action="store_true", help="print the full result as JSON")
    solve.set_defaults(handler=cmd_solve)

    stream = commands.add_parser("stream", help="type clues one by one (Pinpoint)")
    stream.add_argument("--game", required=True, choices=["pinpoint"])
    stream.set_defaults(handler=cmd_stream)

    benchmark = commands.add_parser("benchmark", help="latency / memory / accuracy suite")
    benchmark.add_argument("--game", default="all", choices=["all", *GAMES])
    benchmark.add_argument("--repeats", type=int, default=25)
    benchmark.add_argument("--json-out", help="write rows as JSON")
    benchmark.add_argument("--markdown-out", help="write rows as a Markdown table")
    benchmark.set_defaults(handler=cmd_benchmark)

    serve = commands.add_parser("serve", help="REST backend for the browser extension")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(handler=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, KeyError, GridNotFoundError, UnsolvableError) as error:
        console.print(f"[bold red]error:[/bold red] {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
