"""Local server for the play page: static files plus ``GET /api/puzzle?game=...&level=...``."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from play.levels import LEVEL_NAMES, PLAY_GAMES
from play.pool import Pool

STATIC = Path(__file__).resolve().parent / "static"
CONTENT_TYPES = {  # explicit: Windows' registry can map .js to text/plain, which browsers refuse as a module
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
}


class PlayHandler(BaseHTTPRequestHandler):
    pool: Pool
    timeout = 30

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        url = urlsplit(self.path)
        if url.path == "/api/puzzle":
            self._puzzle(parse_qs(url.query))
        else:
            self._static("index.html" if url.path == "/" else url.path.lstrip("/"))

    def _puzzle(self, query: dict[str, list[str]]) -> None:
        game, level = query.get("game", [""])[0], query.get("level", [""])[0]
        if game not in PLAY_GAMES or level not in LEVEL_NAMES:
            self._json(
                HTTPStatus.BAD_REQUEST, {"error": f"game must be one of {PLAY_GAMES}, level one of {LEVEL_NAMES}"}
            )
            return
        try:
            self._json(HTTPStatus.OK, self.pool.take(game, level))
        except RuntimeError as error:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})

    def _static(self, relative: str) -> None:
        path = (STATIC / relative).resolve()
        if not path.is_relative_to(STATIC) or not path.is_file() or path.suffix not in CONTENT_TYPES:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no file {relative}"})
            return
        self._send(HTTPStatus.OK, path.read_bytes(), CONTENT_TYPES[path.suffix])

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        """Quiet: one line per request would drown the console."""


def make_server(host: str, port: int, pool: Pool) -> ThreadingHTTPServer:
    handler = type("BoundPlayHandler", (PlayHandler,), {"pool": pool})
    return ThreadingHTTPServer((host, port), handler)
