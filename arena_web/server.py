from __future__ import annotations

import json
import os
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena import ArenaConfig, run_batch


PORT = int(os.environ.get("GENERALS_ARENA_PORT", "8910"))
WEB_DIR = Path(__file__).resolve().parent
BOT_SPECS = [
    "heuristic:baseline",
    "heuristic:gambler",
    "heuristic:conservative",
    "heuristic:trickster",
    "heuristic:crazy",
    "random",
]


class ArenaHandler(SimpleHTTPRequestHandler):
    server_version = "GeneralsArenaHTTP/0.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/options":
            self._send_json({"bots": BOT_SPECS})
            return
        if parsed.path == "/api/evaluate":
            self._send_json(self._evaluate(parse_qs(parsed.query)))
            return
        if parsed.path in ("", "/"):
            self.path = "/index.html"
        super().do_GET()

    def _evaluate(self, query: dict[str, list[str]]) -> dict[str, Any]:
        start = time.perf_counter()
        games = self._int_arg(query, "games", 10, 1, 100)
        seed = self._int_arg(query, "seed", 1, 0, 2_000_000_000)
        width = self._int_arg(query, "width", 12, 6, 24)
        height = self._int_arg(query, "height", 12, 6, 24)
        max_turns = self._int_arg(query, "max_turns", 400, 50, 1500)
        strategic_interval = self._int_arg(query, "strategic_interval", 1, 1, 60)
        bot0 = self._choice_arg(query, "bot0", "heuristic:baseline")
        bot1 = self._choice_arg(query, "bot1", "random")

        summary = run_batch(
            [bot0, bot1],
            [seed + offset for offset in range(games)],
            ArenaConfig(
                width=width,
                height=height,
                max_turns=max_turns,
                strategic_interval=strategic_interval,
            ),
        )
        summary["config"] = {
            "games": games,
            "seed": seed,
            "width": width,
            "height": height,
            "max_turns": max_turns,
            "strategic_interval": strategic_interval,
            "bot0": bot0,
            "bot1": bot1,
        }
        summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        return summary

    def _int_arg(
        self,
        query: dict[str, list[str]],
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            value = int((query.get(key) or [default])[0])
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    def _choice_arg(self, query: dict[str, list[str]], key: str, default: str) -> str:
        value = str((query.get(key) or [default])[0]).strip().lower()
        return value if value in BOT_SPECS else default

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), ArenaHandler)
    print(f"Generals 1v1 arena server running at http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
