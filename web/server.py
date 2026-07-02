from __future__ import annotations

import json
import os
import random
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import AGENT_CONFIGS, LLMAgent
from engine.board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from engine.game import Game
from engine.types import Move, PlayerState, Tile, TileType
from execution import Executor


PORT = 8900
WIDTH = 12
HEIGHT = 12
NUM_PLAYERS = 4
HALF_TURNS_PER_ROUND = 6
WEB_DIR = Path(__file__).resolve().parent

PLAYER_PERSONALITIES = ["gambler", "conservative", "trickster", "crazy"]


class GameService:
    def __init__(self) -> None:
        self.running = True
        self.round_index = 0
        self.seed: int | None = None
        self.game: Game
        self.agents: list[LLMAgent]
        self.logs: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.seed = random.randrange(1_000_000_000)
        board = Board.generate_map(WIDTH, HEIGHT, NUM_PLAYERS, seed=self.seed)
        players = [
            PlayerState(
                index=i,
                alive=True,
                name=str(AGENT_CONFIGS[personality]["name"]),
            )
            for i, personality in enumerate(PLAYER_PERSONALITIES)
        ]
        self.game = Game(board, players)
        self.agents = [
            LLMAgent(player_idx=i, personality=personality)
            for i, personality in enumerate(PLAYER_PERSONALITIES)
        ]
        self.round_index = 0
        self.logs = [{
            "round": 0,
            "turn": 0,
            "phase": 0,
            "player": None,
            "message": "新地图已生成，4 名指挥官进入战场。",
        }]
        return self.state()

    def step_round(self, half_turns: int = HALF_TURNS_PER_ROUND) -> dict[str, Any]:
        if self.game.is_finished():
            self._append_log(None, "战局已结束，无法继续推进。")
            return self.state()

        self.round_index += 1
        for _ in range(max(1, half_turns)):
            if self.game.is_finished():
                break

            moves_by_player: dict[int, list[Move]] = {}
            for agent in self.agents:
                player_idx = agent.player_idx
                if not self.game.alive[player_idx]:
                    continue

                view = self.game.get_player_view(player_idx)
                strategy = agent.decide(view)
                executor = Executor(
                    player_idx,
                    view["board"],
                    strategy.get("constraints") or {},
                )
                moves = executor.execute(strategy)
                moves_by_player[player_idx] = moves
                self._append_strategy_log(player_idx, strategy, moves)

            self.game.step(moves_by_player)

        if self.game.is_finished():
            winner = self._winner()
            if winner is not None:
                self._append_log(winner, f"{self._player_label(winner)} 获胜。")
            else:
                self._append_log(None, "战局结束，无幸存者。")

        return self.state()

    def state(self) -> dict[str, Any]:
        rankings = []
        for ranking in self.game.get_rankings():
            player_idx = int(ranking["player"])
            personality = PLAYER_PERSONALITIES[player_idx]
            config = AGENT_CONFIGS[personality]
            rankings.append({
                **ranking,
                "name": config["name"],
                "emoji": config["emoji"],
            })

        winner = self._winner() if self.game.is_finished() else None
        return {
            "turn": self.game.turn,
            "phase": self.game.phase,
            "width": self.game.board.width,
            "height": self.game.board.height,
            "tiles": self._serialize_tiles(self.game.board.tiles),
            "rankings": rankings,
            "finished": self.game.is_finished(),
            "winner": winner,
            "logs": self.logs[-80:],
            "seed": self.seed,
        }

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "turn": self.game.turn,
            "phase": self.game.phase,
            "finished": self.game.is_finished(),
            "players": NUM_PLAYERS,
        }

    def _serialize_tiles(self, tiles: list[list[Tile]]) -> list[list[dict[str, Any]]]:
        return [
            [self._serialize_tile(tile) for tile in row]
            for row in tiles
        ]

    def _serialize_tile(self, tile: Tile) -> dict[str, Any]:
        if tile.occupier == TILE_FOG:
            tile_type = "FOG"
        elif tile.occupier == TILE_FOG_OBSTACLE:
            tile_type = "FOG_OBSTACLE"
        else:
            tile_type = tile.type.name
        return {
            "type": tile_type,
            "occupier": tile.occupier,
            "army": tile.army,
        }

    def _append_strategy_log(self, player_idx: int, strategy: dict[str, Any], moves: list[Move]) -> None:
        plan = str(strategy.get("round_plan") or "执行默认行动。")
        if moves:
            move_text = "，".join(
                f"({move.from_y},{move.from_x})->({move.to_y},{move.to_x})"
                for move in moves[:3]
            )
            if len(moves) > 3:
                move_text += f" 等 {len(moves)} 步"
        else:
            move_text = "无可执行移动"
        self._append_log(player_idx, f"{plan} / {move_text}")

    def _append_log(self, player_idx: int | None, message: str) -> None:
        self.logs.append({
            "round": self.round_index,
            "turn": self.game.turn,
            "phase": self.game.phase,
            "player": player_idx,
            "message": message,
        })

    def _player_label(self, player_idx: int) -> str:
        personality = PLAYER_PERSONALITIES[player_idx]
        config = AGENT_CONFIGS[personality]
        return f"P{player_idx} {config['emoji']}{config['name']}"

    def _winner(self) -> int | None:
        winners = [player.index for player in self.game.players if self.game.alive[player.index]]
        return winners[0] if len(winners) == 1 else None


SERVICE = GameService()


class GeneralsHandler(SimpleHTTPRequestHandler):
    server_version = "GeneralsLLMHTTP/0.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(SERVICE.state())
            return
        if path == "/api/status":
            self._send_json(SERVICE.status())
            return
        if path in ("", "/"):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/step":
            self._send_json(SERVICE.step_round(self._requested_half_turns()))
            return
        if path == "/api/reset":
            self._send_json(SERVICE.reset())
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _requested_half_turns(self) -> int:
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            return HALF_TURNS_PER_ROUND
        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return HALF_TURNS_PER_ROUND
        try:
            return max(1, min(50, int(body.get("half_turns", HALF_TURNS_PER_ROUND))))
        except (TypeError, ValueError):
            return HALF_TURNS_PER_ROUND

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), GeneralsHandler)
    print(f"Generals LLM web server running at http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SERVICE.running = False
        server.server_close()


if __name__ == "__main__":
    main()
