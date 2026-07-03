from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import AGENT_CONFIGS, HeuristicAgent, LLMAgent
from engine.board import TILE_FOG, TILE_FOG_OBSTACLE, Board
from engine.game import Game
from engine.types import Move, PlayerState, Tile, TileType
from execution import Executor


PORT = 8900
WIDTH = 18
HEIGHT = 18
NUM_PLAYERS = 4
HALF_TURNS_PER_ROUND = 12
MAX_STRATEGIC_ROUNDS = 180
MAX_TURNS = MAX_STRATEGIC_ROUNDS * (HALF_TURNS_PER_ROUND // 2)
WEB_DIR = Path(__file__).resolve().parent

PLAYER_PERSONALITIES = ["gambler", "conservative", "trickster", "crazy"]
AGENT_MODE = os.environ.get("GENERALS_AGENT_MODE", "heuristic").strip().lower()


class GameService:
    def __init__(self) -> None:
        self.running = True
        self.round_index = 0
        self.seed: int | None = None
        self._lock = threading.Lock()
        self._stop = False
        self._rounds: list[dict[str, Any]] = []
        self._current_round = -1
        self._thread: threading.Thread | None = None
        self._compute_error: str | None = None
        self._paused = False
        self._terminal_reason: str | None = None
        self.game: Game
        self.agents: list[Any]
        self.logs: list[dict[str, Any]] = []
        self._initial_state: dict[str, Any] = {}
        with self._lock:
            self._new_game_locked()
        self._start_thread()

    def _new_game_locked(self) -> None:
        self._stop = False
        self._rounds = []
        self._current_round = -1
        self._compute_error = None
        self._paused = False
        self._terminal_reason = None
        self.round_index = 0
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
            self._create_agent(i, personality)
            for i, personality in enumerate(PLAYER_PERSONALITIES)
        ]
        self.logs = [{
            "round": 0,
            "turn": 0,
            "phase": 0,
            "player": None,
            "message": "新地图已生成，4 名指挥官进入战场。",
        }]
        self._initial_state = self._state_snapshot()

    def _create_agent(self, player_idx: int, personality: str) -> Any:
        if AGENT_MODE == "llm":
            return LLMAgent(player_idx=player_idx, personality=personality)
        return HeuristicAgent(player_idx=player_idx, personality=personality)

    def _start_thread(self) -> None:
        self._thread = threading.Thread(target=self._compute_all, daemon=True)
        self._thread.start()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._stop = True
            thread = self._thread
        if thread and thread.is_alive():
            thread.join()
        with self._lock:
            self._new_game_locked()
        self._start_thread()
        return self.current_state()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._stop = True
            self._paused = True
            thread = self._thread
        if thread and thread.is_alive():
            thread.join()
        return self.current_state()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self._terminal_reason or self.game.is_finished():
                return self.current_state()
            self._stop = False
            self._paused = False
            thread = self._thread
            should_start = thread is None or not thread.is_alive()
        if should_start:
            self._start_thread()
        return self.current_state()

    def _compute_all(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._stop or self.game.is_finished() or self._terminal_reason:
                        return
                    if self.round_index >= MAX_STRATEGIC_ROUNDS or self.game.turn >= MAX_TURNS:
                        self._terminal_reason = "max_rounds_or_turns"
                        self._append_log(None, "达到推演上限，后台计算已停止。")
                        return
                round_data = self._compute_one_round()
                with self._lock:
                    if self._stop:
                        return
                    self._rounds.append(round_data)
                time.sleep(0.05)
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                self._compute_error = repr(exc)
                self._append_log(None, f"后台计算异常: {exc!r}")

    def _compute_one_round(self) -> dict[str, Any]:
        if self.game.is_finished():
            self._append_log(None, "战局已结束，无法继续推进。")
            return {
                "round": self.round_index,
                "frames": [self._state_snapshot()],
                "logs": [],
                "finished": True,
                "winner": self._winner(),
            }

        self.round_index += 1
        log_start = len(self.logs)
        strategies: dict[int, dict[str, Any]] = {}

        # Parallel agent decisions
        alive_agents = [(a, self.game.get_player_view(a.player_idx)) for a in self.agents if self.game.alive[a.player_idx]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for agent, view in alive_agents:
                future = pool.submit(agent.decide, view)
                futures[future] = (agent, view)

            for future in as_completed(futures):
                agent, view = futures[future]
                try:
                    strategy = future.result()
                except Exception:
                    strategy = agent.simulate_llm(view)
                strategies[agent.player_idx] = strategy

        frames = [self._state_snapshot()]
        first_half_turn = True
        total_moves_by_player: dict[int, int] = {idx: 0 for idx in strategies}
        zero_half_turns_by_player: dict[int, int] = {idx: 0 for idx in strategies}
        reverse_moves_by_player: dict[int, int] = {idx: 0 for idx in strategies}
        previous_edges_by_player: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {
            idx: set() for idx in strategies
        }
        half_turns_by_player: dict[int, int] = {idx: 0 for idx in strategies}

        for _ in range(max(1, HALF_TURNS_PER_ROUND)):
            if self.game.is_finished():
                break

            moves_by_player: dict[int, list[Move]] = {}
            diagnostics_by_player: dict[int, list[str]] = {}
            for agent in self.agents:
                if not self.game.alive[agent.player_idx] or agent.player_idx not in strategies:
                    continue
                strategy = strategies[agent.player_idx]
                view = self.game.get_player_view(agent.player_idx)
                execution_strategy = {
                    **strategy,
                    "_context": {
                        "stats": view.get("stats") or {},
                        "rankings": view.get("rankings") or [],
                    },
                }
                executor = Executor(
                    agent.player_idx,
                    view["board"],
                    strategy.get("constraints") or {},
                )
                result = executor.execute_with_diagnostics(execution_strategy)
                moves_by_player[agent.player_idx] = result.moves
                half_turns_by_player[agent.player_idx] += 1
                total_moves_by_player[agent.player_idx] += len(result.moves)
                if not result.moves:
                    zero_half_turns_by_player[agent.player_idx] += 1
                reverse_moves_by_player[agent.player_idx] += self._count_reverse_moves(
                    agent.player_idx,
                    result.moves,
                    previous_edges_by_player,
                )
                diagnostics_by_player[agent.player_idx] = [
                    f"{diag.objective_type}:{diag.status}:{diag.reason}:{diag.generated_moves}"
                    for diag in result.diagnostics
                ]
                if first_half_turn:
                    self._append_strategy_log(agent.player_idx, strategy, result.moves, result.diagnostics)

            self.game.step(moves_by_player)
            first_half_turn = False

            if self.game.is_finished():
                winner = self._winner()
                if winner is not None:
                    self._append_log(winner, f"{self._player_label(winner)} 获胜。")
                else:
                    self._append_log(None, "战局结束，无幸存者。")

            frames.append(self._state_snapshot())

        for player_idx, total_moves in sorted(total_moves_by_player.items()):
            self._append_log(
                player_idx,
                f"执行汇总: {HALF_TURNS_PER_ROUND} 半回合内重算动作，总计 {total_moves} 动，空转 {zero_half_turns_by_player[player_idx]} 次。",
            )

        return {
            "round": self.round_index,
            "frames": frames,
            "logs": self.logs[log_start:],
            "finished": self.game.is_finished(),
            "winner": self._winner() if self.game.is_finished() else None,
            "terminal_reason": self._terminal_reason,
            "metrics": self._round_metrics(
                frames[-1],
                half_turns_by_player,
                total_moves_by_player,
                zero_half_turns_by_player,
                reverse_moves_by_player,
            ),
        }

    def advance(self, target: int | None = None) -> dict[str, Any]:
        with self._lock:
            if target is not None:
                if self._rounds:
                    target = max(0, min(target, len(self._rounds) - 1))
                else:
                    target = max(0, target)
                self._current_round = target - 1
            next_round = self._current_round + 1
            if next_round < len(self._rounds):
                self._current_round = next_round
                return {"ready": True, **self._rounds[next_round]}
            if self._rounds:
                finished = bool(self._rounds[-1]["finished"])
                winner = self._rounds[-1]["winner"]
            else:
                finished = bool(self._initial_state["finished"])
                winner = self._initial_state["winner"]
            return {
                "ready": False,
                "finished": finished,
                "winner": winner,
                "error": self._compute_error,
                "paused": self._paused,
                "terminal_reason": self._terminal_reason,
            }

    def rounds(self) -> dict[str, Any]:
        with self._lock:
            if self._rounds:
                finished = bool(self._rounds[-1]["finished"])
            else:
                finished = bool(self._initial_state["finished"])
            return {
                "rounds": list(range(len(self._rounds))),
                "current": self._current_round,
                "computed": len(self._rounds),
                "finished": finished,
                "paused": self._paused,
                "terminal_reason": self._terminal_reason,
                "agent_mode": AGENT_MODE,
            }

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return self._quality_metrics_locked()

    def current_state(self) -> dict[str, Any]:
        with self._lock:
            if 0 <= self._current_round < len(self._rounds):
                return self._rounds[self._current_round]["frames"][-1]
            return self._initial_state

    def _state_snapshot(self) -> dict[str, Any]:
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
            "terminal_reason": self._terminal_reason,
            "agent_mode": AGENT_MODE,
        }

    def _count_reverse_moves(
        self,
        player_idx: int,
        moves: list[Move],
        previous_edges_by_player: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]],
    ) -> int:
        previous_edges = previous_edges_by_player.get(player_idx, set())
        reverse_count = 0
        current_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for move in moves:
            edge = ((move.from_x, move.from_y), (move.to_x, move.to_y))
            reverse = (edge[1], edge[0])
            if reverse in previous_edges:
                reverse_count += 1
            current_edges.add(edge)
        previous_edges_by_player[player_idx] = current_edges
        return reverse_count

    def _round_metrics(
        self,
        state: dict[str, Any],
        half_turns_by_player: dict[int, int],
        total_moves_by_player: dict[int, int],
        zero_half_turns_by_player: dict[int, int],
        reverse_moves_by_player: dict[int, int],
    ) -> dict[str, Any]:
        total_half_turns = sum(half_turns_by_player.values())
        total_moves = sum(total_moves_by_player.values())
        zero_half_turns = sum(zero_half_turns_by_player.values())
        reverse_moves = sum(reverse_moves_by_player.values())
        return {
            "round": self.round_index,
            "turn": state["turn"],
            "total_moves": total_moves,
            "avg_moves_per_half_turn": round(total_moves / total_half_turns, 2) if total_half_turns else 0,
            "zero_half_turns": zero_half_turns,
            "zero_half_turn_rate": round(zero_half_turns / total_half_turns, 4) if total_half_turns else 0,
            "reverse_moves": reverse_moves,
            "reverse_move_rate": round(reverse_moves / total_moves, 4) if total_moves else 0,
            "by_player": {
                str(player_idx): {
                    "half_turns": half_turns_by_player.get(player_idx, 0),
                    "moves": total_moves_by_player.get(player_idx, 0),
                    "zero_half_turns": zero_half_turns_by_player.get(player_idx, 0),
                    "reverse_moves": reverse_moves_by_player.get(player_idx, 0),
                }
                for player_idx in range(NUM_PLAYERS)
            },
        }

    def _quality_metrics_locked(self) -> dict[str, Any]:
        rounds = self._rounds
        latest_state = rounds[-1]["frames"][-1] if rounds else self._initial_state
        round_metrics = [round_data.get("metrics") or {} for round_data in rounds]
        total_moves = sum(int(item.get("total_moves", 0)) for item in round_metrics)
        zero_half_turns = sum(int(item.get("zero_half_turns", 0)) for item in round_metrics)
        reverse_moves = sum(int(item.get("reverse_moves", 0)) for item in round_metrics)
        total_half_turns = sum(
            sum(int(player.get("half_turns", 0)) for player in (item.get("by_player") or {}).values())
            for item in round_metrics
        )
        dominance = self._dominance_summary(rounds)
        return {
            "seed": self.seed,
            "agent_mode": AGENT_MODE,
            "computed_rounds": len(rounds),
            "turn": latest_state["turn"],
            "finished": bool(rounds[-1]["finished"]) if rounds else bool(latest_state["finished"]),
            "winner": rounds[-1]["winner"] if rounds else latest_state["winner"],
            "terminal_reason": self._terminal_reason,
            "total_moves": total_moves,
            "avg_moves_per_half_turn": round(total_moves / total_half_turns, 2) if total_half_turns else 0,
            "zero_half_turn_rate": round(zero_half_turns / total_half_turns, 4) if total_half_turns else 0,
            "reverse_move_rate": round(reverse_moves / total_moves, 4) if total_moves else 0,
            "dominance": dominance,
            "players": self._player_metric_rows(latest_state),
            "recent_rounds": round_metrics[-12:],
        }

    def _dominance_summary(self, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        first_round: int | None = None
        leader: int | None = None
        latest_margin = 0.0
        for round_data in rounds:
            state = round_data["frames"][-1]
            alive = [row for row in state.get("rankings", []) if row.get("alive")]
            if len(alive) < 2:
                continue
            top = alive[0]
            second = alive[1]
            second_army = max(1, int(second.get("army", 0)))
            second_tiles = max(1, int(second.get("tiles", 0)))
            army_ratio = int(top.get("army", 0)) / second_army
            tile_ratio = int(top.get("tiles", 0)) / second_tiles
            latest_margin = round(army_ratio, 2)
            if army_ratio >= 2.0 and tile_ratio >= 1.2:
                first_round = int(round_data.get("round", 0))
                leader = int(top.get("player", -1))
                break
        last_round = int(rounds[-1].get("round", 0)) if rounds else 0
        return {
            "leader": leader,
            "first_round": first_round,
            "age_rounds": (last_round - first_round) if first_round is not None else 0,
            "latest_army_ratio": latest_margin,
        }

    def _player_metric_rows(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "player": int(row["player"]),
                "name": row.get("name"),
                "army": int(row.get("army", 0)),
                "tiles": int(row.get("tiles", 0)),
                "alive": bool(row.get("alive")),
                "kills": int(row.get("kills", 0)),
            }
            for row in state.get("rankings", [])
        ]

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._rounds:
                state = self._rounds[-1]["frames"][-1]
                finished = bool(self._rounds[-1]["finished"])
            else:
                state = self._initial_state
                finished = bool(state["finished"])
            return {
                "running": self.running,
                "turn": state["turn"],
                "phase": state["phase"],
                "finished": finished,
                "players": NUM_PLAYERS,
                "computed_rounds": len(self._rounds),
                "current_round": self._current_round,
                "ready": self._current_round + 1 < len(self._rounds),
                "error": self._compute_error,
                "paused": self._paused,
                "terminal_reason": self._terminal_reason,
                "agent_mode": AGENT_MODE,
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

    def _append_strategy_log(self, player_idx: int, strategy: dict[str, Any], moves: list[Move], diagnostics: list[Any]) -> None:
        plan = str(strategy.get("round_plan") or "执行默认行动。")
        reasoning = str(strategy.get("reasoning") or "")
        stance = str(strategy.get("stance", "?"))
        n_obj = len(strategy.get("objectives") or [])
        n_dir = len(strategy.get("direct_orders") or [])
        cons = strategy.get("constraints") or {}
        garrison = cons.get("min_general_garrison", "?")
        if moves:
            move_text = "，".join(
                f"({move.from_y},{move.from_x})->({move.to_y},{move.to_x})"
                for move in moves[:3]
            )
            if len(moves) > 3:
                move_text += f" 等 {len(moves)} 步"
            total_army = sum(abs(m.from_x - m.to_x) + abs(m.from_y - m.to_y) + 1 for m in moves[:3])
        else:
            move_text = "\u7a7a"
        label = f"{plan}"
        if reasoning and reasoning not in plan:
            label += f" | {reasoning[:120]}"
        diag_text = "；".join(
            f"{getattr(diag, 'objective_type', '?')}:{getattr(diag, 'reason', '?')}"
            for diag in diagnostics[:3]
        )
        self._append_log(player_idx, f"[{stance}] {label} / 首半回合 {len(moves)}\u52a8 {n_obj}\u76ee\u6807 {n_dir}\u76f4\u4ee4 g{garrison} / {move_text} / {diag_text}")

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
            self._send_json(SERVICE.current_state())
            return
        if path == "/api/status":
            self._send_json(SERVICE.status())
            return
        if path == "/api/rounds":
            self._send_json(SERVICE.rounds())
            return
        if path == "/api/metrics":
            self._send_json(SERVICE.metrics())
            return
        if path in ("", "/"):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/advance":
            target = self._read_target()
            payload = SERVICE.advance(target)
            for _ in range(30):
                if payload.get("ready") or payload.get("finished") or payload.get("error"):
                    break
                time.sleep(0.1)
                payload = SERVICE.advance(target)
            self._send_json(payload)
            return
        if path == "/api/reset":
            self._send_json(SERVICE.reset())
            return
        if path == "/api/pause":
            self._send_json(SERVICE.pause())
            return
        if path == "/api/resume":
            self._send_json(SERVICE.resume())
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_target(self) -> int | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return None
        try:
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body) if body else {}
            if "target" not in data or data["target"] is None:
                return None
            return int(data["target"])
        except (ValueError, TypeError, json.JSONDecodeError):
            return None


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
